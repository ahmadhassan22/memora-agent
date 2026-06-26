"""
Conflict Detection for Memora.
Checks whether a new fact contradicts an existing memory.
Uses an ADAPTIVE relevance gate instead of a single hardcoded threshold.
Detection only — resolution comes next.
"""

import json
from app.agent.llm_client import chat
from app.memory.store import search_memory, save_memory, delete_memory


# Absolute safety ceiling: beyond this, memories are clearly unrelated.
# Acts as an outer bound so the adaptive logic never checks nonsense.
ABSOLUTE_CEILING = 1.4

# How much closer the nearest memory must be vs. the others (relative gap).
# If nearest is at least this fraction closer than the average of the rest,
# it "stands out" as related and is worth checking.
RELATIVE_GAP = 0.85


CONFLICT_PROMPT = """You compare two facts about a user and decide their relationship.

EXISTING fact: "{existing}"
NEW fact: "{new}"

Decide the relationship:
- "contradict": the new fact conflicts with the existing one; the existing is now outdated (e.g. "is vegetarian" vs "started eating chicken"). This includes temporal changes where past tense negates a present belief (e.g. "lives in Beijing" vs "used to live in Beijing" — no longer lives there)
- "duplicate": they say essentially the same thing (e.g. "is vegetarian" vs "does not eat meat")
- "compatible": both can be true at once; no conflict (e.g. "works in Shenzhen" vs "lives in Shenzhen")

Return ONLY valid JSON, no extra text:
{{"relationship": "contradict" | "duplicate" | "compatible", "reason": "short explanation"}}"""


def _is_relevant(candidates):
    """
    Adaptive gate: is the nearest memory relevant enough to check?
    Two checks combined:
      1. Absolute ceiling — nearest must not be too far away to matter at all.
      2. Relative gap — nearest must stand out as meaningfully closer than
         the average of the other candidates. A memory that's only barely
         closer than everything else isn't specifically related — it's
         just sitting in a generally similar topic area.
    Qwen makes the real contradiction decision; this only filters out
    memories that clearly aren't worth checking.
    """
    nearest = candidates[0]["distance"]

    if nearest > ABSOLUTE_CEILING:
        return False

    rest = candidates[1:]
    if not rest:
        # Nothing to compare against — absolute ceiling is the only signal.
        return True

    avg_rest = sum(c["distance"] for c in rest) / len(rest)
    if avg_rest == 0:
        return True

    return nearest <= RELATIVE_GAP * avg_rest


def detect_conflict(user_id, new_fact_text):
    """
    Check if a new fact conflicts with existing memories.
    Returns a dict:
      {
        "status": "contradict" | "duplicate" | "compatible" | "none",
        "existing_memory": {...} or None,
        "reason": "..."
      }
    """
    # Pull several neighbors so the adaptive gate has data to compare.
    similar = search_memory(user_id, new_fact_text, top_k=5)

    # No memories at all → brand new fact.
    if not similar:
        return {"status": "none", "existing_memory": None, "reason": "No memories exist yet"}

    # Adaptive relevance gate.
    if not _is_relevant(similar):
        return {"status": "none", "existing_memory": None, "reason": "No memory stood out as related"}

    existing = similar[0]

    # Ask Qwen to judge the relationship.
    prompt = CONFLICT_PROMPT.format(existing=existing["text"], new=new_fact_text)
    raw = chat([{"role": "user", "content": prompt}], temperature=0).strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    # Parse safely; on bad output, fail safe to "compatible" (never destroy data).
    try:
        judgment = json.loads(raw)
        relationship = judgment.get("relationship", "compatible")
        reason = judgment.get("reason", "")
    except json.JSONDecodeError:
        return {"status": "compatible", "existing_memory": existing, "reason": "Could not parse judgment"}

    if relationship not in {"contradict", "duplicate", "compatible"}:
        relationship = "compatible"

    return {"status": relationship, "existing_memory": existing, "reason": reason}

def resolve_fact(user_id, fact):
    """
    Detect any conflict for a new fact and act on it.
      user_id: whose memory
      fact: a dict {"text":..., "type":..., "importance":...} from the extractor
    Returns a dict describing what happened:
      {"action": "added" | "updated" | "skipped", "detail": "...", "memory_id": ...}
    """
    text = fact["text"]
    mem_type = fact.get("type", "other")
    importance = fact.get("importance", 5)

    # Run detection
    result = detect_conflict(user_id, text)
    status = result["status"]
    existing = result["existing_memory"]

    if status == "contradict":
        # Outdated belief → remove old, save new
        delete_memory(existing["id"])
        new_id = save_memory(user_id, text, mem_type, importance)
        return {
            "action": "updated",
            "detail": f"Replaced outdated memory: '{existing['text']}' → '{text}'",
            "memory_id": new_id,
        }

    if status == "duplicate":
        # Already known → don't store again
        return {
            "action": "skipped",
            "detail": f"Duplicate of existing memory: '{existing['text']}'",
            "memory_id": existing["id"],
        }

    # compatible or none → save as a new memory
    new_id = save_memory(user_id, text, mem_type, importance)
    return {
        "action": "added",
        "detail": "Saved as new memory",
        "memory_id": new_id,
    }
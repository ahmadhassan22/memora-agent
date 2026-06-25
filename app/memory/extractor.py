"""
Memory Extractor for Memora.
Reads a user message and pulls out only memory-worthy facts as clean JSON.
Filler, chitchat, and temporary statements are ignored.
Includes validation so only safe, well-formed facts pass through.
"""

import json
from app.agent.llm_client import chat


# Allowed fact types — anything else gets normalized to "other"
VALID_TYPES = {"preference", "personal", "goal", "decision", "other"}


EXTRACTION_PROMPT = """You are a memory extraction system. Read the user's message and extract ONLY durable, memory-worthy facts about the user.

EXTRACT facts like:
- Preferences (likes, dislikes, dietary choices)
- Personal details (name, job, location, relationships)
- Goals and plans
- Important decisions or changes

IGNORE:
- Greetings, small talk, filler ("how are you", "thanks", "lol")
- Temporary states ("I'm tired right now")
- Questions the user asks
- Anything not durably true about the user

For each fact, assign:
- "text": the fact, rewritten as a clean third-person statement (e.g. "User is vegetarian")
- "type": one of [preference, personal, goal, decision, other]
- "importance": integer 1-10 (10 = critical identity info, 1 = minor)

Return ONLY a valid JSON array. No explanation, no markdown, no extra text.
If there are NO memory-worthy facts, return an empty array: []

Example output:
[{"text": "User is going vegan starting Monday", "type": "decision", "importance": 7}]"""


def _clean_fact(fact):
    """
    Validate and clean a single fact dict.
    Returns a safe fact dict, or None if it's unusable.
    """
    # Must be a dictionary
    if not isinstance(fact, dict):
        return None

    # Must have non-empty text
    text = fact.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    text = text.strip()

    # Normalize type — default to "other" if missing/invalid
    mem_type = fact.get("type")
    if mem_type not in VALID_TYPES:
        mem_type = "other"

    # Normalize importance — must be int 1-10, else default to 5
    importance = fact.get("importance")
    if not isinstance(importance, int):
        importance = 5
    importance = max(1, min(10, importance))   # clamp into 1-10 range

    return {"text": text, "type": mem_type, "importance": importance}


def extract_facts(message):
    """
    Extract memory-worthy facts from a user message.
      message: raw user text (string)
    Returns: a list of clean, validated fact dicts, or [] if nothing usable.
    """
    # FIX #4: skip empty/whitespace messages — don't waste an API call
    if not message or not message.strip():
        return []

    messages = [
        {"role": "system", "content": EXTRACTION_PROMPT},
        {"role": "user", "content": message},
    ]

    raw = chat(messages, temperature=0)

    # Strip markdown code fences if the model added them
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    # Safely parse JSON; bad output returns nothing instead of crashing
    try:
        facts = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if not isinstance(facts, list):
        return []

    # FIX #2 & #3: clean every fact, drop unusable ones
    clean_facts = []
    for fact in facts:
        cleaned = _clean_fact(fact)
        if cleaned is not None:
            clean_facts.append(cleaned)

    return clean_facts
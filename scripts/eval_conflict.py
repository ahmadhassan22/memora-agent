"""
Conflict-resolution evaluation for Memora.

Question this answers, with a NUMBER instead of a claim:
  When new information contradicts, duplicates, or coexists with an existing
  memory, does Memora end up in the CORRECT belief state more often than a
  naive append-only memory (what most basic RAG-over-history agents do)?

Two systems, same scenarios:
  - MEMORA  : the real resolve_fact() pipeline (detect conflict -> update/
              skip/add). Uses live Qwen for the contradiction judgment.
  - NAIVE   : naive_resolve() below — always just saves the new fact. No
              conflict check. This is a fair, honest baseline: it's exactly
              what an append-only memory does.

Three scenario categories (so the score isn't cherry-picked):
  1. CONTRADICT  — new fact makes the old one outdated.
                   Correct end state: NEW present, OLD gone.
  2. DUPLICATE   — new fact restates the old one.
                   Correct end state: exactly ONE memory (no redundant copy).
  3. COMPATIBLE  — both facts are true at once.
                   Correct end state: BOTH present.

Note on COMPATIBLE: the naive baseline also keeps both here, so it can score
as well as Memora on this category. That's intentional and honest — we don't
rig the baseline to lose. Memora's advantage shows up in CONTRADICT/DUPLICATE.

This calls the real Qwen API (needs your key + network). Run from project root:
    python -m scripts.eval_conflict
"""

import time

from app.memory.store import collection, save_memory
from app.memory.conflict import resolve_fact


# ---- The honest baseline: append-only, no conflict handling ----

def naive_resolve(user_id, fact):
    """What a basic memory agent does: just store everything, always."""
    save_memory(user_id, fact["text"], fact.get("type", "other"), fact.get("importance", 5))
    return {"action": "added"}


# ---- Labeled test scenarios ----
# Each: an OLD fact to pre-seed, a NEW fact to process, and the category.

SCENARIOS = [
    # 1. CONTRADICTIONS — Memora should UPDATE (old gone, new present)
    ("contradict", "User is vegetarian", "User has started eating chicken"),
    ("contradict", "User lives in Beijing", "User moved to Shanghai last month"),
    ("contradict", "User works at a startup", "User now works at a large bank"),
    ("contradict", "User's favorite language is Python", "User now prefers Rust over everything"),
    ("contradict", "User is single", "User got married recently"),
    ("contradict", "User drives a Toyota", "User sold the Toyota and bought a Tesla"),

    # 2. DUPLICATES — Memora should SKIP (stay at exactly one memory)
    ("duplicate", "User is vegetarian", "User does not eat meat"),
    ("duplicate", "User is allergic to shellfish", "User cannot eat shellfish due to allergy"),
    ("duplicate", "User lives in Shenzhen", "User's home is in Shenzhen"),
    ("duplicate", "User is a software engineer", "User works as a software developer"),
    ("duplicate", "User enjoys hiking", "User loves to go hiking"),
    ("duplicate", "User speaks English", "User is an English speaker"),

    # 3. COMPATIBLE — Memora should ADD alongside (both present)
    ("compatible", "User works in Shenzhen", "User lives in Shenzhen"),
    ("compatible", "User likes coffee", "User also enjoys tea"),
    ("compatible", "User has a dog", "User has a cat"),
    ("compatible", "User plays guitar", "User is learning piano"),
    ("compatible", "User is allergic to peanuts", "User enjoys swimming"),
    ("compatible", "User studies AI", "User is based in China"),
]


def _wipe(user_id):
    """Remove all memories for a user so each scenario starts clean."""
    existing = collection.get(where={"user_id": user_id})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])


def _count(user_id):
    return len(collection.get(where={"user_id": user_id})["ids"])


def _texts(user_id):
    return collection.get(where={"user_id": user_id})["documents"]


def _is_correct(category, user_id, old_text, new_text):
    """Judge whether the end state is correct for this category."""
    texts = _texts(user_id)
    count = len(texts)

    if category == "contradict":
        # New belief present, old belief gone.
        return (new_text in texts) and (old_text not in texts)

    if category == "duplicate":
        # No redundant copy: exactly one memory remains.
        return count == 1

    if category == "compatible":
        # Both facts kept.
        return count == 2

    return False


def run_system(label, resolver):
    """Run all scenarios through one resolver; return per-category results."""
    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    results = {"contradict": [0, 0], "duplicate": [0, 0], "compatible": [0, 0]}  # [correct, total]

    for i, (category, old_text, new_text) in enumerate(SCENARIOS):
        user_id = f"eval_{label.lower()}_{i}"
        _wipe(user_id)

        # Pre-seed the OLD belief.
        save_memory(user_id, old_text, "other", 6)

        # Process the NEW statement through the system under test.
        resolver(user_id, {"text": new_text, "type": "other", "importance": 6})

        correct = _is_correct(category, user_id, old_text, new_text)
        results[category][0] += int(correct)
        results[category][1] += 1

        mark = "OK " if correct else "XX "
        print(f"  {mark}[{category:11}] '{old_text}' + '{new_text}' -> {_count(user_id)} mem(s)")

        _wipe(user_id)  # clean up after scoring

    return results


def _summary(label, results):
    total_correct = sum(c for c, _ in results.values())
    total = sum(t for _, t in results.values())
    print(f"\n  {label} results:")
    for cat, (c, t) in results.items():
        print(f"    {cat:11}: {c}/{t}")
    print(f"    {'OVERALL':11}: {total_correct}/{total}  ({100*total_correct/total:.0f}%)")
    return total_correct, total


def main():
    start = time.time()
    print("Memora conflict-resolution evaluation")
    print(f"{len(SCENARIOS)} scenarios x 2 systems = {len(SCENARIOS)*2} runs (live Qwen)\n")

    memora_results = run_system("MEMORA", resolve_fact)
    naive_results = run_system("NAIVE", naive_resolve)

    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
    m_correct, total = _summary("MEMORA", memora_results)
    n_correct, _ = _summary("NAIVE", naive_results)

    print(f"\n  Memora correctly handled {m_correct}/{total} belief states.")
    print(f"  Naive baseline handled    {n_correct}/{total}.")
    gain = m_correct - n_correct
    print(f"  Improvement: +{gain} scenarios ({100*gain/total:.0f} percentage points).")
    print(f"\n  Completed in {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()
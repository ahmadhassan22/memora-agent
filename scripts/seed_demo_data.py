"""
Seed demo data for Memora — makes the DECAY differentiator visibly demoable.

Problem this solves:
  Memories created live during a demo are fresh and high-ish importance, so
  the decay pass never prunes anything on camera — there's nothing stale yet.
  Decay also (correctly) protects memories that are important OR recently used.

  So to *show* decay working, we need to plant a memory that is:
    - low importance        (importance dominates the keep-score)
    - old                   (created long ago)
    - never re-accessed      (last_accessed = creation time, not bumped)
  ...which forces its relevance score below decay's RELEVANCE_FLOOR.

  We ALSO plant two memories that SHOULD survive, so the demo shows decay is
  selective, not indiscriminate:
    - a high-importance memory that is equally old   -> survives via importance
    - a low-importance but fresh memory              -> survives via recency

Run from the project root (so 'app' imports resolve):
    python -m scripts.seed_demo_data
Optionally pass a user id (defaults to 'demo_user' to match the UI):
    python -m scripts.seed_demo_data demo_user

After running, open the UI for that user, look at the stored memories, then
click "Run Decay" — the stale low-value memory should disappear while the
others remain.
"""

import sys
import time

from app.memory.store import collection
from app.agent.llm_client import get_embedding


DAY = 24 * 60 * 60  # one day in seconds


def _insert(user_id, text, mem_type, importance, age_days, since_access_days):
    """
    Insert one memory with CONTROLLED timestamps.

    We write to the Chroma collection directly instead of using save_memory(),
    because save_memory() always stamps created_at / last_accessed with the
    current time — and this script's whole point is to back-date them.

      age_days:           how long ago the memory was created
      since_access_days:  how long ago it was last accessed
                          (>= age_days means "never accessed after creation")
    """
    now = time.time()
    created_at = now - (age_days * DAY)
    last_accessed = now - (since_access_days * DAY)

    memory_id = f"seed-{user_id}-{int(now*1000)}-{importance}"
    vector = get_embedding(text)  # real embedding so search still works on it

    collection.add(
        ids=[memory_id],
        embeddings=[vector],
        documents=[text],
        metadatas=[{
            "user_id": user_id,
            "mem_type": mem_type,
            "importance": importance,
            "created_at": created_at,
            "last_accessed": last_accessed,
        }],
    )
    return memory_id, text


def seed(user_id="demo_user"):
    """Plant the demo memories for a given user."""
    print(f"Seeding demo memories for user '{user_id}'...\n")

    planted = []

    # 1) The STALE one — designed to be pruned by decay.
    #    importance=2, 120 days old, never accessed since.
    #    Relevance score works out to ~0.13, well under the 0.35 floor.
    planted.append(_insert(
        user_id,
        text="User briefly tried a budgeting app called PennyTrack last year",
        mem_type="other",
        importance=2,
        age_days=120,
        since_access_days=120,
    ))

    # 2) SURVIVOR via importance — equally old, but critical.
    #    importance=9, 120 days old. Score ~0.48 -> survives.
    planted.append(_insert(
        user_id,
        text="User is allergic to shellfish",
        mem_type="personal",
        importance=9,
        age_days=120,
        since_access_days=120,
    ))

    # 3) SURVIVOR via freshness — low importance but brand new.
    #    importance=4, created just now. Score ~0.70 -> survives.
    planted.append(_insert(
        user_id,
        text="User is currently reading a book about habit formation",
        mem_type="other",
        importance=4,
        age_days=0,
        since_access_days=0,
    ))

    print("Planted memories:")
    for _id, text in planted:
        print(f"  - {text}")

    print(
        "\nDone. Open the UI for this user, then click 'Run Decay'.\n"
        "Expected: the PennyTrack memory is forgotten (stale + low importance),\n"
        "while the shellfish allergy (important) and the habit book (fresh) remain."
    )


if __name__ == "__main__":
    uid = sys.argv[1] if len(sys.argv) > 1 else "demo_user"
    seed(uid)
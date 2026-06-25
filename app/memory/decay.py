"""
Memory Decay for Memora — differentiator #2 (forgetting).
Scores memories by relevance and prunes the stale, low-value ones.
High-importance memories resist decay.
"""

import time
from app.memory.store import collection


# A memory scoring below this relevance is considered forgettable.
RELEVANCE_FLOOR = 0.35

# How fast memories age. Larger = slower decay (memories last longer).
# Expressed in seconds; here ~30 days as the reference half-life.
AGE_HALFLIFE_SECONDS = 30 * 24 * 60 * 60


def _relevance_score(metadata, now):
    """
    Compute a 0-1 relevance score for one memory.
    Combines: importance (1-10) + recency of last access + age.
      metadata: the memory's stored metadata
      now: current timestamp
    Returns a float (higher = more worth keeping).
    """
    importance = metadata.get("importance", 5) / 10.0        # normalize to 0-1
    last_accessed = metadata.get("last_accessed", now)
    created_at = metadata.get("created_at", now)

    # Recency: how long since last accessed, decayed over the half-life.
    seconds_since_access = now - last_accessed
    recency = 0.5 ** (seconds_since_access / AGE_HALFLIFE_SECONDS)

    # Age: older memories decay slightly, also over the half-life.
    seconds_old = now - created_at
    freshness = 0.5 ** (seconds_old / AGE_HALFLIFE_SECONDS)

    # Weighted blend. Importance dominates so critical facts survive.
    score = (0.5 * importance) + (0.3 * recency) + (0.2 * freshness)
    return score


def run_decay(user_id):
    """
    Score all of a user's memories and prune the forgettable ones.
      user_id: whose memories to process
    Returns a summary dict:
      {"checked": N, "pruned": M, "pruned_texts": [...]}
    """
    now = time.time()

    # Fetch all this user's memories
    data = collection.get(where={"user_id": user_id})

    ids = data["ids"]
    docs = data["documents"]
    metas = data["metadatas"]

    pruned_ids = []
    pruned_texts = []

    for i in range(len(ids)):
        score = _relevance_score(metas[i], now)
        if score < RELEVANCE_FLOOR:
            pruned_ids.append(ids[i])
            pruned_texts.append(docs[i])

    # Delete all forgettable memories at once
    if pruned_ids:
        collection.delete(ids=pruned_ids)

    return {
        "checked": len(ids),
        "pruned": len(pruned_ids),
        "pruned_texts": pruned_texts,
    }
"""
Memory Retriever for Memora — differentiator #3 (smart recall).
Two-stage: vector search -> custom score -> qwen3-rerank final ordering.
Returns the best memories within a limited context budget.
"""

import time
from openai import OpenAI
from app.config import settings
from app.memory.store import search_memory, touch_memory


# Half-life for recency scoring (matches decay.py's time-based logic)
RECENCY_HALFLIFE_SECONDS = 30 * 24 * 60 * 60

# Separate client for rerank — note the DIFFERENT base URL
# (compatible-api, not compatible-mode used for chat/embeddings)
_rerank_client = OpenAI(
    api_key=settings.QWEN_API_KEY,
    base_url="https://dashscope-intl.aliyuncs.com/compatible-api/v1",
)


def _custom_score(memory, now):
    """
    Combine three signals into one relevance score for ranking.
      memory: a result dict from search_memory (has distance + metadata)
      now: current timestamp
    Returns a float (higher = more relevant, should be recalled first).
    """
    # Vector similarity: distance is lower-is-better, so invert it.
    similarity = 1.0 / (1.0 + memory["distance"])

    importance = memory["metadata"].get("importance", 5) / 10.0

    last_accessed = memory["metadata"].get("last_accessed", now)
    seconds_since = now - last_accessed
    recency = 0.5 ** (seconds_since / RECENCY_HALFLIFE_SECONDS)

    # Weighted blend: similarity matters most (it's WHY it was found at all),
    # importance and recency adjust the ordering among similar candidates.
    return (0.6 * similarity) + (0.25 * importance) + (0.15 * recency)


def _rerank(query, memories, top_n):
    """
    Use qwen3-rerank to reorder memories by true relevance to the query.
      query: the search text
      memories: list of memory dicts (each has 'text')
      top_n: how many to return
    Returns: reordered list of memory dicts (best-first).
    Falls back to the input order if the rerank call fails.
    """
    documents = [m["text"] for m in memories]

    try:
        response = _rerank_client.post(
            "/reranks",
            body={
                "model": settings.RERANK_MODEL,
                "query": query,
                "documents": documents,
                "top_n": top_n,
            },
            cast_to=object,
        )

        # Response returns ranked results with original indices.
        # Map them back to the full memory dicts (which carry metadata).
        reranked = []
        for item in response["results"]:
            idx = item["index"]
            reranked.append(memories[idx])

        print(f"[rerank] qwen3-rerank succeeded — reordered {len(reranked)} memories")
        return reranked

    except Exception as e:
        # If rerank fails, fall back to the custom-score order we already have.
        print(f"[rerank] qwen3-rerank FAILED, falling back to custom-score order: {e}")
        return memories[:top_n]


def retrieve(user_id, query, top_k=3, candidate_pool=8):
    """
    Get the best memories for a query, within a limited budget.
      user_id: whose memory to search
      query: the question/context to retrieve for
      top_k: how many memories to actually return (the context budget)
      candidate_pool: how many to fetch before reranking (wider net)
    Returns: list of memory dicts, best-first, length <= top_k.
    """
    candidates = search_memory(user_id, query, top_k=candidate_pool)

    if not candidates:
        return []

    now = time.time()

    # Stage 1: score every candidate with the custom blend, sort best-first.
    for memory in candidates:
        memory["relevance_score"] = _custom_score(memory, now)

    candidates.sort(key=lambda m: m["relevance_score"], reverse=True)

    # Stage 2: take the top custom-scored candidates, then let qwen3-rerank
    # do the final precise ordering for maximum recall accuracy.
    top_candidates = candidates[:max(top_k, 5)]
    final = _rerank(query, top_candidates, top_k)

    # Returning a memory counts as accessing it — bump last_accessed so the
    # recency signal in scoring/decay reflects real use, not just age.
    touch_memory([m["id"] for m in final])

    return final
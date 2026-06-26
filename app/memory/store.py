"""
Memory Store for Memora — ChromaDB vector storage.
Saves memories as vectors + text + metadata, searchable by meaning.
"""

import time
import uuid
import chromadb
from app.agent.llm_client import get_embedding

# Persistent ChromaDB client — saves to disk so memories survive restarts
chroma_client = chromadb.PersistentClient(path="data/chroma_db")

# A "collection" is like a table. get_or_create = reuse if exists, else make it.
collection = chroma_client.get_or_create_collection(name="memories")


def save_memory(user_id, text, mem_type="fact", importance=5):
    """
    Save one memory.
      user_id: who this memory belongs to
      text: the fact, e.g. "User is vegetarian"
      mem_type: category (fact, preference, event...)
      importance: 1-10, used later for ranking/decay
    Returns: the new memory's id
    """
    memory_id = str(uuid.uuid4())          # unique id for this memory
    vector = get_embedding(text)           # 1024-number meaning vector

    collection.add(
        ids=[memory_id],
        embeddings=[vector],
        documents=[text],
        metadatas=[{
            "user_id": user_id,
            "mem_type": mem_type,
            "importance": importance,
            "created_at": time.time(),     # timestamp (for decay later)
            "last_accessed": time.time(),
        }],
    )
    return memory_id

def search_memory(user_id, query, top_k=3):
    """
    Find the most meaning-similar memories for a query.
      user_id: only search this user's memories
      query: what we're looking for, e.g. "what does the user eat?"
      top_k: how many results to return
    Returns: list of dicts with id + text + metadata + distance
    """
    query_vector = get_embedding(query)    # embed the query

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        where={"user_id": user_id},        # filter: only this user's memories
    )

    # Repackage Chroma's raw output into a clean list
    memories = []
    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    for i in range(len(docs)):
        memories.append({
            "id": ids[i],
            "text": docs[i],
            "metadata": metas[i],
            "distance": dists[i],          # lower = more similar
        })

    return memories

def touch_memory(ids):
    """
    Mark one or more memories as accessed right now.
    Updates each memory's 'last_accessed' timestamp to the current time,
    while preserving all other metadata. Called when memories are actually
    retrieved, so the 'recency' signal in scoring/decay reflects real use
    instead of just age.
      ids: a list of memory ids to touch
    """
    if not ids:
        return

    existing = collection.get(ids=ids)
    if not existing["ids"]:
        return

    now = time.time()
    new_metas = []
    for meta in existing["metadatas"]:
        updated = dict(meta)            # copy so we keep importance, created_at, etc.
        updated["last_accessed"] = now
        new_metas.append(updated)

    collection.update(ids=existing["ids"], metadatas=new_metas)


def delete_memory(memory_id):
    """
    Delete one memory by its id.
    Used by conflict resolution to remove an outdated belief.
    """
    collection.delete(ids=[memory_id])


def get_memory_by_id(memory_id):
    """
    Fetch a single memory by its id.
    Returns a dict with text + metadata, or None if not found.
    """
    result = collection.get(ids=[memory_id])

    if not result["ids"]:
        return None

    return {
        "id": result["ids"][0],
        "text": result["documents"][0],
        "metadata": result["metadatas"][0],
    }
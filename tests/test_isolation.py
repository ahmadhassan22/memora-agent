"""
Multi-user isolation tests.

The single most important safety property: one user must NEVER be able to
see, retrieve, or accidentally delete another user's memories. If this leaks,
it's a credibility-killing bug. These tests prove the storage layer enforces
per-user separation.
"""


def test_search_only_returns_own_memories(store):
    """User A's search must never surface User B's memories."""
    store.save_memory("alice", "Alice is vegetarian", "personal", 8)
    store.save_memory("bob", "Bob loves steak", "personal", 8)

    # Alice searches for something close to Bob's memory.
    results = store.search_memory("alice", "what does the user eat", top_k=5)

    texts = [r["text"] for r in results]
    assert "Alice is vegetarian" in texts
    assert "Bob loves steak" not in texts
    # Every result must belong to alice.
    assert all(r["metadata"]["user_id"] == "alice" for r in results)


def test_users_have_independent_memory_sets(store):
    """Counts per user must be independent."""
    store.save_memory("alice", "Alice fact 1", "other", 5)
    store.save_memory("alice", "Alice fact 2", "other", 5)
    store.save_memory("bob", "Bob fact 1", "other", 5)

    alice_mems = store.collection.get(where={"user_id": "alice"})
    bob_mems = store.collection.get(where={"user_id": "bob"})

    assert len(alice_mems["ids"]) == 2
    assert len(bob_mems["ids"]) == 1


def test_empty_user_returns_nothing(store):
    """A user with no memories returns an empty result, not someone else's."""
    store.save_memory("alice", "Alice fact", "other", 5)

    results = store.search_memory("charlie", "anything at all", top_k=5)
    assert results == []


def test_touch_does_not_cross_users(store):
    """Touching Alice's memory must not alter Bob's."""
    a_id = store.save_memory("alice", "Alice memory", "other", 5)
    b_id = store.save_memory("bob", "Bob memory", "other", 5)

    bob_before = store.get_memory_by_id(b_id)["metadata"]["last_accessed"]

    store.touch_memory([a_id])

    bob_after = store.get_memory_by_id(b_id)["metadata"]["last_accessed"]
    assert bob_before == bob_after  # Bob untouched
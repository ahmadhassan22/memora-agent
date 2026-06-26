"""
Conflict resolution logic tests (differentiator #1).

We do NOT test whether Qwen correctly *classifies* a contradiction here —
that's an LLM behavior, covered by the manual checklist. And we do NOT rely on
embedding distances (the fake test embeddings aren't semantic). Instead we
patch detect_conflict's inputs to feed a controlled judgment, then assert that
resolve_fact does the right STRUCTURAL thing:
  - "contradict" -> old memory deleted, new one saved (UPDATED)
  - "duplicate"  -> nothing saved, existing kept (SKIPPED)
  - "compatible" -> new memory saved alongside (ADDED)
We also test the adaptive relevance gate math directly.
"""

import app.memory.conflict as conflict


def _wire_store(store, monkeypatch):
    """Point conflict.py's storage calls at the in-memory test store."""
    monkeypatch.setattr(conflict, "save_memory", store.save_memory)
    monkeypatch.setattr(conflict, "delete_memory", store.delete_memory)


def _force_detection(monkeypatch, status, existing):
    """Make detect_conflict return a fixed verdict, bypassing embeddings/LLM."""
    monkeypatch.setattr(
        conflict, "detect_conflict",
        lambda user_id, text: {"status": status, "existing_memory": existing, "reason": "test"},
    )


def test_contradict_replaces_old_memory(store, monkeypatch):
    _wire_store(store, monkeypatch)
    old_id = store.save_memory("alice", "User is vegetarian", "preference", 7)
    existing = {"id": old_id, "text": "User is vegetarian"}
    _force_detection(monkeypatch, "contradict", existing)

    result = conflict.resolve_fact(
        "alice", {"text": "User started eating chicken", "type": "preference", "importance": 7}
    )

    assert result["action"] == "updated"
    remaining = store.collection.get(where={"user_id": "alice"})
    assert old_id not in remaining["ids"]            # old deleted
    assert len(remaining["ids"]) == 1                # exactly one left
    assert remaining["documents"][0] == "User started eating chicken"


def test_duplicate_is_skipped(store, monkeypatch):
    _wire_store(store, monkeypatch)
    old_id = store.save_memory("alice", "User is vegetarian", "preference", 7)
    existing = {"id": old_id, "text": "User is vegetarian"}
    _force_detection(monkeypatch, "duplicate", existing)

    result = conflict.resolve_fact(
        "alice", {"text": "User does not eat meat", "type": "preference", "importance": 7}
    )

    assert result["action"] == "skipped"
    remaining = store.collection.get(where={"user_id": "alice"})
    assert len(remaining["ids"]) == 1                # nothing new added


def test_compatible_adds_alongside(store, monkeypatch):
    _wire_store(store, monkeypatch)
    old_id = store.save_memory("alice", "User works in Shenzhen", "personal", 7)
    existing = {"id": old_id, "text": "User works in Shenzhen"}
    _force_detection(monkeypatch, "compatible", existing)

    result = conflict.resolve_fact(
        "alice", {"text": "User lives in Shenzhen", "type": "personal", "importance": 7}
    )

    assert result["action"] == "added"
    remaining = store.collection.get(where={"user_id": "alice"})
    assert len(remaining["ids"]) == 2                # both kept


def test_first_fact_is_added_when_no_memories(store, monkeypatch):
    _wire_store(store, monkeypatch)
    _force_detection(monkeypatch, "none", None)

    result = conflict.resolve_fact(
        "alice", {"text": "User is a photographer", "type": "personal", "importance": 6}
    )
    assert result["action"] == "added"
    remaining = store.collection.get(where={"user_id": "alice"})
    assert len(remaining["ids"]) == 1


# ---- Adaptive relevance gate math (pure logic, no store/LLM) ----

def test_gate_passes_clear_standout():
    candidates = [{"distance": 0.3}, {"distance": 1.1}, {"distance": 1.2}, {"distance": 1.3}]
    assert conflict._is_relevant(candidates) is True


def test_gate_rejects_flat_neighborhood():
    candidates = [{"distance": 1.0}, {"distance": 1.05}, {"distance": 1.08}, {"distance": 1.1}]
    assert conflict._is_relevant(candidates) is False


def test_gate_rejects_past_ceiling():
    candidates = [{"distance": 1.5}, {"distance": 1.6}]
    assert conflict._is_relevant(candidates) is False


def test_gate_single_candidate_under_ceiling():
    assert conflict._is_relevant([{"distance": 0.9}]) is True
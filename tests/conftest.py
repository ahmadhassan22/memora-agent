"""
Shared test setup for Memora.

Two jobs:
  1. Replace the real Qwen embedding call with a fast, deterministic fake,
     so logic tests run with no API key, no VPN, no token cost.
  2. Give each test a fresh IN-MEMORY ChromaDB collection, so tests never
     touch the real on-disk data and never bleed into each other.

The fake embedding is deterministic: the same text always maps to the same
vector, and different texts map to different vectors — enough for the storage
and isolation logic to behave realistically without a real model.
"""

import uuid
import hashlib

import pytest
import chromadb


def _fake_embedding(text):
    """Map text -> a stable 16-dim vector derived from its hash."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return [b / 255.0 for b in h[:16]]


@pytest.fixture
def store(monkeypatch):
    """
    Provide a clean in-memory store module for one test.
    Each test gets a UNIQUELY NAMED collection so state never leaks
    between tests.
    """
    import app.agent.llm_client as llm_client
    import app.memory.store as store_module

    monkeypatch.setattr(llm_client, "get_embedding", _fake_embedding)
    monkeypatch.setattr(store_module, "get_embedding", _fake_embedding)

    client = chromadb.EphemeralClient()
    # Unique name per test prevents cross-test state reuse.
    name = "mem_test_" + uuid.uuid4().hex[:8]
    test_collection = client.get_or_create_collection(name=name)
    monkeypatch.setattr(store_module, "collection", test_collection)

    return store_module
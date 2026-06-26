"""
FastAPI service for Memora.
Exposes the memory agent over HTTP so it can be deployed and used by a UI.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.agent.agent_loop import process_message
from app.memory.store import collection
from app.memory.decay import run_decay

app = FastAPI(title="Memora", description="Self-Managing Memory Agent")


# ---- Request body shapes (Pydantic validates incoming JSON) ----

class ChatRequest(BaseModel):
    user_id: str
    message: str


# ---- Endpoints ----

@app.get("/health")
def health():
    """Simple liveness check — deployment platforms ping this."""
    return {"status": "ok", "service": "memora"}


@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    """
    Main endpoint: process a user message through the full agent pipeline.
    Returns the reply, memory actions taken, and memories used.
    On failure (e.g. the upstream Qwen API is unreachable), returns a clean
    error instead of a raw stack trace.
    """
    # Guard: empty/whitespace messages have nothing to process.
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        return process_message(req.user_id, req.message)
    except Exception as e:
        # Most failures here are upstream (Qwen API timeout, rate limit,
        # network/VPN blip), so 502 is more accurate than a generic 500.
        # Still print the real error so it's debuggable from the terminal.
        print(f"[chat] pipeline error: {e}")
        raise HTTPException(status_code=502, detail=f"Memory pipeline failed: {e}")


@app.get("/memories/{user_id}")
def list_memories(user_id: str):
    """Return all stored memories for a user (used by the UI to display state)."""
    try:
        data = collection.get(where={"user_id": user_id})

        memories = []
        for i in range(len(data["ids"])):
            memories.append({
                "id": data["ids"][i],
                "text": data["documents"][i],
                "metadata": data["metadatas"][i],
            })

        return {"user_id": user_id, "count": len(memories), "memories": memories}
    except Exception as e:
        print(f"[memories] read error: {e}")
        raise HTTPException(status_code=500, detail=f"Could not read memories: {e}")


@app.post("/decay/{user_id}")
def decay_endpoint(user_id: str):
    """Trigger memory decay (forgetting) for a user. Used in the demo."""
    try:
        return run_decay(user_id)
    except Exception as e:
        print(f"[decay] error: {e}")
        raise HTTPException(status_code=500, detail=f"Decay failed: {e}")
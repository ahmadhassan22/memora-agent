"""
Agent Loop for Memora — the conductor.
Ties extraction, conflict resolution, retrieval, and response generation
into one end-to-end flow. This is what makes Memora a working agent.
"""

from app.agent.llm_client import chat
from app.memory.extractor import extract_facts
from app.memory.conflict import resolve_fact
from app.memory.retriever import retrieve


# System instruction that tells Qwen how to use the retrieved memories.
RESPONSE_PROMPT = """You are Memora, a helpful assistant with persistent memory of the user.

Below are relevant memories about the user. Use them naturally to give a personalized, accurate response. Do not list the memories back robotically — weave them in only when relevant. If no memory applies, just answer normally.

Known memories about the user:
{memories}"""


def process_message(user_id, message):
    """
    Run the full agent pipeline for one user message.
      user_id: who is talking
      message: their message text
    Returns a dict:
      {
        "reply": "the assistant's response",
        "memory_actions": [...],   # what was stored/updated/skipped
        "memories_used": [...]     # which memories informed the reply
      }
    """
    # STAGE 1: Extract memory-worthy facts from the message.
    facts = extract_facts(message)

    # STAGE 2: For each fact, resolve conflicts and store it.
    memory_actions = []
    for fact in facts:
        action = resolve_fact(user_id, fact)
        memory_actions.append(action)

    # STAGE 3: Retrieve memories relevant to the current message.
    relevant = retrieve(user_id, message, top_k=3)
    memories_used = [m["text"] for m in relevant]

    # STAGE 4: Build the context block from retrieved memories.
    if memories_used:
        memory_block = "\n".join(f"- {text}" for text in memories_used)
    else:
        memory_block = "(no relevant memories yet)"

    # STAGE 5: Generate a memory-grounded response.
    messages = [
        {"role": "system", "content": RESPONSE_PROMPT.format(memories=memory_block)},
        {"role": "user", "content": message},
    ]
    reply = chat(messages)

    return {
        "reply": reply,
        "memory_actions": memory_actions,
        "memories_used": memories_used,
    }
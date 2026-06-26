# Memora — Self-Managing Memory Agent

**A persistent memory agent that remembers what matters, forgets what doesn't, and resolves contradictions automatically — measured, not just claimed.**

Built for the **Global AI Hackathon Series with Qwen Cloud** · **Track 1: MemoryAgent** · powered by Alibaba Cloud's Qwen models.

---

## Results — Measured, Not Claimed

Memora's core differentiator (conflict resolution) is evaluated against a naive append-only baseline — the approach most basic "memory" agents actually use — across 18 labeled scenarios. The eval harness is in [`scripts/eval_conflict.py`](scripts/eval_conflict.py); it's reproducible and the baseline is deliberately fair (not rigged to lose).

| System | Contradictions | Duplicates | Compatible | **Overall** |
|---|---|---|---|---|
| **Memora** | 6/6 | 6/6 | 6/6 | **18/18 (100%)** |
| Naive append-only | 0/6 | 0/6 | 6/6 | 6/18 (33%) |

**+67 percentage points** over the baseline on clear-cut cases.

We then stress-tested Memora on 6 **deliberately ambiguous** scenarios (temporal negation, exception-vs-identity, degree changes). It scored **5/6** — and the single miss is a documented design tension, not a hidden bug (see [Limitations](#limitations--future-work)). A perfect score on easy cases plus an honest failure on a hard one is the point: the system is stress-tested, not cherry-picked.

---

## The Problem

LLMs are stateless — every conversation starts from zero. Most "memory" solutions either dump entire chat history into context (expensive, breaks at scale) or do naive vector retrieval (returns stale, contradictory, or irrelevant memories). Almost none handle the two behaviors that actually make memory *trustworthy*:

- **Forgetting** outdated or low-value information over time
- **Resolving conflicts** when new information contradicts what's already stored

Memora is built specifically to solve both.

---

## Three Engineered Differentiators

### 1. Conflict Resolution ★
When a new fact contradicts an existing memory (*"User is vegetarian"* → *"started eating chicken"*), Memora detects it and **updates the belief** rather than storing both and retrieving randomly.

- An **adaptive relevance gate** — not a flat threshold — uses an absolute distance ceiling *combined with* a relative-gap check: a candidate must stand out as meaningfully closer than its neighbors to be worth an LLM call, so it isn't flagged just for sitting in a generally similar topic area.
- A Qwen judgment call classifies the relationship: `contradict` / `duplicate` / `compatible`.
- Outdated memories are deleted and replaced; duplicates are skipped; compatible facts are kept alongside.

### 2. Decay (Forgetting) ★
Memories are scored by a blend of **importance (50%)**, **recency (30%)**, and **freshness (20%)** with exponential decay (30-day half-life). Memories below a relevance floor are pruned — while high-importance memories (e.g. allergies) resist decay even as they age. Retrieval updates a memory's `last_accessed` timestamp, so *frequently recalled* memories stay alive while genuinely stale ones fade.

### 3. Smart Retrieval ★
A two-stage pipeline mirroring Qwen's recommended RAG pattern — *retrieve many → rerank to top few → pass to the LLM*:
1. Vector search narrows a wide candidate pool using a custom relevance score (similarity + importance + recency).
2. **`qwen3-rerank`** performs final precision reordering on the narrowed set — ensuring the most critical memories are recalled within a limited context window.

---

## Architecture

![Memora Architecture](docs/architecture_diagram.png)
User → Streamlit UI → FastAPI Backend → Agent Loop

│

┌───────────────┬───────────┼───────────────┐

▼               ▼           ▼               ▼

Extractor     Conflict ★    Retriever ★      Decay ★

│               │           │               │

└───────────────┴─────┬─────┴───────────────┘

▼

ChromaDB Vector Store

│

▼

Qwen Cloud (Alibaba Cloud DashScope API)

Chat · Embeddings · Rerank

Both the AI models (Qwen Cloud) and the target backend hosting (Alibaba Cloud ECS) run on Alibaba Cloud infrastructure.

---

## Engineering Story — How It Was Built and Hardened

Memora wasn't just written once and declared done. The interesting engineering is in the diagnosis-and-fix cycle, where each claim was verified against real output before moving on.

**Four concrete bugs found and fixed (each verified, not assumed):**

- **Silent rerank fallback.** The `qwen3-rerank` call was wrapped in a bare `except` that swallowed all errors and quietly fell back to the custom score — meaning the headline two-stage retrieval could have been silently running as one stage with no way to tell. Added explicit success/failure logging; confirmed in the live terminal that rerank actually fires and succeeds.
- **Dead recency signal.** `last_accessed` was stamped once at creation and never updated, so "recency" and "freshness" were mathematically identical in both scoring paths. Added `touch_memory()` and wired it into retrieval; verified with real timestamp deltas (a retrieved memory's `last_accessed` correctly advanced past its `created_at`).
- **Non-adaptive "adaptive" gate.** The conflict gate's relative-gap constant was defined but never used — it was really a flat threshold. Wired in the relative-gap check and verified it rejects flat-neighborhood candidates that the old code would have wastefully sent to the LLM.
- **No error handling.** The API returned raw stack traces on any upstream failure. Wrapped endpoints so an upstream Qwen timeout returns a clean `502` (not a generic `500`), an empty message returns `400`, and the real error still prints server-side for debugging.

**One measured improvement:** the eval's hard set initially exposed a real weakness — Memora failed *temporal* contradictions ("lives in Beijing" vs "used to live in Beijing") because the surface words overlap heavily and the negation lives in the tense, not the content. A targeted prompt example for temporal negation lifted the hard set from 4/6 → 5/6 **with zero regression** on the 18/18 clear-cut set (re-verified after the change).

**Verification discipline:** a 12-test offline suite ([`tests/`](tests/)) covers multi-user isolation (one user can never see, retrieve, or delete another's memories) and the structural correctness of conflict resolution (contradict → replace, duplicate → skip, compatible → keep both), plus the adaptive-gate math. Tests use deterministic fake embeddings and an in-memory store, so they run in under a second with no API key.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM (reasoning) | `qwen-plus-latest` via Qwen Cloud |
| Embeddings | `text-embedding-v4` (1024-dimensional vectors) |
| Reranking | `qwen3-rerank` |
| Vector store | ChromaDB (persistent, local disk) |
| Backend | FastAPI |
| Frontend | Streamlit |
| Deployment target | Docker → Alibaba Cloud ECS |

---

## Project Structure
memora-agent/

├── app/

│   ├── main.py              # FastAPI app — /chat, /memories, /decay, /health (+ error handling)

│   ├── config.py            # Central settings, loads .env safely

│   ├── memory/

│   │   ├── extractor.py     # Pulls memory-worthy facts from raw messages

│   │   ├── store.py         # ChromaDB read/write + touch_memory

│   │   ├── conflict.py      # Conflict detection + resolution (★)

│   │   ├── decay.py         # Forgetting / relevance decay (★)

│   │   └── retriever.py     # Custom scoring + qwen3-rerank (★)

│   └── agent/

│       ├── llm_client.py    # Qwen API wrapper (chat + embeddings)

│       └── agent_loop.py    # Orchestrates the full pipeline

├── ui/

│   └── app.py               # Streamlit chat + live memory visualization

├── scripts/

│   ├── seed_demo_data.py    # Plants aged/low-value memories so decay is demoable

│   └── eval_conflict.py     # Conflict-resolution eval vs naive baseline

├── tests/

│   ├── conftest.py          # In-memory store + fake-embedding fixtures

│   ├── test_isolation.py    # Multi-user isolation guarantees

│   └── test_conflict.py     # Conflict logic + adaptive-gate math

├── docs/

│   └── architecture_diagram.png

├── deploy/

│   ├── Dockerfile

│   ├── docker-compose.yml

│   └── deploy_notes.md

├── data/                    # ChromaDB storage (gitignored)

├── requirements.txt

└── .env                     # API keys (gitignored, never committed)
---

## Setup

**1. Clone and create a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Configure environment variables** — create a `.env` file:
QWEN_API_KEY=your-key-here

QWEN_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
**4. Run the backend**
```bash
uvicorn app.main:app --reload
```

**5. Run the UI** (in a separate terminal)
```bash
streamlit run ui/app.py
```

The UI opens at `http://localhost:8501` and connects to the API at `http://127.0.0.1:8000`.

**Run the tests** (no API key needed):
```bash
python -m pytest tests/ -v
```

**Reproduce the conflict eval** (needs API access):
```bash
python -m scripts.eval_conflict
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/chat` | Send a message, get a memory-grounded reply + memory actions taken |
| `GET` | `/memories/{user_id}` | List all stored memories for a user |
| `POST` | `/decay/{user_id}` | Trigger forgetting for a user |

Interactive API docs at `/docs` once the server is running.

---

## How It Works — End to End

1. **User sends a message** → arrives at `/chat`.
2. **Extraction** — clean, structured facts are pulled from the raw message, filtering out filler and chitchat.
3. **Conflict resolution** — each fact is checked against existing memories; contradictions trigger an update, duplicates are skipped, new facts are added.
4. **Retrieval** — relevant memories are pulled via vector search, custom-scored, and reranked.
5. **Response generation** — Qwen generates a reply grounded in the retrieved memories.
6. **Decay** (on demand) — stale, low-importance memories are pruned.

---

## Deployment

Memora is containerized with Docker and **designed for Alibaba Cloud ECS**, with the FastAPI backend exposed via a public endpoint and ChromaDB persisted to a mounted volume. The Qwen models are already served from Alibaba Cloud (DashScope). **Status: ECS deployment in progress** — provisioning is currently blocked on account payment verification (a known friction point for foreign residents in China), with a support ticket open. The deployment configuration lives in [`deploy/`](deploy/).

---

## Limitations & Future Work

Honest boundaries, surfaced by the eval rather than hidden:

- **Exception vs. identity (the one hard miss).** Given *"is vegetarian"* → *"ate fish once at a wedding,"* Memora treats the exception as a belief update and replaces the original. This is a genuine design tension — whether a single counter-instance overturns a standing identity — with no clean right answer. Memora currently errs toward responsiveness over stability; a more nuanced policy would distinguish one-off exceptions from durable changes.
- **Temporal subtlety.** Temporal negations are now handled for common phrasings, but judgment still rests on the LLM and may vary on unusual constructions.
- **Hard-prune decay.** Decay currently deletes pruned memories outright. A future version could *consolidate/summarize* low-value memories instead of dropping them — preserving a compressed trace rather than forgetting entirely. The simpler hard-prune was a deliberate ship-on-time choice.

---

## License

MIT

---

## Author

**Ahmad Hassan** — Master's student in AI/NLP, Harbin Institute of Technology, Shenzhen
Built for the Global AI Hackathon Series with Qwen Cloud (2026)
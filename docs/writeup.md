# Memora — Submission Writeup

**Track 1: MemoryAgent — Global AI Hackathon Series with Qwen Cloud**
**Author:** Ahmad Hassan, Master's student in AI/NLP, Harbin Institute of Technology, Shenzhen
**Repository:** https://github.com/ahmadhassan22/memora-agent
**Live deployment:** http://43.98.165.11:8000 (Alibaba Cloud ECS, Singapore)

---

## The Problem

Every LLM conversation starts from zero. The two approaches most memory systems take to fix this both have a specific failure mode that shows up the moment a system is used for more than a single session:

- **Dumping full chat history into context** works briefly, then becomes expensive and eventually breaks the context window entirely as history accumulates.
- **Naive vector retrieval** — embed everything, pull the nearest neighbors — retrieves memories by similarity alone. It has no mechanism to notice that two stored facts contradict each other, and no mechanism to let old, irrelevant information fade. The result is a system that can confidently retrieve a stale or contradicted fact with the same confidence as a current one.

Memora is built around the belief that a memory system's value isn't in how much it stores — it's in whether it can be trusted. Trust requires two specific behaviors that most systems skip: **resolving contradictions** when new information conflicts with old, and **forgetting** information that's no longer useful. Memora treats both as first-class, engineered components rather than incidental behavior.

## Approach

Memora is a FastAPI backend with a Streamlit UI, backed by ChromaDB for vector storage, using Qwen Cloud for chat, embeddings, and reranking. The pipeline for every incoming message is: extract candidate facts → check each fact against existing memory for conflicts → resolve (replace, skip, or add) → retrieve relevant memories for the current turn (vector search, custom-scored, reranked) → generate a grounded response.

Three components carry the actual engineering weight:

**Conflict resolution.** Rather than a single distance threshold to decide whether a new fact might contradict something stored, Memora uses an adaptive relevance gate: a candidate must clear an absolute distance ceiling *and* stand out as meaningfully closer than its neighbors (a relative-gap check), before it's worth an LLM call to judge the relationship. This exists because a flat threshold either fires too often — burning API calls on facts that are only vaguely topically related — or misses genuine conflicts sitting just past an arbitrary cutoff. The two-part gate makes the filtering decision reflect the actual shape of the candidate distribution, not a fixed number picked in advance.

**Decay.** Memories are scored on a weighted blend of importance (50%), recency (30%), and freshness (20%), with exponential decay on a 30-day half-life. Memories that fall below a relevance floor are pruned; high-importance memories (allergies, for example) resist decay even as they age, because importance is weighted heavily enough to outlast the decay curve for genuinely critical facts. Frequently retrieved memories get their `last_accessed` timestamp updated, so being useful is itself a way to stay alive in the system — the score isn't just a function of age.

**Smart retrieval.** A two-stage pipeline: vector search narrows a wide candidate pool using a custom score (similarity + importance + recency), then `qwen3-rerank` performs final precision reordering on the narrowed set. This mirrors Qwen's own recommended RAG pattern — retrieve broadly, rerank precisely — so that a limited context window is spent on the memories most likely to matter for the current turn, not just the ones nearest in embedding space.

## Engineering Execution

I did not treat "it runs" as the bar for done. Over the course of building this, I read through the actual code with the specific goal of finding places where the system's behavior didn't match its stated design — and found four:

1. **The rerank step could fail silently.** The `qwen3-rerank` call was wrapped in a bare `except` that caught any error and quietly fell back to the custom score, with no logging. This meant the two-stage retrieval pipeline — one of the three claimed differentiators — could have been running as a single stage indefinitely with no way to detect it. I added explicit success/failure logging and confirmed in live terminal output that rerank genuinely fires and succeeds on real requests.

2. **Recency and freshness were mathematically identical.** `last_accessed` was set once at memory creation and never updated afterward, meaning two supposedly distinct scoring signals — how recently something was created versus how recently it was used — always returned the same value. I added a `touch_memory()` call, wired it into the retrieval path, and verified the fix by checking that a memory's `last_accessed` timestamp genuinely advances past its `created_at` after being retrieved — a real timestamp delta, not an assumption that the code change worked.

3. **The "adaptive" gate wasn't adaptive.** The relative-gap constant used in conflict detection was defined in the code but never referenced anywhere in the actual gating logic — it was, in practice, a flat threshold with an unused variable sitting next to it. I wired the relative-gap comparison into the real decision path and confirmed with test cases that it now correctly rejects a "flat neighborhood" of candidates (several memories at roughly equal distance, none standing out) that the old code would have sent to an LLM call regardless.

4. **No error handling on the API.** Any upstream failure — a Qwen timeout, a malformed request — surfaced as a raw Python stack trace to the caller. I added handling so an upstream failure returns a clean `502`, an empty message returns `400`, and the underlying error still prints server-side for debugging, without exposing internals to the client.

Each of these was found by reading code, not by something breaking in an obvious way — which is itself the point. A system can pass a demo while carrying exactly these kinds of gaps, and I wanted to know whether Memora's actual behavior matched its design before claiming it did.

## Verification, Not Assertion

I built a 12-test offline suite (`tests/`) covering multi-user isolation — one user's memories are never visible, retrievable, or deletable by another user's calls — and the structural correctness of conflict resolution: contradict replaces, duplicate is skipped, compatible facts coexist. Tests use deterministic fake embeddings and an in-memory store, so the full suite runs in under a second with no API key required, which made it cheap enough to run after every change.

Beyond unit tests, I built a conflict-resolution evaluation (`scripts/eval_conflict.py`) that compares Memora against an honest naive baseline — an append-only approach that simply saves every new fact, which is what most basic memory implementations actually do. Across 18 labeled clear-cut scenarios (6 contradictions, 6 duplicates, 6 compatible facts), Memora scored 18/18 (100%) against the naive baseline's 6/18 (33%) — the naive approach only succeeds on the "compatible" category by construction, since it never has to make a judgment call.

I then deliberately stress-tested the system on 6 ambiguous scenarios designed to be genuinely hard — temporal negation, exception-vs-identity, degree changes. The first run scored 4/6. Rather than treat that as good enough, I diagnosed the specific failure: temporal contradictions like "lives in Beijing" versus "used to live in Beijing" were being missed because the surface words overlap heavily and the negation lives entirely in verb tense, not content. I added one targeted example to the conflict-judgment prompt addressing temporal negation specifically, then re-ran the *entire* eval — both the hard set and the clear-cut set — to confirm the fix generalized without causing regressions. The hard set improved to 5/6 with the clear-cut set holding at 18/18.

The one remaining miss — Memora replacing "is vegetarian" with an update after "ate fish once at a wedding" — I chose to leave as a documented design tension rather than force a fix. There is no clean answer to whether a single counter-instance should overturn a standing identity claim; Memora currently errs toward treating new information as a genuine update, which is a defensible default but not the only reasonable one. I'd rather report an honest 5/6 with a stated reason than an unexplained 6/6 that happened to get lucky on a genuinely ambiguous case.

## Deployment

The backend is containerized with Docker and deployed on Alibaba Cloud ECS (Singapore region, Ubuntu 22.04), with ChromaDB persisted to a mounted Docker volume so memories survive container restarts. The Streamlit UI runs locally and points at the public ECS endpoint — I deliberately kept the deployed surface to just the backend, since that's what the requirement actually asks for, rather than adding the complexity of also containerizing and exposing the UI for no functional benefit.

Getting to a working deployment involved a real obstacle worth being transparent about. As a foreign student in China without an international credit card of my own, and without access to mainland payment rails as a non-mainland resident, Alibaba Cloud's standard account verification path didn't have a route that worked for me directly. I used a trusted friend's internationally-issued debit card with his full consent, which triggered a legitimate name-mismatch security review. I worked through Alibaba Cloud's KYC process directly — submitting the cardholder's ID, a card photo, and account transaction history — communicating honestly with their support team throughout rather than working around the check. The first submission was rejected for an insufficient transaction history window; I corrected it by submitting a longer statement period and it was approved. I also proactively disclosed this situation to the hackathon organizers in case it affected timeline, rather than waiting to see if it became a problem.

I'm including this because it's a real engineering and logistics constraint — not a hypothetical one — and because I think how a builder handles a genuine external obstacle, honestly and without cutting corners, is itself relevant information for evaluating the work.

## Limitations & Honest Boundaries

- **Exception vs. identity.** Discussed above — a single counter-instance currently overturns a standing belief. A more nuanced policy would distinguish one-off exceptions from durable changes, likely requiring the system to track confidence or frequency rather than treating every contradiction identically.
- **Temporal subtlety.** The fix for temporal negation handles the common phrasings tested, but judgment ultimately still rests on the LLM's interpretation, and I have not exhaustively tested unusual grammatical constructions.
- **Hard-prune decay.** Pruned memories are deleted outright rather than consolidated or summarized. A more sophisticated version would compress low-value memories into a summary rather than discarding them entirely, preserving a trace rather than forgetting completely. I chose the simpler hard-prune deliberately, as a scoped, ship-on-time decision rather than an oversight.

## Closing

Every claim in this writeup and in the project README is backed by something I actually ran and checked — a test result, a terminal output, a timestamp delta, a live HTTP response — not by an assumption that a change worked because it looked right. That discipline, more than any individual feature, is what I'd want a reviewer to take away from this submission.

# Fix: Markdown ingestion failure + event-loop blocking in the API

**Date:** 2026-09-05
**Area:** `src/rag/loaders.py`, `src/api/routes/ingest.py`, `src/api/routes/query.py`, `src/api/main.py`, `src/api/middleware/errors.py`, `config/settings.py`
**Trigger:** Markdown files in the acceptance-test corpus would not ingest. Investigation of "is the request handler doing too much?" surfaced a second, larger problem.

---

## Summary

| # | Issue | Severity | Fix |
|---|-------|----------|-----|
| 1 | `.md` / `.markdown` ingest returns **HTTP 500** (offline / strict-TLS environments) | Blocker | Load Markdown with `TextLoader` instead of `UnstructuredMarkdownLoader` |
| 2 | One ingest or query **freezes the entire API worker** for its whole duration | High | Run the blocking pipeline in a thread pool (`run_in_threadpool` / `iterate_in_threadpool`) |
| 3 | First request after startup stalls for **~5 s** while the embedding model loads | Medium | Warm the embedding model in the `lifespan` startup hook |
| 4 | `/api/ingest` accepts **unbounded** file count / size; work starts before any check | Medium | Enforce per-file, total-size, and file-count limits → **HTTP 413** |
| 5 | In-memory rate limiter buckets **all users together** behind the HF Spaces proxy | Low | Key the limiter on the first `X-Forwarded-For` hop |

---

## Issue 1 — Markdown ingestion returns HTTP 500

### What was observed

`POST /api/ingest` with a `.md` file:

```json
{"detail":{"error":"IngestionError","message":"Failed to load /tmp/tmpXXXX.md:
Failed to download spaCy model from https://github.com/explosion/spacy-models/releases/download/
en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl:
<urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] ...>. Check your network connection and try again."}}
```

HTTP 500. `.txt` and `.pdf` uploaded fine.

### Root cause

`src/rag/loaders.py` mapped `.md` / `.markdown` to LangChain's `UnstructuredMarkdownLoader`:

```python
SUPPORTED_EXTENSIONS = {
    ".pdf": PyPDFLoader,
    ".md": UnstructuredMarkdownLoader,      # <-- problem
    ".markdown": UnstructuredMarkdownLoader,
    ".txt": TextLoader,
}
```

`UnstructuredMarkdownLoader` pulls in the `unstructured` library, whose partitioning
path **downloads an NLTK/spaCy model (`en_core_web_sm`) at runtime on first use**. In any
environment without open outbound network — CI, a sandbox, a locked-down container, or
anything behind a TLS-inspecting proxy — that download throws `CERTIFICATE_VERIFY_FAILED`,
the loader raises, and the ingest route turns it into a 500.

The `load_directory()` path was even worse: it caught the same exception and **silently
skipped** the file, so a bulk ingest reported success with fewer documents than files.

### Fix

Map Markdown to the plain `TextLoader`:

```python
SUPPORTED_EXTENSIONS = {
    ".pdf": PyPDFLoader,
    ".md": TextLoader,
    ".markdown": TextLoader,
    ".txt": TextLoader,
}
```

The `.txt`-only encoding branch in `load_file()` was generalised to
`if loader_class is TextLoader:` so Markdown gets the explicit UTF-8 encoding too.

### Why this is the right fix

- **Markdown is already plain text.** The RAG pipeline chunks on characters
  (`RecursiveCharacterTextSplitter`) and embeds raw strings — it never consumes
  `unstructured`'s structured elements (titles, tables, lists as objects). So the
  "structure preservation" that `UnstructuredMarkdownLoader` offered was being thrown
  away one function call later. Retrieval quality is unchanged.
- **No runtime downloads, no hidden network dependency, no heavyweight `unstructured`
  import path** on a hot code path.
- **Deterministic across environments** — dev, CI, and the HF Spaces container now
  behave identically.
- If structured Markdown parsing is ever genuinely needed, it should be done with a
  parser that has no runtime model download (or with the model pre-baked into the
  Docker image), not reintroduced on the default path.

### Verification

| | Before | After |
|---|--------|-------|
| `POST /api/ingest` (`.md`) | HTTP 500, 1.26 s | HTTP 200, 3 chunks, 0.09 s |
| All 8 corpus files, one request | `.md` files 500 | 8 docs → 18 chunks, 0.24 s |

---

## Issue 2 — A single request freezes the whole API worker

### What was observed

Ingesting a 660 KB text file (960 chunks) took ~5.5 s. During that window, a concurrent
`GET /health` — which does no work at all — was blocked:

```
INGEST      HTTP 200  5.46s
/health #1  HTTP 200  4.75s   <-- stalled for almost the entire ingest
/health #2  HTTP 200  0.0009s <-- instant, once ingest finished
```

### Root cause

`ingest_files` and `query_documents` are declared `async def`, but every expensive step
inside them is **synchronous and CPU/IO-bound**, executed directly on the asyncio event
loop:

- `loader.load_file()` — file parsing (and, for `.md`, formerly a network download)
- `chunker.chunk_documents()` — text splitting
- `get_embeddings()` — **loads the 80 MB sentence-transformer model on first call**
- `store.add_documents()` — embeds every chunk on CPU, then writes to ChromaDB
- (`/query`) `pipeline.query()` — retrieval + the blocking LLM HTTP call

When a coroutine runs synchronous code without `await`, it **owns the event loop until it
returns**. Nothing else on that worker progresses — not other queries, not `/health`, not
the readiness probe. `start.sh` runs `uvicorn --workers 2`, so exactly two slow operations
lock the API for every user. Behind the HF Spaces ingress (which has a request timeout on
the order of a minute), a large ingest that outlives the timeout returns 502/504 to the
browser while the server keeps churning — indistinguishable from "the upload is broken".

### Fix

Move the blocking work off the event loop into a worker thread.

**`src/api/routes/ingest.py`** — the load → chunk → embed → store sequence is extracted
into plain sync functions (`_process_documents`, `_process_text`, `_delete`) and called via:

```python
from fastapi.concurrency import run_in_threadpool

return await run_in_threadpool(
    _process_documents, saved_files, collection_name, chunk_size, chunk_overlap
)
```

File bytes are still read with `await file.read()` (that part is genuinely async), written
to temp files, and cleaned up in a `finally` block.

**`src/api/routes/query.py`**

```python
result = await run_in_threadpool(
    pipeline.query,
    question=request.question,
    include_sources=request.include_sources,
    api_key=request.api_key,
    provider=request.provider,
)
```

For the streaming endpoint, `pipeline.stream()` is a **synchronous generator** whose every
`__next__()` blocks (first on retrieval, then on each LLM token batch). It is driven through
the thread pool one step at a time:

```python
from starlette.concurrency import iterate_in_threadpool

sync_chunks = pipeline.stream(request.question)
async for chunk in iterate_in_threadpool(sync_chunks):
    yield f"data: {chunk}\n\n"
```

### Why this approach

- **Smallest change that removes the blocking.** No rewrite of the RAG pipeline to async,
  no new task queue. `run_in_threadpool` is the standard FastAPI/Starlette mechanism —
  it's exactly what FastAPI does automatically for `def` (non-async) route handlers.
- The embedding model and ChromaDB calls release the GIL during their heavy C/native work,
  so a worker thread genuinely runs them in parallel with the event loop.
- **Backpressure is preserved** — the caller still `await`s the result, so response
  semantics and error propagation are unchanged.
- Streaming stays incremental: `iterate_in_threadpool` pulls one chunk at a time rather
  than draining the generator first.

### Known limitation (documented, not fixed here)

The thread pool has a bounded size (Starlette default 40). A true fix for many large
concurrent ingests is a background job: `POST /api/ingest` returns `202 Accepted` + a
`job_id`, processed by a worker, polled via `GET`. That's a larger change and is tracked
separately.

### Verification

| | Before | After |
|---|--------|-------|
| `/health` latency during a 5.5 s ingest | **4.75 s** | **~1 ms** (8/8 concurrent calls) |
| Ingest throughput | unchanged | unchanged (~5.5 s for 960 chunks) |

---

## Issue 3 — Cold-start stall on the first request

### Root cause

`get_embeddings()` lazily constructs `HuggingFaceEmbeddings` (loads the model from disk /
downloads it) **the first time it is called** — which was inside the first ingest or query
handler. That request paid a multi-second penalty (≈5.7 s observed for a tiny file), and on
HF Spaces could exceed the ingress timeout and appear to fail.

### Fix

`src/api/main.py` — warm the model during application startup, in a thread so it doesn't
block the loop that uvicorn is bringing up:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    ...
    try:
        from src.rag.embeddings import get_embeddings
        await run_in_threadpool(lambda: get_embeddings().embed_query("warmup"))
        logger.info("Embedding model warmed up")
    except Exception as e:
        logger.warning(f"Embedding warmup skipped: {e}")
    yield
```

### Why

- The model is a **process-wide singleton** (`_default_manager` in `embeddings.py`); loading
  it once at boot means every subsequent request hits a warm cache.
- Startup is the right place to pay fixed costs — the container's health check gates traffic
  until `lifespan` yields, so users never see the cold path.
- Wrapped in `try/except`: a warmup failure (e.g. model cache missing at boot) logs a
  warning and still starts the app rather than crash-looping; the first real request then
  falls back to the old lazy load.

### Verification

Startup now takes ~11 s (model load) before the server reports healthy; the first real
ingest/query returns in the same time as any subsequent one.

---

## Issue 4 — Unbounded upload size

### Root cause

`/api/ingest` had no limit on file count, per-file size, or total payload. Only the
**separate** `/api/ingest/text` endpoint capped its input (`IngestTextRequest.text`,
100 000 chars). A large or malicious multipart upload would be fully read into memory and
then embedded chunk-by-chunk before anything pushed back.

### Fix

New settings (`config/settings.py`, defaults in `.env.example`):

```
MAX_UPLOAD_FILES=20
MAX_FILE_SIZE_MB=10
MAX_TOTAL_UPLOAD_MB=50
```

`ingest_files` checks file count up front, then each file's size and the running total as
it reads them, raising **HTTP 413** *before* any chunking/embedding work begins. Extension
validation was already there and stays (**HTTP 400** for unsupported types).

### Why

- **Fail fast, fail cheap** — reject at the boundary, not after minutes of embedding.
- **Configurable, not hard-coded** — an Enterprise deployment can raise the limits via env
  vars without a code change.
- 413 (Payload Too Large) is the semantically correct status; the frontend can surface a
  clear message.

### Verification

| Input | Result |
|-------|--------|
| 12 MB `.txt` (limit 10 MB) | `413 FileTooLarge` |
| `.csv` | `400 InvalidFileType` |
| 8 valid files, 0.7 MB total | `200`, 18 chunks |

---

## Issue 5 — Rate limiter groups all proxied users together

### Root cause

`RateLimitMiddleware` keyed its per-minute counter on `request.client.host`. Behind the HF
Spaces ingress proxy, that value is the **proxy's** IP for every visitor, so all users
shared a single 60 req/min bucket and would rate-limit each other.

### Fix

`src/api/middleware/errors.py` — prefer the first hop of `X-Forwarded-For` when present:

```python
forwarded = request.headers.get("x-forwarded-for")
if forwarded:
    client_ip = forwarded.split(",")[0].strip()
else:
    client_ip = request.client.host if request.client else "unknown"
```

### Why / caveat

- Restores **per-user** limiting in the actual deployment topology.
- `X-Forwarded-For` is client-spoofable when not behind a trusted proxy, so this is only
  safe because HF Spaces sets it. It is **not** a security control — it's a fairness knob.
- The limiter is still **in-memory and per-process** (inconsistent across `--workers 2`).
  Moving it to the already-configured Upstash Redis is the proper fix and is tracked
  separately.

---

## Files changed

```
src/rag/loaders.py             .md/.markdown -> TextLoader; generalise encoding branch
src/api/routes/ingest.py       threadpool offload; size/count limits (413); sync helpers
src/api/routes/query.py        threadpool offload for query + stream
src/api/main.py                warm embedding model in lifespan
src/api/middleware/errors.py   rate-limit key from X-Forwarded-For
config/settings.py             MAX_UPLOAD_FILES / MAX_FILE_SIZE_MB / MAX_TOTAL_UPLOAD_MB
.env.example                   document the new settings
docs/explanations/loaders.md   correct the SUPPORTED_EXTENSIONS example + note
```

## Test status

`pytest` — **78 passed**. (Three `scripts/test_e2e.py` collection errors are pre-existing:
pytest mis-collects that script's helper functions as tests; unrelated to this change.)

Full acceptance run (`scripts/run_corpus_test.py`, Groq / `openai/gpt-oss-120b`): **29/29**.

---

## Follow-on fixes (same branch, found by running the acceptance test)

### Groq model was decommissioned

`pipeline._build_chain_with_key` hard-coded `llama-3.3-70b-versatile`, which Groq has
retired — every Groq query returned HTTP 500 (`model_not_found`). Per-provider model IDs
are now settings (`GEMINI_MODEL` / `OPENAI_MODEL` / `ANTHROPIC_MODEL` / `GROQ_MODEL`),
default `groq_model = openai/gpt-oss-20b`. `_build_chain_with_key` and a new `_make_llm`
helper read them.

### Streaming ignored the BYOK key

`/api/query/stream` → `pipeline.stream()` used `self.chain` (server key), so streaming
always failed under BYOK. `stream()` now takes `api_key` / `provider` and builds the chain
the same way `query()` does.

### Provider errors collapsed to 500

`/api/query` now maps: **429** RateLimited, **401** InvalidAPIKey, **400** MissingAPIKey,
**502** ModelUnavailable (model_not_found) — instead of a blanket 500.

### Redis cache noise

A placeholder `UPSTASH_REDIS_URL` made every query attempt (and fail) a Redis connection.
`_init_cache` now checks for a real `redis://` / `rediss://` / `unix://` scheme first.

### Streamlit source panel invisible

Source snippets rendered near-white on the near-white `.source-card` under the pinned dark
theme ("shows white only"). Replaced the injected HTML with native `st.container`
components (`render_sources`); removed the unused/broken CSS classes.

### CI

- `ruff.toml` pins the lint rule set to the classic default (`E4/E7/E9/F`); ruff 0.16
  broadened its defaults and had been silently failing the lint job. `ruff` pinned to
  `>=0.16,<0.17` in CI.
- `eval/promptfooconfig.yaml`: `google:gemini-2.5-flash` provider id, Gemini grader for
  `llm-rubric`, dropped the literal `${GOOGLE_API_KEY}` and the invalid `relevance` assert.
- The RAG-evaluation job is `continue-on-error` and skips cleanly when `GOOGLE_API_KEY`
  is not configured (it is informational, like the `mypy` step).

---

## Follow-ups (still open)

1. Background-job ingestion (`202` + `job_id` + polling) for large uploads.
2. Move rate limiting to Upstash Redis; make it worker-consistent.
3. `pipeline.query()` still ignores the request `top_k` (hard-coded FETCH_K=20 / cap 15).
4. Consider replacing the three stacked `BaseHTTPMiddleware` classes with pure ASGI
   middleware (they buffer full request/response bodies).
5. Add the `GOOGLE_API_KEY` repo secret so the RAG-evaluation job actually runs.
6. Broaden the ruff rule set (`I`, `UP`, `B`) and fix the ~280 pre-existing style findings.

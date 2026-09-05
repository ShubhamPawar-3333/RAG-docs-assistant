# DocuMind AI — Acceptance Test Corpus

A self-contained set of documents plus a Q&A matrix for verifying that the RAG
application works end to end: ingestion → chunking → embedding → retrieval →
grounded generation, across all supported formats, plus grounding/refusal and
multi-tenant isolation.

Everything here is about a **fictional** SaaS product, *Nimbus Analytics*. Because
the facts are invented, a correct answer can only come from retrieval — the LLM
cannot know them from pretraining. That is what makes this a real test of the
pipeline and not just of the model.

## Files

| File | Format | Exercises |
|------|--------|-----------|
| `01_nimbus_overview.md` | Markdown | prose + a small table; synonym / semantic retrieval ("performance bottlenecks") |
| `02_nimbus_pricing.md` | Markdown | table extraction, numeric facts, "what is *not* included" |
| `03_nimbus_api_reference.md` | Markdown | fenced code blocks (curl/python), endpoint tables, auth details |
| `04_nimbus_security.txt` | Plain text | multi-fact answers, explicit negations (HIPAA not on Free/Pro) |
| `05_nimbus_faq.txt` | Plain text | Q&A-shaped source, close-but-distinct facts (iOS app vs Android SDK) |
| `06_nimbus_changelog.md` | Markdown | multi-hop ("which version added X, and when") over dated entries |
| `07_nimbus_onboarding.pdf` | PDF | the PyPDFLoader path; trial length, default workspace settings |
| `08_edge_cases.txt` | Plain text | Unicode/accents/CJK/emoji, a single-occurrence "needle" fact, chunk-boundary padding |
| `tenant_acme/acme_handbook.md` | Markdown | **separate collection** — used to prove collections don't leak into each other |
| `unsupported/notes.csv` | CSV | must be **rejected** by `/api/ingest` with HTTP 400 |
| `qa_pairs.yaml` | — | machine-readable expected answers consumed by the runner |

## Collections used

| Collection | Contents |
|------------|----------|
| `test_corpus` | files `01`–`08` (md, txt, pdf) |
| `tenant_acme` | `tenant_acme/acme_handbook.md` only |

## How to run (automated)

```bash
# terminal 1 — backend
source venv/bin/activate
uvicorn src.api.main:app --port 8000

# terminal 2 — the test
export NIMBUS_TEST_API_KEY=<your Gemini/OpenAI/Anthropic/Groq key>
export NIMBUS_TEST_PROVIDER=gemini          # or openai | anthropic | groq
./venv/bin/python scripts/run_corpus_test.py
```

The script ingests both collections, runs every case in `qa_pairs.yaml` against
`POST /api/query`, checks the streaming endpoint, and exits non-zero if anything
fails. Re-run with `SKIP_INGEST=1` to reuse already-ingested collections.

If the PDF ever needs regenerating: `./venv/bin/python scripts/_make_test_pdf.py`.

## How to run (manual, via the Streamlit UI)

1. Log in with your API key.
2. Create a collection `test_corpus`, upload files `01`–`08`.
3. Work down the "What to verify" table below, asking each question and comparing
   the answer and the cited sources.
4. Create a second collection `tenant_acme`, upload only the handbook, and repeat
   the isolation checks.

## What to verify

### Ingestion
- [ ] All 8 Nimbus files ingest without error and report a non-zero chunk count.
      *(`.md`, `.txt` and `.pdf` all load with no external/runtime dependency.)*
- [ ] `notes.csv` is rejected with HTTP 400 and a clear "unsupported file type" message.
- [ ] Re-ingesting the same files does not crash (idempotency / duplicate handling).

### Retrieval & grounded answers (`collection = test_corpus`)
| # | Question | Expected answer contains | Source | Capability |
|---|----------|--------------------------|--------|------------|
| 1 | What is Nimbus Analytics? | product analytics platform, SaaS | 01 | basic recall |
| 2 | How much is the Pro plan per month? | $99 | 02 | numeric fact |
| 3 | Events per month on the Free plan? | 100,000 | 02 | table cell |
| 4 | How do I authenticate to the Events API? | `Authorization: Bearer <key>` | 03 | code block |
| 5 | Max events per POST /events request? | 500 | 03 | basic recall |
| 6 | API rate limit and what happens on exceed? | 600/min, 429, Retry-After | 03 | multi-fact |
| 7 | How is data encrypted at rest / in transit? | AES-256, TLS 1.3 | 04 | multi-fact |
| 8 | Data retention after workspace deletion? | 90 days | 04 | basic recall |
| 9 | Supported browsers? | Chrome/Firefox/Safari/Edge, not IE | 05 | list + negation |
| 10 | Password reset link validity? | 60 minutes | 05 | basic recall |
| 11 | Free trial length / credit card? | 14 days, no credit card | 07 (PDF) | PDF path |
| 12 | Default workspace timezone? | UTC | 07 / 05 | PDF + corroboration |
| 13 | Why are dashboards slow on a huge workspace? | 50M events, pre-aggregated views | 01 | **semantic** (wording differs) |
| 14 | Can staff log in with our corporate IdP? | SAML / SSO (Pro & Enterprise) | 04 | **semantic** |
| 15 | Is HIPAA available on Pro? | No — Enterprise only, needs a BAA | 04 (+02) | **faithfulness / negation** |
| 16 | Which release added the Python SDK, and when? | v3.1.0, 2026-04-08 | 06 | **multi-hop** |
| 17 | When did EU data residency ship? | v3.3.0 / 2026-09-02 | 06 | **multi-hop** |
| 18 | Codename for the 2027 UI redesign? | Project Aurora | 08 | **needle in haystack** |
| 19 | Japanese word for "analysis" in the notes? | 分析 / bunseki | 08 | **Unicode** |

### Grounding / refusal — must NOT answer (`collection = test_corpus`)
| # | Question | Expected behaviour |
|---|----------|--------------------|
| 20 | Weather in Tokyo today? | "not in the documentation"; no invented forecast |
| 21 | Nimbus's current stock price? | says the info isn't in the documents; no `$` figure |
| 22 | Who is the CEO of Nimbus Analytics? | admits no document names a CEO |

### Multi-tenant isolation
| # | Question | Collection | Expected behaviour |
|---|----------|-----------|--------------------|
| 23 | What is the company mascot? | `test_corpus` | does **not** say "red panda" / "Pip" (that fact is only in `tenant_acme`) |
| 24 | Company mascot + vacation days? | `tenant_acme` | "red panda / Pip", "25 days" |
| 25 | How much does the Pro plan cost? | `tenant_acme` | does **not** return $99 (Nimbus pricing isn't in this collection) |

### Non-functional
- [ ] `POST /api/query/stream` returns incremental `data:` chunks ending in `data: [DONE]`.
- [ ] `include_sources: true` returns source snippets with `file_name` metadata matching the table above.
- [ ] Typos still retrieve ("How does retrevial work?") — optional, model-dependent.
- [ ] Latency per query is within the project's target (see `eval/promptfooconfig.yaml`: 5s).

## Pass criteria

- 100% of ingestion checks pass.
- All of cases 1–19 contain the expected facts.
- All of cases 20–25 refuse / stay isolated as described.
- Streaming check passes.

Model wording varies, so the runner matches on **keywords**, not exact strings.
A case that fails only on phrasing (right fact, unmatched synonym) should be
confirmed by eye and the keyword list widened in `qa_pairs.yaml`.

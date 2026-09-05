"""
Acceptance test for DocuMind AI, driven through the public HTTP API.

It ingests the Nimbus Analytics test corpus (data/test_corpus/), then runs the
Q&A matrix in data/test_corpus/qa_pairs.yaml against POST /api/query and checks
that each answer contains the facts it should (and none of the facts it should
not). It also checks a few behaviours that have nothing to do with any single
answer: rejection of unsupported file types, collection isolation, and the
streaming endpoint.

Usage:
    # 1. start the backend in another terminal:
    #    uvicorn src.api.main:app --port 8000
    # 2. run:
    ./venv/bin/python scripts/run_corpus_test.py

Environment variables:
    DOCUMIND_API_URL      base URL of the backend   (default http://localhost:8000)
    NIMBUS_TEST_API_KEY   LLM API key to pass as BYOK (default: none -> server key)
    NIMBUS_TEST_PROVIDER  gemini | openai | anthropic | groq (default gemini)
    SKIP_INGEST=1         reuse collections from a previous run
"""

from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "data" / "test_corpus"
QA_FILE = CORPUS_DIR / "qa_pairs.yaml"

API_URL = os.getenv("DOCUMIND_API_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("NIMBUS_TEST_API_KEY") or None
PROVIDER = os.getenv("NIMBUS_TEST_PROVIDER", "gemini")
SKIP_INGEST = os.getenv("SKIP_INGEST") == "1"

NIMBUS_COLLECTION = "test_corpus"
ACME_COLLECTION = "tenant_acme"

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"

results: list[tuple[str, bool, str]] = []


def record(name: str, passed: bool, detail: str = "") -> None:
    results.append((name, passed, detail))
    mark = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
    line = f"  [{mark}] {name}"
    if detail:
        line += f"\n         {DIM}{detail}{RESET}"
    print(line)


def section(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
def ingest(paths: list[Path], collection: str) -> dict:
    files = [("files", (p.name, p.read_bytes())) for p in paths]
    resp = requests.post(
        f"{API_URL}/api/ingest",
        files=files,
        data={"collection_name": collection},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def do_ingestion() -> None:
    section("Ingestion")

    # Corpus docs are the numbered files (01_..08_); TEST_PLAN.md etc. are not corpus.
    nimbus_files = sorted(
        p for p in CORPUS_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".md", ".txt", ".pdf"}
        and p.name[:1].isdigit()
    )
    try:
        info = ingest(nimbus_files, NIMBUS_COLLECTION)
        ok = info.get("success") and info.get("chunks_created", 0) > 0
        record(
            "ingest test_corpus",
            ok,
            f"{info.get('documents_processed')} docs -> {info.get('chunks_created')} chunks "
            f"from {len(nimbus_files)} files",
        )
        if info.get("documents_processed", 0) < len(nimbus_files):
            record(
                "all corpus files ingested",
                False,
                f"only {info.get('documents_processed')} of {len(nimbus_files)} "
                "files produced documents; the loader skipped some",
            )
    except Exception as e:  # noqa: BLE001
        record("ingest test_corpus", False, repr(e))

    try:
        info = ingest([CORPUS_DIR / "tenant_acme" / "acme_handbook.md"], ACME_COLLECTION)
        record("ingest tenant_acme", bool(info.get("success")), json.dumps(info))
    except Exception as e:  # noqa: BLE001
        record("ingest tenant_acme", False, repr(e))

    # Unsupported file type must be rejected with 400.
    csv_path = CORPUS_DIR / "unsupported" / "notes.csv"
    try:
        resp = requests.post(
            f"{API_URL}/api/ingest",
            files=[("files", (csv_path.name, csv_path.read_bytes()))],
            data={"collection_name": "should_not_exist"},
            timeout=60,
        )
        record(
            "reject unsupported file type (.csv -> 400)",
            resp.status_code == 400,
            f"got HTTP {resp.status_code}",
        )
    except Exception as e:  # noqa: BLE001
        record("reject unsupported file type (.csv -> 400)", False, repr(e))


# --------------------------------------------------------------------------- #
# Querying
# --------------------------------------------------------------------------- #
def query(question: str, collection: str, top_k: int = 8) -> dict:
    payload = {
        "question": question,
        "collection_name": collection,
        "top_k": top_k,
        "include_sources": True,
        "provider": PROVIDER,
    }
    if API_KEY:
        payload["api_key"] = API_KEY

    # Free provider tiers rate-limit aggressively; retry a few times on 429/503.
    last_exc = None
    for attempt in range(4):
        resp = requests.post(f"{API_URL}/api/query", json=payload, timeout=120)
        if resp.status_code in (429, 503):
            last_exc = requests.HTTPError(f"{resp.status_code} from /api/query")
            time.sleep(8 * (attempt + 1))
            continue
        resp.raise_for_status()
        return resp.json()
    raise last_exc


# Normalise typographic dashes and collapse whitespace so keyword checks are not
# defeated by an LLM emitting "AES‑256" instead of "AES-256".
_DASHES = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-",
                         "—": "-", "−": "-", "­": "-"})


def _norm(s: str) -> str:
    return " ".join(str(s).translate(_DASHES).lower().split())


def _contains(haystack: str, needle: str) -> bool:
    return _norm(needle) in _norm(haystack)


def evaluate_case(case: dict) -> None:
    name = f"{case['id']} ({case.get('category', '?')})"
    try:
        data = query(case["question"], case["collection"])
    except Exception as e:  # noqa: BLE001
        record(name, False, f"query error: {e!r}")
        return

    answer = data.get("answer", "") or ""
    problems: list[str] = []

    for kw in case.get("keywords_all", []):
        if not _contains(answer, str(kw)):
            problems.append(f"missing required '{kw}'")

    any_kws = case.get("keywords_any", [])
    if any_kws and not any(_contains(answer, str(kw)) for kw in any_kws):
        problems.append(f"none of {any_kws} present")

    for kw in case.get("must_not_contain", []):
        if _contains(answer, str(kw)):
            problems.append(f"contains forbidden '{kw}'")

    behavior = case.get("expected_behavior")
    passed = not problems
    detail = "; ".join(problems) if problems else f"answer: {answer[:160]}"
    if behavior and passed:
        detail = f"[manual check: {behavior}] answer: {answer[:140]}"
    record(name, passed, detail)


def do_queries() -> None:
    spec = yaml.safe_load(QA_FILE.read_text())
    cases = spec["cases"]

    section(f"Q&A matrix ({len(cases)} cases)")
    for case in cases:
        evaluate_case(case)
        time.sleep(float(os.getenv("NIMBUS_TEST_DELAY", "1")))


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #
def do_streaming() -> None:
    section("Streaming endpoint")
    payload = {
        "question": "What is Nimbus Analytics?",
        "collection_name": NIMBUS_COLLECTION,
        "provider": PROVIDER,
    }
    if API_KEY:
        payload["api_key"] = API_KEY
    try:
        with requests.post(
            f"{API_URL}/api/query/stream", json=payload, stream=True, timeout=120
        ) as resp:
            resp.raise_for_status()
            chunks = []
            for raw in resp.iter_lines():
                if raw and raw.startswith(b"data: "):
                    chunks.append(raw[6:].decode("utf-8", "replace"))
            got_done = "[DONE]" in chunks
            body = "".join(c for c in chunks if c != "[DONE]")
            record(
                "stream returns SSE chunks + [DONE]",
                got_done and len(body) > 0,
                f"{len(chunks)} chunks, {len(body)} chars, done={got_done}",
            )
    except Exception as e:  # noqa: BLE001
        record("stream returns SSE chunks + [DONE]", False, repr(e))


# --------------------------------------------------------------------------- #
def main() -> int:
    print(f"DocuMind acceptance test  ->  {API_URL}")
    print(f"provider={PROVIDER}  byok_key={'set' if API_KEY else 'server default'}")

    try:
        requests.get(f"{API_URL}/health", timeout=10).raise_for_status()
    except Exception as e:  # noqa: BLE001
        print(f"{RED}Backend not reachable at {API_URL}: {e}{RESET}")
        print("Start it with:  uvicorn src.api.main:app --port 8000")
        return 2

    if not SKIP_INGEST:
        do_ingestion()
        time.sleep(1)
    do_queries()
    do_streaming()

    total = len(results)
    failed = [r for r in results if not r[1]]
    section("Summary")
    print(f"  {total - len(failed)}/{total} passed")
    if failed:
        print(f"\n  {RED}Failures:{RESET}")
        for n, _, d in failed:
            print(f"    - {n}: {d}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

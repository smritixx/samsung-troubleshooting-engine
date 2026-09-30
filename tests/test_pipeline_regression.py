"""Regression suite for the Smart Guided Troubleshooting pipeline.

Run as pytest:
    .venv\\Scripts\\python.exe -m pytest tests/test_pipeline_regression.py -v

Run as a diagnostic script (prints each query's outcome):
    .venv\\Scripts\\python.exe tests/test_pipeline_regression.py

Three categories:

  REQUIRED_CONTEXTS_LOCAL   Hard assert: local SIIS must produce a plan
  REQUIRED_CONTEXTS_WEB     Hard assert: web fallback must produce a plan
  REQUIRED_ABSTAIN          Hard assert: no plan exists; must abstain
  INFORMATIONAL             Printed only, not asserted -- Tavily's index
                            and the SIIS corpus legitimately change these
"""
import os
import sys
from pathlib import Path

# Add the project root to sys.path so `import app` works when this
# file is run directly (not via pytest, which handles it automatically).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env before importing the pipeline
from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

import pytest

from app.main import pipeline


HAS_TAVILY = bool(os.environ.get("TAVILY_API_KEY", "").strip())


# --- Test cases ------------------------------------------------------

REQUIRED_CONTEXTS_LOCAL = [
    "My Galaxy S24 Ultra screen is completely black and won't turn on",
    "My Galaxy phone's screen is completely cracked and I can't use it",
    "My Galaxy S22 screen inputs are delayed and the touch responsiveness is laggy",
]

REQUIRED_CONTEXTS_WEB = [
    "phone not charging",
    "pictures are coming out blurry",
]

REQUIRED_ABSTAIN = [
    "The weather on Mars is nice today",
    "My new Samsung phone's main screen stays small and doesn't fill the whole display",
]

INFORMATIONAL = [
    "why are all pictures from my camera blurry",
    "screen is green",
    "My Galaxy S24 screen goes completely blank and I can't use Smart Switch",
    "my screen is green",
    "battery drains fast",
]


# --- Helpers ---------------------------------------------------------

def _run(q):
    """Fresh cache, run pipeline, return response dict."""
    pipeline.cache.clear()
    return pipeline.run(q)


def _summarize(r, q):
    """Compact string for assertion messages."""
    m = r.get("meta", {})
    return (
        f"query={q!r}\n"
        f"  contexts={len(r['response']['contexts'])}\n"
        f"  web_fallback={m.get('web_fallback')} "
        f"siis_score={m.get('siis_match_score')} "
        f"fallback={m.get('fallback')}\n"
        f"  tier={(m.get('siis_components') or {}).get('tier')}"
    )


# --- Pytest tests ----------------------------------------------------

@pytest.mark.parametrize("query", REQUIRED_CONTEXTS_LOCAL)
def test_local_match_returns_contexts(query):
    r = _run(query)
    assert len(r["response"]["contexts"]) >= 1, (
        f"Local SIIS query returned no contexts:\n{_summarize(r, query)}"
    )


@pytest.mark.skipif(not HAS_TAVILY, reason="TAVILY_API_KEY not configured")
@pytest.mark.parametrize("query", REQUIRED_CONTEXTS_WEB)
def test_web_fallback_returns_contexts(query):
    r = _run(query)
    assert len(r["response"]["contexts"]) >= 1, (
        f"Web fallback returned no contexts:\n{_summarize(r, query)}"
    )


@pytest.mark.skipif(not HAS_TAVILY, reason="TAVILY_API_KEY not configured")
@pytest.mark.parametrize("query", REQUIRED_ABSTAIN)
def test_abstains_correctly(query):
    r = _run(query)
    ctxs = r["response"]["contexts"]
    assert len(ctxs) == 0, (
        f"Query should have abstained but returned contexts:\n"
        f"{_summarize(r, query)}\n"
        f"  first title: {ctxs[0]['title'] if ctxs else ''!r}"
    )


# --- Standalone diagnostic runner ------------------------------------

def _diagnostic_main():
    cases = []
    for q in REQUIRED_CONTEXTS_LOCAL:
        cases.append(("REQUIRE contexts (local)", q))
    for q in REQUIRED_CONTEXTS_WEB:
        cases.append(("REQUIRE contexts (web)", q))
    for q in REQUIRED_ABSTAIN:
        cases.append(("REQUIRE abstain", q))
    for q in INFORMATIONAL:
        cases.append(("INFO only", q))

    print("=" * 78)
    print("Pipeline regression sweep")
    print("=" * 78)
    print(f"HAS_TAVILY = {HAS_TAVILY}")

    results = []
    for label, q in cases:
        try:
            r = _run(q)
            n = len(r["response"]["contexts"])
            m = r["meta"]
            first = r["response"]["contexts"][0]["title"] if n else ""

            verdict = "OK"
            if label.startswith("REQUIRE contexts") and n == 0:
                verdict = "FAIL"
            if label == "REQUIRE abstain" and n > 0:
                verdict = "FAIL"

            results.append((verdict, q))
            print(f"\n[{verdict}] {label}")
            print(f"  query : {q[:75]}")
            print(f"  out   : contexts={n}  web={m.get('web_fallback')}  "
                  f"fallback={m.get('fallback')}  "
                  f"score={m.get('siis_match_score')}")
            if first:
                print(f"  title : {first[:75]}")
        except Exception as e:
            results.append(("ERROR", q))
            print(f"\n[ERROR] {label}")
            print(f"  query : {q[:75]}")
            print(f"  error : {type(e).__name__}: {e}")

    n_fail = sum(1 for v, _ in results if v != "OK")
    print()
    print("=" * 78)
    print(f"Summary: {len(results) - n_fail}/{len(results)} passed")
    if n_fail:
        print("\nFailures:")
        for v, q in results:
            if v != "OK":
                print(f"  [{v}] {q[:75]}")
    print("=" * 78)
    return n_fail


if __name__ == "__main__":
    sys.exit(_diagnostic_main())

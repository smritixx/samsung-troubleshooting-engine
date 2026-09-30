# Smart Guided Troubleshooting Engine

A Samsung PRISM (Theme 2) reference implementation that converts an informal, natural-language complaint about a Galaxy device into a validated, ordered, one-tap troubleshooting plan.

The engine retrieves trusted Samsung support text, extracts a structured plan from it (either with an LLM or a deterministic parser), maps every step to an exact masked Settings deeplink from a trusted catalog, orders the plan so non-destructive actions run first, and validates the entire response before returning it. If no trusted evidence supports a plan, the engine abstains instead of hallucinating.

## What problem it solves

Today, a Samsung support agent reads a vague complaint like *"My phone got slow after the update"* or *"The screen flickers and dies fast"*, manually interprets it against a knowledge base, selects and orders troubleshooting actions, and the customer then navigates nested Settings menus on their own. The brief estimates this manual triage at roughly 15 minutes per scenario.

This project compiles that entire flow into an executable plan:

- **Input:** free-form complaint (e.g. *"My Galaxy S24 Ultra screen is completely black and won't turn on"*)
- **Output:** a Pydantic-validated JSON plan with a goal, a title, categorized actions, discrete steps, and one-tap `bixby://` deeplinks pointing at the exact Settings screen each step needs.

The engine never invents a fix. If the trusted knowledge base has no matching procedure, it returns an empty plan with a fallback message.

## Architecture at a glance

    Complaint
       |
       v
    [ Semantic cache lookup ]
       |
       +-- HIT  -> validated response returned (LLM not called)
       |
       +-- MISS
             |
             v
           Query enrichment (canonical form + variants)
             |
             v
           Hybrid SIIS retrieval  (BM25 + character trigram)
             |
             v
           LLM structure extraction  (Phase 1 - Gemini by default)
             |
             +-- on failure -> deterministic extractor
             |
             v
           Deterministic deeplink resolution
             |
             v
           Risk-aware sequencing (auto -> manual -> critical)
             |
             v
           Pydantic + URL-leak + catalog-membership validation
             |
             v
           Store validated plan in semantic cache
             |
             v
           Final JSON response

The LLM is used **only on a cache miss** and **only for structure extraction**. It never sees the deeplink catalog, never produces a `bixby://` URI, and never bypasses validation. See `app/llm.py`.

## File inventory

### Data files

| Path | Purpose |
|---|---|
| `data/deeplinks.json` | Trusted catalog of ~578 masked Galaxy Settings deeplinks. Every `bixby://` string in a response is copied verbatim from here. Nothing else in the project may construct one. |
| `data/siis_responses.json` | Pre-cleaned Samsung support text (20 rows). The only knowledge source the pipeline draws troubleshooting steps from. |

### Application modules

| Path | Purpose |
|---|---|
| `app/__init__.py` | Marks `app/` as a Python package so relative imports resolve. Empty. |
| `app/config.py` | Central configuration: file paths, retrieval depth and thresholds, cache size and similarity gate, action caps, and the LLM environment variable bindings. Nothing is hard-coded elsewhere. |
| `app/schema.py` | Pydantic v2 response contract. Defines `ContextDeeplinkResponse`, `Goal`, `Action`, `StepGroup`, `Deeplink`, `ValidationDeepLink`, and the category / result-type / condition enums. |
| `app/data_loader.py` | Loads both JSON files once, memoized with `functools.lru_cache`. |
| `app/retrieval.py` | Hybrid retrieval: BM25 (exact terminology) + character trigram cosine (paraphrase matching). One `HybridIndex.search()` method serves both the SIIS corpus and the deeplink catalog. |
| `app/cache.py` | Safety-aware semantic cache. Stores only validated plans. Jaccard similarity over normalized tokens. Thread-safe. FIFO eviction. |
| `app/extractor.py` | Deterministic parser that turns Markdown-like SIIS content into candidate actions and steps. Returns an empty list when no procedure is present - which triggers abstention. |
| `app/llm.py` | The only place the pipeline talks to a language model. Sends the complaint plus one retrieved SIIS passage to an OpenAI-compatible endpoint and returns a JSON plan. Never returns a deeplink. All failure modes return `None`. |
| `app/resolver.py` | Maps the natural-language text of a step to the correct catalog entry, copying the URI verbatim. The only module allowed to write deeplink fields. |
| `app/sequencer.py` | Classifies actions by risk (auto / manual / critical) and sorts the plan so auto runs first and critical last. Also enforces the "manual actions carry no deeplink" rule. |
| `app/validator.py` | Final safety gate. URL-leak scan across the serialized response, plus catalog-membership checks on every deeplink field. |
| `app/pipeline.py` | The orchestrator. Owns the entire flow: cache lookup, retrieval, LLM extraction with deterministic fallback, resolution, ordering, Pydantic construction, validation, cache storage. |
| `app/main.py` | FastAPI service. Exposes `POST /v1/troubleshoot` and `GET /health`. Serves the frontend at `/`. |

### Frontend

| Path | Purpose |
|---|---|
| `frontend/index.html` | Single-page demo UI. Vanilla JavaScript, no build step. Posts to `/v1/troubleshoot` and renders action cards, step lists, category badges, and one-tap deeplink buttons. |

### Tests

| Path | Purpose |
|---|---|
| `tests/test_smoke.py` | Four API tests: `/health` payload, plan generation, no URL leakage, cache hit on repeat. |
| `tests/test_llm_integration.py` | Nine tests covering the LLM path: cache-hit skips the LLM, cache-miss invokes it, URL-bearing output is rejected, `bixby://` injection is blocked, every final deeplink is a catalog member, LLM failures fall back gracefully, LLM is off by default. |

### Debug helpers

| Path | Purpose |
|---|---|
| `debug_pipeline.py` | Runs three sample queries stage-by-stage and prints retrieval scores, extraction counts, and validation errors. Use when a plan comes back empty. |
| `debug_llm.py` | Posts directly to the LLM endpoint and shows the raw HTTP response. Bypasses the exception handling in `llm.py` so real errors are visible. |

### Project files

| Path | Purpose |
|---|---|
| `requirements.txt` | Pinned Python dependencies: `fastapi`, `uvicorn[standard]`, `pydantic`, `rank-bm25`, `numpy`, `pytest`, `httpx`. |
| `run.sh` | Bash bootstrap for macOS / Linux: creates the venv, installs deps, launches uvicorn. |
| `instructions.md` | Step-by-step guide to run the project locally. |
| `generate_reference_pdf.py` | Optional. Generates a PDF reference of the project. Uses only the Python standard library. |

## LLM provider

The LLM path is wired through an OpenAI-compatible chat-completions interface, so any provider that exposes that shape works without code changes.

**This project is currently configured for Google Gemini** on the free tier. The relevant settings live in `app/config.py` and are read from environment variables:

    LLM_PROVIDER  = gemini
    LLM_API_KEY   = <your AI Studio key>
    LLM_MODEL     = gemini-3.5-flash-lite
    LLM_BASE_URL  = https://generativelanguage.googleapis.com/v1beta/openai/

To disable the LLM and run purely on the deterministic extractor:

    LLM_PROVIDER = off

This is the default. The project runs end-to-end offline with no API key.

## Safety guarantees

- **No fabricated deeplinks.** Only `resolver.py` writes deeplink fields, and it copies strings verbatim from `deeplinks.json`.
- **No URL leakage.** The validator scans every string in the response and rejects anything that is not an exact catalog member. This includes `http://`, `https://`, `bixby://`, `www.`, and Markdown link syntax.
- **No invented steps.** The deterministic extractor returns an empty list if the reference text has no procedure. The LLM is instructed to abstain in the same situation.
- **No bypassed validation.** The LLM output is a *candidate*. It must pass Pydantic, URL-leak, and catalog-membership checks before reaching the user.
- **No poisoned cache.** Only validated plans are stored, so a cache hit is a guarantee of a previously approved response.
- **No destructive actions first.** Sequencing forces factory reset, wipe, and recovery operations to the end of the plan.

## Quick start

See `instructions.md` for the full local-run guide. The short version:

    cd samsung-troubleshoot
    python -m venv .venv
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    .venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000

Open http://localhost:8000/ in a browser.

## API contract

The competition requires exactly two endpoints.

**`GET /health`**

    {"status": "ok"}

**`POST /v1/troubleshoot`**

Request:

    {
      "query": "My Galaxy S24 Ultra screen is completely black and won't turn on",
      "sis_response": ""
    }

Response (abridged):

    {
      "query": "...",
      "query_variations": ["..."],
      "response": {
        "contexts": [
          {
            "goal": "Follow these steps to troubleshoot: ...",
            "title": "...",
            "score": 0.73,
            "actions": [
              {
                "actionName": "...",
                "description": "...",
                "category": "auto",
                "stepGroups": [
                  {
                    "steps": ["...", "..."],
                    "actionableDeeplink": {
                      "deeplink": "bixby://masked/act/...",
                      "description": "...",
                      "message": "...",
                      "originalType": "onURL"
                    },
                    "validationDeeplink": {
                      "deeplink": "bixby://masked/val/...",
                      "key": "...",
                      "resultType": "boolean",
                      "condition": "equal",
                      "value": "True"
                    }
                  }
                ]
              }
            ]
          }
        ]
      },
      "meta": {
        "cache_hit": false,
        "siis_match_score": 0.733,
        "llm_enabled": true,
        "llm_used": true,
        "llm_fallback": false,
        "schema_valid": true,
        "validation_errors": [],
        "latency_ms": 2672.87
      }
    }

The `meta` block is informational and not part of the competition schema.

## Licence

Hackathon reference code. Reuse freely for the PRISM Theme 2 submission.
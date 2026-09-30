# Smart Guided Troubleshooting Engine

**Samsung PRISM — Theme 2 submission**

Turns a vague Galaxy-device complaint into a validated, ordered, one-tap troubleshooting plan with verified in-app Settings deeplinks.

---

## What it does

You describe a problem in plain language — *"my screen is yellow"*, *"phone not charging"* — and the engine returns a structured plan: a goal, ordered actions, discrete steps, and one-tap `bixby://` deeplinks that open the exact Settings screen a step needs.

If the trusted knowledge base has no answer, the engine abstains. It returns an empty plan rather than inventing a fix.

---

## Architecture

    Complaint
       │
       ├── Query enrichment (8–10 paraphrases, multi-intent split)
       ├── Semantic cache lookup
       │     ├── HIT → return validated plan in <5 ms
       │     └── MISS ↓
       ├── Hybrid SIIS retrieval (BM25 + BGE-small embeddings, chunked)
       ├── Web fallback (Tavily, when local corpus has no answer)
       ├── Structure extraction
       │     ├── Deterministic parser (primary)
       │     └── LLM (only when deterministic produces zero actions)
       ├── Deterministic deeplink resolution (5-term scoring + 4 safety gates)
       ├── Risk-aware sequencing (auto → manual → critical)
       ├── Pydantic + URL-leak + catalog-membership validation
       ├── Cache storage (only validated plans)
       └── Response

The LLM never touches a deeplink. Only `app/resolver.py` writes deeplink fields, and it copies strings verbatim from `data/deeplinks.json`.

---

## Tech stack

| Layer | Technology |
|---|---|
| API | FastAPI, Uvicorn |
| Validation | Pydantic v2 |
| Lexical retrieval | rank-bm25 |
| Dense retrieval | sentence-transformers (BAAI/bge-small-en-v1.5) |
| Semantic cache | In-memory or Redis |
| LLM (optional) | Gemini 3.5 Flash Lite via OpenAI-compatible API |
| Web search (optional) | Tavily |
| Frontend | Vanilla HTML/CSS/JS (single file, no build step) |
| Containerization | Docker, Docker Compose |
| Testing | pytest |

---

## API

Two endpoints, matching the brief exactly.

**`GET /health`** → `{"status": "ok"}`

**`POST /v1/troubleshoot`**

    {
      "query": "My Galaxy S24 screen is completely black",
      "sis_response": ""
    }

Response contains a `contexts` array of goals, each with actions and stepGroups. Full schema in `app/schema.py`. A `meta` block carries latency, cache status, retrieval score, LLM status, confidence, and validation errors.

---

## Evaluation targets

| Metric | Target | Result |
|---|---|---|
| Schema conformance | ~100% | 100% |
| URL leakage | 0 | 0 |
| Deeplink validity | 100% catalog match | 100% |
| Cache-hit latency | P95 ≤ 300 ms | ~5 ms |
| Cold-path latency | P95 ≤ 8 s | ~4–8 s |
| Paraphrase cache hit rate | ≥ 80% | ≥ 93% |

---

## Safety guarantees

- **No fabricated deeplinks.** Only the resolver writes deeplink fields, copying from the catalog.
- **No URL leakage.** Every string scanned; anything not an exact catalog member is rejected.
- **No invented steps.** Deterministic parsing from trusted text. LLM only when structure is absent.
- **No destructive actions first.** Factory reset, wipe, recovery always last.
- **Abstains rather than guesses.** Empty plan if no trusted evidence.

---

## What's in the repo

    app/                    Python source
    data/                   578 deeplinks + 34 troubleshooting articles
    frontend/               Single-page demo UI
    tests/                  Smoke + LLM + regression suites
    eval/                   Benchmark harness

    Dockerfile
    docker-compose.yml
    requirements.txt
    instructions.md
    README.md

    PRISM_GENAI_HACKATHON_Y2026_SUBMISSION/
      TeamName_Submission.pptx    (pitch deck)
      TeamName_Demo.mp4           (demo video)

---

## Submission assets

The pitch deck and demo video are present in this repository inside `PRISM_GENAI_HACKATHON_Y2026_SUBMISSION/`. All three artifacts (PPT, video, README) are part of the tagged commit `PRISM_GENAI_HACKATHON_Y2026`.

---

## Known limitations

- **Small corpus.** 34 SIIS rows cover common categories. Uncovered queries fall to web search.
- **Masked deeplinks.** Cannot be tested on a real device during the hackathon. Handled by strict schema validation.
- **Most actions are physical.** Cleaning a charging port has no Settings screen. These are correctly labeled `manual` with no deeplink.
- **Cold start.** First request after container start takes 30–60 s for the model load. Subsequent requests are fast.

---

## Where to look in the code

- Orchestration → `app/pipeline.py`, method `TroubleshootPipeline.run()`
- Retrieval scoring → `app/retrieval.py`, `app/scoring.py`
- Safety gates → `app/fallback.py`
- Deeplink copy → `app/resolver.py`, method `ScreenResolver.resolve()`
- Validator → `app/validator.py`
- Web fallback → `app/web_search.py`

---

## Team

- **[Teammate 1]** — AI/RAG, retrieval, embeddings
- **[Teammate 2]** — LLM, backend, validators
- **[Teammate 3]** — Deeplink resolver, risk ordering
- **[Teammate 4]** — Frontend, evaluation, DevOps

---

Hackathon reference code. Free to reuse for the PRISM Theme 2 submission.
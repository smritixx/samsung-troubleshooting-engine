# Local Setup and Run Guide

How to run the Smart Guided Troubleshooting Engine.

The engine runs **fully offline by default** — no API keys required. Two optional API keys can be added for the LLM extraction path and the web search fallback.

There are two ways to run it. Pick one:

- **Docker** (recommended — one command, includes Redis, no Python install needed)
- **Manual** (Python 3.10+ on your machine)

---

## 1. Docker path (recommended)

Prerequisites: **Docker Desktop** installed and running. The whale icon in the system tray must be steady (not animating) before you proceed.

### 1.1 First-time build

Open PowerShell in the project folder and create the config file:

    Copy-Item .env.example .env

Then open `.env` in a text editor and fill in the API keys (optional — see section 3). Save.

Build the image:

    docker compose build

The first build takes 5–8 minutes. It downloads:
- The Python 3.11 base image (~50 MB)
- System packages for building Python wheels (~100 MB)
- Python dependencies including CPU-only torch (~150 MB) and the BGE-small embedding model (~130 MB)

Subsequent builds after editing Python code take under 10 seconds because Docker caches every layer.

### 1.2 Start the stack

    docker compose up

This launches two containers: the app and Redis. You should see:

    [+] Running 2/2
     ✔ Container samsung-troubleshoot-redis-1  Started
     ✔ Container samsung-troubleshoot-app-1    Started

Then the app logs stream to this terminal:

    samsung-troubleshoot-app-1  | INFO:     Uvicorn running on http://0.0.0.0:8000
    samsung-troubleshoot-app-1  | INFO:     Application startup complete.

**Leave this terminal open.** Closing it stops the containers.

Open **http://localhost:8000/** in a browser.

**First query takes 30–60 seconds** while the embedding model loads. Every query after that returns in 1–3 seconds.

### 1.3 Background mode (optional)

To free up the terminal, run detached:

    docker compose up -d

Then stream the logs separately:

    docker compose logs -f app

`Ctrl+C` stops following the logs; the containers keep running.

### 1.4 Container management

| Task | Command |
|---|---|
| Stop containers (keep them) | `docker compose stop` |
| Start stopped containers | `docker compose start` |
| Restart the app only | `docker compose restart app` |
| Stop and remove containers | `docker compose down` |
| Also remove the Redis cache volume | `docker compose down -v` |
| Rebuild after editing code | `docker compose up --build` |

### 1.5 Docker troubleshooting

**Build fails with `no valid drivers found`.** Docker Desktop is still initialising. Quit it fully (right-click tray icon → Quit), wait 10 seconds, reopen, and wait for the whale icon to stop animating. Retry.

**Build hangs on `pip install ...` for more than 10 minutes.** It's downloading the CUDA version of torch (1.5 GB). The Dockerfile installs CPU-only torch first to avoid this. If it still happens, verify the Dockerfile contains:

    RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

**`port is already allocated`.** An old Redis container from a previous run is holding port 6379:

    docker stop troubleshoot-redis
    docker rm troubleshoot-redis

Then retry `docker compose up`.

**App container exits immediately.** Check the logs:

    docker compose logs app

The last 20 lines tell you what crashed. Usually a missing `.env` (not fatal — the app runs offline without it), a malformed `data/siis_responses.json`, or a port conflict.

**Port 8000 already in use.** Edit `docker-compose.yml` and change the host port:

    ports:
      - "8001:8000"

Then open `http://localhost:8001/`.

---

## 2. Manual path (Python)

Open PowerShell in the project folder (the one containing `app/`, `data/`, `requirements.txt`).

Create a virtual environment. You only do this once:

    python -m venv .venv

Install dependencies:

    .venv\Scripts\python.exe -m pip install --upgrade pip
    .venv\Scripts\python.exe -m pip install -r requirements.txt

The first install downloads the embedding model (~130 MB). It takes 2–5 minutes. You only do this once.

If PowerShell refuses to run `.venv\Scripts\activate`, ignore it — the `.venv\Scripts\python.exe -m ...` form used above works without activation.

**Optional Redis.** To use Redis instead of the in-process cache:

    docker run -d --name troubleshoot-redis -p 6379:6379 redis:7-alpine

Then set `$env:REDIS_URL = "redis://localhost:6379/0"` before starting uvicorn. Without Redis, the cache falls back to memory and everything still works.

Start the server:

    .venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000

Open **http://localhost:8000/**. Stop with `Ctrl + C`.

---

## 3. Configure API keys (optional)

If you used the Docker path, `.env` is already in place — just fill it in. If you used the manual path, create `.env` in the project folder:

    LLM_PROVIDER=gemini
    LLM_API_KEY=paste-your-gemini-key-here
    LLM_MODEL=gemini-3.5-flash-lite
    LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
    LLM_TIMEOUT_S=8

    TAVILY_API_KEY=paste-your-tavily-key-here

**Gemini key (free):** https://aistudio.google.com/apikey — sign in with any Google account. No credit card required.

**Tavily key (free, 1000 queries/month):** https://tavily.com — create a free account.

The `.env` file is read automatically by `app/config.py`. No shell env vars required for the manual path either.

If you skip this section, the engine runs offline using only the local knowledge base. That covers all the queries the corpus handles.

To disable the LLM at any time, change `LLM_PROVIDER=off` in `.env` and restart the server.

---

## 4. Confirm which path is active

Send a query in the browser and open the "Raw JSON" expander at the bottom. Look at the `meta` block:

    "llm_enabled": true,      ← LLM is configured
    "llm_used": true,         ← LLM was actually called for this query
    "llm_fallback": false,    ← LLM succeeded; deterministic fallback not used
    "web_fallback": true      ← Tavily was used (false means local only)

- `llm_enabled: false` → no key set, or `LLM_PROVIDER=off`
- `llm_used: false` but `llm_enabled: true` → deterministic extraction was sufficient; LLM skipped for speed
- `web_fallback: false` and empty contexts → nothing matched; the engine abstained

---

## 5. Run the tests

With the server stopped (the tests spin up their own in-process client):

    .venv\Scripts\python.exe -m pytest tests/ -q

Expected: **20 passed**. Runs offline even if API keys are set — the LLM tests use mocks.

End-to-end pipeline regression suite (12 real queries against expected outcomes):

    .venv\Scripts\python.exe tests\test_pipeline_regression.py

Expected: **12/12 passed**. Takes 30–60 seconds because it calls Tavily several times.

---

## 6. Try the API directly

Leave the server running. Open a **new** PowerShell window.

Health check:

    curl.exe http://127.0.0.1:8000/health

Expected: `{"status":"ok"}`

Troubleshoot:

    curl.exe -X POST http://127.0.0.1:8000/v1/troubleshoot -H "Content-Type: application/json" -d "{\"query\":\"My Galaxy S24 screen is black\",\"sis_response\":\"\"}"

Or open **http://localhost:8000/docs** for the interactive Swagger UI.

---

## 7. What success looks like

1. `http://localhost:8000/` shows the dark-navy UI with a complaint box and example chips.
2. Clicking a chip and pressing the button produces action cards with step lists, category badges, and a `bixby://masked/...` deeplink under the steps.
3. Clicking the same chip and button again returns instantly with `"cache_hit": true`.
4. Running `pytest tests/ -q` prints `20 passed`.
5. Running `python tests\test_pipeline_regression.py` prints `12/12 passed`.

If all five hold, the project is running correctly.

---

## 8. Where to look in the code

- **The orchestration.** `app/pipeline.py` — read `TroubleshootPipeline.run()`. That method is the entire flow.
- **The LLM call.** `app/llm.py` — read `generate_structured_plan()`. Only fires when deterministic extraction produces too few actions.
- **The retrieval scoring.** `app/retrieval.py` and `app/scoring.py` — BM25, dense cosine, parent-menu penalty.
- **The semantic gate.** `app/fallback.py` — read `semantic_relevance()`. Uses embeddings + specific-token overlap to reject wrong-topic matches.
- **The safety gate.** `app/validator.py` — read `validate_response()`.
- **The deeplink copy.** `app/resolver.py` — read `ScreenResolver.resolve()`. Includes an ambiguity gate that refuses to attach a deeplink when two catalog entries score too similarly.
- **The cache.** `app/cache.py` — versioned, domain-gated, optionally Redis-backed.
- **The web fallback.** `app/web_search.py` — Tavily search, community forum rejection, marketing filter, and 3-variant retry for Tavily's non-determinism.

For architecture context, see `README.md`.
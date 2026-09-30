FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install CPU-only torch FIRST so pip does not pull the 1.5 GB CUDA stack.
# The CPU wheel is ~150 MB vs ~550 MB for the CUDA version.
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY data/ ./data/
COPY frontend/ ./frontend/
COPY tests/ ./tests/
COPY eval/ ./eval/
COPY README.md ./
COPY instructions.md* ./

# Cache the HuggingFace embedding model inside the image at build time
# so the first request is not slowed by a 130MB download.
ENV HF_HOME=/app/.hf_cache
RUN python -c "from app.embeddings import get_embedder; get_embedder()" || true

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD curl -fs http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
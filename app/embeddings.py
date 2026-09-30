"""Pluggable text embedder with sentence-transformers fallback."""
import hashlib
import math
import os
import re
from typing import List

EMBED_DIM = 256
MODEL_NAME = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
_TOKEN_RE = re.compile(r"[a-z0-9]+")

_EMBEDDER = None
_EMBEDDER_KIND = "hash"


def _hash_tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _l2_normalize(v: List[float]) -> List[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def hash_embed(text: str, dim: int = EMBED_DIM) -> List[float]:
    vec = [0.0] * dim
    for tok in _hash_tokens(text):
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16)
        vec[h % dim] += 1.0
    return _l2_normalize(vec)


def _load_sentence_transformer():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)

    def _embed(text: str) -> List[float]:
        v = model.encode(text, normalize_embeddings=True)
        return v.tolist()
    return _embed


def get_embedder():
    global _EMBEDDER, _EMBEDDER_KIND
    if _EMBEDDER is None:
        if os.environ.get("USE_REAL_EMBEDDINGS", "auto").lower() in ("1", "true", "yes", "on", "auto"):
            try:
                _EMBEDDER = _load_sentence_transformer()
                _EMBEDDER_KIND = "sentence-transformer"
            except Exception:
                _EMBEDDER = hash_embed
                _EMBEDDER_KIND = "hash"
        else:
            _EMBEDDER = hash_embed
            _EMBEDDER_KIND = "hash"
    return _EMBEDDER


def get_embedder_kind() -> str:
    get_embedder()
    return _EMBEDDER_KIND


def set_embedder(fn):
    global _EMBEDDER
    _EMBEDDER = fn


def cosine(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))
"""Cross-encoder reranker with deterministic fallback."""
import os
from typing import List

RERANK_MODEL = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

_RERANKER = None
_RERANKER_KIND = "deterministic"


def _deterministic(query: str, candidate: str) -> float:
    """Substring overlap as a cheap rerank signal."""
    from .retrieval import tokenize
    q = [t for t in tokenize(query) if len(t) >= 4]
    if not q:
        return 0.0
    c = candidate.lower()
    return sum(1 for t in q if t in c) / len(q)


def get_reranker():
    global _RERANKER, _RERANKER_KIND
    if _RERANKER is not None:
        return _RERANKER
    if os.environ.get("USE_REAL_RERANKER", "auto").lower() in ("1", "true", "yes", "on", "auto"):
        try:
            from sentence_transformers import CrossEncoder
            model = CrossEncoder(RERANK_MODEL)

            def _rerank(query: str, candidate: str) -> float:
                score = model.predict([(query, candidate)])[0]
                # Map logit to 0-1 with a sigmoid.
                import math
                return 1.0 / (1.0 + math.exp(-float(score)))
            _RERANKER = _rerank
            _RERANKER_KIND = "cross-encoder"
            return _RERANKER
        except Exception:
            pass
    _RERANKER = _deterministic
    _RERANKER_KIND = "deterministic"
    return _RERANKER


def rerank(query: str, candidates: List[str]) -> List[float]:
    """Return one score per candidate, in [0, 1]."""
    fn = get_reranker()
    return [float(fn(query, c)) for c in candidates]


def get_reranker_kind() -> str:
    get_reranker()
    return _RERANKER_KIND
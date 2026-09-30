"""Screen scoring: score = a*BM25 + b*dense + g*rerank + d*domain - l*parent."""
from typing import Optional

from .config import (
    DOMAIN_KEYWORDS, PARENT_MENU_NAMES,
    SCORE_ALPHA, SCORE_BETA, SCORE_DELTA, SCORE_GAMMA, SCORE_LAMBDA,
)
from .reranker import get_reranker
from .retrieval import tokenize


def _query_domain(query: str) -> Optional[str]:
    tokens = set(tokenize(query))
    best, best_hits = None, 0
    for dom, kws in DOMAIN_KEYWORDS.items():
        hits = sum(1 for kw in kws if kw in tokens or kw in query.lower())
        if hits > best_hits:
            best_hits, best = hits, dom
    return best


def _candidate_domain(text: str) -> Optional[str]:
    t = text.lower()
    for dom, kws in DOMAIN_KEYWORDS.items():
        if any(kw in t for kw in kws):
            return dom
    return None


def _parent_penalty(query: str, candidate_text: str, desc: str) -> float:
    q_lower = query.lower()
    hay = (candidate_text + " " + desc).lower()
    for parent in PARENT_MENU_NAMES:
        if parent in hay and parent not in q_lower:
            return 1.0
    return 0.0


def score_candidate(query: str, candidate_text: str,
                    candidate_description: str,
                    bm25_norm: float, dense_norm: float) -> float:
    rerank_fn = get_reranker()
    rerank = float(rerank_fn(query, candidate_text))

    q_dom = _query_domain(query)
    c_dom = _candidate_domain(candidate_text)
    domain_match = 1.0 if q_dom and q_dom == c_dom else 0.0
    parent = _parent_penalty(query, candidate_text, candidate_description)

    return (
        SCORE_ALPHA * bm25_norm
        + SCORE_BETA * dense_norm
        + SCORE_GAMMA * rerank
        + SCORE_DELTA * domain_match
        - SCORE_LAMBDA * parent
    )
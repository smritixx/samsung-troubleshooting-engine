"""Hybrid retrieval: BM25 + dense cosine via embedder + ANN store."""
import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from .config import SIIS_CHUNK_MAX_CHARS, SIIS_CHUNK_MIN_CHARS
from .embeddings import cosine, get_embedder
from .vector_store import VectorStore

TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "being",
    "it", "its", "i", "my", "me", "you", "your", "we", "our",
    "but", "if", "as", "at", "by", "so", "do", "does", "did",
    "can", "will", "would", "should", "could", "may", "might",
    "not", "no", "yes", "when", "then", "than", "from", "into",
    "up", "out", "about", "what", "which", "who", "whom",
    "this", "that", "these", "those", "there", "here", "have", "has",
    "had", "get", "got", "just", "also", "any", "some", "all", "one",
}


def tokenize(text: str) -> List[str]:
    return [t for t in TOKEN_RE.findall(text.lower())
            if t not in STOPWORDS and len(t) > 1]


def _char_ngrams(text: str, n: int = 3) -> Counter:
    text = " " + re.sub(r"\s+", " ", text.lower()) + " "
    return Counter(text[i:i + n] for i in range(len(text) - n + 1))


# K for the saturating BM25 transform. Higher K means a larger raw BM25
# score is needed to reach a given normalized value. K=3 puts a decent
# match (BM25 ~5) around 0.62 and a weak match (BM25 ~1.5) around 0.33.
_BM25_SATURATION_K = 3.0


class HybridIndex:
    """BM25 + dense embeddings + optional ANN store."""

    def __init__(self, docs: List[Dict[str, Any]],
                 use_dense: bool = True):
        self.docs = docs
        self.use_dense = use_dense
        self.tokenized = [tokenize(d["text"]) for d in docs]
        self.bm25 = BM25Okapi(self.tokenized)

        if use_dense:
            embedder = get_embedder()
            self.embeddings: List[List[float]] = [embedder(d["text"]) for d in docs]
            dim = len(self.embeddings[0]) if self.embeddings else 0
            self.store = VectorStore(dim) if dim else None
            if self.store:
                self.store.build(self.embeddings)
        else:
            self.embeddings = []
            self.store = None

        # Trigram fallback
        self.ngrams = [_char_ngrams(d["text"]) for d in docs]
        self.doc_norm = [
            float(sum(v * v for v in c.values())) ** 0.5 or 1.0
            for c in self.ngrams
        ]

    def _bm25_norm(self, query: str) -> np.ndarray:
        """Saturating normalization.

        Unlike dividing by max (which forces the top result to 1.0 even
        when the top result is a weak match), this maps raw BM25 through
        x / (x + K). A garbage query with raw BM25 around 1 returns a
        normalized score around 0.25, which then fails the retrieval
        threshold instead of being passed through.
        """
        q_tokens = tokenize(query)
        scores = np.asarray(self.bm25.get_scores(q_tokens), dtype=float)
        scores = np.clip(scores, 0.0, None)
        return scores / (scores + _BM25_SATURATION_K)

    def _dense_scores(self, query: str) -> np.ndarray:
        if not self.use_dense or not self.embeddings:
            q_ngrams = _char_ngrams(query)
            q_norm = float(sum(v * v for v in q_ngrams.values())) ** 0.5 or 1.0
            cos = np.zeros(len(self.docs), dtype=float)
            for i, c in enumerate(self.ngrams):
                dot = sum(q_ngrams[k] * c.get(k, 0) for k in q_ngrams)
                cos[i] = dot / (q_norm * self.doc_norm[i])
            return cos
        qv = get_embedder()(query)
        return np.asarray([cosine(qv, v) for v in self.embeddings], dtype=float)

    def search(self, query: str, top_k: int = 5,
               scorer: Optional[Callable] = None
               ) -> List[Tuple[int, float, Dict[str, float]]]:
        if not self.docs:
            return []
        bm25 = self._bm25_norm(query)
        dense = self._dense_scores(query)

        results = []
        for i, doc in enumerate(self.docs):
            if scorer is not None:
                final = scorer(query, doc["text"], doc.get("description", ""),
                               float(bm25[i]), float(dense[i]))
            else:
                final = 0.5 * float(bm25[i]) + 0.5 * float(dense[i])
            results.append((i, final, {
                "bm25": float(bm25[i]),
                "dense": float(dense[i]),
            }))
        results.sort(key=lambda r: -r[1])
        return results[:top_k]


# --- Chunking (unchanged) -------------------------------------------------

_CHUNK_HEADING_RE = re.compile(r"^(#{2,4}\s+.+?)\s*$", re.MULTILINE)


def chunk_siis(content: str, title: str) -> List[Dict[str, str]]:
    if not content:
        return []
    headings = list(_CHUNK_HEADING_RE.finditer(content))
    if not headings:
        return [{"heading": title, "text": content.strip()}]
    raw: List[Dict[str, str]] = []
    for i, m in enumerate(headings):
        start = m.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(content)
        heading = m.group(1).lstrip("# ").strip()
        body = content[start:end].strip()
        if body:
            raw.append({"heading": heading, "text": body})
    merged: List[Dict[str, str]] = []
    for c in raw:
        if merged and len(c["text"]) < SIIS_CHUNK_MIN_CHARS:
            merged[-1]["text"] += "\n" + c["heading"] + "\n" + c["text"]
        else:
            merged.append({"heading": c["heading"], "text": c["text"]})
    out: List[Dict[str, str]] = []
    for c in merged:
        if len(c["text"]) <= SIIS_CHUNK_MAX_CHARS:
            out.append(c)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", c["text"])
        buf = ""
        for s in sentences:
            if len(buf) + len(s) + 1 <= SIIS_CHUNK_MAX_CHARS:
                buf = (buf + " " + s).strip()
            else:
                if buf:
                    out.append({"heading": c["heading"], "text": buf})
                buf = s
        if buf:
            out.append({"heading": c["heading"], "text": buf})
    return out
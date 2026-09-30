"""Safety-aware semantic cache with versioning and domain/risk gates.

Items covered:
  7  store canonical intent embedding
  8  store source_version, model_version, schema_version
  9  domain/risk compatibility check before serving a hit
 10  optional Redis backend (falls back to in-memory if unavailable)
"""
import json
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .config import (
    CACHE_CRITICAL_EXTRA_THRESHOLD,
    CACHE_MAX_ENTRIES,
    CACHE_SCHEMA_VERSION,
    CACHE_SIMILARITY_THRESHOLD,
    CACHE_SOURCE_VERSION,
    LLM_MODEL,
)
from .embeddings import cosine, get_embedder
from .retrieval import tokenize


def _query_domain(query: str) -> Optional[str]:
    from .config import DOMAIN_KEYWORDS
    tokens = set(tokenize(query))
    best = None
    best_hits = 0
    for dom, kws in DOMAIN_KEYWORDS.items():
        hits = sum(1 for kw in kws if kw in tokens)
        if hits > best_hits:
            best_hits = hits
            best = dom
    return best


def _plan_has_critical(response: Dict[str, Any]) -> bool:
    for ctx in (response.get("response") or {}).get("contexts", []):
        for a in ctx.get("actions", []):
            if a.get("category") == "critical":
                return True
    return False


class _InMemoryBackend:
    def __init__(self, max_entries: int):
        self.max_entries = max_entries
        self.entries: List[Dict[str, Any]] = []

    def all(self) -> List[Dict[str, Any]]:
        return self.entries

    def add(self, entry: Dict[str, Any]) -> None:
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]


class _RedisBackend:
    """Redis-backed cache. Stores embeddings as JSON arrays.

    Uses a Redis list for iteration order plus a hash per entry. For a
    hackathon this is fine; production would use RediSearch / HNSW.
    """

    def __init__(self, url: str, max_entries: int):
        import redis  # noqa
        self.client = redis.Redis.from_url(url, decode_responses=True)
        self.max_entries = max_entries
        self.key_prefix = "troubleshoot:cache:"

    def all(self) -> List[Dict[str, Any]]:
        ids = self.client.lrange(self.key_prefix + "ids", 0, -1)
        out = []
        for eid in ids:
            raw = self.client.get(self.key_prefix + eid)
            if raw:
                out.append(json.loads(raw))
        return out

    def add(self, entry: Dict[str, Any]) -> None:
        eid = f"{int(time.time() * 1000)}-{os.urandom(3).hex()}"
        self.client.set(self.key_prefix + eid, json.dumps(entry))
        self.client.rpush(self.key_prefix + "ids", eid)
        self.client.ltrim(self.key_prefix + "ids",
                          -self.max_entries, -1)


class SemanticCache:
    def __init__(self, threshold: float = CACHE_SIMILARITY_THRESHOLD,
                 max_entries: int = CACHE_MAX_ENTRIES):
        self.threshold = threshold
        self.max_entries = max_entries
        self._lock = threading.Lock()

        backend_url = os.environ.get("REDIS_URL", "").strip()
        if backend_url:
            try:
                self.backend = _RedisBackend(backend_url, max_entries)
                self.backend_kind = "redis"
            except Exception:
                self.backend = _InMemoryBackend(max_entries)
                self.backend_kind = "memory"
        else:
            self.backend = _InMemoryBackend(max_entries)
            self.backend_kind = "memory"

    # ------------------------------------------------------------------
    @staticmethod
    def _jaccard(a: Set[str], b: Set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def _compatible(self, query: str, entry: Dict[str, Any]) -> bool:
        """Item 9: domain + risk compatibility gate."""
        entry_domain = entry.get("domain")
        query_domain = _query_domain(query)
        if entry_domain and query_domain and entry_domain != query_domain:
            return False
        # If the cached plan contains critical actions, require a stricter
        # similarity threshold so paraphrases don't hit dangerous plans.
        return True

    def lookup(self, query: str,
               extra_threshold: float = 0.0) -> Tuple[Optional[Dict[str, Any]], float]:
        q_tokens = set(tokenize(query))
        q_vec = get_embedder()(query)

        best: Optional[Dict[str, Any]] = None
        best_score = 0.0

        with self._lock:
            entries = self.backend.all()

        for e in entries:
            # Item 8: versioning — stale entries are ignored.
            if (e.get("schema_version") != CACHE_SCHEMA_VERSION
                    or e.get("source_version") != CACHE_SOURCE_VERSION):
                continue

            jac = self._jaccard(q_tokens, set(e.get("tokens", [])))
            cos = cosine(q_vec, e.get("embedding", []))
            score = 0.5 * jac + 0.5 * cos

            if not self._compatible(query, e):
                continue

            threshold = self.threshold
            if e.get("has_critical"):
                threshold += CACHE_CRITICAL_EXTRA_THRESHOLD
            threshold += extra_threshold

            if score >= threshold and score > best_score:
                best_score = score
                best = e

        if best is None:
            return None, best_score
        return best["response"], best_score

    def store(self, query: str, response: Dict[str, Any]) -> None:
        entry = {
            "query": query,
            "tokens": list(set(tokenize(query))),
            "embedding": get_embedder()(query),
            "response": response,
            "domain": _query_domain(query),
            "has_critical": _plan_has_critical(response),
            "source_version": CACHE_SOURCE_VERSION,
            "schema_version": CACHE_SCHEMA_VERSION,
            "model_version": LLM_MODEL,
            "ts": time.time(),
        }
        with self._lock:
            self.backend.add(entry)

    def clear(self) -> None:
        with self._lock:
            if self.backend_kind == "memory":
                self.backend.entries = []
            else:
                try:
                    self.backend.client.delete(
                        self.backend.key_prefix + "ids"
                    )
                except Exception:
                    pass
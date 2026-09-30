"""Metric helpers for benchmarks."""
from typing import Iterable, List


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(latencies: List[float]) -> dict:
    if not latencies:
        return {"count": 0}
    return {
        "count": len(latencies),
        "p50": round(percentile(latencies, 50), 2),
        "p95": round(percentile(latencies, 95), 2),
        "p99": round(percentile(latencies, 99), 2),
        "min": round(min(latencies), 2),
        "max": round(max(latencies), 2),
        "mean": round(sum(latencies) / len(latencies), 2),
    }


def ratio(num: int, den: int) -> float:
    return (num / den) if den else 0.0
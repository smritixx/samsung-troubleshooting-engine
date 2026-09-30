"""Cache-hit and cold-path latency benchmark."""
import time
from typing import Dict, List

from app.main import pipeline

from .metrics import summarize

QUERIES = [
    "My Galaxy S24 Ultra screen is completely black and won't turn on",
    "My Galaxy phone screen is totally cracked",
    "My Galaxy S22 touch is laggy",
    "My Galaxy A15 screen suddenly went black",
    "My Galaxy S24 camera app crashes when I open it",
]


def _time_once(q: str) -> float:
    t = time.perf_counter()
    pipeline.run(q)
    return (time.perf_counter() - t) * 1000.0


def run(cold_rounds: int = 2, warm_rounds: int = 20) -> Dict[str, Dict[str, float]]:
    pipeline.cache.clear()

    cold: List[float] = []
    for _ in range(cold_rounds):
        for q in QUERIES:
            pipeline.cache.clear()
            cold.append(_time_once(q))

    for q in QUERIES:
        pipeline.run(q)   # warm
    warm: List[float] = []
    for _ in range(warm_rounds):
        for q in QUERIES:
            warm.append(_time_once(q))

    cold_sum = summarize(cold)
    warm_sum = summarize(warm)
    print("Cold path:", cold_sum)
    print("Warm path:", warm_sum)
    print(f"Cold P95 target <= 8000 ms: {'PASS' if cold_sum.get('p95', 1e9) <= 8000 else 'FAIL'}")
    print(f"Warm P95 target <=  300 ms: {'PASS' if warm_sum.get('p95', 1e9) <= 300 else 'FAIL'}")
    return {"cold": cold_sum, "warm": warm_sum}


if __name__ == "__main__":
    run()
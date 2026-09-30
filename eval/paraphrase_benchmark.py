"""Paraphrase cache-hit benchmark (item 11: >=80%).

Seed each intent with its canonical query, then issue paraphrases of the
same intent and measure how many hit the cache.
"""
import random
from typing import Dict, List, Tuple

from app.main import pipeline
from app.query_processing import make_paraphrases

from .metrics import ratio

SEEDS = [
    "My Galaxy S24 Ultra screen is completely black and won't turn on",
    "My Galaxy phone screen is totally cracked and I cannot use it",
    "My Galaxy S22 touch is laggy and unresponsive",
    "My Galaxy S22 battery drains very fast",
    "My Galaxy A15 screen suddenly went black on its own",
    "My Galaxy S24 camera app crashes when I open it",
    "My Galaxy S23 won't connect to Wi-Fi",
    "My Galaxy S21 keeps restarting on its own",
    "My Galaxy Z Flip 6 inner screen stopped working",
    "My Galaxy S20 microphone is not working during calls",
]


def _permute(query: str, rng: random.Random) -> str:
    """Return a random paraphrase of the query that is NOT the query itself."""
    variants = [v for v in make_paraphrases(query) if v.lower() != query.lower()]
    if not variants:
        return query
    return rng.choice(variants)


def run(rounds: int = 3) -> Dict[str, float]:
    rng = random.Random(42)
    pipeline.cache.clear()

    # Warm cache with the canonical seeds.
    for seed in SEEDS:
        pipeline.run(seed)

    hits = 0
    total = 0
    per_intent: List[Tuple[str, int, int]] = []
    for seed in SEEDS:
        i_hits = 0
        i_total = 0
        for _ in range(rounds):
            para = _permute(seed, rng)
            resp = pipeline.run(para)
            i_total += 1
            total += 1
            if resp.get("meta", {}).get("cache_hit"):
                i_hits += 1
                hits += 1
        per_intent.append((seed, i_hits, i_total))

    rate = ratio(hits, total)
    print(f"Paraphrase cache hit rate: {hits}/{total} = {rate:.2%}")
    print(f"Target: 80.00%  ->  {'PASS' if rate >= 0.80 else 'FAIL'}")
    for seed, h, t in per_intent:
        print(f"  {h}/{t}  {seed[:60]}")
    return {"hit_rate": rate, "hits": hits, "total": total}


if __name__ == "__main__":
    run()
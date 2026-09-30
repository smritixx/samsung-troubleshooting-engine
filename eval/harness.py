"""Full evaluation harness (item 19).

Runs every benchmark and writes a JSON report to eval/report.json.
"""
import json
import time
from pathlib import Path
from typing import Any, Dict

from app.main import pipeline
from app.validator import validate_response

from . import latency_benchmark, paraphrase_benchmark

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "report.json"


def _schema_conformance() -> Dict[str, Any]:
    """Run the standard set of queries and check schema validity."""
    queries = [
        "My Galaxy S24 Ultra screen is completely black and won't turn on",
        "My Galaxy phone screen is totally cracked",
        "My Galaxy S22 touch is laggy and unresponsive",
        "The weather on Mars is nice today",   # irrelevant -> abstain
    ]
    total = 0
    valid = 0
    url_leaks = 0
    deeplink_invalid = 0
    fallbacks = 0

    for q in queries:
        pipeline.cache.clear()
        resp = pipeline.run(q)
        total += 1
        m = resp.get("meta", {})
        if m.get("schema_valid"):
            valid += 1
        if m.get("fallback"):
            fallbacks += 1
        if m.get("validation_errors"):
            # Look for URL leaks specifically
            for e in m["validation_errors"]:
                if "URL leakage" in e:
                    url_leaks += 1
                if "deeplink" in e.lower():
                    deeplink_invalid += 1

    return {
        "queries": total,
        "schema_valid": valid,
        "schema_pass_rate": valid / total if total else 0.0,
        "fallbacks": fallbacks,
        "url_leaks": url_leaks,
        "deeplink_invalid": deeplink_invalid,
    }


def main() -> None:
    print("=== Schema conformance ===")
    sc = _schema_conformance()
    print(json.dumps(sc, indent=2))

    print("\n=== Paraphrase benchmark ===")
    para = paraphrase_benchmark.run()

    print("\n=== Latency benchmark ===")
    lat = latency_benchmark.run()

    report = {
        "generated_at": time.time(),
        "schema_conformance": sc,
        "paraphrase": para,
        "latency": lat,
    }
    REPORT.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {REPORT}")


if __name__ == "__main__":
    main()
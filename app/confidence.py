"""Confidence scoring (items 13 and 14)."""
from typing import Any, Dict, List

from .config import CONFIDENCE_WEIGHTS


def compute_confidence(siis_score: float,
                       screen_scores: List[Dict[str, Any]],
                       schema_valid: bool,
                       num_steps_extracted: int) -> float:
    """Weighted combination of four signals, all in [0, 1]."""

    # Signal 1: retrieval strength
    siis = max(0.0, min(1.0, siis_score))

    # Signal 2: fraction of non-manual actions that got a deeplink attached
    auto = [s for s in screen_scores if s.get("score", 0.0) > 0.0]
    screen_match = (sum(s["score"] for s in auto) / len(auto)) if auto else 0.0

    # Signal 3: source coverage — proxy: was any step extracted at all
    source_coverage = 1.0 if num_steps_extracted > 0 else 0.0

    # Signal 4: schema validity
    schema = 1.0 if schema_valid else 0.0

    conf = (
        CONFIDENCE_WEIGHTS["siis_score"] * siis
        + CONFIDENCE_WEIGHTS["screen_match"] * screen_match
        + CONFIDENCE_WEIGHTS["source_coverage"] * source_coverage
        + CONFIDENCE_WEIGHTS["schema_valid"] * schema
    )
    return round(max(0.0, min(1.0, conf)), 4)
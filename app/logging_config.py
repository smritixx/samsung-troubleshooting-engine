"""Structured JSON logging (item 18)."""
import json
import logging
import sys
import time
import uuid
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": round(time.time(), 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for k in ("request_id", "stage", "latency_ms", "cache_hit",
                  "cache_similarity", "siis_match_score", "model",
                  "input_tokens", "output_tokens", "cost_usd",
                  "deeplink_ids", "validation_failures", "intent",
                  "stage_latencies"):
            v = getattr(record, k, None)
            if v is not None:
                payload[k] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


_LOGGER = None


def setup_logging(level: int = logging.INFO) -> None:
    global _LOGGER
    logger = logging.getLogger("troubleshoot")
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    logger.handlers = [handler]
    logger.propagate = False
    _LOGGER = logger


def get_logger() -> logging.Logger:
    global _LOGGER
    if _LOGGER is None:
        setup_logging()
    return _LOGGER


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]
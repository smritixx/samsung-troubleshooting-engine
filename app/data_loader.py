import json
from functools import lru_cache
from typing import Any, Dict, List

from .config import DEEPLINKS_FILE, SIIS_FILE


@lru_cache(maxsize=1)
def load_deeplinks() -> List[Dict[str, Any]]:
    with open(DEEPLINKS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["deeplinks"]


@lru_cache(maxsize=1)
def load_siis_rows() -> List[Dict[str, Any]]:
    with open(SIIS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["responses"]
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root. If the file doesn't exist, this is a
# no-op and all values fall back to environment variables or defaults.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

DEEPLINKS_FILE = DATA_DIR / "deeplinks.json"
SIIS_FILE = DATA_DIR / "siis_responses.json"
FRONTEND_DIR = BASE_DIR / "frontend"

# --- Retrieval ------------------------------------------------------------
SIIS_TOP_K = 5
DEEPLINK_TOP_K = 5

# Raised from 0.05. With saturating BM25, real matches land above ~0.45
# and garbage queries land below ~0.35.
SIIS_MIN_SCORE = 0.40

# Fraction of the query's content tokens that must appear verbatim in the
# top chunk. Guards against the case where BM25 is fooled by a common word.
SIIS_MIN_COVERAGE = 0.30

DEEPLINK_MATCH_THRESHOLD = 0.30

# --- Screen scoring weights -----------------------------------------------
SCORE_ALPHA = 0.40
SCORE_BETA = 0.30
SCORE_GAMMA = 0.15
SCORE_DELTA = 0.10
SCORE_LAMBDA = 0.15

# Three-tier SIIS retrieval thresholds.
SIIS_MIN_SCORE = 0.40          # strong match (already exists)
SIIS_SOFT_MIN_SCORE = 0.35     # approximate match floor

# Router: minimum number of chunks a token must appear in to be usable
# as a routing key. Tokens appearing in more chunks are too generic.
ROUTER_MIN_TOKEN_DOCS = 1

# Router: when abstaining, list this many known topics to the user.
ABSTAIN_TOPIC_SUGGESTIONS = 6

PARENT_MENU_NAMES = [
    "settings", "display", "battery", "connections",
    "general management", "advanced features", "notifications",
    "sound and vibration", "security and privacy", "apps",
    "device care", "about phone", "accounts and backup",
]

DOMAIN_KEYWORDS = {
    "battery":   ["battery", "charge", "charging", "power", "drain", "saver"],
    "display":   ["display", "screen", "brightness", "touch", "flicker", "black", "blank", "crack"],
    "connectivity": ["wifi", "wi-fi", "bluetooth", "network", "mobile data", "hotspot"],
    "sound":     ["sound", "volume", "ringtone", "vibration", "audio"],
    "camera":    ["camera", "lens", "photo", "video", "flash"],
    "performance": ["performance", "slow", "lag", "freeze", "memory", "ram"],
    "notifications": ["notification", "alert", "badge", "pop-up"],
    "security":  ["security", "lock", "fingerprint", "face", "password", "pattern"],
}

# --- Chunking -------------------------------------------------------------
SIIS_CHUNK_MIN_CHARS = 200
SIIS_CHUNK_MAX_CHARS = 1200

# --- Cache ----------------------------------------------------------------
CACHE_SIMILARITY_THRESHOLD = 0.65
CACHE_MAX_ENTRIES = 512
CACHE_SCHEMA_VERSION = "1.0.0"
CACHE_SOURCE_VERSION = "siis_v61"
CACHE_CRITICAL_EXTRA_THRESHOLD = 0.10

# --- Confidence -----------------------------------------------------------
CONFIDENCE_WEIGHTS = {
    "siis_score":      0.30,
    "screen_match":    0.30,
    "source_coverage": 0.20,
    "schema_valid":    0.20,
}

# --- Repair ---------------------------------------------------------------
MAX_REPAIR_ATTEMPTS = 2
MAX_REGEN_ATTEMPTS = 1

# --- Action caps ----------------------------------------------------------
MAX_ACTIONS = 8
MAX_STEPS_PER_GROUP = 6
MAX_CONTEXTS = 3

# --- Field-length constraints ---------------------------------------------
GOAL_MAX_CHARS = 200
TITLE_MAX_CHARS = 80
ACTION_NAME_MAX_CHARS = 140
DESCRIPTION_MAX_CHARS = 180
STEP_MAX_WORDS = 150

# --- LLM ------------------------------------------------------------------
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "off").strip()
LLM_API_KEY = os.environ.get("LLM_API_KEY", "").strip()
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-3.5-flash-lite").strip()
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/").strip()
LLM_TIMEOUT_S = float(os.environ.get("LLM_TIMEOUT_S", "8"))
LLM_JSON_MODE = os.environ.get("LLM_JSON_MODE", "true").lower() in ("1", "true", "yes", "on")

LLM_COST_PER_1M_INPUT_USD = float(os.environ.get("LLM_COST_PER_1M_INPUT_USD", "0.0"))
LLM_COST_PER_1M_OUTPUT_USD = float(os.environ.get("LLM_COST_PER_1M_OUTPUT_USD", "0.0"))

# --- Wave 4 toggles -------------------------------------------------------
USE_REAL_EMBEDDINGS = os.environ.get("USE_REAL_EMBEDDINGS", "auto").lower()
USE_REAL_RERANKER = os.environ.get("USE_REAL_RERANKER", "auto").lower()
USE_FAISS = os.environ.get("USE_FAISS", "auto").lower()
REDIS_URL = os.environ.get("REDIS_URL", "").strip()

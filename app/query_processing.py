"""Query processing: paraphrase generation and multi-intent splitting."""
import re
from typing import List

from .config import MAX_CONTEXTS
from .retrieval import tokenize


# --- Paraphrases ----------------------------------------------------------

_SYNONYMS = {
    "drain": "battery", "drains": "battery", "dying": "battery",
    "dies": "battery", "dead": "battery",
    "flicker": "flickers", "flickering": "flickers",
    "crack": "cracked", "cracks": "cracked",
    "lag": "laggy", "laggy": "laggy", "slow": "laggy", "sluggish": "laggy",
    "frozen": "freeze", "freezes": "freeze", "freezing": "freeze",
    "black": "blank", "dark": "blank",
    "disappears": "drains", "quickly": "fast",
    "cell": "phone", "mobile": "phone", "device": "phone",
    "won't": "will not", "wont": "will not", "can't": "cannot", "cant": "cannot",
}

_PLEASANTRIES = re.compile(
    r"\b(please|kindly|urgent(ly)?|asap|help( me)?|thanks|thank you|"
    r"hi|hello|hey|good morning|good evening)\b",
    re.IGNORECASE,
)

_SYMPTOM_OPENERS = [
    "my galaxy phone", "my galaxy", "my samsung phone", "my samsung",
    "my phone", "my tablet", "my device", "the phone",
]


def _strip_pleasantries(q: str) -> str:
    return _PLEASANTRIES.sub("", q).strip()


def _strip_opener(q: str) -> str:
    ql = q.lower()
    for op in _SYMPTOM_OPENERS:
        if ql.startswith(op):
            return q[len(op):].strip(" ,.")
    return q


def make_paraphrases(query: str) -> List[str]:
    q = query.strip().rstrip(".!?")
    if not q:
        return []

    base = q.lower()
    tokens = tokenize(base)
    keyword_only = " ".join(tokens)
    normalized = " ".join(_SYNONYMS.get(t, t) for t in tokens)
    stripped = _strip_pleasantries(q)
    openerless = _strip_opener(q)

    question = f"how do I fix this: {stripped.lower()}" if stripped else base
    complaint = f"i have a problem with my phone: {openerless.lower()}" if openerless else base
    formal = f"the following issue occurs on my galaxy device: {openerless.lower()}" if openerless else base
    short_kw = " ".join(tokens[:6])

    candidates = [
        q, base, stripped, openerless, keyword_only, short_kw,
        normalized, question, complaint, formal,
    ]

    seen = set()
    out: List[str] = []
    for c in candidates:
        c = (c or "").strip()
        if not c:
            continue
        k = c.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(c)
    return out[:10]


# --- Multi-intent splitting -----------------------------------------------

_SPLIT_MARKERS = [
    r"\s+and\s+also\s+",
    r"\s+and\s+then\s+",
    r"\s+plus\s+",
    r";\s*",
    r"\s+also\s+",
    r"\s+as well as\s+",
    r"\s+and\s+",
]

_SYMPTOM_NOUNS = {
    "screen", "display", "battery", "charge", "charger", "charging",
    "touch", "touchscreen", "sound", "volume", "camera", "photo",
    "photos", "video", "wifi", "wi-fi", "bluetooth", "network",
    "connection", "notifications", "notification", "alert", "microphone",
    "mic", "speaker", "speakers", "phone", "tablet", "device",
    "flash", "brightness", "lock", "unlock", "app", "apps",
    "keyboard", "sim", "hotspot", "storage", "memory", "performance",
}

# Word-boundary noun pattern.
_NOUN_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(n) for n in
                        sorted(_SYMPTOM_NOUNS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def _looks_like_symptom(text: str) -> bool:
    """A clause counts as a symptom only if it mentions a concrete device
    area (screen, battery, camera, ...). Predicates alone like 'won't turn
    on' are not treated as independent symptoms."""
    tokens = tokenize(text)
    if len(tokens) < 2:
        return False
    return bool(_NOUN_PATTERN.search(text))


def split_intents(query: str) -> List[str]:
    q = query.strip()
    if not q:
        return []

    for marker in _SPLIT_MARKERS:
        parts = re.split(marker, q)
        if len(parts) < 2:
            continue
        cleaned = [p.strip(" ,.") for p in parts if p.strip()]
        if len(cleaned) < 2:
            continue
        if all(_looks_like_symptom(p) for p in cleaned):
            return cleaned[:MAX_CONTEXTS]

    return [q]
"""Fallback layers, content-coverage, and title-relevance helpers."""
import re
from typing import Any, Dict, List, Optional, Set

from .retrieval import tokenize


_NOISE_RE = re.compile(
    r"^(?:"
    r"s\d+|a\d+|z\d+|note|flip|fold|ultra|plus|"
    r"galaxy|samsung|series|phone|mobile|device|tablet|new|old|"
    r"suddenly|recently|sometimes|occasionally|"
    r"completely|totally|extremely|very|really|"
    r"just|again|now|then|still|already|"
    r"always|never|often|rarely|"
    r"anyway|though|however|"
    # Conversational adverbs and hedges
    r"barely|hardly|somehow|almost|nearly|"
    r"insanely|crazy|super|super|hugely|"
    r"forever|ages|endlessly|"
    r"anymore|already|"
    r"please|kindly|thanks|"
    r"cannot|wont|cant|dont|isnt|arent|wasnt|werent|"
    r"didnt|doesnt|hasnt|havent|wouldnt|couldnt|shouldnt|"
    r"use|used|using|uses|"
    r"try|tried|trying|tries|"
    r"take|takes|taking|took|"
    r"come|comes|coming|came|"
    r"get|got|getting|"
    r"go|goes|going|went|gone|"
    r"make|makes|made|making|"
    r"do|does|doing|done|"
    r"see|seen|seeing|"
    r"plug|plugs|plugged|plugging|"
    r"keep|keeps|keeping|kept|"
    r"become|becomes|becoming|became|"
    r"start|starts|starting|started|"
    r"stop|stops|stopping|stopped|"
    r"act|acts|acting|acted|"
    r"feel|feels|feeling|felt|"
    r"look|looks|looking|looked|"
    r"issue|issues|problem|problems|"
    r"thing|things|way|ways|"
    r"lot|lots|bit|bits|kind|kinds|sort|sorts|"
    r"anyone|someone|everyone|nobody|"
    r"work|works|working|worked|"
    r"time|times|day|days|week|weeks|month|months|"
    r"use|using"
    r")$",
    re.IGNORECASE,
)


# Common semantic equivalences. When the query says "fuzzy" and the
# article says "blurry", they mean the same thing even though the
# tokens differ.
_TOKEN_SYNONYMS = {
    "fuzzy":       {"blurry", "blur", "unclear", "unfocused"},
    "blurry":      {"blur", "fuzzy", "unclear"},
    "sluggish":    {"slow", "laggy", "slowing", "lag"},
    "sluggish_":   {"slow", "laggy"},
    "laggy":       {"slow", "sluggish", "lag"},
    "slow":        {"sluggish", "laggy", "slowing"},
    "hot":         {"heat", "heats", "heating", "overheat",
                    "overheating", "overheated", "warm", "temperature"},
    "overheating": {"hot", "heat", "warm"},
    "heat":        {"hot", "overheat", "heating", "warm"},
    "dead":        {"drained", "empty", "flat", "depleted"},
    "drained":     {"dead", "empty", "depleted"},
    "empty":       {"dead", "drained"},
    "fuzzy_":      {"blurry"},
    "weird":       {"unexpected", "unusual", "abnormal", "strange"},
    "strange":     {"unexpected", "unusual", "weird"},
    "off":         {"unexpected", "abnormal"},
    "normal":      {"standard", "default"},
    "responding":  {"working", "responsive"},
    "responding_": {"working"},
    "stuck":       {"frozen", "unresponsive", "stalled"},
    "frozen":      {"stuck", "unresponsive"},
    "unresponsive": {"stuck", "frozen"},
    "yellow":      {"yellowish", "yellowed", "warm"},
    "yellowish":   {"yellow", "yellowed", "warm"},
    "dim":         {"dark", "dimmed", "dull"},
    "dark":        {"dim", "dimmed"},
    "bright":      {"brightness", "brighter"},
    "brightness":  {"bright", "brighter"},
    "drain":       {"draining", "drains", "drained"},
    "draining":    {"drain", "drains", "drained"},
    "charging":    {"charge", "charger", "charges"},
    "charge":      {"charging", "charger", "charges"},
    "photo":       {"picture", "photos", "pictures", "image", "images"},
    "picture":     {"photo", "photos", "pictures", "image"},
    "photos":      {"photo", "picture", "pictures", "images"},
    "pictures":    {"photo", "picture", "photos", "images"},
    "crash":       {"crashing", "crashes", "crashed", "stop"},
    "crashing":    {"crash", "crashes", "stop"},
    "apps":        {"app", "application", "applications"},
    "app":         {"apps", "application", "applications"},
    "responding__": {"working"},
}




# ---------------------------------------------------------------------------
# Semantic relevance
#
# Uses the loaded sentence-transformer to compare query and article text in
# embedding space. Replaces token-based and synonym-based matching with a
# single cosine similarity threshold. The model already knows that "fuzzy"
# and "blurry" are close and that "screen size" and "touch screen" are not.
# ---------------------------------------------------------------------------

_SEMANTIC_CACHE = {}


def _embed_many(texts):
    """Embed a list of texts in a single batch. Falls back to per-item
    embedding if the model does not support batching."""
    if not texts:
        return []
    # Try the batch path once
    try:
        from .embeddings import get_embedder
        embed = get_embedder()
        # Sentence-transformer models expose .encode(list) -- the
        # wrapper in embeddings.py wraps .encode(text). Check if the
        # model exposes a batch API.
        model = getattr(embed, "__self__", None)
        if model is not None and hasattr(model, "encode"):
            vecs = model.encode(list(texts), normalize_embeddings=True)
            return [v.tolist() for v in vecs]
    except Exception:
        pass
    # Fallback: per-item
    return [_embed(t) for t in texts]


def _embed(text: str):
    """Cached embedding lookup. The model is loaded once at startup,
    and results are cached so repeated calls for the same text are
    free."""
    if not text:
        return None
    # Cache key uses the full text for short strings, first 400 chars
    # for long ones. Enough to distinguish distinct inputs.
    key = text if len(text) <= 200 else text[:400]
    cached = _SEMANTIC_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        from .embeddings import get_embedder
        embed = get_embedder()
        vec = embed(text)
        _SEMANTIC_CACHE[key] = vec
        return vec
    except Exception:
        return None
    key = text[:200]
    cached = _SEMANTIC_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        from .embeddings import get_embedder
        embed = get_embedder()
        vec = embed(text)
        _SEMANTIC_CACHE[key] = vec
        return vec
    except Exception:
        return None


def semantic_similarity(a: str, b: str) -> float:
    """Cosine similarity between two texts. Returns 0.0 on failure."""
    from .embeddings import cosine
    va = _embed(a)
    vb = _embed(b)
    if va is None or vb is None:
        return 0.0
    try:
        return float(cosine(va, vb))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Specific-token gate
#
# Semantic similarity alone cannot distinguish "screen size doesn't fill"
# from "touch screen doesn't work" -- both are screen problems. This
# second gate requires the query and article to share at least one
# content word that names a specific feature, not just a category.
# ---------------------------------------------------------------------------

_CATEGORY_WORDS = {
    # Device and domain nouns
    "screen", "display", "settings", "setting",
    "phone", "device", "tablet", "samsung", "galaxy",
    "main", "restart", "reboot",
    "network", "connection", "connect", "wifi",
    "battery", "charge", "charging", "charger",
    # Issue nouns
    "issue", "issues", "problem", "problems",
    "fix", "fixes", "troubleshoot",
    # Verbs
    "enable", "disable", "turn", "toggle",
    "open", "tap", "check", "select", "view",
    "update", "updates", "page", "show",
    "options", "option", "feature", "features",
    # Negations and contractions
    "doesn", "didn", "isn", "aren", "wasn", "weren",
    "hasn", "haven", "hadn", "wouldn", "couldn", "shouldn",
    "don", "won", "cant", "cannot", "wont",
    # Connectives and modifiers
    "when", "where", "while", "with", "without",
    "from", "into", "onto", "over", "under",
    "very", "really", "quite", "still", "just", "only",
    "always", "never", "often", "sometimes",
    "much", "many", "more", "most", "less",
    "barely", "hardly", "somehow", "almost", "nearly",
    "insanely", "super", "hugely", "forever",
    "anymore", "already",
    # Generic state words
    "stays", "stay", "goes", "comes",
    "turns", "turn", "keeps", "keep",
    "works", "work", "worked", "working",
    "properly", "correctly", "normal", "normally",
    "take", "takes", "taking", "took",
    "come", "coming", "came",
    "plug", "plugs", "plugged", "plugging",
    "becoming", "became", "become",
    "start", "starts", "starting", "started",
    "stop", "stops", "stopping", "stopped",
    "act", "acts", "acting", "acted",
    "feel", "feels", "feeling", "felt",
    "look", "looks", "looking", "looked",
    "time", "times", "day", "days",
}


# Aspect-level bridges. The embedding model rates some wrong matches
# (screen size vs touch screen, 0.76) higher than some correct matches
# (sluggish vs slowing down, 0.72). Only a lexical signal can separate
# them, and the specific-token gate is that signal -- but it has false
# negatives for morphological variants of the same aspect.
#
# This map bridges those aspect-level equivalences. It is deliberately
# small (~15 aspects) and organized by problem area, not by arbitrary
# word pairs.
_ASPECT_BRIDGES = {
    # Performance
    "sluggish": ("slow", "slowing", "lag", "lagging", "performance",
                 "slowness", "slowdown"),
    "slow":     ("sluggish", "slowing", "lag", "lagging", "performance"),
    "lag":      ("sluggish", "slow", "slowing", "laggy", "performance"),
    "laggy":    ("sluggish", "slow", "slowing", "lag"),
    # Charging / battery
    "battery":  ("charg", "charging", "charge", "charger", "power"),
    "plug":     ("charg", "charging", "charge", "charger", "power"),
    "charger":  ("charg", "charging", "charge", "power"),
    # Photo / camera
    "fuzzy":    ("blurry", "blur", "unclear", "unfocused"),
    "photo":    ("camera", "picture", "image", "photos", "pictures"),
    "camera":   ("photo", "picture", "image", "photos", "pictures"),
    "picture":  ("camera", "photo", "image", "pictures"),
    # Display color cast. Every color bridges to "tint" and "hue"
    # because "yellow tint" and "yellow screen" describe the same
    # appearance problem.
    "yellow":    ("yellowish", "yellowed", "warm", "tint", "hue"),
    "yellowish": ("yellow", "yellowed", "warm", "tint", "hue"),
    "green":     ("greenish", "tint", "hue"),
    "greenish":  ("green", "tint", "hue"),
    "blue":      ("bluish", "tint", "hue"),
    "bluish":    ("blue", "tint", "hue"),
    "red":       ("reddish", "tint", "hue"),
    "reddish":   ("red", "tint", "hue"),
    "pink":      ("pinkish", "tint", "hue"),
    "pinkish":   ("pink", "tint", "hue"),
    "orange":    ("tint", "hue"),
    "purple":    ("tint", "hue"),
    "brown":     ("tint", "hue"),
    "dim":       ("dark", "dimmed", "dull", "brightness"),
    # Display visual defects. "Spots", "line", "tint", "stain" are all
    # anomalies on the panel -- they bridge to each other. "Dot" is
    # intentionally excluded because it names the camera-in-use
    # indicator, not a screen defect.
    # Display visual defects. Bridges list only OTHER anomaly words,
    # never category words like "screen" or "display" -- those appear
    # in every article and defeat the specificity gate.
    "spots":    ("tint", "line", "stain", "pixel", "dead", "discolor"),
    "spot":     ("tint", "line", "stain", "pixel", "dead", "discolor"),
    "tint":     ("spots", "spot", "line", "stain", "discolor"),
    "line":     ("tint", "spots", "spot", "stain", "discolor"),
    "stain":    ("tint", "spots", "spot", "line", "discolor"),
    "stains":   ("tint", "spots", "spot", "line", "discolor"),
    "discolor": ("tint", "spots", "spot", "line", "stain"),
    # Heat
    "hot":      ("heat", "overheat", "overheating", "warm", "temperature"),
    "heat":     ("hot", "overheat", "overheating", "warm", "temperature"),
    # Sound
    "quiet":    ("volume", "loud", "sound", "audio"),
    "loud":     ("volume", "sound", "audio"),
}


def _aspect_bridge_match(q_token: str, hay: str) -> bool:
    """True if the query token appears in hay, or if any of its aspect
    bridges appear in hay."""
    if q_token in hay:
        return True
    for bridge in _ASPECT_BRIDGES.get(q_token, ()):
        if bridge in hay:
            return True
    return False


# Color words describe an appearance, not a feature. "Green" alone
# doesn't identify a problem -- it appears in "green dot" (camera
# indicator), "green tint" (display), "green line" (panel damage),
# "green spots" (dead pixels). When a query also has a non-color
# specific token ("spots", "blurry", "laggy"), that token must match.
# Colors alone are insufficient signal.
_COLOR_WORDS = {
    "green", "greenish",
    "yellow", "yellowish", "yellowed",
    "blue", "bluish",
    "red", "reddish",
    "pink", "pinkish",
    "orange", "purple",
    "black", "white",
    "gray", "grey",
    "brown", "silver", "gold",
}


def has_specific_overlap(query: str, title: str,
                         article_head: str = "") -> bool:
    """True if the query shares at least one specific (non-category)
    content token with the title or article head.

    The small-screen query's specific tokens are {small, fill, whole}.
    A touch-screen article doesn't have those, so the check rejects.
    The blurry-photo query's specific token is {blurry}. A blurry-photo
    article has it, so the check passes.
    """
    q_all = content_tokens(query)
    if not q_all:
        return True
    q_specific = [t for t in q_all if t not in _CATEGORY_WORDS]
    if not q_specific:
        # Query is entirely category words -- fall back to bridge-aware
        # overlap so "battery" bridges to "charging" in the title.
        hay = ((title or "") + " " + (article_head or "")).lower()
        return any(_aspect_bridge_match(t, hay) for t in q_all)

    hay = ((title or "") + " " + (article_head or "")).lower()

    # Split specific tokens into colors and non-colors.
    non_color = [t for t in q_specific if t not in _COLOR_WORDS]
    colors = [t for t in q_specific if t in _COLOR_WORDS]

    if non_color:
        # The query has at least one specific non-color token
        # ("spots", "blurry", "charging"). That token MUST match.
        # Colors alone are not sufficient to identify the topic.
        return any(_aspect_bridge_match(t, hay) for t in non_color)

    # Query has only colors as specific tokens. Allow a color match.
    return any(_aspect_bridge_match(t, hay) for t in colors)


def semantic_relevance(query: str, article_title: str,
                       article_head: str = "",
                       threshold: float = 0.60) -> bool:
    """Three independent gates:

    1. Distinguishing feature / problem-class check.
       Rejects titles that name a feature or problem the query
       doesn't mention (e.g. "overheating while charging" for a
       "phone not charging" query).

    2. Specific-token overlap (with aspect bridges).

    3. Semantic similarity.
    """
    # Gate 1: distinguishing feature / problem-class check.
    if _has_different_feature(query, article_title, article_head or ""):
        return False

    # Gate 2: specific-token overlap.
    if not has_specific_overlap(query, article_title, article_head):
        return False

    # Gate 3: semantic similarity.
    haystack = (article_title or "").strip()
    if article_head:
        haystack = (haystack + ". " + article_head[:400]).strip()
    if not haystack:
        return True
    return semantic_similarity(query, haystack) >= threshold

    # Gate 1: semantic similarity
    sim = semantic_similarity(query, haystack)
    if sim < threshold:
        return False

    # Gate 2: specific token overlap
    if not has_specific_overlap(query, article_title, article_head):
        return False

    return True


def content_tokens(query: str) -> List[str]:
    """Query tokens with noise filtered out."""
    return [t for t in tokenize(query)
            if len(t) >= 3 and not _NOISE_RE.match(t)]




def _synonym_match(query: str, text: str) -> bool:
    """True if any query token matches any text token directly or via
    the synonym map."""
    q_tokens = content_tokens(query)
    if not q_tokens:
        return True
    hay = (text or "").lower()
    for t in q_tokens:
        if t in hay:
            return True
        for syn in _TOKEN_SYNONYMS.get(t, ()):
            if syn in hay:
                return True
    return False


def rarest_token(query: str, chunks: List[Dict[str, Any]]):
    """Return (token, count). count is 0 when no chunk contains the token."""
    tokens = content_tokens(query)
    if not tokens:
        return None, 0
    counts = []
    for tok in tokens:
        n = sum(1 for c in chunks if tok in c["text"].lower())
        counts.append((tok, n))
    tok, n = min(counts, key=lambda p: p[1])
    return tok, n


def query_domains(query: str) -> Set[str]:
    from .config import DOMAIN_KEYWORDS
    tokens = set(tokenize(query))
    q_lower = query.lower()
    matched: Set[str] = set()
    for dom, kws in DOMAIN_KEYWORDS.items():
        if any(kw in tokens or kw in q_lower for kw in kws):
            matched.add(dom)
    return matched


def chunk_domains(chunk_text: str) -> Set[str]:
    from .config import DOMAIN_KEYWORDS
    t = chunk_text.lower()
    matched: Set[str] = set()
    for dom, kws in DOMAIN_KEYWORDS.items():
        if any(kw in t for kw in kws):
            matched.add(dom)
    return matched


def domains_compatible(query: str, chunk_text: str) -> bool:
    q = query_domains(query)
    c = chunk_domains(chunk_text)
    if not q or not c:
        return True
    return bool(q & c)



# Specific Samsung features/actions. If an article title contains one
# of these and the user's query does not, the article is about a
# different topic -- e.g. "Screen mirroring to your Samsung TV" is
# about casting, not about a phone's display being small.
_DISTINGUISHING_WORDS = {
    "mirroring", "mirror", "casting", "cast", "streaming", "stream",
    "smart view", "smartview", "smartthings",
    "screenshot", "screenshots", "screen recording", "screen recorder",
    "screen rotation", "rotate", "rotation", "orientation",
    "split screen", "multi window", "pop-up view", "app pair",
    "backup", "back up", "restore", "sync", "syncing",
    "safe mode", "developer options", "recovery",
    "do not disturb", "dnd",
    "reset", "factory reset", "format",
    "voice assistant", "bixby", "google assistant", "alexa",
    "screen saver", "screensaver",
    "screen timeout", "auto rotate",
    "flex mode", "cover screen",
    "flip", "fold", "spen", "s pen", "air command",
    "wireless power share", "power share", "powershare",
    "reverse wireless", "reverse charging",
    # Cross-device and app names. Only FEATURE names belong here.
    # Device categories ("tablet", "pc", "tv") are excluded because
    # they appear in the titles of almost every relevant article.
    "samsung flow",
    "samsung dex",
    "link to windows",
    "smartthings", "smart view",
    "bixby",
    "samsung notes", "samsung health", "samsung pay", "samsung wallet",
    "samsung internet", "samsung pass", "samsung kids",
    "secure folder", "edge panel", "edge lighting",
    "one-handed mode",
    "wireless dex",
    # Problem classes -- titles that name a problem the query doesn't
    # mention are about a different issue. Bridges below handle
    # synonyms so "hot" still matches "overheating".
    "overheating", "overheat", "overheated",
    "laggy", "sluggish", "slowness",
    "blurry", "fuzzy", "unfocused",
    "shattered",
    "crashing", "freezes", "frozen",
}


# Device categories that indicate the article is about a different
# primary device. A phone query should not match a PC article.
_OTHER_DEVICES_RE = __import__("re").compile(
    r"\b(?:"
    r"pc|windows|chromebook|mac(?:book|os)?|"
    r"tv|smart\s+tv|qled|the\s+frame|"
    r"galaxy\s+watch|galaxy\s+buds|wearable|"
    r"monitor|display\s+monitor|"
    r"s95[a-z]|s90[a-z]|s85[a-z]|s80[a-z]|q60[a-z]|q70[a-z]|q80[a-z]|q90[a-z]|"
    r"neo\s+qled|oled\s+tv"
    r")\b",
    __import__("re").IGNORECASE,
)




def _shares_content_tokens(query, title, min_tokens=None):
    """Require content-token overlap between the query and the title.

    When min_tokens is None (the default), the requirement scales with
    the query length:
      - 1-2 content tokens in query  ->  require 1 overlap
      - 3+ content tokens in query   ->  require 2 overlap

    Short queries have too few tokens to demand two; a query like
    "phone not charging" reduces to just {"charging"} after filtering,
    and a title that mentions charging is relevant.
    """
    q = set(content_tokens(query))
    if not q:
        return True
    t = set(content_tokens(title))
    if min_tokens is None:
        # Most short queries have 1-2 semantic tokens after filtering
        # ("phone not charging" -> {charging}). Requiring two overlap
        # for a 3-token query is too strict because 1-2 of those tokens
        # are usually filler ("pictures are coming out blurry" ->
        # only "blurry" carries signal).
        min_tokens = 1 if len(q) <= 3 else 2
    return len(q & t) >= min_tokens


def _has_different_feature(query, title, heading=""):
    """True if the title names a specific feature or problem that the
    query does not mention (directly or via an aspect bridge).

    Catches two failure modes:
      1. Feature mismatch -- title names a feature the query doesn't
         (e.g. "Screen mirroring" for a "screen size" query).
      2. Problem-class mismatch -- title names a different problem
         (e.g. "overheating while charging" for a "phone not
         charging" query).
    """
    hay = (title + " " + (heading or "")).lower()
    q_lower = (query or "").lower()
    # Collect query tokens plus raw lowercased words so bridges can
    # consult the full query text.
    q_tokens = set(content_tokens(query)) | set(
        re.findall(r"[a-z]+", q_lower)
    )

    for word in _DISTINGUISHING_WORDS:
        if word not in hay:
            continue
        if word in q_lower:
            continue
        # Check whether any query token bridges to this word.
        bridged = False
        for qt in q_tokens:
            if word in _ASPECT_BRIDGES.get(qt, ()):
                bridged = True
                break
        if not bridged:
            return True
    return False


def shares_content(query: str, title: str, heading: str = "") -> bool:
    """True if at least one content token from the query appears in
    the row title or chunk heading -- directly or via a synonym.
    """
    if _has_different_feature(query, title, heading):
        return False
    hay = (title or "") + " " + (heading or "")
    return _synonym_match(query, hay)

    """True if at least one content token from the query appears in the
    row title or chunk heading.

    Uses substring matching in both directions so morphological variants
    (charging/charge, screen/screens, touch/touchscreen) match. Returns
    True when the query has no content tokens, to avoid over-filtering
    abbreviated queries like 'help'.

    This is the gate that catches false positives such as
    'phone not charging' matching 'Transfer Secure folder with Smart
    Switch' -- the titles share no content words, so the match is
    rejected and the router tries something else.
    """
    q_tokens = content_tokens(query)
    if not q_tokens:
        return True
    hay_text = (title + " " + (heading or "")).lower()
    hay_words = tokenize(hay_text)
    for t in q_tokens:
        if t in hay_text:
            return True
        for h in hay_words:
            if len(h) >= 4 and h in t:
                return True
    return False


def router_pick(query: str, chunks: List[Dict[str, Any]],
                results: List) -> Optional[Dict[str, Any]]:
    """Pick a chunk to route to when the score is low but the query
    references content that does exist in the corpus.

    Requires the rarest token in the chunk text, domain compatibility,
    and content overlap with the row title or chunk heading.
    """
    tok, count = rarest_token(query, chunks)
    if not tok or count == 0:
        return None

    for idx, _score, _comp in results:
        chunk = chunks[idx]
        if tok not in chunk["text"].lower():
            continue
        if not domains_compatible(query, chunk["text"]):
            continue
        row_title = chunk["ref"]["siis_response"].get("title", "")
        if not shares_content(query, row_title, chunk.get("description", "")):
            continue
        return chunk
    return None


def known_topics(chunks: List[Dict[str, Any]],
                 limit: int = 6) -> List[str]:
    """Return distinct article titles, not chunk headings."""
    titles = []
    seen = set()
    for c in chunks:
        # Use the parent row's title, falling back to chunk heading.
        ref = c.get("ref") or {}
        row_title = (ref.get("siis_response") or {}).get("title", "")
        title = row_title.strip() or (c.get("description") or "").strip()
        key = title.lower()
        if title and key not in seen:
            seen.add(key)
            titles.append(title)
        if len(titles) >= limit:
            break
    return titles

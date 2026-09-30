"""Web search fallback using Tavily.

Tavily returns pre-extracted article text. Samsung region sites often
return navigation chrome in raw_content, so we clean it before use and
fall back to Tavily's snippet when the cleaned text is still junk.
"""
import json
import os
import re
import urllib.error
import urllib.request
from typing import Dict, List, Optional


TAVILY_ENDPOINT = "https://api.tavily.com/search"

ALLOWED_DOMAINS = [
    "samsung.com",
    "community.samsung.com",
    "r1.community.samsung.com",
    "r2.community.samsung.com",
    "r3.community.samsung.com",
    "eu.community.samsung.com",
    "us.community.samsung.com",
]

TAVILY_TIMEOUT_S = float(os.environ.get("TAVILY_TIMEOUT_S", "20"))
TAVILY_MAX_RESULTS = int(os.environ.get("TAVILY_MAX_RESULTS", "5"))
MIN_CONTENT_CHARS = 200


# Signals that the extracted text is site chrome, not article body.
_BOILERPLATE_MARKERS = [
    # Cookies
    "this site uses cookies",
    "samsung and cookies",
    "by clicking accept",
    "we use cookies",
    "manage cookies",
    "accept all cookies",
    "cookie settings",
    # Footer / legal
    "privacy policy",
    "terms and conditions",
    "all rights reserved",
    "copyright",
    "about us",
    "site map",
    "sign up for our newsletter",
    # Marketing
    "shop by category",
    "shop by brand",
    "your cart is empty",
    "open my menu",
    "add to cart",
    # Page structure
    "global navigation",
    "footer navigation",
    "side navigation",
    "top navigation",
    "main navigation",
    "breadcrumb",
    "skip to main content",
    "skip to content",
    "alternative text",
    # Support widgets
    "chat with us",
    "thank you for your feedback",
    "please answer all questions",
    "was this helpful",
    "did this answer your question",
    "contact info",
    "contact us",
    "online",
    "call",
    "face to face support",
    "face-to-face support",
    "related questions",
    "related articles",
    "related topics",
    "you may also like",
    "see more",
    "learn more",
    # Section labels from Samsung support chrome
    "product help",
    "repairs & warranty",
    "repairs and warranty",
    "additional support",
    "support home",
    "search results",
    "popular topics",
    "recently viewed",
    # Empty placeholders
    "rc01_",
    "static content",
]


# Headings that are always page chrome. Match against the heading
# line only (after "##" stripped), case-insensitive.
_CHROME_HEADING_RE = re.compile(
    r"^\s*(?:"
    r"global navigation bar|footer navigation|side navigation|"
    r"top navigation|main navigation|breadcrumb|"
    r"contact info|contact us|"
    r"product help|repairs\s*&\s*warranty|repairs and warranty|"
    r"additional support|support home|"
    r"chat with us|thank you for your feedback|"
    r"related questions|related articles|related topics|"
    r"online|call|face to face support|face-to-face support|"
    r"glossary|"
    r"menu|navigation"
    r")\s*$",
    re.IGNORECASE,
)

# Markdown image / alt-text artifacts from Tavily.
_IMAGE_ARTIFACT_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_ALT_TEXT_RE = re.compile(r"Alternative Text\([^)]*\)", re.IGNORECASE)
_MD_HEADING_RE = re.compile(r"^#+\s*$", re.MULTILINE)
_REPEATED_PUNCT_RE = re.compile(r"([!?.,])\1{2,}")

# Common article-body start markers on Samsung support pages.
_ARTICLE_ANCHORS = [
    "what to do if",
    "how to fix",
    "how do i",
    "if your",
    "troubleshooting",
    "follow these steps",
    "try the following",
    "here are some",
]


def _is_enabled() -> bool:
    return bool(os.environ.get("TAVILY_API_KEY", "").strip())


def _tavily_search(query: str, max_results: int) -> List[Dict[str, str]]:
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        return []

    body = {
        "query": query,
        "search_depth": "basic",   # faster round-trip; advanced triples latency
        "max_results": max_results,
        "include_domains": ALLOWED_DOMAINS,
        "topic": "general",
        "include_answer": False,
        "include_raw_content": "markdown",
        "include_images": False,
    }

    data = json.dumps(body).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    req = urllib.request.Request(
        TAVILY_ENDPOINT, data=data, headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=TAVILY_TIMEOUT_S) as resp:
            raw = resp.read().decode("utf-8")
        parsed = json.loads(raw)
        out: List[Dict[str, str]] = []
        for r in parsed.get("results", []):
            if not r.get("url"):
                continue
            out.append({
                "title": str(r.get("title", "")).strip(),
                "url": str(r.get("url", "")).strip(),
                "content": str(r.get("content", "")).strip(),
                "raw_content": str(r.get("raw_content") or "").strip(),
            })
        return out
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            json.JSONDecodeError, ValueError, OSError):
        return []


def _boilerplate_score(text: str) -> int:
    """Count boilerplate markers in the text. Higher = more junk."""
    lowered = text.lower()
    return sum(1 for m in _BOILERPLATE_MARKERS if m in lowered)


# Match markdown list items that appear mid-line so we can
# push each one onto its own line before extraction.
_BULLET_SPLIT_RE = re.compile(
    r"(?<![\n\-\*])\s+(?=(?:-|\*)\s+\S)"
)

_NUMBERED_HEADING_RE = re.compile(r"^#{1,4}\s*\d+\.\s+\S", re.MULTILINE)




# Sentences that are pure marketing / store-widget / nav junk.
_SENTENCE_JUNK_RE = re.compile(
    r"\b(?:"
    r"select\s+your\s+province|"
    r"shipping\s+destination|"
    r"store\s+pick[-\s]?up|"
    r"insufficient\s+stock|"
    r"applied\s+coupon|"
    r"promotion\s+code|"
    r"cannot\s+be\s+redeemed|"
    r"tick\s+this\s+box|"
    r"to\s+proceed\s+to\s+samsung|"
    r"check\s+preferences|"
    r"recommendations?\s+for\s+you|"
    r"no\s+suggestions|"
    r"suggested\s+searches|"
    r"popular\s+searches|"
    r"recent\s+searches|"
    r"related\s+products|"
    r"recommended\s+products|"
    r"product\s+and\s+service\s+announcements|"
    r"special\s+offers,?\s+events|"
    r"newsletters|"
    r"update\s+your\s+product\s+preferences|"
    r"your\s+cart|"
    r"page\s+will\s+be\s+reloaded|"
    r"corresponding\s+prices|"
    r"remove\s+a\s+product|"
    r"are\s+you\s+sure|"
    r"confirm\s+your\s+selection|"
    r"galaxy\s+ai|"
    r"sign\s+in|"
    r"sign\s+up|"
    r"log\s+in|"
    r"register\s+now|"
    r"download\s+the\s+app|"
    r"learn\s+more|"
    r"read\s+more|"
    r"see\s+all|"
    r"view\s+all|"
    r"show\s+more|"
    r"load\s+more|"
    r"hp'?s\s+website|"
    r"visit\s+hp|"
    r"continue\s+to\s+visit|"
    r"select\s+continue|"
    r"go\s+to\s+hp|"
    r"www\.[a-z]+\.com|"
    r"click\s+here"
    r")\b",
    re.IGNORECASE,
)




# Markdown emphasis, headings, and forum-specific metadata.
_MD_HEADING_RE = __import__("re").compile(r"^\s*#{1,6}\s*", __import__("re").MULTILINE)
_MD_BOLD_RE = __import__("re").compile(r"\*\*(.+?)\*\*")
_MD_ITALIC_RE = __import__("re").compile(r"(?<!\*)\*([^*\n]+?)\*(?!\*)")
_MD_LINK_RE = __import__("re").compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_IMAGE_RE = __import__("re").compile(r"!\[[^\]]*\]\([^)]*\)")
_TIMESTAMP_RE = __import__("re").compile(
    r"\b\d{1,2}-\d{1,2}-\d{4}\s+\d{1,2}:\d{2}\s*(?:AM|PM)\b",
    __import__("re").IGNORECASE,
)
_FORUM_META_RE = __import__("re").compile(
    r"(?:"
    r"Original\s+topic:?|"
    r"Topic\s+created:?|"
    r"Last\s+edited\s+by|"
    r"Float\s+this\s+Topic|"
    r"Please\s+mark\s+any\s+response\s+that\s+helped\s+you|"
    r"as\s+your\s+solution|"
    r"Accepted\s+Solution|"
    r"Reply\s+to\s+this\s+topic"
    r")",
    __import__("re").IGNORECASE,
)


def _strip_markdown_and_forum(text):
    """Remove markdown formatting and forum metadata."""
    if not text:
        return ""
    text = _MD_IMAGE_RE.sub("", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _MD_BOLD_RE.sub(r"\1", text)
    text = _MD_ITALIC_RE.sub(r"\1", text)
    text = _MD_HEADING_RE.sub("", text)
    text = _TIMESTAMP_RE.sub("", text)
    text = _FORUM_META_RE.sub("", text)
    text = __import__("re").sub(r"\s{2,}", " ", text)
    return text.strip()


def _sentence_clean(text):
    """Split into sentences and drop any that are marketing chrome."""
    if not text:
        return ""
    parts = __import__("re").split(r"(?<=[.!?])\s+", text)
    kept = []
    for s in parts:
        s = s.strip()
        # Strip stray markdown emphasis markers.
        s = s.replace("**", "").strip()
        if not s or len(s) < 15:
            continue
        if _SENTENCE_JUNK_RE.search(s):
            continue
        kept.append(s)
    return " ".join(kept)


def _clean_article(text: str) -> str:
    """Strip nav chrome, cookie banners, SEO meta, and markdown artifacts.

    Drops everything before the first "## N. Title" numbered heading,
    which is where the real step-by-step content begins on Samsung
    support pages.
    """
    if not text:
        return ""

    # Strip image and alt-text artifacts.
    text = _IMAGE_ARTIFACT_RE.sub("", text)
    text = _ALT_TEXT_RE.sub("", text)
    text = _REPEATED_PUNCT_RE.sub(r"\1\1", text)

    text = re.sub(r"(?<!\n)\s*(#{1,4}\s+)", r"\n\1", text)

    # Also split mid-line markdown bullets so each '- step'
    # ends up on its own line. Without this, '- a - b' reads as
    # a single long step instead of two.
    text = _BULLET_SPLIT_RE.sub("\n", text)

    # If a numbered heading exists, drop everything before it. This
    # skips the title, the SEO meta description, and any top-of-page
    # navigation in one slice.
    match = _NUMBERED_HEADING_RE.search(text)
    if match:
        text = text[match.start():]

    # Drop empty markdown headings (## with no title).
    text = _MD_HEADING_RE.sub("", text)

    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]

    # Drop lines that are mostly boilerplate markers.
    cleaned = []
    for ln in lines:
        low = ln.lower()
        if any(marker in low for marker in _BOILERPLATE_MARKERS):
            continue
        # Drop chrome headings outright.
        bare = ln.lstrip("#").strip()
        if _CHROME_HEADING_RE.match(bare):
            continue
        # Drop short standalone nav labels.
        if len(ln) <= 20 and not ln.endswith("."):
            stripped = re.sub(r"[^a-zA-Z]", "", ln).lower()
            if stripped in ("discover", "shop", "support", "explore",
                            "learn", "offers", "deals", "news", "blog",
                            "contact", "call", "online"):
                continue
        cleaned.append(ln)

    joined = "\n".join(cleaned)
    # Sentence-level pass: split the whole joined text into sentences
    # and drop any that are marketing/store-widget chrome. This catches
    # cases where the source page is one run-on line, which the
    # line-based filter above cannot see.
    return _sentence_clean(joined)


_MD_LINK_RE = __import__("re").compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_IMAGE_RE = __import__("re").compile(r"!\[[^\]]*\]\([^)]*\)")


def _strip_md(text: str) -> str:
    """Remove markdown links and images from a string."""
    if not text:
        return ""
    text = _MD_IMAGE_RE.sub("", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    return text


def _content_from_result(result: Dict[str, str]) -> Optional[Dict[str, str]]:
    """Pick the best usable content from a Tavily result.

    Order of preference:
      1. Cleaned raw_content (full article) if it passes both the
         length and the boilerplate checks.
      2. Tavily's snippet (usually the most relevant paragraph).
    """
    title = result.get("title") or "Samsung support article"
    title = _strip_md(title)

    raw = result.get("raw_content", "")
    if raw:
        cleaned_raw = _strip_md(_clean_article(raw))
        # Require length and reject text still dominated by nav.
        if (len(cleaned_raw) >= MIN_CONTENT_CHARS
                and _boilerplate_score(cleaned_raw) == 0):
            return {"title": title[:200], "content": cleaned_raw}

    snippet = _strip_md(_clean_article(result.get("content", "")))
    if len(snippet) >= MIN_CONTENT_CHARS:
        return {"title": title[:200], "content": snippet}

    return None


def _has_real_prose(text: str, min_chars: int = 300) -> bool:
    """Reject cleaned text that has too little real content. Chrome-only
    pages end up as scattered labels with no paragraphs."""
    if len(text) < min_chars:
        return False
    # Count sentences of length > 30 chars.
    long_sentences = [s for s in __import__("re").split(r"(?<=[.!?])\s+", text)
                      if len(s) > 30]
    return len(long_sentences) >= 3




# Filler words that Tavily doesn't need. Stripped from a query when
# producing a fallback search phrase.
_QUERY_FILLER_RE = __import__("re").compile(
    r"\b(?:"
    r"why|what|how|when|where|which|who|"
    r"is|are|was|were|be|being|been|"
    r"the|a|an|and|or|but|so|if|then|"
    r"my|mine|me|i|we|our|you|your|"
    r"all|every|some|any|each|"
    r"from|for|with|without|about|of|to|in|on|at|by|"
    r"this|that|these|those|it|its|"
    # Added: prepositions and adverbs that slip through
    r"out|up|down|over|under|into|onto|off|"
    r"here|there|now|then|just|only|still|"
    r"very|really|quite|too|much|many|more|most|"
    r"also|even|ever|never|always|sometimes|often"
    r")\b",
    __import__("re").IGNORECASE,
)


def _simplify_query(query: str) -> str:
    """Strip filler words to produce a keyword-focused search phrase."""
    if not query:
        return ""
    cleaned = _QUERY_FILLER_RE.sub(" ", query).strip()
    cleaned = __import__("re").sub(r"\s+", " ", cleaned).strip()
    return cleaned or query


def _keyword_query(query: str) -> str:
    """Keep only nouns and adjectives that carry topical signal.

    Strips verbs too, since Tavily matches better on bare nouns
    ("blurry photo camera") than on conversational phrasing.
    """
    if not query:
        return ""
    verb_like = __import__("re").compile(
        r"\b(?:is|are|was|were|be|been|being|"
        r"come|comes|coming|came|"
        r"go|goes|going|went|"
        r"get|gets|getting|got|"
        r"keep|keeps|keeping|kept|"
        r"show|shows|showing|shown|"
        r"look|looks|looking|"
        r"seem|seems|seeming|"
        r"appear|appears|appearing|"
        r"stay|stays|staying|"
        r"turn|turns|turning|"
        r"happen|happens|happening)\b",
        __import__("re").IGNORECASE,
    )
    tokens = verb_like.sub(" ", query)
    tokens = _QUERY_FILLER_RE.sub(" ", tokens).strip()
    tokens = __import__("re").sub(r"\s+", " ", tokens).strip()
    return tokens or query





# Common semantic equivalences that token matching alone misses.
# "hot" and "overheating" describe the same problem; without this map
# the resolver rejects them as unrelated.
_TOKEN_SYNONYMS = {
    "hot":         {"heat", "heats", "heating", "overheat",
                    "overheating", "overheated", "warm", "temperature"},
    "heat":        {"hot", "heats", "heating", "overheat",
                    "overheating", "overheated", "warm"},
    "warm":        {"hot", "heat", "heats", "heating",
                    "overheat", "overheating"},
    "overheating": {"hot", "heat", "heats", "heating", "warm"},
    "fast":        {"quick", "quickly", "rapid", "rapidly", "speedy"},
    "quick":       {"fast", "rapid", "rapidly", "speedy"},
    "slow":        {"slowly", "sluggish", "lag", "laggy"},
    "laggy":       {"slow", "sluggish", "lag", "lagging"},
    "blurry":      {"blur", "fuzzy", "unclear", "unfocused"},
    "fuzzy":       {"blurry", "blur", "unclear"},
    "broken":      {"cracked", "damaged", "shattered"},
    "cracked":     {"broken", "damaged", "shattered"},
    "damaged":     {"broken", "cracked"},
    "dead":        {"drained", "empty", "flat"},
    "drained":     {"dead", "empty", "flat"},
    "won":         {"not", "cannot"},
    "cannot":      {"not", "unable"},
    "unable":      {"cannot", "not"},
    "stuck":       {"frozen", "unresponsive", "stalled"},
    "frozen":      {"stuck", "unresponsive"},
    "unresponsive": {"stuck", "frozen", "dead"},
    "yellow":      {"yellowish", "yellowed"},
    "yellowish":   {"yellow", "yellowed"},
    "green":       {"greenish"},
    "loud":        {"noisy", "noise"},
    "noisy":       {"loud", "noise"},
    "dim":         {"dark", "dimmed"},
    "dark":        {"dim", "dimmed"},
    "bright":      {"brightness", "brighter"},
    "brightness":  {"bright", "brighter"},
}


def _token_or_synonym_match(q_token: str, title_set: set) -> bool:
    if q_token in title_set:
        return True
    for syn in _TOKEN_SYNONYMS.get(q_token, ()):
        if syn in title_set:
            return True
    return False


def _title_coverage_ok(query: str, title: str,
                      article_head: str = "") -> bool:
    """Relevance check: semantic similarity + specific token overlap."""
    try:
        from .fallback import semantic_relevance
    except Exception:
        return True
    return semantic_relevance(query, title, article_head, threshold=0.60)




# Product categories that are never phones. If the URL contains one of
# these paths, the article is about a different product and cannot
# answer a phone question, regardless of word overlap.
_NON_PHONE_URL_PATTERNS = __import__("re").compile(
    r"(?:"
    r"/tv/|/televisions?/|/soundbar/|/sound-tower/|/audio/|"
    r"/monitors?/|/display[s]?/monitors?/|/monitor/|"
    r"/refrigerators?/|/washers?/|/dryers?/|/ovens?/|/vacuum/|"
    r"/air-?conditioners?/|/air-?purifiers?/|"
    r"/watches?/|/galaxy-?watch/|/wearables?/|/buds/|"
    r"/tablets?/|/galaxy-?tab/|"
    r"/laptops?/|/galaxy-?book|/notebooks?/|"
    r"/smartthings/|"
    r"/smartphones/galaxy-(?:z-)?(?:fold|flip)\d*/?$"  # product landing
    r")",
    __import__("re").IGNORECASE,
)


def _is_non_phone_url(url: str) -> bool:
    return bool(_NON_PHONE_URL_PATTERNS.search(url or ""))




# Semantic relevance threshold. Below this, the title+summary is not
# meaningfully related to the query even if words overlap.
_SEMANTIC_MIN_SIMILARITY = 0.35


def _semantic_relevance(query: str, title: str, content_head: str) -> float:
    """Cosine similarity between query and title+content_head.

    Uses the same embedder the retrieval layer uses, so the model is
    loaded once and shared across the pipeline.
    """
    try:
        from .embeddings import get_embedder, cosine
    except Exception:
        return 1.0  # no embeddings -> do not reject on this signal

    embed = get_embedder()
    if embed is None:
        return 1.0

    haystack = (title + ". " + (content_head or "")[:400]).strip()
    if not haystack:
        return 1.0

    try:
        q_vec = embed(query)
        h_vec = embed(haystack)
        return float(cosine(q_vec, h_vec))
    except Exception:
        return 1.0


def _try_fetch(query: str) -> Optional[Dict[str, str]]:
    """One pass: search Tavily and return the first usable result."""
    if not query:
        return None

    try:
        from .fallback import _has_different_feature
    except Exception:
        _has_different_feature = None

    results = _tavily_search(query, TAVILY_MAX_RESULTS)

    # Drop community forum threads. They contain dates, timestamps,
    # replies, and "please mark as solution" meta text. Prefer an
    # official support article, or abstain.
    def _is_forum(url):
        u = url.lower()
        return ("community.samsung.com" in u
                or "/t5/" in u
                or "r1.community" in u
                or "r2.community" in u
                or "r3.community" in u
                or "eu.community" in u
                or "us.community" in u)

    results = [r for r in results if not _is_forum(r.get("url", ""))]

    # Pages that describe products or list specs, not troubleshoot.
    _MARKETING_RE = __import__("re").compile(
        r"\b(?:"
        r"comparisons?|compare|versus|vs\.?|"
        r"faqs?|"
        r"specs?|specifications?|"
        r"product\s+page|"
        r"buy\s+now|shop\s+now|order\s+now|"
        r"model\s+explained|models\s+explained|"
        r"what\s+is\s+a\s+galaxy|"
        r"introducing\s+the|"
        r"discover\s+the|"
        r"explore\s+the"
        r")\b",
        __import__("re").IGNORECASE,
    )
    # If the title has a marketing term AND doesn't mention a
    # troubleshooting verb, it is a marketing page, not a fix.
    _TROUBLESHOOT_HINT_RE = __import__("re").compile(
        r"\b(?:fix|fixes|fixing|troubleshoot|troubleshooting|"
        r"not\s+working|doesn'?t|won'?t|"
        r"issue|issues|problem|problems|error|errors|"
        r"resolve|resolving|solve|solving|"
        r"how\s+to\s+(?:fix|troubleshoot|resolve))\b",
        __import__("re").IGNORECASE,
    )

    # Evaluate every candidate and pick the one with the most
    # extractable steps. Tavily's ranking is non-deterministic across
    # calls, and a bad top result can otherwise shadow a better one
    # further down. Scoring by step count makes the pipeline stable
    # regardless of which article Tavily ranks first.
    try:
        from .extractor import extract_actions as _extract
    except Exception:
        _extract = None

    best = None
    best_steps = -1

    for result in results:
        url = result["url"]
        if not any(d in url for d in ALLOWED_DOMAINS):
            continue
        # URL-based device check: TV, audio, appliance, watch pages
        # are never phone troubleshooting.
        if _is_non_phone_url(url):
            continue
        title = _strip_markdown_and_forum(result.get("title", "") or "")
        if _has_different_feature is not None and _has_different_feature(query, title):
            continue
        if _MARKETING_RE.search(title) and not _TROUBLESHOOT_HINT_RE.search(title):
            continue
        if not _title_coverage_ok(query, title):
            continue
        try:
            from .fallback import _shares_content_tokens
            if not _shares_content_tokens(query, title):
                continue
        except Exception:
            pass

        built = _content_from_result(result)
        if built is None:
            continue
        if not _has_real_prose(built.get("content", "")):
            continue

        # Semantic relevance gate: the title + first 400 chars of the
        # article body must be meaningfully similar to the query.
        # This catches wrong-topic matches that pass keyword filters
        # ("Big Screen TVs" for "screen looks big").
        sim = _semantic_relevance(
            query, title, built.get("content", ""),
        )
        if sim < _SEMANTIC_MIN_SIMILARITY:
            continue

        # Score by the number of extractable steps. More steps means
        # the article has a real procedure, not just a stray sentence.
        step_count = 0
        if _extract is not None:
            try:
                actions = _extract({"title": title,
                                    "content": built.get("content", "")})
                step_count = sum(len(a["steps"]) for a in actions)
            except Exception:
                step_count = 0

        # Prefer more steps; break ties by keeping the first (Tavily
        # already ranked it higher).
        if step_count > best_steps:
            best_steps = step_count
            best = built

    return best


def fetch_fallback_steps(query: str) -> Optional[Dict[str, str]]:
    """Try three query variants in order.

    Tavily's result ranking is non-deterministic across calls, and the
    same query can return different top results within seconds. Trying
    multiple phrasings increases the chance of finding a support
    article.
    """
    if not _is_enabled():
        return None

    variants = [
        query,
        _simplify_query(query),
        _keyword_query(query),
    ]
    seen = set()
    for v in variants:
        v = (v or "").strip()
        if not v or v in seen:
            continue
        seen.add(v)
        result = _try_fetch(v)
        if result:
            return result

    return None

    # First attempt: use the query as-is.
    result = _try_fetch(query)
    if result:
        return result

    # Second attempt: strip filler words.
    simplified = _simplify_query(query)
    if simplified and simplified != query:
        result = _try_fetch(simplified)
        if result:
            return result

    return None

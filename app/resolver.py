"""Screen resolver.

Maps a step's text to the correct catalog entry using the 5-term scoring
function. Never constructs a URI: only returns catalog entries verbatim.

A content-coverage gate rejects matches that come from a single common
word shared with the entry. For short signals (single-step actions) the
coverage bar is relaxed slightly, since fewer tokens means each miss
costs proportionally more.
"""
import re
from typing import Any, Dict, List, Optional

from .config import DEEPLINK_MATCH_THRESHOLD, DEEPLINK_TOP_K
from .retrieval import HybridIndex, tokenize
from .scoring import score_candidate


# Words that carry no topical signal in an action name. Excluded from
# the coverage check so "check", "try", "use" don't dilute the ratio.
_GENERIC_VERBS = {
    # Generic verbs
    "check", "try", "use", "used", "using", "make", "made",
    "get", "got", "getting", "see", "seen", "seeing",
    "go", "goes", "going", "went", "gone",
    "do", "did", "does", "doing", "done",
    "have", "has", "had", "take", "took", "put", "keep",
    "increase", "decrease", "adjust", "change", "set",
    "disable", "enable", "prevent", "avoid", "resolve", "fix",
    # Navigation verbs -- carry no topical signal for coverage
    "open", "tap", "select", "navigate", "press", "choose",
    "scroll", "click", "touch", "swipe", "hold", "find",
    "locate", "access", "reach",
    # Filler nouns, ubiquitous enough that their presence
    # does not discriminate one entry from another
    "options", "option", "menu", "screen", "screen.",
    "display", "buttons", "button", "full", "function",
    "turn",
    # Direction and state prepositions
    "off", "on", "up", "down", "in", "out", "before", "after",
    # Brand and ubiquitous device names -- present in every entry
    "phone", "samsung", "galaxy", "device", "tablet",
    # Negations and connectives -- carry no topical signal, but they
    # appear in almost every sentence and create false token overlaps.
    "doesn", "didn", "isn", "aren", "wasn", "weren",
    "hasn", "haven", "hadn", "wouldn", "couldn", "shouldn",
    "don", "won",
    "while", "when", "where", "with", "without", "from",
}



# Read-only entries only *display* a value; they don't change anything.
# They should never be returned for a troubleshooting step, which always
# asks the user to act on a setting.
_READ_ONLY_DESC_RE = re.compile(
    r"^\s*(retrieves|checks|displays|shows|views|reads|monitors|"
    r"view|check|display|show|read|monitor)\b",
    re.IGNORECASE,
)



# TV, refrigerator, air conditioner, and other non-phone devices share
# the catalog but must not match phone troubleshooting steps.
_NON_PHONE_DEVICE_RE = re.compile(
    r"\b(?:"
    r"via TV Settings|TV Settings|on TV|TV Bixby|"
    r"Air Conditioner|Refrigerator|"
    r"Anynet\+|Motion Lighting|"
    r"Picture Mode|Dynamic Contrast|Energy Saving Solution|"
    r"Auto Screen Dim|Audio Format|Voice Wake-up|Voice Guide|"
    r"Sound Mode|Brightness Optimization"
    r")\b",
    re.IGNORECASE,
)


def _is_non_phone_entry(entry):
    """True for entries that target non-phone devices."""
    desc = entry.get("description") or ""
    qna = entry.get("qna_description") or ""
    msg = entry.get("message") or ""
    return bool(_NON_PHONE_DEVICE_RE.search(desc + " " + qna + " " + msg))


def _is_read_only_entry(entry):
    return bool(_READ_ONLY_DESC_RE.match((entry.get("description") or "").strip()))


def _content_coverage(text: str, entry: Dict[str, Any]) -> float:
    """Fraction of the text's substantive tokens that appear in the
    matched entry's description / qna / message."""
    seen = set()
    tokens = []
    for t in tokenize(text):
        # Strip a single trailing "s" so "gestures" matches "gesture"
        # and "buttons" matches "button".
        if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        if len(t) < 3 or t in _GENERIC_VERBS or t in seen:
            continue
        seen.add(t)
        tokens.append(t)
    if not tokens:
        return 1.0
    # Only `description` and `qna_description` are semantic. `message`
    # is the UI label shown on the Settings toggle and often contains
    # words from other features (e.g. corner-actions has message
    # "Disable Gesture Controls"), which pollutes the match.
    hay = " ".join([
        entry.get("description", "") or "",
        entry.get("qna_description", "") or "",
    ]).lower()
    hits = sum(1 for t in tokens if t in hay)
    return hits / len(tokens)


# Category words that appear in dozens of catalog entries. A query
# that shares ONLY these words with an entry has not identified a
# specific feature, so no deeplink should be attached.
_CATEGORY_WORDS = {
    "restart", "reboot", "device", "phone", "tablet",
    "screen", "display", "settings", "setting",
    "mode", "network", "connection", "connect",
    "enable", "disable", "enables", "disables",
    "turn", "toggle", "change", "adjust", "adjusts",
    "reset", "resets", "open", "opens", "view", "views",
    "check", "checks", "page", "show", "shows",
    "update", "updates", "set", "sets",
    "option", "options", "feature", "features",
}


def _has_specific_overlap(query: str, entry: dict) -> bool:
    """True if the query and entry share at least one token that is
    NOT a generic category word. Restart/screen/network alone are
    not enough to identify a feature."""
    from .retrieval import tokenize
    q_tokens = [
        t for t in tokenize(query)
        if len(t) >= 3 and t not in _CATEGORY_WORDS
    ]
    if not q_tokens:
        return False
    hay = " ".join([
        entry.get("description", "") or "",
        entry.get("qna_description", "") or "",
    ]).lower()
    return any(t in hay for t in q_tokens)


def _all_share_feature(query: str, entries: list, min_shared: int = 2) -> bool:
    """True if every entry shares at least min_shared of the query's
    content tokens in its description or qna.

    Detects enable/disable pairs and other same-feature variants.
    "Turn off Eye Comfort Shield" and both "Enables..." / "Disables..."
    entries share the tokens eye, comfort, shield -- so they are the
    same feature, not ambiguous.
    """
    from .retrieval import tokenize
    q_tokens = [t for t in tokenize(query) if len(t) >= 3]
    if not q_tokens:
        return False
    required = min(min_shared, len(q_tokens))
    for entry in entries:
        hay = (
            (entry.get("description") or "") + " "
            + (entry.get("qna_description") or "") + " "
            + (entry.get("message") or "")
        ).lower()
        shared = sum(1 for t in q_tokens if t in hay)
        if shared < required:
            return False
    return True


class ScreenResolver:
    def __init__(self, deeplinks: List[Dict[str, Any]]):
        self.deeplinks = deeplinks
        docs = []
        for d in deeplinks:
            text = " ".join(filter(None, [
                d.get("description", ""),
                d.get("qna_description", ""),
                d.get("message", ""),
            ]))
            docs.append({
                "id": d["id"],
                "text": text,
                "description": d.get("description", ""),
            })
        self.index = HybridIndex(docs)

    def resolve(self, text: str,
                top_k: int = DEEPLINK_TOP_K) -> Optional[Dict[str, Any]]:
        """Return the top catalog entry if it clears score and coverage."""
        entry, _score, _comp, _ids = self.resolve_with_components(
            text, top_k=top_k, topic=text,
        )
        return entry

    def resolve_with_components(self, text: str,
                                top_k: int = DEEPLINK_TOP_K,
                                topic: str = ""):
        """Walk candidates in score order and return the best that
        passes all filters.

        Adds an ambiguity gate that refuses to pick when the top two
        candidates describe genuinely different features. Pairs like
        "Enable Fast charging" / "Disable Fast charging" are not
        ambiguous -- they are the same feature with opposite toggles,
        and the query verb tells us which one to pick.
        """
        results = self.index.search(text, top_k=max(top_k, 20), scorer=None)
        if not results:
            return None, 0.0, {}, []

        ranked_ids = [self.deeplinks[i]["id"] for i, _, _ in results]
        coverage_input = topic if topic else text

        tokens = [t for t in tokenize(coverage_input) if len(t) >= 3]
        min_coverage = 0.6 if len(tokens) <= 4 else 0.5

        passing = []
        for idx, score, comp in results:
            if score < DEEPLINK_MATCH_THRESHOLD:
                break
            entry = self.deeplinks[idx]
            if _is_read_only_entry(entry):
                continue
            if _is_non_phone_entry(entry):
                continue
            coverage = _content_coverage(coverage_input, entry)
            if coverage < min_coverage:
                continue
            # Require a specific (non-category) token to overlap. This
            # stops generic step text like "Restart the device" from
            # matching the wrong restart-related entry.
            if not _has_specific_overlap(coverage_input, entry):
                continue
            passing.append((idx, score, comp, entry))

        if not passing:
            top_idx, top_score, top_comp = results[0]
            return None, top_score, top_comp, ranked_ids

        # Ambiguity gate. Only refuse if the top two candidates are
        # about DIFFERENT features. If they are enable/disable pairs
        # of the same feature, pick the highest-scored one.
        if len(passing) >= 2:
            top_score = passing[0][1]
            second_score = passing[1][1]
            if top_score > 0 and (second_score / top_score) >= 0.92:
                top_two_entries = [passing[0][3], passing[1][3]]
                if not _all_share_feature(coverage_input, top_two_entries):
                    top_idx, top_score, top_comp, _ = passing[0]
                    return None, top_score, top_comp, ranked_ids
                # else: same feature, fall through and pick the top

        top_idx, top_score, top_comp, entry = passing[0]
        return entry, top_score, top_comp, ranked_ids

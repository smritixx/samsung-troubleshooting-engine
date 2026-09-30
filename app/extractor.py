"""Deterministic extraction of actions and step groups from SIIS or web content.

Parses the Markdown-ish text into a list of (heading, body) blocks, then
turns each block into an Action. Applies filters to keep only actionable
steps and strip promissory/lead-in/note chrome.

No LLM is used here -- the point of the engine is to stay grounded in the
reference text and run offline.
"""
import re
from typing import Any, Dict, List, Tuple

from .config import MAX_ACTIONS, MAX_STEPS_PER_GROUP

STEP_HEADING_RE = re.compile(
    r"^#{2,4}\s*(?:\*\*)?Step\s*\d+\s*[:.\-]?\s*(.+?)(?:\*\*)?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
H2_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
LIST_MARKER_RE = re.compile(r"^(?:[-\*\u2022]|\d+\.)\s+")
MD_EMPHASIS_RE = re.compile(r"\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`")


# --- Filters --------------------------------------------------------------

_PROMISSORY_RE = re.compile(
    r"[\s,]*"
    r"(?:"
    r"(?:follow|try)\s+these\s+(?:steps|actions|instructions)"
    r"(?:\s+to\s+[^:.\n]{1,100})?"
    r"|here'?s\s+how\s+to\s+[^:.\n]{1,100}"
    r"|here\s+is\s+how\s+to\s+[^:.\n]{1,100}"
    r"|here\s+are\s+the\s+steps"
    r"|try\s+the\s+following"
    r"|do\s+the\s+following"
    r"|complete\s+the\s+following"
    r"|perform\s+the\s+following"
    r")"
    r"[:.]?\s*$",
    re.IGNORECASE,
)

_LEADIN_RE = re.compile(
    r"^\s*to\s+\w+(?:\s+\w+){0,5}\s*[:.]\s*$",
    re.IGNORECASE,
)

_NOTE_PREFIX_RE = re.compile(
    r"^\s*(?:Note|Important|Tip|Warning|Caution)\s*[:\-]\s*",
    re.IGNORECASE,
)

_INLINE_STEP_RE = re.compile(
    r"\s+(?=Step\s+\d+[.:]\s)",
    re.IGNORECASE,
)

# Leading "Step N." prefix that remains after inline splitting.
_STEP_PREFIX_RE = re.compile(
    r"^\s*Step\s+\d+\s*[.:]\s*",
    re.IGNORECASE,
)

_OBSERVATION_RE = re.compile(
    r"^\s*(?:"
    r"the\s+\w+|"
    r"it(?:'s| is)\s+|"
    r"they(?:'s|\s+are)\s+|"
    r"this\s+(?:means|is|will)\s+|"
    r"next,\s+let(?:'s| us)\s+|"
    r"note\s+that\s+|"
    r"keep\s+in\s+mind\s+|"
    r"depending\s+on\s+|"
    r"when\s+you\s+|"
    r"if\s+you\s+have\s+"
    r")\b",
    re.IGNORECASE,
)

_IMPERATIVE_RE = re.compile(
    r"^\s*(?:"
    r"check|verify|inspect|examine|look|remove|insert|connect|"
    r"disconnect|plug|unplug|press|hold|release|tap|open|close|"
    r"navigate|select|choose|enter|type|swipe|drag|toggle|enable|"
    r"disable|turn|set|change|adjust|increase|decrease|raise|lower|"
    r"try|use|charge|let|wait|restart|reboot|power|boot|clean|wipe|"
    r"dry|place|put|keep|make|ensure|confirm|test|run|start|stop|"
    r"apply|save|delete|clear|reset|update|install|download|go|"
    r"contact|visit|call|send|bring|report|uninstall|backup|back\s+up|"
    r"restore|reinstall|take|move|find|locate|access|reach|scroll"
    r")\b",
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    text = MD_EMPHASIS_RE.sub(lambda m: next(g for g in m.groups() if g), text)
    return text.strip()


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


_IF_CLAUSE_RE = re.compile(
    r"^\s*(?:if|when)\s+[^,]{3,80},\s*",
    re.IGNORECASE,
)


def _is_imperative(s: str) -> bool:
    t = s.strip()
    if not t:
        return False
    if _OBSERVATION_RE.match(t):
        return False
    if _IMPERATIVE_RE.match(t):
        return True
    # "If X, Y." / "When X, Y." -- a conditional intro followed by a
    # substantive clause is treated as a step. This matches Samsung's
    # writing style, e.g. "If this is your first time using the app,
    # the app may require permissions from you."
    stripped = _IF_CLAUSE_RE.sub("", t)
    if stripped != t and len(stripped) >= 15:
        return True
    return False


def _split_inline_steps(text: str) -> List[str]:
    if not text:
        return []
    parts = _INLINE_STEP_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _strip_prefixes(s: str) -> str:
    s = _clean(s)
    s = _STEP_PREFIX_RE.sub("", s).strip()
    s = _NOTE_PREFIX_RE.sub("", s).strip()
    s = _PROMISSORY_RE.sub("", s).strip()
    return s.strip()




# Light tokenizer for dedup -- keeps "check", "tap", etc. so we can
# tell which sentences are different in meaning.
_DEDUP_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "for", "in", "on",
    "at", "by", "with", "is", "are", "was", "were", "be", "as",
    "it", "its", "this", "that", "these", "those",
    "then", "next", "also",
    "your", "you", "we", "our",
}

# A lone trailing number after sentence-ending punctuation:
# "...updates. 2" -> "...updates."  This catches the case where a
# numbered list marker got absorbed into the previous sentence.
_TRAILING_LIST_NUM_RE = __import__("re").compile(
    r"([.!?])\s+\d{1,2}\s*$"
)


def _dedup_tokens(text):
    """Return a set of content tokens for similarity checks."""
    if not text:
        return set()
    raw = __import__("re").findall(r"[a-zA-Z0-9]+", text.lower())
    return {t for t in raw if len(t) > 1 and t not in _DEDUP_STOPWORDS}


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _dedupe_steps(steps):
    """Drop steps that are redundant with an earlier one.

    Two rules:
      1. Subset rule: if step B's tokens are entirely contained in
         step A's, drop B (A is more specific).
      2. Near-duplicate rule: if step B shares >=80% of its tokens
         with step A, drop B (A and B say the same thing with
         different phrasing).

    Steps with fewer than 2 tokens are always kept -- they carry no
    redundant content to filter out.
    """
    kept = []
    kept_tokens = []
    for s in steps:
        toks = _dedup_tokens(s)
        if len(toks) < 2:
            kept.append(s)
            kept_tokens.append(toks)
            continue
        drop = False
        for ktoks in kept_tokens:
            if len(ktoks) < 2:
                continue
            # Rule 1: subset
            if len(ktoks) >= len(toks) and toks.issubset(ktoks):
                drop = True
                break
            # Rule 2: Jaccard >= 0.80
            if _jaccard(toks, ktoks) >= 0.75:
                drop = True
                break
        if not drop:
            kept.append(s)
            kept_tokens.append(toks)
    return kept


def _trim_trailing_list_num(s):
    """Strip a lone trailing list number: "updates. 2" -> "updates." """
    return _TRAILING_LIST_NUM_RE.sub(r"\1", s)


def _extract_steps(body: str) -> List[str]:
    body = re.sub(r"^#+\s+.*$", "", body, flags=re.MULTILINE)

    raw: List[str] = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        line = LIST_MARKER_RE.sub("", line)
        line = _strip_prefixes(line)

        if _LEADIN_RE.match(line):
            continue
        if len(line) < 5:
            continue

        inline = _split_inline_steps(line)
        if len(inline) > 1:
            for piece in inline:
                piece = _strip_prefixes(piece)
                if _LEADIN_RE.match(piece):
                    continue
                if 12 <= len(piece) <= 250 and _is_imperative(piece):
                    raw.append(piece)
            continue

        if len(line) <= 250:
            if _is_imperative(line):
                raw.append(line)
        else:
            for s in _split_sentences(line):
                s = _strip_prefixes(s)
                if 12 <= len(s) <= 250 and _is_imperative(s):
                    raw.append(s)

    # Strip trailing list-marker numbers from each step.
    raw = [_trim_trailing_list_num(s).strip() for s in raw]
    raw = [s for s in raw if s]

    # Exact-match dedup (fast path).
    seen = set()
    out: List[str] = []
    for s in raw:
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)

    # Semantic dedup: drop near-duplicates by token subset + Jaccard.
    # Safety: if dedup would drop everything, keep the original list
    # so the pipeline still has at least one step to work with.
    deduped = _dedupe_steps(out)
    if deduped:
        out = deduped
    # else: keep the pre-dedup list

    return out[:MAX_STEPS_PER_GROUP]


def _parse_blocks(siis: Dict[str, Any]) -> List[Tuple[str, str]]:
    content: str = siis.get("content", "")
    title: str = siis.get("title", "Troubleshooting")

    matches = list(STEP_HEADING_RE.finditer(content))
    if matches:
        blocks: List[Tuple[str, str]] = []
        for i, m in enumerate(matches):
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            blocks.append((m.group(1).strip(), content[start:end]))
        return blocks

    h2s = list(H2_HEADING_RE.finditer(content))
    if h2s:
        blocks = []
        for i, m in enumerate(h2s):
            start = m.end()
            end = h2s[i + 1].start() if i + 1 < len(h2s) else len(content)
            heading = m.group(1).strip()
            heading = re.sub(r"^\d+\.\s*", "", heading)
            blocks.append((heading, content[start:end]))
        return blocks

    return [(title, content)]


# Minimum steps for an action to be considered useful. Actions with
# only one step are usually a support-contact link or an ad fragment,
# not a real troubleshooting procedure.
_MIN_USEFUL_STEPS = 2


def _is_useful_action(steps: List[str]) -> bool:
    if len(steps) >= _MIN_USEFUL_STEPS:
        return True
    # Single-step actions are only useful when the step is a real
    # command that mentions a Samsung feature or setting.
    if not steps:
        return False
    s = steps[0].lower()
    if "hp" in s or "website" in s or "continue" in s:
        return False
    if "contact" in s and "support" in s:
        return False
    return True


def extract_actions(siis: Dict[str, Any]) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    for heading, body in _parse_blocks(siis):
        steps = _extract_steps(body)
        if not steps:
            continue
        if not _is_useful_action(steps):
            continue
        actions.append({
            "actionName": heading[:120] or "Troubleshooting step",
            "description": steps[0][:180],
            "steps": steps,
        })
        if len(actions) >= MAX_ACTIONS:
            break
    return actions

"""Deterministic validator with repair (items 15, 16).

Handles two response shapes:
  - {"contexts": [...]}                 (Pydantic dump used by pipeline.py)
  - {"response": {"contexts": [...]}}   (public API wrapper)
"""
import re
from typing import Any, Dict, List, Set, Tuple

from .config import (
    ACTION_NAME_MAX_CHARS, DESCRIPTION_MAX_CHARS, GOAL_MAX_CHARS,
    STEP_MAX_WORDS, TITLE_MAX_CHARS,
)

URL_LEAK_RE = re.compile(
    r"https?://|bixby://|www\.|\[[^\]]*\]\([^)]+\)",
    re.IGNORECASE,
)

_STRIP_EMBEDDED_URL_RE = re.compile(
    r"https?://\S*|bixby://\S*|www\.\S*|\[[^\]]*\]\([^)]*\)",
    re.IGNORECASE,
)


def _walk_strings(obj: Any, path=()):
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_strings(v, path + (str(k),))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _walk_strings(v, path + (str(i),))


def _get_contexts(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the contexts list regardless of which shape was passed in."""
    direct = response.get("contexts")
    if isinstance(direct, list):
        return direct
    wrapped = response.get("response", {})
    if isinstance(wrapped, dict):
        inner = wrapped.get("contexts")
        if isinstance(inner, list):
            return inner
    return []


def validate_response(response: Dict[str, Any],
                      catalog_uris: Set[str]) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    # URL leakage across the entire tree.
    for path, s in _walk_strings(response):
        if s in catalog_uris:
            continue
        if URL_LEAK_RE.search(s):
            where = ".".join(path) or "<root>"
            # Do NOT include the offending string in the message; that
            # would leak the URL back into the response via meta.
            errors.append(f"URL leakage at {where}")
            break

    # Catalog membership + word/char limits.
    contexts = _get_contexts(response)
    for ci, ctx in enumerate(contexts):
        if len(ctx.get("goal", "")) > GOAL_MAX_CHARS:
            errors.append(f"contexts[{ci}].goal exceeds {GOAL_MAX_CHARS} chars")
        if len(ctx.get("title", "")) > TITLE_MAX_CHARS:
            errors.append(f"contexts[{ci}].title exceeds {TITLE_MAX_CHARS} chars")
        for ai, action in enumerate(ctx.get("actions", [])):
            if len(action.get("actionName", "")) > ACTION_NAME_MAX_CHARS:
                errors.append(f"contexts[{ci}].actions[{ai}].actionName too long")
            if len(action.get("description", "")) > DESCRIPTION_MAX_CHARS:
                errors.append(f"contexts[{ci}].actions[{ai}].description too long")
            for sgi, sg in enumerate(action.get("stepGroups", [])):
                ad = sg.get("actionableDeeplink") or {}
                if ad and ad.get("deeplink") and ad["deeplink"] not in catalog_uris:
                    errors.append(
                        f"contexts[{ci}].actions[{ai}].stepGroups[{sgi}]"
                        ".actionableDeeplink not in catalog"
                    )
                vd = sg.get("validationDeeplink") or {}
                if vd and vd.get("deeplink") and vd["deeplink"] not in catalog_uris:
                    errors.append(
                        f"contexts[{ci}].actions[{ai}].stepGroups[{sgi}]"
                        ".validationDeeplink not in catalog"
                    )
                for si, step in enumerate(sg.get("steps", [])):
                    if len(step.split()) > STEP_MAX_WORDS:
                        errors.append(
                            f"contexts[{ci}].actions[{ai}].stepGroups[{sgi}]"
                            f".steps[{si}] exceeds {STEP_MAX_WORDS} words"
                        )

    return (len(errors) == 0, errors)


# --- Item 15: deterministic repair ----------------------------------------

def _strip_urls_from_field(value: Any) -> Tuple[Any, bool]:
    """Strip embedded URLs from a string. Returns (new_value, changed)."""
    if not isinstance(value, str):
        return value, False
    cleaned = _STRIP_EMBEDDED_URL_RE.sub("", value).strip()
    # Collapse any double spaces left by the removal.
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned, (cleaned != value)


def repair_response(response: Dict[str, Any],
                    catalog_uris: Set[str]) -> Tuple[Dict[str, Any], List[str]]:
    """Best-effort deterministic repair. Returns (patched, actions_taken)."""
    actions: List[str] = []
    contexts = _get_contexts(response)

    for ctx in contexts:
        # Goal
        new_goal, changed = _strip_urls_from_field(ctx.get("goal", ""))
        if changed:
            ctx["goal"] = new_goal
            actions.append("stripped URL from goal")
        if len(ctx.get("goal", "")) > GOAL_MAX_CHARS:
            ctx["goal"] = ctx["goal"][:GOAL_MAX_CHARS].rstrip() + "..."
            actions.append("truncated goal")

        # Title
        new_title, changed = _strip_urls_from_field(ctx.get("title", ""))
        if changed:
            ctx["title"] = new_title
            actions.append("stripped URL from title")
        if len(ctx.get("title", "")) > TITLE_MAX_CHARS:
            ctx["title"] = ctx["title"][:TITLE_MAX_CHARS].rstrip()
            actions.append("truncated title")

        for action in ctx.get("actions", []):
            new_name, changed = _strip_urls_from_field(action.get("actionName", ""))
            if changed:
                action["actionName"] = new_name
                actions.append("stripped URL from actionName")
            if len(action.get("actionName", "")) > ACTION_NAME_MAX_CHARS:
                action["actionName"] = action["actionName"][:ACTION_NAME_MAX_CHARS].rstrip()
                actions.append("truncated actionName")

            new_desc, changed = _strip_urls_from_field(action.get("description", ""))
            if changed:
                action["description"] = new_desc
                actions.append("stripped URL from description")
            if len(action.get("description", "")) > DESCRIPTION_MAX_CHARS:
                action["description"] = action["description"][:DESCRIPTION_MAX_CHARS].rstrip()
                actions.append("truncated description")

            for sg in action.get("stepGroups", []):
                new_steps = []
                for s in sg.get("steps", []):
                    cleaned, changed = _strip_urls_from_field(s)
                    if changed:
                        actions.append("stripped URL from step")
                    words = cleaned.split()
                    if len(words) > STEP_MAX_WORDS:
                        # Reserve one token for the ellipsis so the final
                        # count is exactly STEP_MAX_WORDS, not one over.
                        cleaned = (
                            " ".join(words[:STEP_MAX_WORDS - 1]).rstrip()
                            + "..."
                        )
                        actions.append("truncated step")
                    new_steps.append(cleaned)
                sg["steps"] = new_steps

                for key in ("actionableDeeplink", "validationDeeplink"):
                    d = sg.get(key)
                    if isinstance(d, dict) and d.get("deeplink") and d["deeplink"] not in catalog_uris:
                        sg[key] = None
                        actions.append(f"removed non-catalog {key}")

    return response, actions
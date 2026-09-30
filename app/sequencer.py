"""Risk classification and ordering.

Categories are set by the pipeline AFTER deeplink resolution runs,
based on concrete facts rather than keyword guessing:

    critical   = destructive or sensitive operations (keyword match)
    auto       = a verified Settings deeplink was attached
    manual     = everything else (physical actions, support calls)
"""

CRITICAL_KEYWORDS = (
    "factory data reset", "factory reset", "wipe",
    "erase all", "delete all", "clear data",
    "recovery menu", "recovery mode", "format device",
    "reset device", "erase device",
)

RISK_ORDER = {"auto": 0, "manual": 1, "critical": 2}


def is_critical(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in CRITICAL_KEYWORDS)


def classify_action(action_name: str, steps_text: str,
                    has_deeplink: bool) -> str:
    combined = (action_name or "") + " " + (steps_text or "")
    if is_critical(combined):
        return "critical"
    if has_deeplink:
        return "auto"
    return "manual"


def order_actions(actions):
    return sorted(
        actions,
        key=lambda a: RISK_ORDER.get(a.get("category", "manual"), 1),
    )

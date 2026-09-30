"""LLM-based Phase 1 structure extraction with token accounting."""
import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_JSON_MODE, LLM_MODEL,
    LLM_PROVIDER, LLM_TIMEOUT_S, MAX_STEPS_PER_GROUP,
)


SYSTEM_PROMPT = """You are the structure-extraction stage of a Samsung Galaxy
troubleshooting assistant. You convert a natural-language complaint plus one
retrieved support passage into a structured troubleshooting plan.

You must obey every rule below. Violating any rule means the plan is rejected.

CONTENT RULES
- Use ONLY the retrieved support passage as your source of truth.
- Do NOT invent troubleshooting steps that are not supported by that passage.
- Do NOT reference settings, menus, or features the passage does not mention.
- If the passage does not contain a usable procedure, return {"abstain": true}.

OUTPUT RULES
- Return STRICT JSON, no prose, no code fences, no commentary.
- Never emit a URL, a bixby:// URI, an http(s):// URI, a www. host, or a markdown link.
- Never mention the words "deeplink", "URI", "URL", or "bixby" in any field.
- Each step is one short imperative sentence, at most 25 words.
- Categorize every action as one of: "auto", "manual", "critical".
- Order actions: all auto first, then manual, then critical.

REQUIRED JSON SHAPE
{
  "goal": "<one sentence, at most 200 chars>",
  "title": "<short title, at most 80 chars>",
  "actions": [
    {
      "actionName": "<short imperative, at most 60 chars>",
      "description": "<one sentence, at most 180 chars>",
      "category": "auto",
      "steps": ["<step 1>", "<step 2>"]
    }
  ]
}

If you cannot produce a plan from the retrieved passage, return exactly:
{"abstain": true}
"""


def is_llm_enabled() -> bool:
    return LLM_PROVIDER.lower() not in ("", "off", "none", "false", "0")


def _build_user_prompt(query: str, retrieved: Dict[str, Any]) -> str:
    title = str(retrieved.get("title", "")).strip()
    content = str(retrieved.get("content", "")).strip()
    if len(content) > 6000:
        content = content[:6000] + "\n...[truncated]"
    return (
        "USER COMPLAINT:\n"
        f"{query.strip()}\n\n"
        "RETRIEVED SUPPORT PASSAGE (your only source of truth):\n"
        f"Title: {title}\n"
        f"{content}\n\n"
        "Return the structured JSON now."
    )


def _call_chat_completions(user_prompt: str) -> Tuple[Optional[str], Dict[str, int]]:
    """Return (content, usage_dict)."""
    if not LLM_BASE_URL:
        return None, {}
    url = LLM_BASE_URL.rstrip("/") + "/chat/completions"

    body: Dict[str, Any] = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
    }
    if LLM_JSON_MODE:
        body["response_format"] = {"type": "json_object"}

    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=LLM_TIMEOUT_S) as resp:
        raw = resp.read().decode("utf-8")
    parsed = json.loads(raw)
    content = parsed["choices"][0]["message"]["content"]
    usage = parsed.get("usage") or {}
    return content, {
        "input_tokens": int(usage.get("prompt_tokens", 0)),
        "output_tokens": int(usage.get("completion_tokens", 0)),
    }


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def adapt_to_candidates(llm_result: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(llm_result, dict):
        return None
    if llm_result.get("abstain") is True:
        return None
    actions = llm_result.get("actions")
    if not isinstance(actions, list) or not actions:
        return None

    out: List[Dict[str, Any]] = []
    for a in actions:
        if not isinstance(a, dict):
            continue
        steps: List[str] = []
        raw_steps = a.get("steps")
        if isinstance(raw_steps, list):
            steps = [str(s).strip() for s in raw_steps if str(s).strip()]
        if not steps:
            for sg in (a.get("stepGroups") or []):
                if isinstance(sg, dict):
                    for s in (sg.get("steps") or []):
                        s = str(s).strip()
                        if s:
                            steps.append(s)
        if not steps:
            continue
        cat = a.get("category")
        if cat not in ("auto", "manual", "critical"):
            cat = None
        out.append({
            "actionName": str(a.get("actionName", "")).strip()[:120] or "Troubleshooting step",
            "description": str(a.get("description", "")).strip()[:180] or steps[0][:180],
            "steps": steps[:MAX_STEPS_PER_GROUP],
            "category_hint": cat,
        })
    return out or None


def generate_structured_plan(query: str,
                             retrieved_context: Dict[str, Any]
                             ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Return (parsed_plan_or_none, usage_meta)."""
    meta = {
        "input_tokens": 0,
        "output_tokens": 0,
        "model": LLM_MODEL,
    }
    if not is_llm_enabled():
        return None, meta
    try:
        user_prompt = _build_user_prompt(query, retrieved_context)
        raw, usage = _call_chat_completions(user_prompt)
        if not raw:
            return None, meta
        meta.update(usage)
        cleaned = _strip_code_fences(raw)
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            return None, meta
        return parsed, meta
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            json.JSONDecodeError, KeyError, IndexError, ValueError, OSError):
        return None, meta
    except Exception:
        return None, meta


def generate_structured_plan_retry(query: str,
                                   retrieved_context: Dict[str, Any],
                                   previous_errors: List[str]
                                   ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Item 15: a constrained regenerate attempt that tells the model what went wrong."""
    meta = {"input_tokens": 0, "output_tokens": 0, "model": LLM_MODEL}
    if not is_llm_enabled():
        return None, meta
    try:
        extra = (
            "\n\nYour previous response failed validation with these errors:\n"
            + "\n".join(f"- {e}" for e in previous_errors[:5])
            + "\nPlease return a corrected JSON plan that satisfies all rules."
        )
        prompt = _build_user_prompt(query, retrieved_context) + extra
        raw, usage = _call_chat_completions(prompt)
        if not raw:
            return None, meta
        meta.update(usage)
        parsed = json.loads(_strip_code_fences(raw))
        if not isinstance(parsed, dict):
            return None, meta
        return parsed, meta
    except Exception:
        return None, meta
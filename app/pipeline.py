"""End-to-end pipeline (Waves 1 and 2)."""
import json
import re
import time
from typing import Any, Dict, List, Optional

from . import llm
from .action_graph import build_graph
from .cache import SemanticCache
from .confidence import compute_confidence
from .config import (
    ACTION_NAME_MAX_CHARS,
    ABSTAIN_TOPIC_SUGGESTIONS,
    LLM_COST_PER_1M_INPUT_USD, LLM_COST_PER_1M_OUTPUT_USD,
    MAX_CONTEXTS, MAX_REGEN_ATTEMPTS, SIIS_MIN_SCORE, SIIS_TOP_K,
)
from .data_loader import load_deeplinks, load_siis_rows
from .extractor import extract_actions
from .logging_config import get_logger, new_request_id
from .query_processing import make_paraphrases, split_intents
from .resolver import ScreenResolver
from .retrieval import HybridIndex, chunk_siis
from .schema import (
    Action, ContextDeeplinkResponse, Deeplink, Goal, StepGroup,
    ValidationDeepLink, actionCategory,
)
from .sequencer import classify_action, is_critical, order_actions
from .validator import repair_response, validate_response

FALLBACK_TITLE = "No matching troubleshooting content found"
FALLBACK_GOAL = (
    "We could not find a verified troubleshooting plan in the trusted "
    "knowledge base for this complaint. Please contact Samsung Support."
)

WEB_CONTENT_MAX_CHARS = 3000

LOG = get_logger()


# --- Web-content gating helpers -------------------------------------------

_EXPLANATORY_PREFIX_RE = re.compile(
    r"^\s*(?:\d+\.\s*)?(?:why|what is|what are|what causes|"
    r"how does|introduction|overview)\b",
    re.IGNORECASE,
)

_SETTINGS_VERB_RE = re.compile(
    r"\b(open|tap|navigate|select|enable|disable|toggle|switch|"
    r"turn on|turn off|go to|check the|set the|adjust the)\b",
    re.IGNORECASE,
)


def _is_explanatory_action(name):
    """True if the action name looks like a Why/What explanation."""
    return bool(_EXPLANATORY_PREFIX_RE.match((name or "").strip()))



# Navigation verbs: the user is moving through menus, not performing
# an action. A step that only has these is not a deeplink target --
# the user is expected to tap through to the next screen.
_NAV_VERB_RE = re.compile(
    r"^\s*(?:tap|select|open|navigate|go to|choose|press|scroll to)\b",
    re.IGNORECASE,
)

# Action verbs: the user toggles, adjusts, or changes a specific
# setting. Steps with these get deeplink resolution.
_ACTION_VERB_RE = re.compile(
    r"\b(enable|disable|turn on|turn off|toggle|set the|set to|"
    r"change|adjust|increase|decrease|raise|lower|switch to|"
    r"activate|deactivate|clear|reset|apply|confirm)\b",
    re.IGNORECASE,
)



# Hardware button presses, charging, cable handling, cleaning, battery
# removal. These are physical procedures that never correspond to a
# Settings screen, even when the action name contains words that match
# a catalog entry ("restart", "reset", "power").
_PHYSICAL_PROCEDURE_RE = re.compile(
    r"\b(?:"
    # Hardware buttons
    r"press\s+(?:and\s+)?hold\b|"
    r"hold\s+(?:both|the)\b|"
    r"press\s+both\b|"
    r"press\s+and\s+hold\s+both\b|"
    # Inspection
    r"inspect\s+(?:the|for)\b|"
    r"examine\s+(?:the|for)\b|"
    r"remove\s+any\s+(?:cases|covers)\b|"
    r"check\s+for\s+(?:physical|damage|corrosion|moisture)\b|"
    # Cable / charger handling
    r"connect\s+(?:the|your)\s+(?:phone|device|tablet)\b|"
    r"disconnect\s+(?:the|your)\s+(?:phone|device|tablet)\b|"
    r"let\s+it\s+charge\b|"
    r"charge\s+(?:the|your)\s+(?:phone|device|tablet)\b|"
    r"plug\s+(?:in|the)\b|"
    r"unplug\b|"
    # Battery
    r"remove\s+the\s+battery\b|"
    r"reinsert\s+the\s+battery\b|"
    # Physical cleaning
    r"clean\s+the\s+charging\s+port\b|"
    r"wipe\s+(?:the|with)\b|"
    r"dry\s+(?:the|with)\b|"
    # Visiting support
    r"visit\s+(?:a|an|the)\s+(?:samsung|authorized)\b|"
    r"contact\s+(?:samsung|support)\b"
    r")\b",
    re.IGNORECASE,
)



# Action names that are never critical, regardless of what keyword
# matching says. Restart, reboot, and safe mode are diagnostics, not
# destructive operations.
_NEVER_CRITICAL_RE = re.compile(
    r"\b(?:"
    r"restart|reboot|soft\s+reset|"
    r"safe\s+mode|recovery\s+mode\s*\(safe\)"
    r")\b",
    re.IGNORECASE,
)



# The only phrases that can ever justify a critical rating. Everything
# else -- including the words "restart", "safe", "reset", "erase" in
# isolation -- is at most manual.
_TRULY_DESTRUCTIVE_RE = re.compile(
    r"\b(?:"
    r"factory\s+data\s+reset|"
    r"factory\s+reset|"
    r"wipe\s+(?:the\s+)?(?:device|data|cache)|"
    r"erase\s+all\s+(?:data|content)|"
    r"delete\s+all\s+(?:data|content)|"
    r"format\s+(?:the\s+)?device|"
    r"recovery\s+menu"
    r")\b",
    re.IGNORECASE,
)


def _is_truly_destructive(text):
    """Returns True only for phrases that unambiguously destroy data."""
    return bool(_TRULY_DESTRUCTIVE_RE.search(text or ""))


def _is_never_critical(text):
    return bool(_NEVER_CRITICAL_RE.search(text or ""))


def _content_tokens_for_step(text):
    """Return content tokens from a step, filtered the same way the
    resolver's coverage check filters them."""
    from .resolver import _GENERIC_VERBS
    from .retrieval import tokenize
    out = []
    for t in tokenize(text or ""):
        if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        if len(t) < 3 or t in _GENERIC_VERBS:
            continue
        out.append(t)
    return out


def _is_physical_procedure(text):
    """True if the text describes a physical action that has no
    corresponding Settings screen."""
    return bool(_PHYSICAL_PROCEDURE_RE.search(text or ""))


# Compound-step expander. Turns "Enable X and Y" into two separate
# steps so each gets its own deeplink resolution.
_ACTION_LEAD_RE = re.compile(
    r"^(enable|disable|turn on|turn off|toggle|set|change|adjust|"
    r"activate|deactivate)\s+",
    re.IGNORECASE,
)

_ANY_VERB_LEAD_RE = re.compile(
    r"^(enable|disable|turn on|turn off|toggle|set|change|adjust|"
    r"activate|deactivate|tap|select|open|navigate|go to|check|"
    r"verify|press|choose|scroll|look for|look at|review|read|"
    r"confirm|wait|hold|swipe|drag|move|remove|insert|plug|"
    r"unplug|connect|disconnect|restart|reboot)\b",
    re.IGNORECASE,
)

_CONTINUATION_RE = re.compile(
    r"^(then|after|before|next|finally|or|but|so|if|otherwise)\b",
    re.IGNORECASE,
)


def _expand_compound_step(step):
    """Split "Enable Fast charging and Fast wireless charging" into
    two independent steps. Only splits when:
      - left half starts with an action verb
      - right half does NOT start with any verb
      - right half is not a continuation word
      - right half has at least two words
    """
    if not step:
        return [step]
    parts = re.split(r"\s+and\s+", step, maxsplit=1)
    if len(parts) != 2:
        return [step]
    left, right = parts[0].strip(), parts[1].strip()

    verb_match = _ACTION_LEAD_RE.match(left)
    if not verb_match:
        return [step]
    if _ANY_VERB_LEAD_RE.match(right):
        return [step]
    if _CONTINUATION_RE.match(right):
        return [step]
    if len(right.split()) < 2:
        return [step]

    verb = verb_match.group(1)
    return [left, verb + " " + right]


def _is_actionable_step(text):
    """True if the step describes an action on a specific setting,
    rather than just navigation through menus."""
    if not text:
        return False
    t = text.strip()
    # Action verb always wins.
    if _ACTION_VERB_RE.search(t):
        return True
    # Pure navigation without an action verb -- skip resolution.
    if _NAV_VERB_RE.match(t):
        return False
    # Short steps that don't clearly do anything -- be conservative.
    if len(t.split()) < 4:
        return False
    # Longer descriptive steps -- allow the resolver a shot.
    return True

def _looks_like_settings_step(text):
    """True if the text is a good candidate for deeplink resolution.

    Long prose paragraphs fail (they dilute the resolver signal), but
    short, directive action names pass without requiring a specific
    verb, so phrases like "Try wireless charging" resolve correctly.
    """
    if not text:
        return False
    if len(text) > 400:
        return False
    # Short text is already directive enough.
    if len(text) <= 150:
        return True
    # For longer prose, require a settings verb so we don't feed a
    # marketing paragraph to the resolver.
    return bool(_SETTINGS_VERB_RE.search(text))


class TroubleshootPipeline:
    def __init__(self):
        self.deeplinks = load_deeplinks()

        self.catalog_uris = set()
        for d in self.deeplinks:
            if d.get("deeplink"):
                self.catalog_uris.add(d["deeplink"])
            v = d.get("validation")
            if isinstance(v, dict) and v.get("deeplink"):
                self.catalog_uris.add(v["deeplink"])

        self.siis_rows = load_siis_rows()

        self.siis_chunks: List[Dict[str, Any]] = []
        for row in self.siis_rows:
            payload = row["siis_response"]
            title = payload.get("title", "")
            content = payload.get("content", "")
            for chunk in chunk_siis(content, title):
                text = " ".join(filter(None, [
                    row.get("original_query", ""),
                    title,
                    chunk["heading"],
                    chunk["text"],
                ]))
                self.siis_chunks.append({
                    "id": row["id"],
                    "text": text,
                    "description": chunk["heading"],
                    "ref": row,
                })

        self.siis_index = HybridIndex(self.siis_chunks)
        self.resolver = ScreenResolver(self.deeplinks)
        self.cache = SemanticCache()

    def _retrieve_siis_chunk(self, query: str):
        """Three-tier SIIS retrieval with content-coverage and
        title-relevance gates.

        Returns (chunk_or_none, score, components, approximate_flag).
        """
        from .config import SIIS_SOFT_MIN_SCORE
        from .fallback import (
            domains_compatible, rarest_token, router_pick, shares_content,
        )

        results = self.siis_index.search(query, top_k=SIIS_TOP_K)
        if not results:
            return None, 0.0, {}, False

        top_idx, top_score, comp = results[0]
        comp = dict(comp)
        top_chunk = self.siis_chunks[top_idx]
        row_title = top_chunk["ref"]["siis_response"].get("title", "")
        chunk_heading = top_chunk.get("description", "")

        # Strict preflight: at least half of the query's substantive
        # content tokens must appear in the top chunk. This catches
        # cases where the score is high purely from shared generic
        # words like "screen" or "display" without real topic overlap.
        # Semantic preflight: the top chunk must be meaningfully
        # similar to the query. Uses the same embedding comparison
        # as the web search tier for consistency.
        from .fallback import semantic_similarity
        _sim = semantic_similarity(query, top_chunk["text"][:800])
        comp["semantic_similarity"] = round(_sim, 3)
        if _sim < 0.62:
            comp["tier"] = "corpus_miss"
            return None, top_score, comp, False

        rare_tok, rare_count = rarest_token(query, self.siis_chunks)
        if rare_tok and rare_count == 0:
            comp["tier"] = "no_coverage"
            comp["missing_token"] = rare_tok
            return None, top_score, comp, False

        if top_score >= SIIS_MIN_SCORE:
            if not domains_compatible(query, top_chunk["text"]):
                comp["tier"] = "domain_mismatch"
            elif not shares_content(query, row_title, chunk_heading):
                comp["tier"] = "title_mismatch"
            else:
                comp["tier"] = "exact"
                return top_chunk, top_score, comp, False

        if top_score >= SIIS_SOFT_MIN_SCORE:
            if (domains_compatible(query, top_chunk["text"])
                    and shares_content(query, row_title, chunk_heading)):
                comp["tier"] = "approximate_domain"
                return top_chunk, top_score, comp, True

        chunk = router_pick(query, self.siis_chunks, results)
        if chunk is not None:
            comp["tier"] = "router"
            return chunk, top_score, comp, True

        comp["tier"] = "none"
        return None, top_score, comp, False

    @staticmethod
    def _normalize_siis_override(raw: Any) -> Optional[Dict[str, Any]]:
        if not raw:
            return None
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            s = raw.strip()
            if not s:
                return None
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                return {"title": "Provided reference", "content": s}
        return None

    @staticmethod
    def _web_content_to_candidate(web_content: Dict[str, str]) -> Dict[str, Any]:
        """Turn unstructured web content into a manual action.

        Splits the body into sentences so the plan shows multiple
        distinct steps instead of one giant paragraph.
        """
        import re as _re

        title = web_content.get("title") or "Samsung support content"
        body = (web_content.get("content") or "").strip()

        # Cap total length.
        if len(body) > WEB_CONTENT_MAX_CHARS:
            body = body[:WEB_CONTENT_MAX_CHARS].rstrip()
            last = max(body.rfind(". "), body.rfind(".\n"),
                       body.rfind("!"), body.rfind("?"))
            if last > WEB_CONTENT_MAX_CHARS * 0.6:
                body = body[:last + 1]

        # Split into sentences and group into readable steps.
        sentences = [s.strip() for s in _re.split(r"(?<=[.!?])\s+", body)
                     if s.strip()]

        # Merge short sentences so a step isn't 3 words long.
        merged = []
        buf = ""
        for s in sentences:
            if len(buf) + len(s) + 1 <= 220:
                buf = (buf + " " + s).strip()
            else:
                if buf:
                    merged.append(buf)
                buf = s
        if buf:
            merged.append(buf)

        # Cap at MAX_STEPS_PER_GROUP.
        from .config import MAX_STEPS_PER_GROUP
        merged = merged[:MAX_STEPS_PER_GROUP]

        if not merged:
            merged = [body] if body else ["(no content)"]

        return {
            "actionName": title[:140],
            "description": "Review this Samsung support content for your issue",
            "steps": merged,
            "category_hint": "manual",
        }


    # Service-center escalation -------------------------------------------------

    @staticmethod
    def _maybe_add_escalation(candidates, source_content):
        """Append a manual "Visit a Samsung Service Center" action when
        the source text mentions service escalation AND no existing
        action already covers it."""
        if not candidates or not source_content:
            return candidates

        body = source_content.lower()
        escalation_phrases = [
            "service center", "support center", "authorized service",
            "require service", "needs service", "walk-in",
        ]
        mentions_service = any(p in body for p in escalation_phrases)
        if not mentions_service:
            return candidates

        already_has = any(
            "service" in (c["actionName"] + " " + c["description"]).lower()
            for c in candidates
        )
        if already_has:
            return candidates

        return list(candidates) + [{
            "actionName": "Visit a Samsung Service Center",
            "description": (
                "If none of the above steps resolve the issue, the "
                "device may require in-person service."
            ),
            "steps": [
                "Back up any data you can still access.",
                "Contact Samsung Support or visit an authorized "
                "Samsung Service Center.",
            ],
            "category_hint": "manual",
        }]


    def _build_step_group(self, steps, entry):
        actionable = validation = None
        if entry:
            actionable = Deeplink(
                deeplink=entry["deeplink"],
                description=entry.get("description", "") or "",
                message=entry.get("message", "") or "",
                originalType=entry.get("originalType"),
            )
            v = entry.get("validation")
            if isinstance(v, dict) and v.get("deeplink"):
                validation = ValidationDeepLink(
                    deeplink=v["deeplink"],
                    key=v.get("key", "") or "",
                    resultType=v.get("resultType"),
                    condition=v.get("condition"),
                    value=v.get("value"),
                )
        return StepGroup(
            steps=steps,
            validationDeeplink=validation,
            actionableDeeplink=actionable,
        )

    def _build_goal(self, title, goal_text, candidates, score):
        if not candidates:
            return None, {"deeplink_ids": [], "screen_scores": []}
        built = []
        deeplink_ids = []
        screen_scores = []
        for cand in candidates:
            joined = " ".join(str(s) for s in cand["steps"])
            combined = cand["actionName"] + " " + joined

            # Categorization order:
            #   1. Critical keywords match -> critical (safety override)
            #   2. Otherwise try to resolve a deeplink
            #   3. Deeplink attached -> auto
            #   4. Otherwise -> manual
            llm_hint = cand.get("category_hint")
            # Only the keyword list can escalate to critical. The LLM
            # over-classifies benign diagnostics (safe mode, restart)
            # as critical, and misfires here are worse than misses:
            # the user loses trust when "Restart your phone" is
            # flagged as dangerous.
            critical_match = is_critical(combined)

            dl_entry = None
            dl_score = 0.0
            dl_comp = {}
            ranked_ids = []
            if not critical_match:
                # Build a focused resolution signal: the action name plus
                # at most the first two short steps. Passing the full
                # joined step list dilutes the signal and lets common
                # words like "charging" or "screen" dominate the match.
                signal_parts = [cand["actionName"]]
                for s in cand["steps"][:2]:
                    if len(s) < 120:
                        signal_parts.append(s)
                signal_text = ". ".join(signal_parts)
                # Gate on the action name only -- it's the topical
                # summary. Step text can be long prose that trips the
                # length check even when the action itself is fine.
                if (_looks_like_settings_step(cand["actionName"])
                        and _is_actionable_step(cand["actionName"])
                        and not _is_physical_procedure(combined)):
                    dl_entry, dl_score, dl_comp, ranked_ids = \
                        self.resolver.resolve_with_components(
                            signal_text, topic=cand["actionName"],
                        )

            category = classify_action(
                cand["actionName"], joined,
                has_deeplink=dl_entry is not None,
            )
            if critical_match and not _is_never_critical(cand["actionName"]):
                category = "critical"

            if dl_entry and dl_entry["id"] not in deeplink_ids:
                deeplink_ids.append(dl_entry["id"])

            screen_scores.append({
                "action": cand["actionName"],
                "score": round(dl_score, 4),
                "components": dl_comp,
                "top_ids": ranked_ids,
            })
            desc_text = cand["description"][:180]
            steps_list = list(cand["steps"])
            # If the first step is identical to the description, drop
            # it from the visible list so the UI doesn't render the
            # same sentence twice. Keep at least one step.
            if (len(steps_list) > 1
                    and steps_list[0].strip() == desc_text.strip()):
                steps_list = steps_list[1:]

            # Build step groups. When the action has multiple steps,
            # each one becomes its own group with its own deeplink
            # resolution. The schema allows a list of step groups
            # precisely for this: one action spanning several screens.
            # For single-step actions we keep one group.
            if len(steps_list) > 1 and category != "manual":
                groups = []
                _any_step_dl = False
                expanded = []
                for _s in steps_list:
                    expanded.extend(_expand_compound_step(_s))
                for one_step in expanded:
                    one_dl = None
                    one_score = 0.0
                    one_comp = {}
                    one_ids = []
                    if (_is_actionable_step(one_step)
                            and _looks_like_settings_step(one_step)
                            and not _is_physical_procedure(one_step)):
                        one_dl, one_score, one_comp, one_ids = \
                            self.resolver.resolve_with_components(
                                one_step, topic=one_step,
                            )
                    if one_dl:
                        _any_step_dl = True
                        if one_dl["id"] not in deeplink_ids:
                            deeplink_ids.append(one_dl["id"])
                    groups.append(self._build_step_group([one_step], one_dl))
                    screen_scores.append({
                        "action": cand["actionName"],
                        "step": one_step[:80],
                        "score": round(one_score, 4),
                        "components": one_comp,
                        "top_ids": one_ids,
                    })

                # If no individual step resolved but the whole action
                # did, attach the whole-action deeplink to the last
                # step group so the user gets a shortcut. Otherwise
                # downgrade the category to manual so the label matches
                # the actual behavior.
                if not _any_step_dl:
                    if dl_entry is not None:
                        last_steps = list(groups[-1].steps)
                        groups[-1] = self._build_step_group(
                            last_steps, dl_entry,
                        )
                        if dl_entry["id"] not in deeplink_ids:
                            deeplink_ids.append(dl_entry["id"])
                    else:
                        category = "manual"

                built.append({
                    "actionName": cand["actionName"][:ACTION_NAME_MAX_CHARS],
                    "description": desc_text,
                    "stepGroups": groups,
                    "category": actionCategory(category),
                })
            else:
                built.append({
                    "actionName": cand["actionName"][:ACTION_NAME_MAX_CHARS],
                    "description": desc_text,
                    "stepGroups": [self._build_step_group(steps_list, dl_entry)],
                    "category": actionCategory(category),
                })
        # Final normalization: force any critical action to manual
        # unless its text contains an unambiguously destructive phrase.
        # This runs regardless of which upstream step set the category,
        # so neither the keyword list nor an LLM hint can flag a benign
        # diagnostic as critical.
        for a in built:
            if a.get("category") == "critical":
                combined = a["actionName"] + " " + a.get("description", "")
                if not _is_truly_destructive(combined):
                    a["category"] = "manual"

        # Manual actions carry no deeplinks. A manual action has no
        # Settings screen by definition, so any deeplink on it is a
        # false positive. Strip here, at the end, to sidestep the
        # layered resolver filtering entirely.
        for _a in built:
            if _a.get("category") == "manual":
                for _sg in _a.get("stepGroups", []):
                    _sg.actionableDeeplink = None
                    _sg.validationDeeplink = None

        built = order_actions(built)
        return Goal(
            goal=goal_text,
            title=title,
            actions=[Action(**a) for a in built],
            score=float(score),
        ), {"deeplink_ids": deeplink_ids, "screen_scores": screen_scores}

    def _fallback(self, query, variations, meta, t0):
        from .fallback import known_topics
        topics = known_topics(self.siis_chunks, limit=ABSTAIN_TOPIC_SUGGESTIONS)
        return {
            "query": query,
            "query_variations": variations,
            "response": {"contexts": []},
            "meta": {
                **meta,
                "fallback": True,
                "schema_valid": False,
                "fallback_title": FALLBACK_TITLE,
                "fallback_message": FALLBACK_GOAL,
                "suggested_topics": topics,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            },
        }

    def _run_single_intent(self, query: str, sis_override: Any = None,
                           request_id: str = "") -> Dict[str, Any]:
        t0 = time.perf_counter()
        stage_latencies: Dict[str, float] = {}

        def mark(name, t_start):
            stage_latencies[name] = round((time.perf_counter() - t_start) * 1000, 2)

        variations = make_paraphrases(query) if query else []
        meta: Dict[str, Any] = {
            "request_id": request_id or new_request_id(),
            "cache_hit": False,
            "cache_similarity": 0.0,
            "cache_backend": self.cache.backend_kind,
            "siis_match_score": 0.0,
            "siis_components": {},
            "approximate": False,
            "web_fallback": False,
            "web_source_title": "",
            "validation_errors": [],
            "variations": variations,
            "llm_enabled": llm.is_llm_enabled(),
            "llm_used": False,
            "llm_fallback": False,
            "input_tokens": 0,
            "output_tokens": 0,
            "model": llm.LLM_MODEL if hasattr(llm, "LLM_MODEL") else "",
            "cost_usd": 0.0,
            "repair_attempts": 0,
            "regenerate_attempts": 0,
            "extraction_preference": "",
        }

        # Cache lookup
        t = time.perf_counter()
        if query and not sis_override:
            cached, sim = self.cache.lookup(query)
            meta["cache_similarity"] = round(sim, 4)
            if cached is not None:
                mark("cache", t)
                meta["cache_hit"] = True
                meta["stage_latencies"] = stage_latencies
                meta["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                meta["confidence"] = cached.get("confidence", 0.0)
                cached = dict(cached)
                cached["meta"] = meta
                return cached
        mark("cache", t)

        # SIIS retrieval
        t = time.perf_counter()
        siis = self._normalize_siis_override(sis_override)
        score = 1.0
        components: Dict[str, Any] = {}
        approximate = False
        if siis is None:
            chunk, score, components, approximate = \
                self._retrieve_siis_chunk(query)
            if chunk:
                siis = chunk["ref"]["siis_response"]
                # Quality pre-check: run the extractor and reject chunks
                # that produce only low-value actions. This must run
                # before the web fallback so the pipeline can use web
                # content when the local chunk is junk.
                from .extractor import extract_actions as _ea
                _probe = _ea(siis)
                if not _probe:
                    siis = None
                    components = dict(components)
                    components["tier"] = "low_quality"
                else:
                    _good = 0
                    for _c in _probe:
                        _steps = _c.get("steps", [])
                        _text = (_c.get("actionName", "") + " " +
                                 _c.get("description", "")).lower()
                        if len(_steps) >= 2:
                            _good += 1
                        elif ("contact" in _text and "support" in _text):
                            continue
                        elif "visit" in _text and (
                                "website" in _text or "hp" in _text):
                            continue
                        else:
                            _good += 1
                    if _good == 0:
                        siis = None
                        components = dict(components)
                        components["tier"] = "low_quality"

        meta["siis_match_score"] = round(float(score), 4)
        meta["siis_components"] = {
            k: (round(v, 4) if isinstance(v, (int, float)) else v)
            for k, v in components.items()
        }
        meta["approximate"] = approximate
        mark("retrieval", t)

        # Web fallback (only when the local corpus has no coverage)
        web_content: Optional[Dict[str, str]] = None
                # Web fallback: fire whenever SIIS retrieval returned nothing,
        # regardless of which local tier rejected it. The only guard is
        # a minimum query length so one-word input does not burn a
        # Tavily call.
        web_content: Optional[Dict[str, str]] = None
        if not siis:
            if len(query) > 8:
                try:
                    from .web_search import fetch_fallback_steps
                    web_content = fetch_fallback_steps(query)
                except Exception:
                    web_content = None
                if web_content:
                    siis = web_content
                    meta["web_fallback"] = True
                    meta["web_source_title"] = web_content.get("title", "")
                    components["tier"] = "web_fallback"
                    meta["siis_components"] = {
                        k: (round(v, 4) if isinstance(v, (int, float)) else v)
                        for k, v in components.items()
                    }

            if not siis:
                meta["stage_latencies"] = stage_latencies
                return self._fallback(query, variations, meta, t0)

        # Phase 1: LLM
        t = time.perf_counter()
        title = siis.get("title", "Troubleshooting")
        goal_text = f"Follow these steps to troubleshoot: {title}"
        candidates: Optional[List[Dict[str, Any]]] = None

        # Only call the LLM when deterministic extraction is insufficient.
        # Saves 15-20s of latency on SIIS rows that already have good steps.
        _should_call_llm = False
        if llm.is_llm_enabled():
            try:
                _probe = extract_actions(siis)
            except Exception:
                _probe = []
            try:
                _probe = [c for c in _probe
                          if not _is_explanatory_action(c["actionName"])]
            except Exception:
                pass
            _useful = [c for c in _probe if len(c.get("steps", [])) >= 2]
            if len(_useful) < 2:
                _should_call_llm = True
            else:
                meta["llm_skipped_reason"] = "deterministic_sufficient"

        if _should_call_llm:
            try:
                llm_out = llm.generate_structured_plan(query, siis)
                if isinstance(llm_out, tuple) and len(llm_out) == 2:
                    result, usage = llm_out
                elif isinstance(llm_out, dict):
                    result, usage = llm_out, {}
                else:
                    result, usage = None, {}
            except Exception:
                result, usage = None, {}
            meta["input_tokens"] += usage.get("input_tokens", 0)
            meta["output_tokens"] += usage.get("output_tokens", 0)
            if result and not result.get("abstain"):
                adapted = llm.adapt_to_candidates(result)
                if adapted:
                    candidates = adapted
                    title = str(result.get("title") or title)[:120]
                    goal_text = str(result.get("goal") or goal_text)[:300]
                    meta["llm_used"] = True
        mark("llm", t)

        # Cross-check: for local SIIS sources, run the deterministic
        # extractor too. The LLM sometimes produces a goal that names
        # all the steps but emits fewer actions. When that happens,
        # prefer the deterministic list, which reads the "### Step N:"
        # headings verbatim.
        if candidates is not None and web_content is None:
            _det_check = extract_actions(siis)
            if _det_check:
                _det_check = [
                    c for c in _det_check
                    if not _is_explanatory_action(c["actionName"])
                ]
            if _det_check and len(_det_check) > len(candidates):
                candidates = [{
                    "actionName": c["actionName"],
                    "description": c["description"],
                    "steps": c["steps"],
                    "category_hint": None,
                } for c in _det_check]
                meta["extraction_preference"] = "deterministic"
            else:
                meta["extraction_preference"] = "llm"

        # Deterministic extractor / web content wrapper
        t = time.perf_counter()
        if candidates is None:
            # The LLM did not produce a plan. Record that fact so the
            # meta block reflects the fallback path.
            if llm.is_llm_enabled():
                meta["llm_fallback"] = True

            # Try deterministic extraction on whichever source we have.
            # Both SIIS responses and web articles use the same
            # {title, content} shape, and both often have "## Step N"
            # or "## N." headings that the extractor understands.
            det = extract_actions(siis)
            if det:
                det = [c for c in det
                       if not _is_explanatory_action(c["actionName"])]
            if det:
                candidates = [{
                    "actionName": c["actionName"],
                    "description": c["description"],
                    "steps": c["steps"],
                    "category_hint": None,
                } for c in det]
                if web_content is not None:
                    # Use the web article's title for the goal; it's
                    # more specific than the generic fallback.
                    title = web_content.get("title", title)[:120]
                    goal_text = f"Follow these steps to troubleshoot: {title}"
            elif web_content is not None:
                # Extraction failed on web content -- it's probably a
                # forum thread or discussion post with no real steps.
                # Wrap as one informational manual action instead.
                candidates = [self._web_content_to_candidate(web_content)]
                title = web_content.get("title", title)[:120]
                goal_text = f"Review this Samsung support content: {title}"
                meta["web_informational"] = True
        mark("extraction", t)

        # Quality gate: reject the local chunk if extraction produced
        # only low-value actions (single-step, or support-contact).
        # This catches SIIS rows whose body is mostly footer/ad content
        # and would otherwise produce a "Contact Support" plan.
        def _is_usable_action(cand):
            steps = cand.get("steps", [])
            if len(steps) >= 2:
                return True
            name = (cand.get("actionName", "") + " " + cand.get("description", "")).lower()
            if "contact" in name and "support" in name:
                return False
            if "visit" in name and ("website" in name or "hp" in name):
                return False
            # Single-step action whose text looks like an ad fragment.
            if steps and ("continue to visit" in steps[0].lower()
                          or "hp's website" in steps[0].lower()):
                return False
            return True

        if candidates:
            usable = [c for c in candidates if _is_usable_action(c)]
            if not usable:
                # The local chunk had no real troubleshooting content.
                # Treat this as a miss so the pipeline falls through
                # to the web search.
                meta["local_quality_reject"] = True
                meta["stage_latencies"] = stage_latencies
                siis = None

        if siis is None and web_content is None:
            # Nothing usable from local. Let the caller try the web.
            # Return a sentinel that tells the outer logic to try web.
            # For now just fall through with empty candidates, which
            # triggers the standard fallback path.
            pass

        # Add escalation action if the source mentions a service center
        # and no existing action already covers it.
        candidates = self._maybe_add_escalation(
            candidates or [], siis.get("content", "") if siis else "",
        )

        t = time.perf_counter()
        goal, dep_meta = self._build_goal(title, goal_text, candidates or [], score)
        if goal is None:
            meta["stage_latencies"] = stage_latencies
            return self._fallback(query, variations, meta, t0)
        mark("resolution", t)

        # Validate + repair
        t = time.perf_counter()
        schema_response = ContextDeeplinkResponse(contexts=[goal])
        dumped = schema_response.model_dump(mode="json")
        ok, errors = validate_response(dumped, self.catalog_uris)

        if not ok:
            for _ in range(2):
                dumped, repairs = repair_response(dumped, self.catalog_uris)
                if repairs:
                    meta["repair_attempts"] += 1
                ok, errors = validate_response(dumped, self.catalog_uris)
                if ok:
                    break

        mark("validation", t)
        meta["validation_errors"] = errors
        meta["schema_valid"] = ok
        meta["deeplink_ids"] = dep_meta.get("deeplink_ids", [])
        meta["screen_scores"] = dep_meta.get("screen_scores", [])

        meta["cost_usd"] = round(
            meta["input_tokens"] / 1_000_000 * LLM_COST_PER_1M_INPUT_USD
            + meta["output_tokens"] / 1_000_000 * LLM_COST_PER_1M_OUTPUT_USD,
            6,
        )

        if not ok:
            meta["stage_latencies"] = stage_latencies
            return self._fallback(query, variations, meta, t0)

        num_steps = sum(
            len(sg.get("steps", []))
            for ctx in dumped["contexts"]
            for a in ctx["actions"]
            for sg in a["stepGroups"]
        )
        conf = compute_confidence(
            meta["siis_match_score"], meta["screen_scores"], ok, num_steps,
        )
        meta["confidence"] = conf

        actions_flat = [a for ctx in dumped["contexts"] for a in ctx["actions"]]
        meta["action_graph"] = build_graph(actions_flat)

        meta["stage_latencies"] = stage_latencies
        final = {
            "query": query,
            "query_variations": variations,
            "response": dumped,
            "meta": {**meta, "latency_ms": round((time.perf_counter() - t0) * 1000, 2)},
        }

        if ok and query and not meta.get("web_informational"):
            self.cache.store(query, {
                "query": query,
                "query_variations": variations,
                "response": dumped,
                "confidence": conf,
            })

        return final

    def run(self, query: str, sis_override: Any = None) -> Dict[str, Any]:
        t0 = time.perf_counter()
        request_id = new_request_id()
        query = (query or "").strip()
        if not query:
            return self._fallback(query, [], {"request_id": request_id}, t0)

        cached, sim = self.cache.lookup(query)
        if cached is not None and not sis_override:
            cached = dict(cached)
            cached["meta"] = {
                **cached.get("meta", {}),
                "request_id": request_id,
                "cache_hit": True,
                "cache_similarity": round(sim, 4),
                "cache_backend": self.cache.backend_kind,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            }
            return cached

        intents = split_intents(query)
        multi = len(intents) > 1

        if not multi:
            return self._run_single_intent(query, sis_override, request_id)

        all_contexts: List[Dict[str, Any]] = []
        merged_meta: Dict[str, Any] = {
            "request_id": request_id,
            "multi_intent": True,
            "intents": intents,
            "sub_results": [],
            "cache_hit": False,
            "llm_enabled": llm.is_llm_enabled(),
            "llm_used": False,
            "llm_fallback": False,
            "web_fallback": False,
            "cache_backend": self.cache.backend_kind,
            "siis_match_score": 0.0,
        }

        for sub in intents[:MAX_CONTEXTS]:
            sub_resp = self._run_single_intent(sub, None, request_id)
            sub_meta = sub_resp.get("meta", {})
            merged_meta["sub_results"].append({
                "query": sub,
                "latency_ms": sub_meta.get("latency_ms"),
                "cache_hit": sub_meta.get("cache_hit"),
                "siis_match_score": sub_meta.get("siis_match_score"),
                "confidence": sub_meta.get("confidence"),
                "fallback": sub_meta.get("fallback", False),
            })
            if sub_meta.get("llm_used"):
                merged_meta["llm_used"] = True
            if sub_meta.get("llm_fallback"):
                merged_meta["llm_fallback"] = True
            if sub_meta.get("web_fallback"):
                merged_meta["web_fallback"] = True
            if sub_meta.get("web_source_title") and not merged_meta.get("web_source_title"):
                merged_meta["web_source_title"] = sub_meta["web_source_title"]
            if sub_meta.get("siis_components") and not merged_meta.get("siis_components"):
                merged_meta["siis_components"] = sub_meta["siis_components"]
            for ctx in sub_resp.get("response", {}).get("contexts", []):
                all_contexts.append(ctx)

        sub_scores = [
            s.get("siis_match_score") for s in merged_meta["sub_results"]
            if s.get("siis_match_score") is not None
        ]
        if sub_scores:
            merged_meta["siis_match_score"] = round(max(sub_scores), 4)

        merged_meta["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        variations = make_paraphrases(query)

        if not all_contexts:
            from .fallback import known_topics
            merged_meta["fallback"] = True
            merged_meta["fallback_title"] = FALLBACK_TITLE
            merged_meta["fallback_message"] = FALLBACK_GOAL
            merged_meta["suggested_topics"] = known_topics(
                self.siis_chunks, limit=ABSTAIN_TOPIC_SUGGESTIONS,
            )
            return {
                "query": query,
                "query_variations": variations,
                "response": {"contexts": []},
                "meta": merged_meta,
            }

        merged_response = {
            "query": query,
            "query_variations": variations,
            "response": {"contexts": all_contexts[:MAX_CONTEXTS]},
            "meta": merged_meta,
        }

        self.cache.store(query, {
            "query": query,
            "query_variations": variations,
            "response": merged_response["response"],
        })

        return merged_response

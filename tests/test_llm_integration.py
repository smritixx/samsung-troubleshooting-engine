"""LLM-integration tests."""
import json
import re as _re
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import llm as llm_module
from app.data_loader import load_deeplinks
from app.main import app, pipeline


client = TestClient(app)


def _all_catalog_uris():
    """Every URI in the catalog: top-level AND nested validation URIs."""
    uris = set()
    for d in load_deeplinks():
        if d.get("deeplink"):
            uris.add(d["deeplink"])
        v = d.get("validation")
        if isinstance(v, dict) and v.get("deeplink"):
            uris.add(v["deeplink"])
    return uris


CATALOG_URIS = _all_catalog_uris()
CATALOG_LOWER = {u.lower() for u in CATALOG_URIS}


GOOD_PLAN = {
    "goal": "Follow these steps to troubleshoot your device.",
    "title": "Blank display troubleshooting",
    "actions": [
        {
            "actionName": "Force a restart",
            "description": "Force a restart to clear a temporary hang.",
            "category": "auto",
            "steps": [
                "Press and hold the power and volume down buttons for 20 seconds.",
                "Release the buttons and wait for the Samsung logo to appear.",
            ],
        },
        {
            "actionName": "Visit a Samsung service center",
            "description": "If the screen still does not turn on, seek service.",
            "category": "manual",
            "steps": ["Contact Samsung support for further assistance."],
        },
    ],
}

PLAN_WITH_HTTP_URL = {
    "goal": "Visit https://evil.example.com for help",
    "title": "Bad plan",
    "actions": [
        {
            "actionName": "Open the fix",
            "description": "Go to http://phishing.example.com.",
            "category": "auto",
            "steps": [
                "Open https://evil.example.com in the browser.",
                "Also try www.another-evil.example.com.",
            ],
        },
    ],
}

PLAN_WITH_BIXBY_URL = {
    "goal": "Open battery settings",
    "title": "Bad plan",
    "actions": [
        {
            "actionName": "Open battery",
            "description": "Go to battery settings.",
            "category": "auto",
            "steps": [
                "Navigate to bixby://masked/act/deadbeef0000",
                "Then tap the battery optimisation switch.",
            ],
        },
    ],
}


def _clear_cache():
    pipeline.cache.clear()


def _llm_return(plan):
    """Wrap a plan dict in the tuple that llm.generate_structured_plan returns."""
    return (plan, {"input_tokens": 10, "output_tokens": 20, "model": "mock"})


def test_cache_hit_does_not_invoke_llm():
    _clear_cache()
    query = "My Galaxy S24 Ultra screen is completely black and wont turn on."

    with patch.object(llm_module, "is_llm_enabled", return_value=True), \
         patch.object(llm_module, "generate_structured_plan",
                      return_value=_llm_return(GOOD_PLAN)) as fake:
        client.post("/v1/troubleshoot", json={"query": query, "sis_response": ""})
        first_calls = fake.call_count
        r2 = client.post("/v1/troubleshoot", json={"query": query, "sis_response": ""}).json()
        second_calls = fake.call_count

    assert r2["meta"]["cache_hit"] is True
    assert first_calls >= 1
    assert second_calls == first_calls, "LLM was invoked again on a cache hit"


def test_cache_miss_invokes_llm():
    _clear_cache()
    query = "My Galaxy phone screen is totally cracked and I cannot use it."

    with patch.object(llm_module, "is_llm_enabled", return_value=True), \
         patch.object(llm_module, "generate_structured_plan",
                      return_value=_llm_return(GOOD_PLAN)) as fake:
        r = client.post("/v1/troubleshoot", json={"query": query, "sis_response": ""}).json()

    assert fake.call_count >= 1
    assert r["meta"]["cache_hit"] is False


def test_llm_url_output_is_rejected():
    _clear_cache()
    query = "My tablet screen goes blank sometimes."

    with patch.object(llm_module, "is_llm_enabled", return_value=True), \
         patch.object(llm_module, "generate_structured_plan",
                      return_value=_llm_return(PLAN_WITH_HTTP_URL)):
        r = client.post("/v1/troubleshoot", json={"query": query, "sis_response": ""}).json()

    body = json.dumps(r).lower()
    assert "evil.example.com" not in body
    assert "phishing.example.com" not in body


def test_llm_cannot_inject_bixby_url():
    _clear_cache()
    query = "My Galaxy screen is black."

    with patch.object(llm_module, "is_llm_enabled", return_value=True), \
         patch.object(llm_module, "generate_structured_plan",
                      return_value=_llm_return(PLAN_WITH_BIXBY_URL)):
        r = client.post("/v1/troubleshoot", json={"query": query, "sis_response": ""}).json()

    body = json.dumps(r).lower()
    assert "bixby://masked/act/deadbeef0000" not in body
    for m in _re.finditer(r"bixby://[^\s\"'<>]+", body):
        assert m.group(0) in CATALOG_LOWER, f"fabricated deeplink leaked: {m.group(0)}"


def test_final_deeplinks_all_in_catalog():
    _clear_cache()
    query = "My Galaxy S24 Ultra screen is completely black and wont turn on."

    with patch.object(llm_module, "is_llm_enabled", return_value=True), \
         patch.object(llm_module, "generate_structured_plan",
                      return_value=_llm_return(GOOD_PLAN)):
        r = client.post("/v1/troubleshoot", json={"query": query, "sis_response": ""}).json()

    for ctx in r["response"]["contexts"]:
        for action in ctx["actions"]:
            for sg in action["stepGroups"]:
                ad = sg.get("actionableDeeplink")
                if ad and ad.get("deeplink"):
                    assert ad["deeplink"] in CATALOG_URIS
                vd = sg.get("validationDeeplink")
                if vd and vd.get("deeplink"):
                    assert vd["deeplink"] in CATALOG_URIS


def test_llm_failure_falls_back_to_deterministic():
    _clear_cache()
    query = "My Galaxy phone screen is totally cracked and I cannot use it."

    with patch.object(llm_module, "is_llm_enabled", return_value=True), \
         patch.object(llm_module, "generate_structured_plan",
                      return_value=(None, {})) as fake:
        r = client.post("/v1/troubleshoot", json={"query": query, "sis_response": ""}).json()

    assert fake.call_count >= 1
    assert r["meta"].get("llm_fallback") is True or \
           r["meta"].get("multi_intent") is True
    # Response should still be coherent
    assert "response" in r
    assert "contexts" in r["response"]


def test_llm_exception_falls_back():
    _clear_cache()
    query = "My Galaxy S22 screen inputs are delayed and the touch is laggy."

    with patch.object(llm_module, "is_llm_enabled", return_value=True), \
         patch.object(llm_module, "generate_structured_plan",
                      side_effect=RuntimeError("boom")):
        r = client.post("/v1/troubleshoot", json={"query": query, "sis_response": ""}).json()

    assert "response" in r
    assert "contexts" in r["response"]


def test_health_still_works():
    assert client.get("/health").json() == {"status": "ok"}


def test_llm_off_by_default():
    """When LLM is not enabled, the response must say so explicitly."""
    _clear_cache()
    query = "My Galaxy S22 screen is blank and I cannot use Smart Switch."

    with patch.object(llm_module, "is_llm_enabled", return_value=False):
        r = client.post("/v1/troubleshoot", json={"query": query, "sis_response": ""}).json()

    # Either a single-intent or multi-intent response, but llm_enabled must be False
    # and llm_used must also be False (we never called the model).
    assert r["meta"].get("llm_enabled") is False
    assert r["meta"].get("llm_used") is False
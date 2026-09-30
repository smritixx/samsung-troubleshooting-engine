import json
from fastapi.testclient import TestClient

from app.main import app, pipeline

client = TestClient(app)


def _clear_cache():
    pipeline.cache.clear()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_troubleshoot_blank_screen():
    _clear_cache()
    r = client.post("/v1/troubleshoot", json={
        "query": "My Galaxy S24 Ultra screen is completely black and won't turn on, "
                 "even though the phone powers on and rings.",
        "sis_response": "",
    })
    assert r.status_code == 200
    data = r.json()
    assert "response" in data
    assert "contexts" in data["response"]


def test_cracked_screen_matches_catalog():
    _clear_cache()
    r = client.post("/v1/troubleshoot", json={
        "query": "My Galaxy phone's screen is completely cracked and I can't use it.",
        "sis_response": "",
    })
    assert r.status_code == 200
    data = r.json()
    s = json.dumps(data)
    assert "http://" not in s
    assert "https://" not in s
    assert "www." not in s


def test_cache_hit():
    _clear_cache()
    q = "My Galaxy S22 screen is blank and I cannot use Smart Switch."
    r1 = client.post("/v1/troubleshoot", json={"query": q, "sis_response": ""}).json()
    r2 = client.post("/v1/troubleshoot", json={"query": q, "sis_response": ""}).json()
    assert r2["meta"]["cache_hit"] is True
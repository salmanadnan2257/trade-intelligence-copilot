import pytest
from starlette.testclient import TestClient  # wraps httpx

from app import agent
from app.server import app


def _client():
    return TestClient(app)


def test_health():
    with _client() as c:
        r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["database"] is True
    assert body["vector_store"] is True


def test_index_page_renders():
    with _client() as c:
        r = c.get("/")
    assert r.status_code == 200
    assert "Trade Intelligence Copilot" in r.text


def test_ask_endpoint_with_stubbed_agent(monkeypatch):
    def fake_ask(question):
        return {
            "question": question,
            "answer": "stub",
            "supported": True,
            "plan": {},
            "citations": {"sql": None, "sql_rows": [], "chunks": []},
            "trace": [],
            "cost": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_usd": 0.0, "calls": []},
            "latency_ms": 1.0,
        }

    monkeypatch.setattr(agent, "ask", fake_ask)
    # server imports agent module and calls agent.ask, so patching the attribute works
    with _client() as c:
        r = c.post("/ask", json={"question": "hello"})
    assert r.status_code == 200
    assert r.json()["answer"] == "stub"


@pytest.mark.live
def test_ask_endpoint_live_grounded():
    with _client() as c:
        r = c.post("/ask", json={"question": "How many import shipments came from China?"})
    assert r.status_code == 200
    body = r.json()
    assert body["supported"] is True
    assert "825" in body["answer"]

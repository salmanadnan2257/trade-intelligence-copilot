import pytest

from app import answer, tools


def test_hs_lookup_by_name():
    res = tools.hs_lookup("glucose")
    assert res["matches"]
    assert any("1702" in (m["category_hs"] or "") for m in res["matches"])


def test_hs_lookup_by_code():
    res = tools.hs_lookup("1701")
    assert res["matches"]
    assert all("1701" in (m["product_hs"] or "") for m in res["matches"])


def test_deterministic_refusal_when_no_evidence():
    assert answer.deterministic_refusal(None, [], None) is True
    assert answer.deterministic_refusal({"rows": []}, [], {"matches": []}) is True


def test_deterministic_refusal_false_with_evidence():
    assert answer.deterministic_refusal({"rows": [{"c": 1}]}, [], None) is False
    assert answer.deterministic_refusal(None, [{"id": "x#s1"}], None) is False


# --- live tests (need the API) ---------------------------------------------

@pytest.mark.live
def test_planner_routes_sql_question():
    from app import planner

    p = planner.plan("What was the total USD value of all imports?")
    assert p["sql_needed"] is True


@pytest.mark.live
def test_planner_routes_regulation_question():
    from app import planner

    p = planner.plan("What is a phytosanitary certificate?")
    assert p["retrieval_needed"] is True


@pytest.mark.live
def test_agent_refuses_out_of_scope():
    from app import agent

    r = agent.ask("What were Pakistan's wheat exports to India in 2019?")
    assert r["supported"] is False

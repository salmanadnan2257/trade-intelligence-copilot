"""End-to-end orchestration: plan -> run tools -> synthesize grounded answer.

Returns the answer, its citations, a full tool-call trace, and the per-query cost.
"""
from __future__ import annotations

import time
from typing import Any

from . import answer, config, llm, planner, tools


def ask(question: str) -> dict[str, Any]:
    meter = llm.CostMeter()
    t0 = time.perf_counter()
    trace: list[dict[str, Any]] = []

    the_plan = planner.plan(question, meter=meter)
    trace.append({"step": "plan", "detail": the_plan})

    sql_result: dict | None = None
    hs_result: dict | None = None
    all_hits: list[dict] = []
    strong_chunks: list[dict] = []

    if the_plan["sql_needed"]:
        q = the_plan.get("sql_question") or question
        sql_result = tools.text_to_sql(q, meter=meter)
        trace.append(
            {
                "step": "sql",
                "detail": {
                    "sql": sql_result.get("sql"),
                    "error": sql_result.get("error"),
                    "row_count": len(sql_result.get("rows") or []),
                    "rows_preview": (sql_result.get("rows") or [])[:5],
                },
            }
        )

    if the_plan["retrieval_needed"]:
        rq = the_plan.get("retrieval_query") or question
        all_hits, strong_chunks = tools.relevant_chunks(rq)
        trace.append(
            {
                "step": "retrieval",
                "detail": {
                    "query": rq,
                    "hits": [
                        {"id": h["id"], "section": h["section"], "score": round(h["score"], 3)}
                        for h in all_hits
                    ],
                    "kept": [c["id"] for c in strong_chunks],
                    "min_score": config.RETRIEVAL_MIN_SCORE,
                },
            }
        )

    if the_plan.get("hs_lookup"):
        hs_result = tools.hs_lookup(the_plan["hs_lookup"])
        trace.append(
            {
                "step": "hs_lookup",
                "detail": {"term": hs_result["term"], "match_count": len(hs_result["matches"])},
            }
        )

    # Refuse deterministically when there is no evidence at all, or clearly out of scope
    # with nothing retrieved.
    if answer.deterministic_refusal(sql_result, strong_chunks, hs_result):
        result = {
            "supported": False,
            "answer": answer.REFUSAL_TEXT,
            "cited_chunks": [],
            "used_sql": False,
        }
        trace.append({"step": "refusal", "detail": {"reason": "no supporting evidence"}})
    else:
        result = answer.synthesize(
            question,
            sql_result=sql_result,
            chunks=strong_chunks,
            hs_result=hs_result,
            meter=meter,
        )
        trace.append({"step": "synthesize", "detail": {"supported": result["supported"]}})

    latency_ms = (time.perf_counter() - t0) * 1000.0

    citations = {
        "sql": sql_result.get("sql") if sql_result else None,
        "sql_rows": (sql_result.get("rows") or [])[:20] if (sql_result and result["used_sql"]) else [],
        "hs_matches": (hs_result.get("matches") or [])[:25] if hs_result else [],
        "chunks": [
            {"id": c["id"], "doc_title": c["doc_title"], "section": c["section"], "text": c["text"]}
            for c in strong_chunks
            if c["id"] in set(result.get("cited_chunks", []))
        ],
    }

    return {
        "question": question,
        "answer": result["answer"],
        "supported": result["supported"],
        "plan": the_plan,
        "citations": citations,
        "trace": trace,
        "cost": meter.summary(),
        "latency_ms": round(latency_ms, 1),
    }

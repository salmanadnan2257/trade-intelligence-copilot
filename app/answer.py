"""Grounded answer synthesis with an explicit refusal path.

The synthesizer only sees the evidence gathered by the tools. It must ground every
claim in that evidence and cite it, or refuse with 'not supported by the data'.
There is also a deterministic guard: if no SQL rows and no strong regulation chunks
were found, we refuse before spending a synthesis call reasoning about nothing.
"""
from __future__ import annotations

import json
from typing import Any

from . import config, llm

REFUSAL_TEXT = "This question is not supported by the available data."

ANSWER_SYSTEM = (
    "You answer questions about Pakistani sugar-sector import data and trade "
    "regulations using ONLY the evidence provided. You never invent numbers, "
    "countries, or rules. If the evidence does not support an answer, you refuse."
)


def _evidence_block(sql_result: dict | None, chunks: list[dict], hs_result: dict | None) -> str:
    parts = []
    if sql_result is not None:
        parts.append("SQL QUERY:\n" + str(sql_result.get("sql")))
        if sql_result.get("error"):
            parts.append("SQL ERROR: " + str(sql_result["error"]))
        rows = sql_result.get("rows") or []
        parts.append("SQL ROWS (%d):\n%s" % (len(rows), json.dumps(rows[:50], default=str)))
    if hs_result is not None:
        parts.append("HS LOOKUP:\n" + json.dumps(hs_result.get("matches", [])[:25], default=str))
    if chunks:
        reg = "\n\n".join(
            f"[{c['id']}] ({c['doc_title']} :: {c['section']})\n{c['text']}" for c in chunks
        )
        parts.append("REGULATION CHUNKS:\n" + reg)
    return "\n\n".join(parts) if parts else "(no evidence gathered)"


def synthesize(
    question: str,
    *,
    sql_result: dict | None,
    chunks: list[dict],
    hs_result: dict | None,
    meter: llm.CostMeter | None = None,
) -> dict[str, Any]:
    evidence = _evidence_block(sql_result, chunks, hs_result)
    prompt = f"""Question: {question}

EVIDENCE (this is everything you may use):
{evidence}

Instructions:
- Answer ONLY from the evidence above. Do not use outside knowledge for facts,
  numbers, or specific rules.
- If the evidence (SQL rows and/or regulation chunks) does not support an answer,
  set "supported" to false and give the refusal.
- Cite your sources: list the SQL query if you used numbers from it, and the chunk
  ids (like "customs-duty-concepts#s2") for any rule you state.
- Keep the answer concise and factual.

Return ONLY JSON:
{{
  "supported": true/false,
  "answer": "<answer, or the refusal reason if unsupported>",
  "cited_chunks": ["<chunk id>", ...],
  "used_sql": true/false
}}"""
    out = llm.generate_json(prompt, model=config.MODEL_FAST, system=ANSWER_SYSTEM, meter=meter)
    supported = bool(out.get("supported"))
    answer_text = out.get("answer", "").strip()
    if not supported:
        answer_text = answer_text or REFUSAL_TEXT
    return {
        "supported": supported,
        "answer": answer_text,
        "cited_chunks": out.get("cited_chunks", []),
        "used_sql": bool(out.get("used_sql")),
    }


def deterministic_refusal(sql_result: dict | None, chunks: list[dict], hs_result: dict | None) -> bool:
    """True when there is simply no evidence to answer from."""
    has_rows = bool(sql_result and sql_result.get("rows"))
    has_chunks = bool(chunks)
    has_hs = bool(hs_result and hs_result.get("matches"))
    return not (has_rows or has_chunks or has_hs)

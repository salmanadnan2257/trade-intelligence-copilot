"""The agentic planner: an LLM decides which tools a question needs."""
from __future__ import annotations

from typing import Any

from . import config, llm, schema_prompt

PLANNER_SYSTEM = (
    "You are a query planner for a trade-intelligence copilot. You decide which "
    "tools are needed to answer a question. You do not answer the question yourself."
)

PLANNER_PROMPT = """Available tools:
- sql: run analytics over a database of Pakistani sugar-sector IMPORT shipments.
  Use for anything about values, quantities, counts, countries, buyers, sellers,
  HS-code totals, rankings, or dates that live in the transaction data.
- retrieval: search a corpus of agricultural-trade regulation explainers (HS-code
  basics, phytosanitary/SPS, customs duty concepts, import documents, policy).
  Use when the question asks about rules, procedures, definitions, or "how/why".
- hs_lookup: look up the HS-code hierarchy (heading -> category -> subcategory) by
  code or product name. Use for "what HS code is X" style questions.

{scope}

Decide the plan. A question may need several tools (for example, compute a number
with sql, then explain the rule behind it with retrieval).

Question: {question}

Return ONLY JSON with this shape:
{{
  "sql_needed": true/false,
  "sql_question": "<question restated for SQL>" or null,
  "retrieval_needed": true/false,
  "retrieval_query": "<search query>" or null,
  "hs_lookup": "<code or product name>" or null,
  "in_scope": true/false,   // false if it clearly asks for data outside the dataset scope
  "rationale": "<one sentence>"
}}"""


def plan(question: str, *, meter: llm.CostMeter | None = None) -> dict[str, Any]:
    prompt = PLANNER_PROMPT.format(
        scope=schema_prompt.DATASET_SCOPE, question=question
    )
    result = llm.generate_json(
        prompt, model=config.MODEL_FAST, system=PLANNER_SYSTEM, meter=meter
    )
    # Normalize missing keys.
    return {
        "sql_needed": bool(result.get("sql_needed")),
        "sql_question": result.get("sql_question"),
        "retrieval_needed": bool(result.get("retrieval_needed")),
        "retrieval_query": result.get("retrieval_query"),
        "hs_lookup": result.get("hs_lookup"),
        "in_scope": result.get("in_scope", True),
        "rationale": result.get("rationale", ""),
    }

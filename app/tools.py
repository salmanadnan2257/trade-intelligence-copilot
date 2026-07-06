"""The three tools the planner can invoke: text-to-SQL, regulation retrieval, HS lookup."""
from __future__ import annotations

from typing import Any

from . import config, db, embeddings, llm, schema_prompt, vectorstore

_store: vectorstore.VectorStore | None = None


def store() -> vectorstore.VectorStore:
    global _store
    if _store is None:
        _store = vectorstore.VectorStore.load()
    return _store


# --- Tool 1: text-to-SQL over trade data -----------------------------------

TEXT_TO_SQL_SYSTEM = "You translate questions into safe read-only SQLite SELECT queries."


def text_to_sql(question: str, *, meter: llm.CostMeter | None = None) -> dict[str, Any]:
    prompt = f"""{schema_prompt.SCHEMA_DDL}

{schema_prompt.DATASET_SCOPE}

{schema_prompt.TEXT_TO_SQL_RULES}

Question: {question}

Return ONLY JSON: {{"sql": "<SELECT ...>" or null, "reason": "<short>"}}"""
    plan = llm.generate_json(prompt, model=config.MODEL_FAST, system=TEXT_TO_SQL_SYSTEM, meter=meter)
    sql = plan.get("sql")
    result: dict[str, Any] = {"sql": sql, "reason": plan.get("reason", ""), "rows": [], "error": None}
    if not sql:
        result["error"] = "no_sql_generated"
        return result
    try:
        result["rows"] = db.run_select(sql)
    except db.SqlGuardError as e:
        result["error"] = f"guard_rejected: {e}"
    except Exception as e:  # execution error (bad column, syntax)
        result["error"] = f"execution_error: {e}"
    return result


# --- Tool 2: vector retrieval over regulations ------------------------------

def retrieve(query: str, *, top_k: int | None = None) -> list[dict]:
    qvec = embeddings.embed_one(query, task_type="RETRIEVAL_QUERY")
    hits = store().search(qvec, top_k=top_k)
    return hits


def relevant_chunks(query: str, *, top_k: int | None = None) -> tuple[list[dict], list[dict]]:
    """Return (all_hits, hits_above_threshold)."""
    hits = retrieve(query, top_k=top_k)
    strong = [h for h in hits if h["score"] >= config.RETRIEVAL_MIN_SCORE]
    return hits, strong


# --- Tool 3: HS-code lookup -------------------------------------------------

def hs_lookup(term: str) -> dict[str, Any]:
    """Look up HS-code hierarchy rows by code prefix or name substring."""
    term = (term or "").strip()
    like = f"%{term}%"
    code_like = f"{term}%"
    conn = db.connect(read_only=True)
    try:
        rows = conn.execute(
            """
            SELECT p.hs_code AS product_hs, p.name AS product,
                   c.hs_code AS category_hs, c.name AS category,
                   s.hs_code AS subcategory_hs, s.name AS subcategory
            FROM product_subcategories s
            JOIN product_categories c ON s.category_id = c.id
            JOIN products p ON c.product_id = p.id
            WHERE s.hs_code LIKE ? OR c.hs_code LIKE ? OR p.hs_code LIKE ?
               OR s.name LIKE ? OR c.name LIKE ? OR p.name LIKE ?
            LIMIT 25
            """,
            (code_like, code_like, code_like, like, like, like),
        ).fetchall()
        return {"term": term, "matches": [dict(r) for r in rows]}
    finally:
        conn.close()

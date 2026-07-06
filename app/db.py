"""SQLite access plus a strict read-only SQL guard for the text-to-SQL path.

The guard is the security boundary for LLM-generated SQL. It rejects anything
that is not a single SELECT and blocks write/DDL keywords, multiple statements,
comments, and PRAGMA. Execution also runs against a connection opened in
query-only mode so a bypass still cannot write.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

from . import config

# Keywords that must never appear in an LLM-produced query.
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|attach|"
    r"detach|pragma|vacuum|reindex|grant|revoke|begin|commit|rollback)\b",
    re.IGNORECASE,
)


class SqlGuardError(ValueError):
    """Raised when a query fails the read-only safety checks."""


def validate_select(sql: str) -> str:
    """Return a cleaned single SELECT statement or raise SqlGuardError."""
    if not sql or not sql.strip():
        raise SqlGuardError("empty query")
    cleaned = sql.strip()
    if "--" in cleaned or "/*" in cleaned:
        raise SqlGuardError("comments are not allowed")
    # Allow one optional trailing semicolon, reject multiple statements.
    body = cleaned[:-1] if cleaned.endswith(";") else cleaned
    if ";" in body:
        raise SqlGuardError("multiple statements are not allowed")
    lowered = body.lstrip().lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        raise SqlGuardError("only SELECT (or WITH ... SELECT) queries are allowed")
    if _FORBIDDEN.search(body):
        raise SqlGuardError("query contains a forbidden write/DDL keyword")
    return body


def connect(db_path: Path | None = None, *, read_only: bool = True) -> sqlite3.Connection:
    path = Path(db_path or config.DB_PATH)
    if read_only:
        conn = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
        # Belt and braces: also set query_only on the connection.
        conn.execute("PRAGMA query_only = ON")
    else:
        conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def run_select(sql: str, *, limit: int = 200, db_path: Path | None = None) -> list[dict[str, Any]]:
    """Validate then execute a read-only SELECT, capping the row count."""
    body = validate_select(sql)
    conn = connect(db_path, read_only=True)
    try:
        cur = conn.execute(body)
        rows = cur.fetchmany(limit)
        return [dict(r) for r in rows]
    finally:
        conn.close()

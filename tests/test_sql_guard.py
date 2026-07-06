import pytest

from app import db


@pytest.mark.parametrize(
    "bad",
    [
        "INSERT INTO trade_transactions VALUES (1)",
        "UPDATE trade_transactions SET usd = 0",
        "DELETE FROM trade_transactions",
        "DROP TABLE trade_transactions",
        "SELECT 1; DROP TABLE companies",
        "SELECT * FROM companies -- comment",
        "PRAGMA table_info(companies)",
        "ALTER TABLE companies ADD COLUMN x TEXT",
        "SELECT * FROM companies; SELECT * FROM products",
        "",
        "REPLACE INTO products VALUES (1,'x','y')",
    ],
)
def test_guard_rejects_non_select(bad):
    with pytest.raises(db.SqlGuardError):
        db.validate_select(bad)


@pytest.mark.parametrize(
    "good",
    [
        "SELECT COUNT(*) FROM trade_transactions",
        "select sum(usd) from trade_transactions where country = 'China'",
        "WITH t AS (SELECT usd FROM trade_transactions) SELECT SUM(usd) FROM t",
        "SELECT * FROM companies LIMIT 5;",
    ],
)
def test_guard_accepts_select(good):
    assert db.validate_select(good)


def test_run_select_returns_rows():
    rows = db.run_select("SELECT COUNT(*) AS c FROM trade_transactions")
    assert rows and rows[0]["c"] == 2232


def test_run_select_blocks_write():
    with pytest.raises(db.SqlGuardError):
        db.run_select("DELETE FROM trade_transactions")


def test_readonly_connection_cannot_write():
    import sqlite3

    conn = db.connect(read_only=True)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("CREATE TABLE hack(x)")
    finally:
        conn.close()

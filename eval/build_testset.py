"""Write eval/testset.jsonl, computing every SQL gold answer from the database.

The gold numbers are never typed by hand: each one is the result of the query next
to it, run against data/trade.sqlite. Rebuild the database and rerun this, and the
gold answers follow. Run: python -m eval.build_testset
"""
from __future__ import annotations

import json
from pathlib import Path

from app import db

HERE = Path(__file__).resolve().parent
TESTSET = HERE / "testset.jsonl"


def scalar(sql: str):
    row = db.run_select(sql)[0]
    return next(iter(dict(row).values()))


def money(sql: str) -> float:
    return round(float(scalar(sql)), 2)


TOP_COUNTRY = (
    "SELECT country FROM trade_transactions GROUP BY country "
    "ORDER BY SUM(usd) DESC LIMIT 1"
)


def build() -> list[dict]:
    top_country = scalar(TOP_COUNTRY)
    hs1704 = money("SELECT SUM(usd) FROM trade_transactions WHERE hs_code LIKE '1704%'")
    sql = [
        ("What was the total USD value of all imports in the dataset?",
         {"value": money("SELECT SUM(usd) FROM trade_transactions"), "rel_tol": 0.01}),
        ("How many trade transactions are in the dataset in total?",
         {"value": scalar("SELECT COUNT(*) FROM trade_transactions"), "rel_tol": 0.0}),
        ("How many import shipments came from China?",
         {"value": scalar("SELECT COUNT(*) FROM trade_transactions WHERE country = 'China'"), "rel_tol": 0.0}),
        ("Which country was the largest source of imports by total USD value?",
         {"contains": top_country}),
        ("What was the total imported quantity in kilograms across all shipments?",
         {"value": money("SELECT SUM(qty_kg) FROM trade_transactions"), "rel_tol": 0.01}),
        ("How many distinct source countries appear in the import data?",
         {"value": scalar("SELECT COUNT(DISTINCT country) FROM trade_transactions"), "rel_tol": 0.0}),
        ("What was the total USD value of imports classified under HS heading 1704 (sugar confectionery)?",
         {"value": hs1704, "rel_tol": 0.01}),
        ("How many shipments were classified under HS heading 1701?",
         {"value": scalar("SELECT COUNT(*) FROM trade_transactions WHERE hs_code LIKE '1701%'"), "rel_tol": 0.0}),
        ("What was the total USD value of imports from Germany?",
         {"value": money("SELECT SUM(usd) FROM trade_transactions WHERE country = 'Germany'"), "rel_tol": 0.01}),
        ("How many shipments did Orvella Foods import?",
         {"value": scalar("SELECT COUNT(*) FROM trade_transactions WHERE buyer LIKE '%Orvella%'"), "rel_tol": 0.0}),
        ("What was the total USD value of imports from China?",
         {"value": money("SELECT SUM(usd) FROM trade_transactions WHERE country = 'China'"), "rel_tol": 0.01}),
        ("How many shipments were recorded under HS code 1702.1100 (lactose)?",
         {"value": scalar("SELECT COUNT(*) FROM trade_transactions WHERE hs_code = '1702.1100'"), "rel_tol": 0.0}),
    ]
    items = [
        {"id": f"sql-{i:02d}", "type": "sql", "question": q, "gold": g, "expect_refusal": False}
        for i, (q, g) in enumerate(sql, start=1)
    ]

    retrieval = [
        ("What is a phytosanitary certificate and who issues it?", ["phytosanitary-sps-basics"]),
        ("How is customs duty charged: what is the difference between ad valorem and specific duty?", ["customs-duty-concepts"]),
        ("What documents are typically needed to import goods into Pakistan?", ["pakistan-import-documents"]),
        ("What is a certificate of origin and how does it affect duty?", ["certificate-of-origin-fta"]),
        ("What is an anti-dumping duty and how does it arise?", ["anti-dumping-safeguard"]),
        ("What is the Pakistan Single Window and what does it do?", ["pakistan-single-window-weboc"]),
        ("What does HS heading 1702 cover?", ["sugar-syrups-hs-1702", "hs-codes-basics"]),
    ]
    items += [
        {"id": f"ret-{i:02d}", "type": "retrieval", "question": q, "gold_docs": d, "expect_refusal": False}
        for i, (q, d) in enumerate(retrieval, start=1)
    ]

    items += [
        {"id": "multi-01", "type": "multi",
         "question": "What was the total USD value of HS 1704 confectionery imports, and what kind of products does HS heading 1704 cover?",
         "gold": {"value": hs1704, "rel_tol": 0.01}, "gold_docs": ["confectionery-hs-1704"], "expect_refusal": False},
        {"id": "multi-02", "type": "multi",
         "question": "How much sugar under HS 1701 was imported by value, and why is sugar trade treated as policy-sensitive in Pakistan?",
         "gold": {"value": money("SELECT SUM(usd) FROM trade_transactions WHERE hs_code LIKE '1701%'"), "rel_tol": 0.01},
         "gold_docs": ["sugar-hs-1701"], "expect_refusal": False},
        {"id": "multi-03", "type": "multi",
         "question": "Which country was the top source of imports by value, and what is a certificate of origin that might affect the duty on its goods?",
         "gold": {"contains": top_country}, "gold_docs": ["certificate-of-origin-fta"], "expect_refusal": False},
    ]

    refusals = [
        "What were Pakistan's wheat exports to India in 2019?",
        "What was the total USD value of cotton imports in the dataset?",
        "How many shipments were exported to Dubai?",
        "How many rice imports were recorded during 2010?",
        "Which companies imported steel according to the data?",
        "What was the total value of molasses exports?",
    ]
    items += [
        {"id": f"refuse-{i:02d}", "type": "refusal", "question": q, "expect_refusal": True}
        for i, q in enumerate(refusals, start=1)
    ]
    return items


if __name__ == "__main__":
    items = build()
    TESTSET.write_text("".join(json.dumps(i) + "\n" for i in items), encoding="utf-8")
    print(f"Wrote {len(items)} items to {TESTSET}")

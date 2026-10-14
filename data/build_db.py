"""Generate the demo trade database: synthetic Pakistani sugar-sector import records.

Every company, shipment, price and quantity in here is made up. The generator is
seeded, so the same code always produces the same database byte for byte, and
the evaluation's gold answers can be recomputed from it at any time.

What is real is only the public scaffolding: the HS chapter 17 headings (sugar,
sugar syrups, molasses, confectionery) and the country names. The schema is a
small star: companies, an HS-code hierarchy (product -> category -> subcategory
-> item), and trade_transactions.

Run: python -m data.build_db
"""
from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "trade.sqlite"

SEED = 1717
N_TRANSACTIONS = 2000
PKR_PER_USD = 280.0
WINDOW_START = date(2025, 1, 1)
WINDOW_DAYS = 365  # calendar year 2025

SCHEMA = """
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    hs_code TEXT,
    name TEXT
);
CREATE TABLE product_categories (
    id INTEGER PRIMARY KEY,
    hs_code TEXT,
    name TEXT,
    product_id INTEGER REFERENCES products(id)
);
CREATE TABLE product_subcategories (
    id INTEGER PRIMARY KEY,
    hs_code TEXT,
    name TEXT,
    category_id INTEGER REFERENCES product_categories(id)
);
CREATE TABLE product_items (
    id INTEGER PRIMARY KEY,
    name TEXT,
    subcategory_id INTEGER REFERENCES product_subcategories(id)
);
CREATE TABLE companies (
    id INTEGER PRIMARY KEY,
    name TEXT,
    country TEXT,
    province TEXT,
    year_established INTEGER,
    verification_status TEXT,
    has_trade_data INTEGER
);
CREATE TABLE trade_transactions (
    id INTEGER PRIMARY KEY,
    tx_reference TEXT,
    trade_type TEXT,          -- 'import' or 'export'
    country TEXT,             -- counterparty country
    hs_code TEXT,
    buyer TEXT,
    seller TEXT,
    shipping_agent TEXT,
    product_item_id INTEGER REFERENCES product_items(id),
    usd REAL,                 -- transaction value in USD
    pkr REAL,                 -- transaction value in PKR
    qty_kg REAL,
    qty_mt REAL,
    std_unit TEXT,
    usd_per_kg REAL,
    usd_per_mt REAL,
    reporting_date TEXT       -- ISO date YYYY-MM-DD
);
CREATE INDEX idx_tx_country ON trade_transactions(country);
CREATE INDEX idx_tx_hs ON trade_transactions(hs_code);
CREATE INDEX idx_tx_type ON trade_transactions(trade_type);
CREATE INDEX idx_tx_date ON trade_transactions(reporting_date);
"""

# HS chapter 17, from the public tariff nomenclature. Each subcategory carries
# the price band (USD per metric tonne) and shipment size band (MT) the
# generator draws from, plus a weight for how often it is shipped.
#   heading: (name, [(category_hs, category_name, [(sub_hs, sub_name, usd_mt, mt, weight)])])
HS_TREE = {
    "1701": ("Cane or beet sugar", [
        ("1701.14", "Other raw cane sugar", [
            ("1701.1400", "Raw cane sugar, bulk", (430, 560), (500, 6000), 3),
        ]),
        ("1701.99", "Refined sugar", [
            ("1701.9910", "White crystalline cane sugar", (520, 690), (50, 1500), 6),
            ("1701.9920", "White crystalline beet sugar", (540, 700), (25, 600), 2),
            ("1701.9930", "Chemically pure sucrose", (900, 1400), (1, 40), 1),
        ]),
    ]),
    "1702": ("Other sugars and sugar syrups", [
        ("1702.11", "Lactose", [
            ("1702.1100", "Lactose, 99% or more", (1100, 1700), (5, 120), 7),
        ]),
        ("1702.30", "Glucose and glucose syrup", [
            ("1702.3010", "Dextrose monohydrate", (620, 860), (20, 400), 8),
            ("1702.3020", "Dextrose anhydrous", (780, 1050), (10, 200), 5),
            ("1702.3030", "Liquid glucose", (480, 700), (40, 800), 4),
        ]),
        ("1702.60", "Fructose and fructose syrup", [
            ("1702.6000", "High-fructose syrup", (650, 980), (20, 300), 2),
        ]),
        ("1702.90", "Other sugars", [
            ("1702.9010", "Maltodextrin", (700, 1000), (10, 250), 4),
            ("1702.9020", "Caramel colour", (1300, 2400), (1, 30), 2),
            ("1702.9030", "Invert sugar syrup", (600, 900), (10, 150), 1),
        ]),
    ]),
    "1703": ("Molasses", [
        ("1703.10", "Cane molasses", [
            ("1703.1000", "Cane molasses", (110, 210), (300, 5000), 1),
        ]),
    ]),
    "1704": ("Sugar confectionery, without cocoa", [
        ("1704.10", "Chewing gum", [
            ("1704.1000", "Chewing gum, sugar-coated or not", (2200, 4200), (1, 25), 3),
        ]),
        ("1704.90", "Other sugar confectionery", [
            ("1704.9010", "Hard-boiled candy", (1500, 2900), (2, 40), 4),
            ("1704.9020", "Jelly and gummy sweets", (1800, 3600), (1, 30), 4),
            ("1704.9030", "Toffees and caramels", (1700, 3200), (1, 35), 3),
            ("1704.9040", "White chocolate", (3500, 5500), (1, 15), 1),
        ]),
    ]),
}

# Origin countries with a sampling weight. Brazil and Thailand dominate the
# bulk-sugar lanes on purpose, so "largest source by value" has a clear answer.
COUNTRIES = [
    ("Brazil", 9), ("Thailand", 7), ("India", 6), ("China", 10), ("Malaysia", 5),
    ("Indonesia", 4), ("Turkey", 4), ("Germany", 3), ("Netherlands", 3), ("France", 3),
    ("United States", 3), ("New Zealand", 2), ("Australia", 2), ("Poland", 2),
    ("Belgium", 2), ("Spain", 2), ("Italy", 1), ("Vietnam", 2), ("South Africa", 1),
    ("Egypt", 1), ("Argentina", 1), ("Mexico", 1), ("Ireland", 1), ("Denmark", 1),
    ("Singapore", 1), ("Japan", 1), ("South Korea", 1), ("Saudi Arabia", 1),
]

# Bulk sugar and molasses only travel from a handful of producing countries.
BULK_ORIGINS = [("Brazil", 5), ("Thailand", 3), ("India", 2)]

# Invented importer names. Coined words, so none of them should belong to a real firm.
BUYERS = [
    "Zelvora Foods (Pvt) Ltd", "Tamrin Sweets (Pvt) Ltd", "Qadrel Pharma (Pvt) Ltd",
    "Noorvale Confectionery Ltd", "Kashvi Beverages (Pvt) Ltd", "Brimlow Dairy Industries",
    "Saffrel Biscuits (Pvt) Ltd", "Ravenna Agro Traders", "Dhunmal Sugar Refinery Ltd",
    "Orvella Foods (Pvt) Ltd", "Pelican Crest Trading Co", "Mirzan Chemicals (Pvt) Ltd",
    "Talvik Pharmaceuticals Ltd", "Hezara Candy Works", "Indus Quill Foods",
    "Larkspur Nutrition (Pvt) Ltd", "Sabeel Commodity House", "Vantorr Bakeries (Pvt) Ltd",
    "Ghazni Fold Distillers Ltd", "Corvane Snacks (Pvt) Ltd", "Elmira Feed Mills",
    "Pashtar Juice Company", "Ruvali Ice Cream (Pvt) Ltd", "Kestrin Healthcare (Pvt) Ltd",
]
# A few importers dominate, the rest form a long tail.
BUYER_WEIGHTS = [9, 7, 6, 6, 5, 5, 4, 4, 4, 8, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1]

SELLER_STEMS = [
    "Aurelio", "Bendtsen", "Calvaro", "Dunmere", "Estrova", "Faralon", "Grenwick",
    "Halvard", "Istara", "Jorvik", "Kelmora", "Lumaro", "Mavrik", "Nordale",
    "Orlanza", "Pellucid", "Quorra", "Rostell", "Sundara", "Tirvana",
]
SELLER_SUFFIX = {
    "Brazil": "Acucar Ltda", "Thailand": "Sugar Co Ltd", "India": "Agro Exports Pvt Ltd",
    "China": "Biotech Co Ltd", "Malaysia": "Food Sdn Bhd", "Indonesia": "Pangan PT",
    "Turkey": "Gida AS", "Germany": "Lebensmittel GmbH", "Netherlands": "Ingredients BV",
    "France": "Sucres SAS", "United States": "Ingredients Inc", "New Zealand": "Dairy Ltd",
    "Australia": "Foods Pty Ltd", "Poland": "Cukier Sp zoo", "Belgium": "Food NV",
    "Spain": "Alimentos SL", "Italy": "Dolciaria Srl", "Vietnam": "Food JSC",
    "South Africa": "Sugar Pty Ltd", "Egypt": "Food SAE", "Argentina": "Alimentos SA",
    "Mexico": "Azucar SA de CV", "Ireland": "Dairy Ingredients Ltd", "Denmark": "Foods ApS",
    "Singapore": "Trading Pte Ltd", "Japan": "Shokuhin KK", "South Korea": "Foods Co Ltd",
    "Saudi Arabia": "Food Industries Co",
}

AGENTS = [
    "Harborline Logistics (Pvt) Ltd", "Bluewake Shipping Agency", "Keel & Compass Freight",
    "Portgate Maritime Services", "Tidemark Cargo (Pvt) Ltd", "Saltway Forwarding Co",
]

PROVINCES = ["Sindh", "Punjab", "Khyber Pakhtunkhwa", "Balochistan", "Islamabad Capital Territory"]

PACKS = ["25 kg bags", "50 kg bags", "1 MT jumbo bags", "bulk", "drums", "cartons"]
GRADES = ["food grade", "pharma grade", "industrial grade", "ICUMSA 45", "ICUMSA 150"]


def _weighted(rng: random.Random, pairs):
    items = [p[0] for p in pairs]
    weights = [p[1] for p in pairs]
    return rng.choices(items, weights=weights, k=1)[0]


def build(db_path: Path = DB_PATH) -> dict:
    rng = random.Random(SEED)

    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA)

    # HS hierarchy, with one generated item description per (subcategory, pack, grade) draw.
    subcats = []  # (sub_id, sub_hs, sub_name, usd_band, mt_band, weight, heading)
    pid = cid = sid = 0
    for heading, (pname, cats) in HS_TREE.items():
        pid += 1
        conn.execute("INSERT INTO products VALUES (?,?,?)", (pid, heading, pname))
        for c_hs, c_name, subs in cats:
            cid += 1
            conn.execute("INSERT INTO product_categories VALUES (?,?,?,?)", (cid, c_hs, c_name, pid))
            for s_hs, s_name, usd_band, mt_band, weight in subs:
                sid += 1
                conn.execute("INSERT INTO product_subcategories VALUES (?,?,?,?)", (sid, s_hs, s_name, cid))
                subcats.append((sid, s_hs, s_name, usd_band, mt_band, weight, heading))

    item_id = 0
    sub_items: dict[int, list[int]] = {}
    for sub_id, _hs, s_name, *_ in subcats:
        for _ in range(4):
            item_id += 1
            desc = f"{s_name.upper()}, {rng.choice(GRADES).upper()}, PACKED IN {rng.choice(PACKS).upper()}"
            conn.execute("INSERT INTO product_items VALUES (?,?,?)", (item_id, desc, sub_id))
            sub_items.setdefault(sub_id, []).append(item_id)

    # Directory: every buyer, plus a few firms with no trade rows.
    directory = BUYERS + ["Shalimar Vale Traders", "Kirthar Agro Co", "Margalla Sweets", "Sutlej Bend Foods"]
    for i, name in enumerate(directory, start=1):
        conn.execute(
            "INSERT INTO companies VALUES (?,?,?,?,?,?,?)",
            (
                i, name, "Pakistan", rng.choice(PROVINCES), rng.randint(1978, 2021),
                "verified" if rng.random() < 0.75 else "unverified",
                1 if name in BUYERS else 0,
            ),
        )

    sellers: dict[str, list[str]] = {}
    for country, _w in COUNTRIES:
        stems = rng.sample(SELLER_STEMS, 3)
        sellers[country] = [f"{s} {SELLER_SUFFIX[country]}" for s in stems]

    buyer_pairs = list(zip(BUYERS, BUYER_WEIGHTS))
    sub_pairs = [(s, s[5]) for s in subcats]
    for n in range(1, N_TRANSACTIONS + 1):
        sub = _weighted(rng, sub_pairs)
        sub_id, s_hs, _s_name, (p_lo, p_hi), (m_lo, m_hi), _w, heading = sub
        if heading in ("1701", "1703"):
            country = _weighted(rng, BULK_ORIGINS)
        else:
            country = _weighted(rng, COUNTRIES)
        usd_per_mt = round(rng.uniform(p_lo, p_hi), 2)
        # Log-uniform shipment size: many small lots, a few large ones.
        qty_mt = round(m_lo * (m_hi / m_lo) ** rng.random(), 3)
        qty_kg = round(qty_mt * 1000, 2)
        usd = round(usd_per_mt * qty_mt, 2)
        pkr = round(usd * PKR_PER_USD * rng.uniform(0.98, 1.02), 0)
        day = WINDOW_START + timedelta(days=rng.randrange(WINDOW_DAYS))
        conn.execute(
            "INSERT INTO trade_transactions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                n, f"SYN-{n:05d}", "import", country, s_hs,
                _weighted(rng, buyer_pairs), rng.choice(sellers[country]), rng.choice(AGENTS),
                rng.choice(sub_items[sub_id]), usd, pkr, qty_kg, qty_mt, "MT",
                round(usd_per_mt / 1000, 5), usd_per_mt, day.isoformat(),
            ),
        )
    conn.commit()

    counts = {}
    for t in (
        "products",
        "product_categories",
        "product_subcategories",
        "product_items",
        "companies",
        "trade_transactions",
    ):
        counts[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    conn.close()
    return counts


if __name__ == "__main__":
    counts = build()
    print(f"Built {DB_PATH}")
    for k, v in counts.items():
        print(f"  {k}: {v}")

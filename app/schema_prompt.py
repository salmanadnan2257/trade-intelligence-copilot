"""Schema description and dataset scope used by the text-to-SQL step.

Keeping the scope facts here (all rows are imports, HS chapter 17, a fixed date
window) lets the planner and answerer reason about what is and is not in the data,
which is what makes honest refusal possible.
"""

SCHEMA_DDL = """
-- SQLite. Query with SELECT only.

products(id, hs_code, name)                       -- top HS heading, e.g. 1701 'Cane or beet sugar'
product_categories(id, hs_code, name, product_id) -- e.g. 1702.30 'Glucose and glucose syrup'
product_subcategories(id, hs_code, name, category_id) -- e.g. 1702.3020 'Dextrose anhydrous'
product_items(id, name, subcategory_id)           -- free-text product description on a shipment

companies(id, name, country, province, year_established, verification_status, has_trade_data)

trade_transactions(
    id, tx_reference,
    trade_type,        -- 'import' or 'export'
    country,           -- counterparty (origin) country of the shipment
    hs_code,           -- HS code string, e.g. '1702.1100'
    buyer,             -- importing company (free text)
    seller,            -- foreign supplier (free text)
    shipping_agent,
    product_item_id,   -- FK -> product_items.id
    usd, pkr,          -- shipment value
    qty_kg, qty_mt,    -- quantity
    std_unit,
    usd_per_kg, usd_per_mt,
    reporting_date     -- 'YYYY-MM-DD'
)
"""

# Ground-truth scope facts, computed directly from the database (synthetic demo data).
DATASET_SCOPE = """
DATASET SCOPE (important for deciding if a question is answerable):
- trade_transactions contains 2000 rows. Every row is an IMPORT (trade_type = 'import').
  There are NO export rows in this dataset.
- All shipments are in HS chapter 17 (sugar and sugar products: headings 1701-1704).
  There is no data for other commodities (no rice, wheat, cotton, machinery, etc.).
- reporting_date ranges from 2025-01-01 to 2025-12-31 only. No other years are covered.
- 'country' is the counterparty/origin country of an import. 'buyer' is the Pakistani
  importer; 'seller' is the foreign supplier.
- companies has 28 rows and is a directory; it is not fully linked to every transaction.

If a question asks about data outside this scope (exports, non-sugar commodities,
other years, other countries' domestic markets), it CANNOT be answered from the data.
"""

TEXT_TO_SQL_RULES = """
Rules for the SQL you produce:
- SQLite dialect. A SINGLE read-only SELECT (or WITH ... SELECT). No INSERT/UPDATE/
  DELETE/DDL, no semicolons except one optional trailing one, no comments.
- Prefer SUM(usd) for trade value, SUM(qty_kg)/SUM(qty_mt) for quantity, COUNT(*) for
  shipment counts. Use ROUND for money.
- Match names or free text with LIKE wrapped in % on both sides, for example
  buyer LIKE '%Tamrin%' or country LIKE '%South Africa%'. Never match a name with =.
- HS codes are stored as strings; use LIKE '1702%' for a whole heading.
- Always add a LIMIT (<= 100) unless the query is a single aggregate.
- If the question cannot be answered from these tables, return sql = null.
"""

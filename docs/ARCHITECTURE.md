# Architecture

## Overview

The copilot answers natural-language questions about Pakistani sugar-sector import
data and trade-regulation concepts. A question flows through a planner, one to three
tools, and a grounded-answer synthesizer. Every answer is either backed by cited
evidence or refused.

```
question
   |
   v
[planner]  gemini-2.5-flash -> {sql_needed, retrieval_needed, hs_lookup, in_scope}
   |
   +--> [text-to-SQL tool]   gemini-2.5-flash -> SELECT -> SQL guard -> read-only SQLite -> rows
   +--> [retrieval tool]     embed query (text-multilingual-embedding-002) -> cosine search -> chunks
   +--> [hs_lookup tool]     SQLite lookup over the HS-code hierarchy
   |
   v
[deterministic refusal check] -- no evidence? -> refuse
   |
   v
[answer synthesizer]  gemini-2.5-flash -> grounded answer + citations, or refusal
   |
   v
{answer, supported, citations, trace, cost, latency}
```

## Components

- `app/planner.py` decides which tools to run. It is told the dataset scope so it
  can flag clearly out-of-scope questions.
- `app/tools.py` holds the three tools:
  - text-to-SQL: an LLM writes one SELECT against a schema prompt; the query passes
    through `app/db.py`'s guard and runs on a read-only, immutable SQLite connection.
  - retrieval: the query is embedded and matched against a normalized vector matrix
    (cosine via dot product). Chunks below a score threshold are dropped.
  - hs_lookup: a direct SQL join over products / categories / subcategories.
- `app/answer.py` synthesizes the final answer from only the gathered evidence, or
  refuses. A deterministic guard refuses before the LLM call when no evidence exists.
- `app/agent.py` orchestrates the steps and assembles the trace and cost meter.
- `app/server.py` exposes `POST /ask`, `GET /health`, and the single-page UI.

## Data

- `data/build_db.py` generates `data/trade.sqlite`, a synthetic demo database: companies,
  an HS-code hierarchy (product -> category -> subcategory -> item), and 2,000 made-up
  import transactions for calendar 2025. It is seeded, so every build is byte-identical.
  Only the HS chapter 17 headings and country names are real; every company, price and
  quantity is invented.
- `data/regulations/*.md` are 16 authored explanatory summaries (HS basics, SPS,
  customs duty, import documents, policy). `ingest/ingest.py` splits them on section
  headers, embeds each section, and writes `data/vectors.npz` + `data/chunks.json`.

## Why SQLite + a hand-built vector store

pgvector was the first choice, but the target machine has PostgreSQL 18 without the
`vector` extension and no way to install it (no Docker, no superuser build step). The
regulation corpus is tens of chunks, so an exact brute-force cosine search over a
normalized numpy matrix is instant and removes a moving part. SQLite ships as a small
prebuilt demo database, so the project runs without any database server.

## Safety boundary

LLM-generated SQL is the main risk surface. It is contained three ways: the guard
rejects anything that is not a single SELECT (blocking write/DDL keywords, multiple
statements, comments, PRAGMA); the connection is opened `mode=ro&immutable=1` with
`PRAGMA query_only`; and results are row-capped. The refusal path is enforced both by
a deterministic no-evidence check and by the synthesizer's instructions.

## Cost accounting

`app/llm.py` records prompt and output tokens for every generation call on a per-query
`CostMeter` and prices them with the Vertex list rates in `app/config.py`. The API
returns the real token count and USD cost for each request.

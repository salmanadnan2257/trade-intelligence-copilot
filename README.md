# Trade Intelligence Copilot

An agentic retrieval-augmented-generation (RAG) copilot over agricultural-trade data,
with a real, published evaluation layer. Ask a plain-English question about Pakistani
sugar-sector import data or trade-regulation concepts and get a grounded, cited answer,
or an honest refusal when the data does not support one.

The headline is the evaluation. This project does not just claim it works; it ships a
28-item test set with ground-truth gold answers, runs the whole system against the live
Gemini API, and publishes the real metrics below. The harness also checks the LLM judge
against hand labels so the judge itself is validated; those labels are pending for the
current run (see below).

## Why it exists

Most RAG demos answer confidently and are never measured. Two things make a trade
copilot trustworthy: it must ground numeric answers in the actual database rather than
guessing, and it must refuse when the data cannot support an answer. This project is
built to measure exactly those properties end to end.

## What it does

- **Agentic planning.** An LLM planner (gemini-2.5-flash) reads a question and decides
  which tools to run: text-to-SQL over trade data, vector retrieval over a regulations
  corpus, an HS-code lookup, or several of them in one query.
- **Text-to-SQL with a safety boundary.** The SQL is generated against a strict schema
  prompt, passed through a read-only guard (single SELECT only, write/DDL keywords
  blocked, multiple statements and comments rejected), and executed on an immutable
  read-only SQLite connection.
- **Grounded, cited answers.** The final answer is synthesized from only the gathered
  evidence and cites its sources: the SQL query and rows, the HS-lookup rows, and the
  regulation chunks used.
- **Refusal path.** When there is no supporting evidence, or the question is outside the
  dataset scope, the copilot answers "not supported by the available data." This path is
  enforced deterministically and by the synthesizer, and it is tested.
- **Service + UI.** A FastAPI app exposes `POST /ask`, `GET /health`, and a single-page
  UI that shows the answer, the full tool-call trace, the citations, and the token cost
  and latency of the query.

## Data

- **Trade data (synthetic).** `data/build_db.py` generates a compact relational SQLite
  database: companies, an HS-code hierarchy (product -> category -> subcategory -> item),
  and 2,000 import transactions with country, HS code, USD/PKR value, quantity, and
  date, covering Pakistani sugar-sector (HS chapter 17) imports for calendar 2025. Every
  importer, supplier, shipping agent, price and quantity is made up. Only the HS
  headings and country names are real. The generator is seeded, so it rebuilds the same
  file byte for byte, and `eval/build_testset.py` recomputes every gold answer from it.
- **Regulations corpus.** 16 authored explanatory summaries of agricultural-trade and
  Pakistan import/export concepts (HS-code basics, phytosanitary/SPS, customs duty
  concepts, import documents, policy, trade remedies). These are plain-language
  explainers written for this demo; they do not reproduce any specific statute, SRO
  number, or exact duty rate. See `data/regulations/_README.md`.

## Architecture

```
question -> planner -> [ text-to-SQL | retrieval | hs_lookup ] -> refusal check
         -> grounded-answer synthesizer -> {answer, citations, trace, cost, latency}
```

- Generation: gemini-2.5-flash (planning, text-to-SQL, synthesis); gemini-2.5-pro (the
  evaluation judge).
- Embeddings: text-multilingual-embedding-002 (768-dim), via the Vertex predict API.
- Vector store: a hand-built cosine-similarity store over a normalized numpy matrix.
  pgvector was preferred but the target machine has PostgreSQL 18 without the `vector`
  extension and no way to install it, and the corpus is small, so an exact brute-force
  search is instant and removes a moving part. See `docs/ARCHITECTURE.md`.

## Evaluation results

From one real run against the live Vertex AI API (`python -m eval.run_eval --run`) on
2026-10-14, over the 28-item `eval/testset.jsonl` and the synthetic database. Full
machine-readable output is in `eval/results.json`; methodology and limitations are in
`docs/EVALUATION.md`.

| Metric | Result |
| --- | --- |
| SQL answer correctness (numeric/substring vs ground truth) | **15 / 15 = 100%** |
| Retrieval hit rate (a gold-document chunk was retrieved) | **8 / 10 = 80%** |
| Refusal precision / recall on unanswerable questions | **1.00 / 1.00** |
| LLM-as-judge faithfulness (gemini-2.5-pro) | **21 / 22 = 95.5%** |
| Judge vs human agreement | **pending: the subset has not been labelled for this run** |
| Mean cost per query | **$0.00093** |
| Total cost of the published eval run | **$0.0260** |
| Latency p50 / p95 | **9.0 s / 27.0 s** (8.8 s / 12.5 s without the 3 rate-limited items) |

Reading the numbers honestly:

- **SQL correctness is 100% only after an earlier fix.** In the first version's first
  run the model matched a company name without wildcards and returned 0 shipments for an
  importer that had dozens. Tightening the schema prompt to require `LIKE '%...%'` for
  name matches fixed it. The gold numbers are computed directly from the database, so
  they are true ground truth.
- **The one faithfulness miss is a judgment call.** Asked about "HS code 1702.1100
  (lactose)", the model answered with the right count (227) and repeated the question's
  own "(lactose)" label. The evidence rows never say what 1702.1100 is, so the judge
  marked it unfaithful. A human might call that fair, since the label came from the
  user and is correct. That disagreement is exactly what the judge-vs-human check is
  for, and it is why that number has to be filled in before the 95.5% is leaned on.
- **Latency p95 is inflated by rate limiting, not by the system.** Three items (sql-10,
  sql-11, ret-07) waited out `429 RESOURCE_EXHAUSTED` responses from Vertex during the
  run; their latencies include that waiting. Without them p50 / p95 is 8.8 s / 12.5 s.
  Both figures are reported rather than quietly dropping the slow ones.
- **Refusal was perfect on this set** (all 6 unanswerable questions refused, no false
  refusals), but the set is small, so read it as "the path works," not "it never fails."

## Setup

Requires Python 3.11+ and a Google Cloud service-account key with Vertex AI access.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: set GOOGLE_APPLICATION_CREDENTIALS to your key path and GCP_PROJECT
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
export GCP_PROJECT=your-gcp-project-id
```

The key file must live outside the repo. The code reads only its path from the
environment; no key or project id is committed.

The demo SQLite database (`data/trade.sqlite`) and the cached regulation embeddings
(`data/vectors.npz`, `data/chunks.json`) are prebuilt and shipped, so the app runs
immediately. To rebuild them:

```bash
python -m data.build_db          # regenerate the synthetic trade.sqlite (seeded, identical every time)
python -m eval.build_testset     # recompute the eval gold answers from it
python -m ingest.ingest          # re-chunk + re-embed the regulations (needs the API)
```

## Usage

```bash
# Run the service + UI
uvicorn app.server:app --port 8077
# open http://localhost:8077

# Ask via the API
curl -s -X POST localhost:8077/ask -H 'Content-Type: application/json' \
  -d '{"question":"Which country was the largest source of imports by value?"}'
```

Real example answers from the published live run:

- *"What was the total USD value of all imports in the dataset?"* ->
  "The total USD value of all imports in the dataset was 274,995,234.91." (SQL:
  `SELECT SUM(usd) FROM trade_transactions`)
- *"What is a phytosanitary certificate and who issues it?"* -> a grounded definition
  citing `phytosanitary-sps-basics#s2`.
- *"What were Pakistan's wheat exports to India in 2019?"* -> "This question is not
  supported by the available data." (the dataset has no exports, no wheat, no 2019)

## Tests

```bash
pytest                 # offline unit tests (SQL guard, ingestion, vector store, API)
GOOGLE_APPLICATION_CREDENTIALS=... GCP_PROJECT=... pytest   # also runs live API tests
```

33 tests total: 29 run offline (SQL guard rejects writes, vector-store shape and search,
HS lookup, refusal logic, API `/health`, `/` and `/ask` with a stubbed agent). 4 are
marked `live` and hit the real API (planner routing, out-of-scope refusal, a grounded
`/ask`); they skip automatically when no credentials are present. All 33 pass with
credentials set; all 29 offline tests pass without.

## Evaluation harness

```bash
python -m eval.run_eval --run       # live: run the copilot + judge, save transcripts
python -m eval.run_eval --metrics   # offline: recompute metrics from saved transcripts
```

`--run` writes `eval/transcripts.json` (every answer, trace, cost, and judge verdict)
and `eval/results.json`. `--metrics` recomputes the numbers from the transcripts and the
hand labels in `eval/human_labels.json` without spending on the API.

## What is needed to run end to end

A paid Google Cloud project with Vertex AI enabled and access to gemini-2.5-flash,
gemini-2.5-pro, and text-multilingual-embedding-002 (the models this was built and
verified against). Without credentials, the offline test suite still passes and the
prebuilt database and embeddings are present, but `/ask` and the eval run need the API.
This project was verified end to end against a live project: ingestion, real cited
answers, the refusal path, the full eval run, the test suite, and the server were all
exercised for real.

## Challenges

- **pgvector was not available.** PostgreSQL 18 is installed but without the `vector`
  extension, and the environment has no Docker or superuser build path. Rather than fight
  the infrastructure, I built a small exact cosine-similarity store over a normalized
  numpy matrix. For a corpus of tens of chunks this is instant and removes a dependency;
  the trade-off (no ANN index, no server-side filtering) is documented and would matter
  only at a much larger scale.
- **The LLM judge was blind to some evidence.** The first eval run flagged HS-lookup
  answers as unfaithful because the judge only reconstructed evidence from the SQL and
  regulation citations, not the HS-lookup rows the answerer actually used. I added
  HS-lookup rows to the citations (and therefore to the judge's view and the UI). The
  remaining unfaithful flags in that run turned out to be genuine: the model padded
  enumerations with invented product names, which is a real finding, not a harness
  artifact.
- **Text-to-SQL matched names literally.** Asked how many shipments one importer
  brought in, the model generated a `buyer LIKE` match on the full company name with no
  wildcards and returned zero rows. Because the gold answer was computed from the
  database, the eval caught the wrong count immediately. The fix was a sharper
  instruction in the schema prompt (`LIKE '%...%'`, never `=`, for names).
- **Faithful is not the same as correct.** That zero-row answer was judged faithful (it
  correctly reported what its query returned) while being wrong. This forced a clear
  separation in the metrics between grounding (faithfulness) and ground-truth
  correctness (the SQL gold check), rather than trusting one number.
- **Moving to synthetic data.** The first version ran on a dataset that cannot be
  published, so it was replaced with a seeded synthetic generator and the whole
  evaluation was re-run against it. The gold answers were never typed by hand, which is
  what made the switch cheap: rebuild the database, regenerate the test set, re-run.
- **Making refusal reliable.** An earlier version could answer out-of-scope questions
  because retrieval returned weak chunks. Adding a score threshold plus a deterministic
  no-evidence refusal (before any synthesis call) made the refusal path both cheaper and
  dependable, which the unanswerable test items confirm.
- **Writing the exhaustive project documentation honestly.** The first documentation pass stalled and had to be restarted. Getting the diagrams to actually compile surfaced several genuine LaTeX bugs of its own: unescaped commas breaking a tree diagram's parser, a style name colliding with a built-in PGF key, and a sequence diagram too wide for the page, each one only visible after a real compile attempt failed.

## What I learned

- **Ground-truth gold answers computed from the database are what make an LLM eval
  trustworthy.** Every SQL gold number here comes from a direct query, so "correct" is
  not a judgment call. That single decision is what let me catch the wildcard bug.
- **An LLM judge must be validated, and must see exactly what the answerer saw.** The
  judge-vs-human agreement number is not decoration; it is what makes a faithfulness
  figure interpretable, and building it surfaced a real evidence-plumbing bug in the
  harness. On the first version it was 91.7% (11 of 12); it has to be measured again
  for the current run.
- **Enumeration is where grounded models leak.** In the first version, point-fact
  answers stayed grounded and "list everything under this heading" is where the model
  reached past its evidence. In the synthetic re-run the enumerations stayed inside the
  HS-lookup rows, most likely because the synthetic hierarchy is small enough for the
  lookup to return every item. A practical fix that does not depend on luck is to cap or
  template enumerations to the retrieved rows only.
- **A cheap deterministic guard beats an expensive model call for refusal.** Checking for
  the absence of any evidence before calling the synthesizer is faster, cheaper, and more
  reliable than hoping the model refuses on its own.
- **Read-only really needs layers.** The SQL safety here is a keyword guard, a
  single-statement check, and an immutable read-only connection with `query_only` set.
  Any one alone is bypassable; together they hold, and a test asserts a write fails.

## What I'd do differently

- **The retrieval hit rate (80%) is the weakest number and I would fix it first.** Two
  "what does HS heading X cover" questions were answered from the HS-lookup tool instead
  of retrieving the matching regulation document, so the gold document was never fetched.
  The planner should prefer retrieval for definitional questions, and the retrieval gold
  should probably include the HS-lookup path as an acceptable source.
- **Constrain enumerations to evidence.** In the first version the two genuine
  faithfulness misses were both the model listing product examples beyond the rows it
  was given. I would post-filter enumerated items against the actual evidence, or
  instruct the synthesizer to list at most the rows present and say "and others."
- **The test set is small (28 items) by design and by budget.** The metrics show the
  paths work but are not a large benchmark. With more budget I would expand each category,
  add near-miss numeric questions, and add adversarial refusal cases (plausible but
  out-of-scope) rather than only clearly out-of-scope ones.
- **The judge and the answerer are the same model family.** The human-label check
  partly covers this, but a second judge from a different family would be a stronger
  cross-check on faithfulness.
- **Latency is high (p50 ~9 s).** Each query is three or four sequential model calls.
  Planning and text-to-SQL could be merged into one call, and independent tools could run
  concurrently, to cut this materially.

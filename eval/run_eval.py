"""Evaluation harness. Runs the copilot over eval/testset.jsonl and reports real metrics.

Two phases so the paid API run happens once:
  python -m eval.run_eval --run       # live: calls the copilot + LLM judge, saves transcripts
  python -m eval.run_eval --metrics   # offline: recompute metrics from transcripts + human labels

Metrics:
  - SQL answer correctness (numeric within tolerance, or required substring present)
  - retrieval hit rate (a chunk from a gold doc was retrieved)
  - refusal precision / recall on the unanswerable set
  - LLM-as-judge faithfulness (gemini-2.5-pro), plus judge-vs-human agreement on a
    labelled subset so the judge itself is checked
  - per-query token cost and p50/p95 latency
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from app import agent, config, llm

HERE = Path(__file__).resolve().parent
TESTSET = HERE / "testset.jsonl"
TRANSCRIPTS = HERE / "transcripts.json"
HUMAN_LABELS = HERE / "human_labels.json"
RESULTS = HERE / "results.json"

JUDGE_SYSTEM = (
    "You are a strict evaluator of factual faithfulness. You judge whether an answer "
    "is fully supported by the evidence shown, with no invented facts or numbers."
)


def load_testset() -> list[dict]:
    return [json.loads(l) for l in TESTSET.read_text(encoding="utf-8").splitlines() if l.strip()]


def extract_numbers(text: str) -> list[float]:
    nums = []
    for m in re.findall(r"-?\d[\d,]*\.?\d*", text):
        try:
            nums.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return nums


def sql_correct(answer: str, gold: dict) -> bool:
    if "contains" in gold:
        return gold["contains"].lower() in answer.lower()
    target = gold["value"]
    rel_tol = gold.get("rel_tol", 0.0)
    for n in extract_numbers(answer):
        if rel_tol == 0.0:
            if abs(n - target) < 0.5:
                return True
        elif target != 0 and abs(n - target) / abs(target) <= rel_tol:
            return True
    return False


def retrieval_hit(trace: list[dict], gold_docs: list[str]) -> bool:
    for step in trace:
        if step["step"] == "retrieval":
            hit_ids = [h["id"] for h in step["detail"]["hits"]]
            docs = {hid.split("#")[0] for hid in hit_ids}
            return any(g in docs for g in gold_docs)
    return False


def judge_faithfulness(item_result: dict) -> dict:
    """Ask gemini-2.5-pro whether the answer is grounded in the gathered evidence."""
    cites = item_result["citations"]
    evidence_parts = []
    if cites.get("sql"):
        evidence_parts.append("SQL: " + str(cites["sql"]))
    if cites.get("sql_rows"):
        evidence_parts.append("SQL ROWS: " + json.dumps(cites["sql_rows"], default=str)[:2000])
    if cites.get("hs_matches"):
        evidence_parts.append("HS LOOKUP: " + json.dumps(cites["hs_matches"], default=str)[:2500])
    for c in cites.get("chunks", []):
        evidence_parts.append(f"[{c['id']}] {c['text']}")
    evidence = "\n".join(evidence_parts) or "(no evidence used)"
    prompt = f"""Question: {item_result['question']}

Answer given: {item_result['answer']}

Evidence the answer was allowed to use:
{evidence}

Is every factual claim (numbers, countries, rules) in the answer directly supported
by the evidence above, with nothing invented? A refusal that correctly declines to
answer counts as faithful. Return ONLY JSON:
{{"faithful": true/false, "reason": "<short>"}}"""
    meter = llm.CostMeter()
    out = llm.generate_json(prompt, model=config.MODEL_PRO, system=JUDGE_SYSTEM, meter=meter)
    return {
        "faithful": bool(out.get("faithful")),
        "reason": out.get("reason", ""),
        "judge_cost_usd": meter.cost_usd,
    }


def run_live() -> None:
    if not config.credentials_present():
        raise SystemExit("GOOGLE_APPLICATION_CREDENTIALS not set; cannot run live eval.")
    items = load_testset()
    transcripts = []
    for i, item in enumerate(items, 1):
        print(f"[{i}/{len(items)}] {item['id']}: {item['question'][:60]}...")
        result = agent.ask(item["question"])
        judge = judge_faithfulness(result)
        transcripts.append({"item": item, "result": result, "judge": judge})
    TRANSCRIPTS.write_text(json.dumps(transcripts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(transcripts)} transcripts -> {TRANSCRIPTS}")
    compute_metrics()


def compute_metrics() -> dict:
    transcripts = json.loads(TRANSCRIPTS.read_text(encoding="utf-8"))
    human = {}
    if HUMAN_LABELS.is_file():
        human = json.loads(HUMAN_LABELS.read_text(encoding="utf-8"))

    sql_total = sql_ok = 0
    ret_total = ret_ok = 0
    faithful_total = faithful_ok = 0
    agree_total = agree_ok = 0
    latencies = []
    costs = []
    per_item = []

    # refusal confusion matrix over the whole set
    tp = fp = tn = fn = 0

    for t in transcripts:
        item, result, judge = t["item"], t["result"], t["judge"]
        refused = not result["supported"]
        should_refuse = item.get("expect_refusal", False)
        latencies.append(result["latency_ms"])
        costs.append(result["cost"]["cost_usd"])

        if should_refuse and refused:
            tp += 1
        elif should_refuse and not refused:
            fn += 1
        elif not should_refuse and refused:
            fp += 1
        else:
            tn += 1

        row = {"id": item["id"], "type": item["type"], "refused": refused,
               "supported": result["supported"], "faithful": judge["faithful"]}

        if item["type"] in ("sql", "multi") and "gold" in item:
            sql_total += 1
            ok = sql_correct(result["answer"], item["gold"])
            sql_ok += int(ok)
            row["sql_correct"] = ok
        if item["type"] in ("retrieval", "multi") and "gold_docs" in item:
            ret_total += 1
            ok = retrieval_hit(result["trace"], item["gold_docs"])
            ret_ok += int(ok)
            row["retrieval_hit"] = ok
        # faithfulness is judged on answered (non-refusal) items
        if not should_refuse and result["supported"]:
            faithful_total += 1
            faithful_ok += int(judge["faithful"])
        # judge-vs-human agreement on labelled subset
        if item["id"] in human:
            agree_total += 1
            agree_ok += int(bool(human[item["id"]]) == bool(judge["faithful"]))
        per_item.append(row)

    refusal_precision = tp / (tp + fp) if (tp + fp) else None
    refusal_recall = tp / (tp + fn) if (tp + fn) else None

    metrics = {
        "n_items": len(transcripts),
        "sql_answer_correctness": {
            "correct": sql_ok, "total": sql_total,
            "accuracy": round(sql_ok / sql_total, 3) if sql_total else None,
        },
        "retrieval_hit_rate": {
            "hits": ret_ok, "total": ret_total,
            "rate": round(ret_ok / ret_total, 3) if ret_total else None,
        },
        "refusal": {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(refusal_precision, 3) if refusal_precision is not None else None,
            "recall": round(refusal_recall, 3) if refusal_recall is not None else None,
        },
        "faithfulness_llm_judge": {
            "faithful": faithful_ok, "total": faithful_total,
            "rate": round(faithful_ok / faithful_total, 3) if faithful_total else None,
            "judge_model": config.MODEL_PRO,
        },
        "judge_vs_human_agreement": {
            "agree": agree_ok, "labelled": agree_total,
            "agreement": round(agree_ok / agree_total, 3) if agree_total else None,
        },
        "cost": {
            "total_usd": round(sum(costs), 6),
            "mean_per_query_usd": round(statistics.mean(costs), 6) if costs else None,
        },
        "latency_ms": {
            "p50": round(statistics.median(latencies), 1) if latencies else None,
            "p95": round(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 1) if latencies else None,
            "mean": round(statistics.mean(latencies), 1) if latencies else None,
        },
        "per_item": per_item,
    }
    RESULTS.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_item"}, indent=2))
    print(f"\nWrote {RESULTS}")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="live run against the API, then compute metrics")
    ap.add_argument("--metrics", action="store_true", help="recompute metrics from saved transcripts")
    args = ap.parse_args()
    if args.run:
        run_live()
    else:
        compute_metrics()

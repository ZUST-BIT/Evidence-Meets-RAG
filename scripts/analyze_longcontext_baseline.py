"""Analyze the long-context (top-k=100) baseline against the main top-8 runs.

For Document Corpus and Evidence Graph under the Vanilla RAG path, compares
k=100 (~50k-token contexts) against the main k=8 runs on the same 571 records:
overall Accuracy / Macro-F1, paired record-level deltas with bootstrap CIs,
and the Evidence-vs-Document contrast at k=100.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag_eval_lib import gold_label, macro_f1_from_labels  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dataset" / "rag" / "analysis" / "ecorag_longcontext_baseline"
RUNS = {
    "doc_k8": ROOT / "dataset/rag/results/multi_cancer__raw_observation_view__naive_rag__formalv3_571_classic4_raw_vs_graph.jsonl",
    "doc_k100": ROOT / "dataset/rag/results/multi_cancer__raw_observation_view__naive_rag__longcontext_k100.jsonl",
    "eg_k8": ROOT / "dataset/rag/results/multi_cancer__evidence_graph_complete__naive_rag__formalv3_571_classic4_raw_vs_graph.jsonl",
    "eg_k100": ROOT / "dataset/rag/results/multi_cancer__evidence_graph_complete__naive_rag__longcontext_k100.jsonl",
}
ITER = 10_000
SEED = 20260820


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load(path: Path) -> dict[str, dict[str, str]]:
    out = {}
    for row in read_jsonl(path):
        rid = str(row["record"]["id"])
        out[rid] = {
            "gold": gold_label(row["record"]),
            "pred": str(row.get("normalized_prediction") or ""),
        }
    return out


def metrics(rows: dict[str, dict[str, str]], ids: list[str]) -> dict[str, float]:
    gold = [rows[i]["gold"] for i in ids]
    pred = [rows[i]["pred"] for i in ids]
    acc = sum(g == p for g, p in zip(gold, pred)) / len(ids)
    return {"accuracy": round(acc, 4), "macro_f1": round(macro_f1_from_labels(gold, pred), 4)}


def paired_delta(a: dict, b: dict, ids: list[str]) -> dict[str, Any]:
    diffs = [int(a[i]["gold"] == a[i]["pred"]) - int(b[i]["gold"] == b[i]["pred"]) for i in ids]
    point = sum(diffs) / len(ids)
    rng = random.Random(SEED)
    boots = []
    for _ in range(ITER):
        sample = [diffs[rng.randrange(len(diffs))] for _ in range(len(diffs))]
        boots.append(sum(sample) / len(sample))
    boots.sort()
    lo = boots[int(0.025 * (ITER - 1))]
    hi = boots[int(0.975 * (ITER - 1))]
    return {"point": round(point, 4), "ci95": [round(lo, 4), round(hi, 4)]}


def main() -> None:
    data = {key: load(path) for key, path in RUNS.items()}
    ids = sorted(set.intersection(*(set(v) for v in data.values())))
    print(f"aligned records: {len(ids)}")
    report: dict[str, Any] = {
        "design": {
            "condition": "top-k=100 (~50k-token contexts) vs main top-k=8, Vanilla RAG path, same generator/prompts/normalization",
            "aligned_records": len(ids),
        },
        "overall": {key: metrics(rows, ids) for key, rows in data.items()},
        "paired": {
            "doc_k100_minus_doc_k8": paired_delta(data["doc_k100"], data["doc_k8"], ids),
            "eg_k100_minus_eg_k8": paired_delta(data["eg_k100"], data["eg_k8"], ids),
            "eg_k100_minus_doc_k100": paired_delta(data["eg_k100"], data["doc_k100"], ids),
            "eg_k8_minus_doc_k100": paired_delta(data["eg_k8"], data["doc_k100"], ids),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "longcontext_baseline_summary.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()

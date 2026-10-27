from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

from rag_eval_lib import gold_label, normalize_prediction


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def record_id(row: dict[str, Any]) -> str:
    return str((row.get("record") or {}).get("id") or "")


def gold_answer(record: dict[str, Any]) -> str:
    return gold_label(record)


def predicted_answer(row: dict[str, Any]) -> str:
    value = row.get("normalized_prediction")
    if value not in (None, ""):
        return str(value)
    return normalize_prediction(str(row.get("prediction") or ""), row.get("record") or {})


def align_rows(path_a: Path, path_b: Path, task_type: str | None = None) -> list[dict[str, Any]]:
    rows_a = {record_id(row): row for row in read_jsonl(path_a)}
    rows_b = {record_id(row): row for row in read_jsonl(path_b)}
    common_ids = sorted(set(rows_a).intersection(rows_b))
    aligned: list[dict[str, Any]] = []
    for rid in common_ids:
        row_a = rows_a[rid]
        row_b = rows_b[rid]
        record = row_a.get("record") or row_b.get("record") or {}
        if task_type and str(record.get("task_type") or "") != task_type:
            continue
        aligned.append(
            {
                "id": rid,
                "record": record,
                "gold": gold_answer(record),
                "pred_a": predicted_answer(row_a),
                "pred_b": predicted_answer(row_b),
            }
        )
    return aligned


def accuracy(items: list[dict[str, Any]], key: str) -> float:
    if not items:
        return 0.0
    correct = sum(1 for item in items if item[key] == item["gold"])
    return correct / len(items)


def macro_f1(items: list[dict[str, Any]], key: str) -> float:
    labels = sorted({item["gold"] for item in items} | {item[key] for item in items})
    if not labels:
        return 0.0
    f1_scores = []
    for label in labels:
        tp = sum(1 for item in items if item["gold"] == label and item[key] == label)
        fp = sum(1 for item in items if item["gold"] != label and item[key] == label)
        fn = sum(1 for item in items if item["gold"] == label and item[key] != label)
        if tp == 0 and fp == 0 and fn == 0:
            f1_scores.append(0.0)
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
    return sum(f1_scores) / len(f1_scores)


def exact_mcnemar_pvalue(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def bootstrap_macro_f1(
    items: list[dict[str, Any]],
    *,
    key_a: str,
    key_b: str,
    iters: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    if not items:
        return {"ci95": [0.0, 0.0], "p_value": 1.0, "delta_samples_mean": 0.0}
    deltas = []
    for _ in range(iters):
        sample = [items[rng.randrange(len(items))] for _ in range(len(items))]
        deltas.append(macro_f1(sample, key_a) - macro_f1(sample, key_b))
    deltas.sort()
    lower = deltas[int(0.025 * (len(deltas) - 1))]
    upper = deltas[int(0.975 * (len(deltas) - 1))]
    prob_non_positive = sum(1 for value in deltas if value <= 0) / len(deltas)
    prob_non_negative = sum(1 for value in deltas if value >= 0) / len(deltas)
    p_value = min(1.0, 2 * min(prob_non_positive, prob_non_negative))
    return {
        "ci95": [round(lower, 4), round(upper, 4)],
        "p_value": round(p_value, 6),
        "delta_samples_mean": round(sum(deltas) / len(deltas), 4),
    }


def compare_results(
    *,
    path_a: Path,
    path_b: Path,
    label_a: str,
    label_b: str,
    task_type: str | None,
    bootstrap_iters: int,
    seed: int,
) -> dict[str, Any]:
    items = align_rows(path_a, path_b, task_type=task_type)
    acc_a = accuracy(items, "pred_a")
    acc_b = accuracy(items, "pred_b")
    f1_a = macro_f1(items, "pred_a")
    f1_b = macro_f1(items, "pred_b")

    wins = sum(1 for item in items if item["pred_a"] == item["gold"] and item["pred_b"] != item["gold"])
    losses = sum(1 for item in items if item["pred_a"] != item["gold"] and item["pred_b"] == item["gold"])
    ties = len(items) - wins - losses

    by_task = Counter(str(item["record"].get("task_type") or "") for item in items)
    bootstrap = bootstrap_macro_f1(
        items,
        key_a="pred_a",
        key_b="pred_b",
        iters=bootstrap_iters,
        seed=seed,
    )
    return {
        "comparison": {
            "label_a": label_a,
            "label_b": label_b,
            "result_a": str(path_a.relative_to(ROOT) if path_a.is_absolute() else path_a),
            "result_b": str(path_b.relative_to(ROOT) if path_b.is_absolute() else path_b),
            "task_type": task_type,
            "sample_count": len(items),
        },
        "accuracy": {
            "a": round(acc_a, 4),
            "b": round(acc_b, 4),
            "delta": round(acc_a - acc_b, 4),
            "p_value": exact_mcnemar_pvalue(wins, losses),
            "p_value_unrounded": exact_mcnemar_pvalue(wins, losses),
            "wins": wins,
            "ties": ties,
            "losses": losses,
        },
        "macro_f1": {
            "a": round(f1_a, 4),
            "b": round(f1_b, 4),
            "delta": round(f1_a - f1_b, 4),
            "ci95": bootstrap["ci95"],
            "p_value": bootstrap["p_value"],
        },
        "by_task_type_counts": dict(sorted(by_task.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired significance analysis between two RAG result files.")
    parser.add_argument("--result-a", type=Path, required=True)
    parser.add_argument("--result-b", type=Path, required=True)
    parser.add_argument("--label-a", required=True)
    parser.add_argument("--label-b", required=True)
    parser.add_argument("--task-type", default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--bootstrap-iters", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = compare_results(
        path_a=(args.result_a if args.result_a.is_absolute() else ROOT / args.result_a),
        path_b=(args.result_b if args.result_b.is_absolute() else ROOT / args.result_b),
        label_a=args.label_a,
        label_b=args.label_b,
        task_type=args.task_type,
        bootstrap_iters=args.bootstrap_iters,
        seed=args.seed,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    output_path = args.output_json if args.output_json.is_absolute() else ROOT / args.output_json
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()

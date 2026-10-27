from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from run_significance_analysis import accuracy, align_rows, exact_mcnemar_pvalue


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "dataset" / "rag" / "analysis" / "ecorag_core_significance"

RESULTS = {
    "vanilla": {
        "label": "Vanilla RAG",
        "evidence": ROOT / "dataset/rag/results/fixed_context_component_ablation_571_vanilla__full.jsonl",
        "doc": ROOT / "dataset/rag/results/multi_cancer__raw_observation_view__naive_rag__formalv3_571_classic4_raw_vs_graph.jsonl",
        "semantic": ROOT / "dataset/rag/results/multi_cancer__semantic_kg__naive_rag__formalv3_571_classic4_semantic_kg.jsonl",
    },
    "graph": {
        "label": "Graph RAG",
        "evidence": ROOT / "dataset/rag/results/unified_571/stage/R1C1G1.jsonl",
        "doc": ROOT / "dataset/rag/results/multi_cancer__raw_observation_view__graph_rag__formalv3_571_classic4_raw_vs_graph.jsonl",
        "semantic": ROOT / "dataset/rag/results/multi_cancer__semantic_kg__graph_rag__formalv3_571_classic4_semantic_kg.jsonl",
    },
}

BASELINES = {"doc": "Document Corpus", "semantic": "Semantic KG"}
CORE_TASKS = [
    "cross_source_corroboration_status",
    "cross_source_support_bundle",
    "evidence_signal_interpretation",
    "hierarchical_evidence_aggregation",
    "mechanism_trace_intermediate_gene",
    "precision_oncology_evidence",
]


def fast_macro_f1(items: list[dict[str, Any]], key: str) -> float:
    gold_counts: Counter[str] = Counter()
    pred_counts: Counter[str] = Counter()
    true_positive: Counter[str] = Counter()
    for item in items:
        gold = item["gold"]
        pred = item[key]
        gold_counts[gold] += 1
        pred_counts[pred] += 1
        if gold == pred:
            true_positive[gold] += 1
    labels = gold_counts.keys() | pred_counts.keys()
    scores = []
    for label in labels:
        tp = true_positive[label]
        fp = pred_counts[label] - tp
        fn = gold_counts[label] - tp
        scores.append((2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def bootstrap_deltas(items: list[dict[str, Any]], iterations: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    acc_samples = []
    f1_samples = []
    n = len(items)
    for _ in range(iterations):
        sample = [items[rng.randrange(n)] for _ in range(n)]
        acc_samples.append(accuracy(sample, "pred_a") - accuracy(sample, "pred_b"))
        f1_samples.append(fast_macro_f1(sample, "pred_a") - fast_macro_f1(sample, "pred_b"))
    acc_samples.sort()
    f1_samples.sort()

    def summarize(samples: list[float]) -> dict[str, Any]:
        lower = samples[int(iterations * 0.025)]
        upper = samples[int(iterations * 0.975) - 1]
        p_non_positive = sum(value <= 0 for value in samples) / iterations
        p_non_negative = sum(value >= 0 for value in samples) / iterations
        return {
            "ci95": [round(lower, 4), round(upper, 4)],
            "p_value": round(min(1.0, 2 * min(p_non_positive, p_non_negative)), 6),
        }

    return {"accuracy": summarize(acc_samples), "macro_f1": summarize(f1_samples)}


def compare(path_a: Path, path_b: Path, task: str | None, iterations: int, seed: int) -> dict[str, Any]:
    items = align_rows(path_a, path_b, task_type=task)
    acc_a = accuracy(items, "pred_a")
    acc_b = accuracy(items, "pred_b")
    f1_a = fast_macro_f1(items, "pred_a")
    f1_b = fast_macro_f1(items, "pred_b")
    wins = sum(item["pred_a"] == item["gold"] and item["pred_b"] != item["gold"] for item in items)
    losses = sum(item["pred_a"] != item["gold"] and item["pred_b"] == item["gold"] for item in items)
    bootstrap = bootstrap_deltas(items, iterations, seed)
    return {
        "task_type": task or "all",
        "count": len(items),
        "evidence_accuracy": round(acc_a, 4),
        "baseline_accuracy": round(acc_b, 4),
        "accuracy_delta": round(acc_a - acc_b, 4),
        "accuracy_ci95": bootstrap["accuracy"]["ci95"],
        "mcnemar_p": exact_mcnemar_pvalue(wins, losses),
        "mcnemar_p_unrounded": exact_mcnemar_pvalue(wins, losses),
        "evidence_macro_f1": round(f1_a, 4),
        "baseline_macro_f1": round(f1_b, 4),
        "macro_f1_delta": round(f1_a - f1_b, 4),
        "macro_f1_ci95": bootstrap["macro_f1"]["ci95"],
        "macro_f1_bootstrap_p": bootstrap["macro_f1"]["p_value"],
        "wins": wins,
        "ties": len(items) - wins - losses,
        "losses": losses,
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [1.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (total - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def apply_holm(rows: list[dict[str, Any]]) -> None:
    acc_adjusted = holm_adjust([float(row["mcnemar_p"]) for row in rows])
    f1_adjusted = holm_adjust([float(row["macro_f1_bootstrap_p"]) for row in rows])
    for row, acc_p, f1_p in zip(rows, acc_adjusted, f1_adjusted):
        row["mcnemar_holm_p"] = acc_p
        row["macro_f1_holm_p"] = f1_p
        row["accuracy_significant_positive"] = bool(row["accuracy_delta"] > 0 and acc_p < 0.05)
        row["macro_f1_significant_positive"] = bool(row["macro_f1_delta"] > 0 and f1_p < 0.05)


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "framework", "baseline", "task_type", "count", "evidence_accuracy", "baseline_accuracy",
        "accuracy_delta", "accuracy_ci95", "mcnemar_p", "mcnemar_holm_p", "wins", "ties", "losses",
        "evidence_macro_f1", "baseline_macro_f1", "macro_f1_delta", "macro_f1_ci95",
        "macro_f1_bootstrap_p", "macro_f1_holm_p",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def format_ci(value: list[float]) -> str:
    return f"[{value[0]:+.4f}, {value[1]:+.4f}]"


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Evidence Meets RAG: Core Paired Significance Analysis",
        "",
        "## Design",
        "",
        "We compare Evidence KG with Document Corpus and Semantic KG on the same 571 records under Vanilla RAG and Graph RAG. Accuracy uncertainty uses 10,000 paired bootstrap samples and McNemar's exact test; Macro-F1 uses 10,000 paired bootstrap samples. Six major task types (n=100, 81, 85, 86, 100, and 100 in the full benchmark) are tested separately, with Holm correction within each framework-baseline family.",
        "",
        "## Overall Results",
        "",
        "| Framework | Baseline | Acc Evidence/Base | Acc Delta (95% CI) | McNemar p | W/T/L | F1 Evidence/Base | F1 Delta (95% CI) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["overall"]:
        lines.append(
            f"| {row['framework']} | {row['baseline']} | {row['evidence_accuracy']:.4f}/{row['baseline_accuracy']:.4f} | "
            f"{row['accuracy_delta']:+.4f} {format_ci(row['accuracy_ci95'])} | {row['mcnemar_p']:.6f} | "
            f"{row['wins']}/{row['ties']}/{row['losses']} | {row['evidence_macro_f1']:.4f}/{row['baseline_macro_f1']:.4f} | "
            f"{row['macro_f1_delta']:+.4f} {format_ci(row['macro_f1_ci95'])} |"
        )
    lines.extend([
        "",
        "## Holm-corrected Task Results",
        "",
        "| Framework | Baseline | Task | N | Acc Delta | Holm p | F1 Delta | F1 Holm p | Conclusion |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in payload["tasks"]:
        if row["accuracy_significant_positive"] or row["macro_f1_significant_positive"]:
            conclusion = "positive"
        elif row["accuracy_delta"] < 0 and row["mcnemar_holm_p"] < 0.05:
            conclusion = "negative"
        else:
            conclusion = "not significant"
        lines.append(
            f"| {row['framework']} | {row['baseline']} | {row['task_type']} | {row['count']} | "
            f"{row['accuracy_delta']:+.4f} | {row['mcnemar_holm_p']:.6f} | {row['macro_f1_delta']:+.4f} | "
            f"{row['macro_f1_holm_p']:.6f} | {conclusion} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
    ])
    for row in payload["overall"]:
        def interval_description(ci):
            return "lies above zero" if ci[0] > 0 else "lies below zero" if ci[1] < 0 else "includes zero"
        lines.append(f"{row['framework']} versus {row['baseline']}: the Accuracy interval {interval_description(row['accuracy_ci95'])}; "
                     f"the Macro-F1 interval {interval_description(row['macro_f1_ci95'])}. "
                     f"The two-sided exact McNemar p value is {row['mcnemar_p']:.6g}.")
    lines.extend(["", "Use the Holm-corrected task results to qualify task-specific claims; a positive point estimate alone does not establish a significant improvement.", ""])
    return "\n".join(lines)


def main() -> None:
    iterations = 10000
    overall = []
    tasks = []
    seed = 20260628
    for framework_key, config in RESULTS.items():
        for baseline_key, baseline_label in BASELINES.items():
            overall_row = compare(config["evidence"], config[baseline_key], None, iterations, seed)
            overall_row.update({"framework": config["label"], "baseline": baseline_label})
            overall.append(overall_row)
            family_rows = []
            for task in CORE_TASKS:
                seed += 1
                row = compare(config["evidence"], config[baseline_key], task, iterations, seed)
                row.update({"framework": config["label"], "baseline": baseline_label})
                family_rows.append(row)
            apply_holm(family_rows)
            tasks.extend(family_rows)
            seed += 1
    payload = {
        "design": {
            "record_count": 571,
            "frameworks": ["Vanilla RAG", "Graph RAG"],
            "baselines": list(BASELINES.values()),
            "bootstrap_iterations": iterations,
            "task_multiple_testing": "Holm correction over six core tasks within each framework-baseline family",
        },
        "overall": overall,
        "tasks": tasks,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "core_significance_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_tsv(OUTPUT_DIR / "core_significance_overall.tsv", overall)
    write_tsv(OUTPUT_DIR / "core_significance_tasks.tsv", tasks)
    report = render_report(payload)
    (OUTPUT_DIR / "core_significance_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "dataset" / "rag" / "results" / "model_robustness"
OUTPUT_DIR = ROOT / "dataset" / "rag" / "analysis" / "ecorag_model_robustness"
MODELS = {
    "gpt_5_mini_existing": "GPT-5 mini",
    "gemini_3_8_flash": "Gemini 3.8 Flash",
    "deepseek_v4_flash": "DeepSeek V4 Flash",
    "gpt_5_4_mini": "GPT-5.4 mini",
    "qwen3_5_flash": "Qwen 3.5 Flash",
}
SOURCES = {"doc_corpus": "Document Corpus", "semantic_kg": "Semantic KG", "evidence_kg": "Evidence KG"}
CORE_TASKS = [
    "cross_source_corroboration_status",
    "cross_source_support_bundle",
    "evidence_signal_interpretation",
    "hierarchical_evidence_aggregation",
    "mechanism_trace_intermediate_gene",
    "precision_oncology_evidence",
]
PANEL = ROOT / "dataset" / "evaluated" / "external_reasoning_benchmark_formal_v3_571.jsonl"
BASELINE_RESULTS = {
    "doc_corpus": ROOT / "dataset/rag/results/multi_cancer__raw_observation_view__naive_rag__formalv3_571_classic4_raw_vs_graph.jsonl",
    "semantic_kg": ROOT / "dataset/rag/results/multi_cancer__semantic_kg__naive_rag__formalv3_571_classic4_semantic_kg.jsonl",
    "evidence_kg": ROOT / "dataset/rag/results/multi_cancer__evidence_graph_complete__naive_rag__formalv3_571_classic4_raw_vs_graph.jsonl",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def macro_f1(rows: list[dict[str, Any]], prediction_key: str = "normalized_prediction") -> float:
    gold = Counter()
    pred = Counter()
    tp = Counter()
    for row in rows:
        g = str(row["gold_label"])
        p = str(row[prediction_key])
        gold[g] += 1
        pred[p] += 1
        if g == p:
            tp[g] += 1
    labels = gold.keys() | pred.keys()
    values = []
    for label in labels:
        denominator = 2 * tp[label] + pred[label] - tp[label] + gold[label] - tp[label]
        values.append(2 * tp[label] / denominator if denominator else 0.0)
    return sum(values) / len(values) if values else 0.0


def accuracy(rows: list[dict[str, Any]]) -> float:
    return sum(int(row["correct"]) for row in rows) / len(rows)


def exact_mcnemar(wins: int, losses: int) -> float:
    n = wins + losses
    if not n:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, index) for index in range(k + 1)) / (2**n))


def compare(evidence: list[dict[str, Any]], baseline: list[dict[str, Any]], seed: int, iterations: int = 10000) -> dict[str, Any]:
    base_by_id = {row["record_id"]: row for row in baseline}
    pairs = [(row, base_by_id[row["record_id"]]) for row in evidence]
    wins = sum(a["correct"] and not b["correct"] for a, b in pairs)
    losses = sum(not a["correct"] and b["correct"] for a, b in pairs)
    acc_delta = accuracy(evidence) - accuracy(baseline)
    f1_delta = macro_f1(evidence) - macro_f1(baseline)
    rng = random.Random(seed)
    acc_samples = []
    f1_samples = []
    for _ in range(iterations):
        indices = [rng.randrange(len(pairs)) for _ in pairs]
        sample_a = [pairs[index][0] for index in indices]
        sample_b = [pairs[index][1] for index in indices]
        acc_samples.append(accuracy(sample_a) - accuracy(sample_b))
        f1_samples.append(macro_f1(sample_a) - macro_f1(sample_b))
    acc_samples.sort()
    f1_samples.sort()
    low_index = min(iterations - 1, int(.025 * iterations))
    high_index = min(iterations - 1, max(0, int(.975 * iterations) - 1))
    return {
        "accuracy_delta": round(acc_delta, 4),
        "accuracy_ci95": [round(acc_samples[low_index], 4), round(acc_samples[high_index], 4)],
        "mcnemar_p": round(exact_mcnemar(wins, losses), 6),
        "macro_f1_delta": round(f1_delta, 4),
        "macro_f1_ci95": [round(f1_samples[low_index], 4), round(f1_samples[high_index], 4)],
        "wins": wins,
        "ties": len(pairs) - wins - losses,
        "losses": losses,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"count": len(rows), "accuracy": round(accuracy(rows), 4), "macro_f1": round(macro_f1(rows), 4)}


def load_baseline_rows() -> list[dict[str, Any]]:
    panel_ids = {str(row["id"]) for row in read_jsonl(PANEL)}
    output = []
    for source, path in BASELINE_RESULTS.items():
        for row in read_jsonl(path):
            rid = str(row["record"]["id"])
            if rid not in panel_ids:
                continue
            gold = str((row.get("record") or {}).get("answer") or "")
            from rag_eval_lib import gold_label
            normalized_gold = gold_label(row["record"])
            prediction = str(row.get("normalized_prediction") or "")
            output.append({
                "record_id": rid,
                "task_type": row["record"].get("task_type"),
                "knowledge_source": source,
                "model": "gpt-5-mini",
                "gold_label": normalized_gold,
                "normalized_prediction": prediction,
                "correct": int(prediction == normalized_gold),
            })
    return output


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Evidence Meets RAG: Cross-LLM Robustness",
        "",
        "All models answer the same balanced 145-record panel from identical precomputed Vanilla top-8 retrieval contexts. Only the generator changes.",
        "",
        "## Performance by Model and Knowledge Shape",
        "",
        "| Model | Document Acc/F1 | Semantic Acc/F1 | Evidence Acc/F1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for model_key, model in payload["models"].items():
        values = model["sources"]
        lines.append(
            f"| {model['label']} | {values['doc_corpus']['accuracy']:.4f}/{values['doc_corpus']['macro_f1']:.4f} | "
            f"{values['semantic_kg']['accuracy']:.4f}/{values['semantic_kg']['macro_f1']:.4f} | "
            f"{values['evidence_kg']['accuracy']:.4f}/{values['evidence_kg']['macro_f1']:.4f} |"
        )
    lines.extend([
        "",
        "## Paired Evidence KG Effects",
        "",
        "| Model | Baseline | Acc Delta (95% CI) | McNemar p | F1 Delta (95% CI) | W/T/L |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for model in payload["models"].values():
        for baseline in ["doc_corpus", "semantic_kg"]:
            effect = model["effects"][f"evidence_vs_{baseline}"]
            lines.append(
                f"| {model['label']} | {SOURCES[baseline]} | {effect['accuracy_delta']:+.4f} {effect['accuracy_ci95']} | "
                f"{effect['mcnemar_p']:.6f} | {effect['macro_f1_delta']:+.4f} {effect['macro_f1_ci95']} | "
                f"{effect['wins']}/{effect['ties']}/{effect['losses']} |"
            )
    lines.extend([
        "",
        "## Task-level Accuracy Delta (Evidence KG minus Baseline)",
        "",
        "| Model | Task | vs Document | vs Semantic KG | Evidence Accuracy |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for model in payload["models"].values():
        for task in CORE_TASKS:
            row = model["by_task_type"][task]
            lines.append(
                f"| {model['label']} | {task} | {row['evidence_minus_doc_accuracy']:+.4f} | "
                f"{row['evidence_minus_semantic_accuracy']:+.4f} | {row['evidence_accuracy']:.4f} |"
            )
    lines.extend(["", "## Interpretation", ""])
    positive_doc = sum(model["effects"]["evidence_vs_doc_corpus"]["accuracy_delta"] > 0 for model in payload["models"].values())
    positive_sem = sum(model["effects"]["evidence_vs_semantic_kg"]["accuracy_delta"] > 0 for model in payload["models"].values())
    lines.extend([
        f"Evidence KG has a positive Accuracy delta over Document Corpus in {positive_doc}/{len(payload['models'])} complete model settings and over Semantic KG in {positive_sem}/{len(payload['models'])}. The variation in effect size quantifies generator-dependent evidence utilization rather than a universally invariant answer-level gain.",
        "Task-level results should be used to identify recurring evidence-intensive gains and failure modes; claims should not require every task-model cell to be positive.",
        "Differences across generators quantify evidence-utilization robustness. They do not change the retrieval-side Component Recall and Complete Bundle results because every model receives the same retrieved contexts.",
        "All configured model families are evaluated with the same frozen retrieval contexts, prompts, decoding temperature, and answer normalization.",
        "",
    ])
    if payload["unavailable_models"]:
        lines.extend(["## Incomplete Runs", ""])
        for label, status in payload["unavailable_models"].items():
            lines.append(f"- {label}: {status}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    payload = {
        "design": {"record_count": 145, "retrieval": "fixed Vanilla top-8", "bootstrap_iterations": 10000},
        "models": {},
        "unavailable_models": {},
    }
    seed = 20260628
    for model_key, label in MODELS.items():
        path = INPUT_DIR / f"{model_key}.jsonl"
        rows = load_baseline_rows() if model_key == "gpt_5_mini_existing" else (read_jsonl(path) if path.exists() else [])
        if len(rows) != 435:
            payload["unavailable_models"][label] = f"{len(rows)}/435 rows completed; excluded from paired comparison"
            continue
        by_source = {source: [row for row in rows if row["knowledge_source"] == source] for source in SOURCES}
        model_payload = {
            "label": label,
            "sources": {source: summarize(source_rows) for source, source_rows in by_source.items()},
            "effects": {},
            "by_task_type": {},
        }
        for baseline in ["doc_corpus", "semantic_kg"]:
            model_payload["effects"][f"evidence_vs_{baseline}"] = compare(by_source["evidence_kg"], by_source[baseline], seed)
            seed += 1
        for task in CORE_TASKS:
            task_sources = {source: [row for row in source_rows if row["task_type"] == task] for source, source_rows in by_source.items()}
            evidence_acc = accuracy(task_sources["evidence_kg"])
            model_payload["by_task_type"][task] = {
                "count": len(task_sources["evidence_kg"]),
                "evidence_accuracy": round(evidence_acc, 4),
                "evidence_minus_doc_accuracy": round(evidence_acc - accuracy(task_sources["doc_corpus"]), 4),
                "evidence_minus_semantic_accuracy": round(evidence_acc - accuracy(task_sources["semantic_kg"]), 4),
            }
        payload["models"][model_key] = model_payload
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "model_robustness_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = render(payload)
    (OUTPUT_DIR / "model_robustness_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

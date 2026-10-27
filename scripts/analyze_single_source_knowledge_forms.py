from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

from rag_eval_lib import gold_label, read_jsonl


ROOT = Path(__file__).resolve().parents[1]
GENERATION = ROOT / "dataset/rag/results/single_source_form_generation.jsonl"
RETRIEVAL = ROOT / "dataset/rag/analysis/evidence_meets_rag_single_source_forms/single_source_form_retrieval_571.jsonl"
OUTPUT_DIR = ROOT / "dataset/rag/analysis/evidence_meets_rag_single_source_forms"
SOURCES = ("tcga", "pubmed")
FORMS = ("document_corpus", "semantic_kg", "evidence_graph")
METRICS = ("component_recall", "complete_bundle", "joint_scope_precision", "support_mrr")


def macro_f1(rows: list[dict[str, Any]]) -> float:
    gold = Counter(gold_label(row["record"]) for row in rows)
    pred = Counter(str(row["normalized_prediction"]) for row in rows)
    tp = Counter()
    for row in rows:
        label = gold_label(row["record"])
        if str(row["normalized_prediction"]) == label:
            tp[label] += 1
    scores = []
    for label in gold.keys() | pred.keys():
        denominator = gold[label] + pred[label]
        scores.append(2 * tp[label] / denominator if denominator else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def exact_mcnemar(wins: int, losses: int) -> float:
    n = wins + losses
    if not n:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / (2**n))


def ci(values: list[float], seed: str) -> list[float]:
    rng = random.Random(seed)
    samples = [sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(10000)]
    samples.sort()
    return [round(samples[250], 4), round(samples[9749], 4)]


def paired_answer(evidence: list[dict[str, Any]], baseline: list[dict[str, Any]], source: str, baseline_name: str) -> dict[str, Any]:
    by_id = {row["record_id"]: row for row in baseline}
    pairs = [(row, by_id[row["record_id"]]) for row in evidence]
    deltas = [float(a["correct"]) - float(b["correct"]) for a, b in pairs]
    wins = sum(value > 0 for value in deltas)
    losses = sum(value < 0 for value in deltas)
    rng = random.Random(f"f1:{source}:{baseline_name}")
    f1_samples = []
    for _ in range(10000):
        indices = [rng.randrange(len(pairs)) for _ in pairs]
        f1_samples.append(
            macro_f1([pairs[i][0] for i in indices]) - macro_f1([pairs[i][1] for i in indices])
        )
    f1_samples.sort()
    return {
        "baseline": baseline_name,
        "count": len(pairs),
        "evidence_accuracy": round(sum(a["correct"] for a, _b in pairs) / len(pairs), 4),
        "baseline_accuracy": round(sum(b["correct"] for _a, b in pairs) / len(pairs), 4),
        "accuracy_delta": round(sum(deltas) / len(deltas), 4),
        "accuracy_ci95": ci(deltas, f"acc:{source}:{baseline_name}"),
        "evidence_macro_f1": round(macro_f1(evidence), 4),
        "baseline_macro_f1": round(macro_f1(baseline), 4),
        "macro_f1_delta": round(macro_f1(evidence) - macro_f1(baseline), 4),
        "macro_f1_ci95": [round(f1_samples[250], 4), round(f1_samples[9749], 4)],
        "wins": wins,
        "ties": len(pairs) - wins - losses,
        "losses": losses,
        "mcnemar_p": round(exact_mcnemar(wins, losses), 6),
    }


def paired_retrieval(evidence: list[dict[str, Any]], baseline: list[dict[str, Any]], source: str, baseline_name: str) -> dict[str, Any]:
    by_id = {row["record_id"]: row for row in baseline}
    output = {}
    for metric in METRICS:
        deltas = [float(row["retrieval_audit"][metric]) - float(by_id[row["record_id"]]["retrieval_audit"][metric]) for row in evidence]
        output[metric] = {"delta": round(sum(deltas) / len(deltas), 4), "ci95": ci(deltas, f"ret:{source}:{baseline_name}:{metric}")}
    return output


def main() -> None:
    generation = read_jsonl(GENERATION)
    retrieval = read_jsonl(RETRIEVAL)
    payload = {"design": {"strict_source_evidence_id_alignment": True, "retriever": "BM25", "top_k": 8}, "sources": {}}
    for source in SOURCES:
        generated = {form: [row for row in generation if row["source"] == source and row["form"] == form] for form in FORMS}
        retrieved = {form: [row for row in retrieval if row["source"] == source and row["form"] == form and row["record_id"] in {item["record_id"] for item in generated[form]}] for form in FORMS}
        comparisons = []
        for baseline in ("document_corpus", "semantic_kg"):
            row = paired_answer(generated["evidence_graph"], generated[baseline], source, baseline)
            row["retrieval"] = paired_retrieval(retrieved["evidence_graph"], retrieved[baseline], source, baseline)
            comparisons.append(row)
        payload["sources"][source] = {
            "scope": "pan-cancer 571" if source == "tcga" else "HCC/CRC 127",
            "form_scores": {
                form: {
                    "count": len(generated[form]),
                    "accuracy": round(sum(row["correct"] for row in generated[form]) / len(generated[form]), 4),
                    "macro_f1": round(macro_f1(generated[form]), 4),
                }
                for form in FORMS
            },
            "comparisons": comparisons,
        }
    (OUTPUT_DIR / "single_source_form_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Evidence Meets RAG: Source-Decoupled Knowledge-Form Analysis",
        "",
        "All three forms within a source contain exactly the same source evidence IDs. Retrieval uses BM25 top-8; generation uses the same GPT-5-mini configuration. TCGA is evaluated pan-cancer (N=571), while PubMed answer evaluation is restricted to its systematic HCC/CRC coverage (N=127).",
        "",
        "## Answer Quality",
        "",
        "| Source | Form | N | Accuracy | Macro-F1 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for source in SOURCES:
        for form in FORMS:
            score = payload["sources"][source]["form_scores"][form]
            lines.append(f"| {source.upper()} | {form} | {score['count']} | {score['accuracy']:.4f} | {score['macro_f1']:.4f} |")
    lines.extend([
        "",
        "## Paired Evidence-Graph Effects",
        "",
        "| Source | Baseline | Accuracy contribution (95% CI) | McNemar p | Macro-F1 contribution (95% CI) | W/T/L | Component Recall | Complete Bundle |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for source in SOURCES:
        for row in payload["sources"][source]["comparisons"]:
            lines.append(
                f"| {source.upper()} | {row['baseline']} | {row['accuracy_delta']:+.4f} {row['accuracy_ci95']} | {row['mcnemar_p']:.6f} | "
                f"{row['macro_f1_delta']:+.4f} {row['macro_f1_ci95']} | {row['wins']}/{row['ties']}/{row['losses']} | "
                f"{row['retrieval']['component_recall']['delta']:+.4f} {row['retrieval']['component_recall']['ci95']} | "
                f"{row['retrieval']['complete_bundle']['delta']:+.4f} {row['retrieval']['complete_bundle']['ci95']} |"
            )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Compare Component Recall, Complete Bundle, and answer quality within each isolated source. Interpret answer-level differences using the current paired intervals and McNemar tests rather than ranking alone.",
        "",
    ])
    report = "\n".join(lines)
    (OUTPUT_DIR / "single_source_form_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

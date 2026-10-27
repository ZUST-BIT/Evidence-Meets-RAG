from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rag_eval_lib import gold_label


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "dataset/rag/analysis/ecorag_cross_cancer"
BENCHMARK = ROOT / "dataset/evaluated/external_reasoning_benchmark_formal_v3_571.jsonl"
COVERABILITY = ROOT / "dataset/rag/analysis/ecorag_coverability_leakage/ecorag_coverability_leakage_records.jsonl"
RETRIEVAL = ROOT / "dataset/rag/analysis/ecorag_retrieval_quality/retrieval_quality_records.jsonl"

RESULTS = {
    "vanilla": {
        "label": "Vanilla RAG",
        "doc_corpus": ROOT / "dataset/rag/results/multi_cancer__raw_observation_view__naive_rag__formalv3_571_classic4_raw_vs_graph.jsonl",
        "semantic_kg": ROOT / "dataset/rag/results/multi_cancer__semantic_kg__naive_rag__formalv3_571_classic4_semantic_kg.jsonl",
        "evidence_kg": ROOT / "dataset/rag/results/fixed_context_component_ablation_571_vanilla__full.jsonl",
    },
    "graph": {
        "label": "Graph RAG",
        "doc_corpus": ROOT / "dataset/rag/results/multi_cancer__raw_observation_view__graph_rag__formalv3_571_classic4_raw_vs_graph.jsonl",
        "semantic_kg": ROOT / "dataset/rag/results/multi_cancer__semantic_kg__graph_rag__formalv3_571_classic4_semantic_kg.jsonl",
        "evidence_kg": ROOT / "dataset/rag/results/unified_571/stage/R1C1G1.jsonl",
    },
}
SOURCE_LABELS = {
    "doc_corpus": "Document Corpus",
    "semantic_kg": "Semantic KG",
    "evidence_kg": "Evidence Graph",
}
FAMILY_ORDER = ["aml", "brca", "crc", "gastroesoph", "glioma", "hcc", "hnsc", "nsclc", "ovarian", "paad", "rcc"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def family(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") or {}
    return str(metadata.get("priority_cancer_family") or record.get("oncology_focus") or "unknown").strip().lower()


def macro_f1(gold: list[str], pred: list[str]) -> float:
    gold_counts = Counter(gold)
    pred_counts = Counter(pred)
    tp = Counter(g for g, p in zip(gold, pred) if g == p)
    labels = gold_counts.keys() | pred_counts.keys()
    scores = []
    for label in labels:
        denominator = 2 * tp[label] + (pred_counts[label] - tp[label]) + (gold_counts[label] - tp[label])
        scores.append(2 * tp[label] / denominator if denominator else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def exact_mcnemar(wins: int, losses: int) -> float:
    n = wins + losses
    if not n:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2**n))


def holm(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [1.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def paired_effect(evidence: list[dict[str, Any]], baseline: list[dict[str, Any]], seed: int, iterations: int = 10000) -> dict[str, Any]:
    base_by_id = {row["record_id"]: row for row in baseline}
    pairs = [(row, base_by_id[row["record_id"]]) for row in evidence if row["record_id"] in base_by_id]
    gold = [row[0]["gold"] for row in pairs]
    evidence_pred = [row[0]["pred"] for row in pairs]
    baseline_pred = [row[1]["pred"] for row in pairs]
    evidence_correct = [int(g == p) for g, p in zip(gold, evidence_pred)]
    baseline_correct = [int(g == p) for g, p in zip(gold, baseline_pred)]
    wins = sum(e and not b for e, b in zip(evidence_correct, baseline_correct))
    losses = sum(not e and b for e, b in zip(evidence_correct, baseline_correct))
    acc_delta = sum(e - b for e, b in zip(evidence_correct, baseline_correct)) / len(pairs)
    f1_delta = macro_f1(gold, evidence_pred) - macro_f1(gold, baseline_pred)
    rng = random.Random(seed)
    acc_samples: list[float] = []
    f1_samples: list[float] = []
    for _ in range(iterations):
        indices = [rng.randrange(len(pairs)) for _ in pairs]
        sample_gold = [gold[index] for index in indices]
        sample_evidence = [evidence_pred[index] for index in indices]
        sample_baseline = [baseline_pred[index] for index in indices]
        acc_samples.append(sum(evidence_correct[index] - baseline_correct[index] for index in indices) / len(indices))
        f1_samples.append(macro_f1(sample_gold, sample_evidence) - macro_f1(sample_gold, sample_baseline))
    acc_samples.sort()
    f1_samples.sort()
    low_index = min(iterations - 1, int(.025 * iterations))
    high_index = min(iterations - 1, max(0, int(.975 * iterations) - 1))
    return {
        "count": len(pairs),
        "evidence_accuracy": round(sum(evidence_correct) / len(pairs), 4),
        "baseline_accuracy": round(sum(baseline_correct) / len(pairs), 4),
        "accuracy_delta": round(acc_delta, 4),
        "accuracy_ci95": [round(acc_samples[low_index], 4), round(acc_samples[high_index], 4)],
        "evidence_macro_f1": round(macro_f1(gold, evidence_pred), 4),
        "baseline_macro_f1": round(macro_f1(gold, baseline_pred), 4),
        "macro_f1_delta": round(f1_delta, 4),
        "macro_f1_ci95": [round(f1_samples[low_index], 4), round(f1_samples[high_index], 4)],
        "wins": wins,
        "ties": len(pairs) - wins - losses,
        "losses": losses,
        "mcnemar_p": round(exact_mcnemar(wins, losses), 6),
    }


def load_performance() -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, list[dict[str, Any]]]]]:
    benchmark = read_jsonl(BENCHMARK)
    record_lookup = {str(record["id"]): record for record in benchmark}
    task_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in benchmark:
        task_counts[family(record)][str(record.get("task_type") or "unknown")] += 1

    payload: dict[str, Any] = {}
    normalized_rows: dict[str, dict[str, Any]] = {}
    seed = 20260629
    for framework, spec in RESULTS.items():
        by_source: dict[str, list[dict[str, Any]]] = {}
        for source, path in spec.items():
            if source == "label":
                continue
            rows = []
            for raw in read_jsonl(path):
                record = raw["record"]
                rid = str(record["id"])
                rows.append({
                    "record_id": rid,
                    "family": family(record),
                    "task_type": record.get("task_type"),
                    "gold": gold_label(record),
                    "pred": str(raw.get("normalized_prediction") or ""),
                })
            by_source[source] = rows
        normalized_rows[framework] = by_source
        family_effects = []
        for baseline in ["doc_corpus", "semantic_kg"]:
            comparison_rows = []
            for cancer in FAMILY_ORDER:
                evidence_rows = [row for row in by_source["evidence_kg"] if row["family"] == cancer]
                baseline_rows = [row for row in by_source[baseline] if row["family"] == cancer]
                effect = paired_effect(evidence_rows, baseline_rows, seed)
                seed += 1
                effect.update({
                    "framework": framework,
                    "framework_label": spec["label"],
                    "family": cancer,
                    "baseline": baseline,
                    "baseline_label": SOURCE_LABELS[baseline],
                    "task_count": len(task_counts[cancer]),
                    "dominant_task_share": round(max(task_counts[cancer].values()) / sum(task_counts[cancer].values()), 4),
                })
                comparison_rows.append(effect)
            adjusted = holm([row["mcnemar_p"] for row in comparison_rows])
            for row, value in zip(comparison_rows, adjusted):
                row["mcnemar_holm_p"] = round(value, 6)
            family_effects.extend(comparison_rows)
        payload[framework] = {"label": spec["label"], "effects": family_effects}
    return payload, record_lookup, normalized_rows


def build_task_decomposition(normalized_rows: dict[str, dict[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    output = []
    for framework, by_source in normalized_rows.items():
        evidence_by_id = {row["record_id"]: row for row in by_source["evidence_kg"]}
        for baseline in ["doc_corpus", "semantic_kg"]:
            baseline_by_id = {row["record_id"]: row for row in by_source[baseline]}
            grouped: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
            for rid, evidence in evidence_by_id.items():
                base = baseline_by_id[rid]
                grouped[(evidence["family"], str(evidence["task_type"]))].append((evidence, base))
            for (cancer, task), pairs in sorted(grouped.items()):
                delta_sum = sum(int(e["pred"] == e["gold"]) - int(b["pred"] == b["gold"]) for e, b in pairs)
                output.append({
                    "framework": framework,
                    "family": cancer,
                    "baseline": baseline,
                    "task_type": task,
                    "count": len(pairs),
                    "accuracy_delta": round(delta_sum / len(pairs), 4),
                    "family_accuracy_contribution": round(delta_sum / sum(1 for row in by_source["evidence_kg"] if row["family"] == cancer), 4),
                })
    return output


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row.get(key) or 0.0) for row in rows) / len(rows) if rows else 0.0


def build_mechanism(record_lookup: dict[str, dict[str, Any]], performance: dict[str, Any]) -> list[dict[str, Any]]:
    cover_rows = read_jsonl(COVERABILITY)
    retrieval_rows = read_jsonl(RETRIEVAL)
    for row in cover_rows:
        row["family"] = family(record_lookup[row["record_id"]])
    for row in retrieval_rows:
        row["family"] = family(record_lookup[row["record_id"]])

    vanilla_doc = {row["family"]: row for row in performance["vanilla"]["effects"] if row["baseline"] == "doc_corpus"}
    output = []
    for cancer in FAMILY_ORDER:
        by_cover_source = {
            source: [row for row in cover_rows if row["family"] == cancer and row["knowledge_source"] == source]
            for source in SOURCE_LABELS
        }
        by_retrieval_source = {
            source: [row for row in retrieval_rows if row["family"] == cancer and row["knowledge_source"] == source]
            for source in SOURCE_LABELS
        }
        output.append({
            "family": cancer,
            "full_n": vanilla_doc[cancer]["count"],
            "vanilla_accuracy_delta_vs_doc": vanilla_doc[cancer]["accuracy_delta"],
            "evidence_support_coverability": round(mean(by_cover_source["evidence_kg"], "rule_support_coverable"), 4),
            "doc_support_coverability": round(mean(by_cover_source["doc_corpus"], "rule_support_coverable"), 4),
            "support_coverability_delta_vs_doc": round(mean(by_cover_source["evidence_kg"], "rule_support_coverable") - mean(by_cover_source["doc_corpus"], "rule_support_coverable"), 4),
            "evidence_bundle_coverability": round(mean(by_cover_source["evidence_kg"], "bundle_completeness"), 4),
            "doc_bundle_coverability": round(mean(by_cover_source["doc_corpus"], "bundle_completeness"), 4),
            "bundle_coverability_delta_vs_doc": round(mean(by_cover_source["evidence_kg"], "bundle_completeness") - mean(by_cover_source["doc_corpus"], "bundle_completeness"), 4),
            "panel_n": len(by_retrieval_source["evidence_kg"]),
            "component_recall_delta_vs_doc": round(mean(by_retrieval_source["evidence_kg"], "component_recall") - mean(by_retrieval_source["doc_corpus"], "component_recall"), 4),
            "complete_bundle_delta_vs_doc": round(mean(by_retrieval_source["evidence_kg"], "complete_bundle") - mean(by_retrieval_source["doc_corpus"], "complete_bundle"), 4),
        })
    return output


def pearson(xs: list[float], ys: list[float]) -> float:
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = math.sqrt(sum((x - x_mean) ** 2 for x in xs) * sum((y - y_mean) ** 2 for y in ys))
    return numerator / denominator if denominator else 0.0


def ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    output = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = (index + end - 1) / 2 + 1
        for position in range(index, end):
            output[ordered[position][0]] = rank
        index = end
    return output


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_report(performance: dict[str, Any], mechanism: list[dict[str, Any]], task_decomposition: list[dict[str, Any]], correlations: dict[str, Any]) -> str:
    lines = [
        "# Evidence Meets RAG: Cross-Cancer Performance and Mechanism Analysis",
        "",
        "## Design",
        "",
        "We stratify the existing 571-record main results by 11 cancer families. Evidence Graph is paired with Document Corpus and Semantic KG within each family under Vanilla and Graph RAG. Accuracy and Macro-F1 differences use 10,000 paired bootstrap resamples; Accuracy additionally uses exact McNemar tests with Holm correction across the 11 families within each framework-baseline comparison. Full-benchmark coverability supplies the primary mechanism analysis, while top-8 retrieval metrics from the balanced 145 panel are descriptive because family-level panel sizes are small.",
        "",
    ]
    for framework in ["vanilla", "graph"]:
        lines.extend([
            f"## {performance[framework]['label']}",
            "",
            "| Cancer | N | Baseline | Acc EG/Base | Delta (95% CI) | Holm p | F1 Delta (95% CI) | W/T/L |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        rows = performance[framework]["effects"]
        for cancer in FAMILY_ORDER:
            for baseline in ["doc_corpus", "semantic_kg"]:
                row = next(item for item in rows if item["family"] == cancer and item["baseline"] == baseline)
                lines.append(
                    f"| {cancer.upper()} | {row['count']} | {row['baseline_label']} | {row['evidence_accuracy']:.4f}/{row['baseline_accuracy']:.4f} | "
                    f"{row['accuracy_delta']:+.4f} {row['accuracy_ci95']} | {row['mcnemar_holm_p']:.6f} | "
                    f"{row['macro_f1_delta']:+.4f} {row['macro_f1_ci95']} | {row['wins']}/{row['ties']}/{row['losses']} |"
                )
        lines.append("")

    lines.extend([
        "## Mechanism by Cancer Family",
        "",
        "| Cancer | Full N | EG-Doc Acc Delta | Support Coverability Delta | Bundle Coverability Delta | Panel N | Component Recall Delta | Retrieved Bundle Delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in mechanism:
        lines.append(
            f"| {row['family'].upper()} | {row['full_n']} | {row['vanilla_accuracy_delta_vs_doc']:+.4f} | "
            f"{row['support_coverability_delta_vs_doc']:+.4f} | {row['bundle_coverability_delta_vs_doc']:+.4f} | "
            f"{row['panel_n']} | {row['component_recall_delta_vs_doc']:+.4f} | {row['complete_bundle_delta_vs_doc']:+.4f} |"
        )
    lines.extend([
        "",
        "Across the 11 cancer families, the descriptive Pearson correlation between the Vanilla Evidence-vs-Document Accuracy delta and the support-coverability delta is "
        f"{correlations['accuracy_vs_support_coverability']['pearson']:+.3f} (Spearman {correlations['accuracy_vs_support_coverability']['spearman']:+.3f}); "
        "for bundle coverability it is "
        f"{correlations['accuracy_vs_bundle_coverability']['pearson']:+.3f} (Spearman {correlations['accuracy_vs_bundle_coverability']['spearman']:+.3f}). "
        "With only 11 heterogeneous families, these are mechanism diagnostics rather than inferential mediation estimates.",
        "",
        "## Task-Mixture Diagnostics (Vanilla vs Document Corpus)",
        "",
        "| Cancer | Dominant task (share) | Largest positive task contribution | Largest negative task contribution |",
        "| --- | --- | --- | --- |",
    ])
    vanilla_tasks = [row for row in task_decomposition if row["framework"] == "vanilla" and row["baseline"] == "doc_corpus"]
    for cancer in FAMILY_ORDER:
        rows = [row for row in vanilla_tasks if row["family"] == cancer]
        total = sum(row["count"] for row in rows)
        dominant = max(rows, key=lambda row: row["count"])
        positive = max(rows, key=lambda row: row["family_accuracy_contribution"])
        negative = min(rows, key=lambda row: row["family_accuracy_contribution"])
        lines.append(
            f"| {cancer.upper()} | {dominant['task_type']} ({dominant['count'] / total:.1%}) | "
            f"{positive['task_type']} ({positive['family_accuracy_contribution']:+.4f}) | "
            f"{negative['task_type']} ({negative['family_accuracy_contribution']:+.4f}) |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
    ])
    for framework in ["vanilla", "graph"]:
        for baseline in ["doc_corpus", "semantic_kg"]:
            rows = [row for row in performance[framework]["effects"] if row["baseline"] == baseline]
            positive = sum(row["accuracy_delta"] > 0 for row in rows)
            negative = sum(row["accuracy_delta"] < 0 for row in rows)
            significant = sum(row["accuracy_delta"] > 0 and row["mcnemar_holm_p"] < 0.05 for row in rows)
            lines.append(
                f"- {performance[framework]['label']} vs {SOURCE_LABELS[baseline]}: positive Accuracy delta in {positive}/11 families, negative in {negative}/11, and Holm-significant positive in {significant}/11."
            )
    lines.extend([
        "",
        "Cancer-level effects should not be interpreted as pure biological domain effects. Family sample sizes range from 29 to 93 and task mixtures differ substantially; several families lack some task types. The appropriate claim is directional domain robustness with explicit heterogeneity, not universal significance in every cancer.",
        "",
        "The full-benchmark mechanism table tests whether evidence availability changes in the expected direction within each cancer. The 145-panel retrieval columns are secondary: small family cells make them useful for triangulation but unsuitable for family ranking.",
        "",
    ])
    return "\n".join(lines)


def plot_effect_matrix(performance: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm

    # The main text discusses Graph RAG. The matrix separates absolute Accuracy
    # from signed Evidence Graph gains so readers do not have to infer the
    # comparisons from connected points.
    rows = performance["graph"]["effects"]
    doc_rows = {row["family"]: row for row in rows if row["baseline"] == "doc_corpus"}
    sem_rows = {row["family"]: row for row in rows if row["baseline"] == "semantic_kg"}
    order = sorted(FAMILY_ORDER, key=lambda cancer: doc_rows[cancer]["accuracy_delta"], reverse=True)
    absolute = [
        [
            doc_rows[cancer]["baseline_accuracy"],
            sem_rows[cancer]["baseline_accuracy"],
            doc_rows[cancer]["evidence_accuracy"],
        ]
        for cancer in order
    ]
    gains = [
        [doc_rows[cancer]["accuracy_delta"], sem_rows[cancer]["accuracy_delta"]]
        for cancer in order
    ]

    accuracy_cmap = LinearSegmentedColormap.from_list("accuracy", ["#F1F6FA", "#B7D0E5", "#4C78A8"])
    gain_cmap = LinearSegmentedColormap.from_list("gain", ["#E7A15B", "#FAFAFA", "#55A868"])
    accuracy_norm = Normalize(vmin=0.20, vmax=0.60)
    gain_norm = TwoSlopeNorm(vmin=-0.25, vcenter=0.0, vmax=0.25)

    fig, (ax_abs, ax_gain) = plt.subplots(
        1, 2, figsize=(3.55, 2.42), sharey=True,
        gridspec_kw={"width_ratios": [3.0, 2.0], "wspace": 0.06},
    )
    ax_abs.imshow(absolute, cmap=accuracy_cmap, norm=accuracy_norm, aspect="auto")
    ax_gain.imshow(gains, cmap=gain_cmap, norm=gain_norm, aspect="auto")

    labels = [f"{cancer.upper()} ({doc_rows[cancer]['count']})" for cancer in order]
    ax_abs.set_yticks(range(len(order)), labels)
    ax_abs.set_xticks(range(3), ["Document\nCorpus", "Semantic\nKG", "Evidence\nGraph"])
    ax_gain.set_xticks(range(2), ["vs.\nDocument", "vs.\nSemantic"])
    ax_abs.set_title("Absolute Accuracy", fontsize=6.7, fontweight="bold", pad=20)
    ax_gain.set_title("Evidence Graph gain", fontsize=6.7, fontweight="bold", pad=20)

    for axis in (ax_abs, ax_gain):
        axis.xaxis.tick_top()
        axis.tick_params(axis="x", labelsize=5.8, length=0, pad=1.5)
        axis.tick_params(axis="y", labelsize=5.8, length=0, pad=2.0)
        axis.set_xticks([value - 0.5 for value in range(axis.images[0].get_array().shape[1] + 1)], minor=True)
        axis.set_yticks([value - 0.5 for value in range(len(order) + 1)], minor=True)
        axis.grid(which="minor", color="white", linewidth=0.9)
        axis.tick_params(which="minor", bottom=False, left=False)
        for spine in axis.spines.values():
            spine.set_visible(False)

    for row_index, values in enumerate(absolute):
        for column_index, value in enumerate(values):
            ax_abs.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=5.9)

    for row_index, cancer in enumerate(order):
        for column_index, value in enumerate(gains[row_index]):
            significant = column_index == 0 and doc_rows[cancer]["mcnemar_holm_p"] < 0.05
            label = f"{value:+.2f}{'*' if significant else ''}"
            ax_gain.text(
                column_index, row_index, label, ha="center", va="center", fontsize=5.9,
                fontweight="bold" if significant else "normal",
            )

    fig.subplots_adjust(left=0.245, right=0.995, bottom=0.02, top=0.79)
    fig.savefig(OUTPUT_DIR / "cross_cancer_accuracy_matrix.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUTPUT_DIR / "cross_cancer_accuracy_matrix.png", dpi=240, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUTPUT_DIR / "cross_cancer_accuracy_matrix.svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    performance, record_lookup, normalized_rows = load_performance()
    mechanism = build_mechanism(record_lookup, performance)
    task_decomposition = build_task_decomposition(normalized_rows)
    accuracy_delta = [row["vanilla_accuracy_delta_vs_doc"] for row in mechanism]
    support_delta = [row["support_coverability_delta_vs_doc"] for row in mechanism]
    bundle_delta = [row["bundle_coverability_delta_vs_doc"] for row in mechanism]
    correlations = {
        "accuracy_vs_support_coverability": {
            "pearson": round(pearson(accuracy_delta, support_delta), 4),
            "spearman": round(pearson(ranks(accuracy_delta), ranks(support_delta)), 4),
        },
        "accuracy_vs_bundle_coverability": {
            "pearson": round(pearson(accuracy_delta, bundle_delta), 4),
            "spearman": round(pearson(ranks(accuracy_delta), ranks(bundle_delta)), 4),
        },
    }
    payload = {
        "design": {
            "benchmark_records": 571,
            "families": FAMILY_ORDER,
            "bootstrap_iterations": 10000,
            "multiple_testing": "Holm within framework-baseline across 11 families",
        },
        "performance": performance,
        "mechanism": mechanism,
        "task_decomposition": task_decomposition,
        "correlations": correlations,
    }
    (OUTPUT_DIR / "cross_cancer_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    effect_rows = [row for framework in performance.values() for row in framework["effects"]]
    write_tsv(OUTPUT_DIR / "cross_cancer_effects.tsv", effect_rows)
    write_tsv(OUTPUT_DIR / "cross_cancer_mechanism.tsv", mechanism)
    write_tsv(OUTPUT_DIR / "cross_cancer_task_decomposition.tsv", task_decomposition)
    report = render_report(performance, mechanism, task_decomposition, correlations)
    (OUTPUT_DIR / "cross_cancer_report.md").write_text(report + "\n", encoding="utf-8")
    plot_effect_matrix(performance)
    print(report)


if __name__ == "__main__":
    main()

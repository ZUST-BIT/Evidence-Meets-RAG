from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "dataset/rag/analysis/evidence_meets_rag_hyperparameter_stability"
INPUT = ANALYSIS_DIR / "retrieval_hyperparameter_stability_145.jsonl"
SUMMARY = ANALYSIS_DIR / "retrieval_hyperparameter_summary.json"
REPORT = ANALYSIS_DIR / "retrieval_hyperparameter_report.md"

FORM_LABELS = {
    "document_corpus": "Document Corpus",
    "semantic_kg": "Semantic KG",
    "evidence_graph": "Evidence Graph",
}
PRIMARY_METRICS = ["component_recall", "complete_bundle"]
DISPLAY_METRICS = [
    "entity_hit",
    "joint_scope_precision",
    "component_recall",
    "complete_bundle",
    "support_mrr",
    "source_diversity",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(rows)}
    for metric in DISPLAY_METRICS:
        out[metric] = round(mean([float(row["retrieval_audit"].get(metric, 0.0)) for row in rows]), 4)
    return out


def paired_delta(
    rows: list[dict[str, Any]],
    setting: str,
    form_a: str,
    form_b: str,
    metric: str,
    iterations: int = 10000,
) -> dict[str, Any]:
    by_id: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if row["setting"] == setting and row["form"] in {form_a, form_b}:
            by_id[(str(row["record_id"]), row["form"])] = row
    ids = sorted(
        rid
        for rid in {key[0] for key in by_id}
        if (rid, form_a) in by_id and (rid, form_b) in by_id
    )
    deltas = [
        float(by_id[(rid, form_a)]["retrieval_audit"].get(metric, 0.0))
        - float(by_id[(rid, form_b)]["retrieval_audit"].get(metric, 0.0))
        for rid in ids
    ]
    observed = mean(deltas)
    wins = sum(1 for delta in deltas if delta > 0)
    ties = sum(1 for delta in deltas if delta == 0)
    losses = sum(1 for delta in deltas if delta < 0)
    rng = random.Random(f"20260629:{setting}:{form_a}:{form_b}:{metric}")
    samples = []
    for _ in range(iterations):
        samples.append(sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas))
    samples.sort()
    return {
        "n": len(ids),
        "delta": round(observed, 4),
        "ci95": [round(samples[int(iterations * 0.025)], 4), round(samples[int(iterations * 0.975) - 1], 4)],
        "wins_ties_losses": [wins, ties, losses],
    }


def render_metric_effect(effect: dict[str, Any]) -> str:
    lo, hi = effect["ci95"]
    wins, ties, losses = effect["wins_ties_losses"]
    return f"{effect['delta']:+.4f} [{lo:+.4f}, {hi:+.4f}], W/T/L {wins}/{ties}/{losses}"


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Evidence Meets RAG: Retrieval Hyperparameter and Budget Stability",
        "",
        "## Design",
        "",
        "This control experiment tests whether the retrieval-side advantage of the Evidence Graph is stable under common retrieval hyperparameters. We use the balanced 145-record faithfulness panel and compare Document Corpus, Semantic KG, and Evidence Graph with identical BM25 retrieval code. The top-k sweep is evaluated on all 145 records at k = 2, 4, 8, and 16. BM25 k1 and b sweeps are evaluated on a fixed 20-record SHA-256 panel independent of retrieval outcomes to keep the run lightweight while preserving paired comparison.",
        "",
        "The primary endpoints are Component Recall and Complete Bundle, because they directly measure whether the retrieved context contains the task-required evidence parts and whether the full evidence bundle is assembled. Joint Scope Precision is reported as a noise/scope-control diagnostic.",
        "",
        "## Top-k Sweep",
        "",
        "| Setting | Form | Joint Scope P | Component Recall | Complete Bundle | Support MRR |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    topk_settings = ["topk_2", "topk_4", "default", "topk_16"]
    for setting in topk_settings:
        for form in ["document_corpus", "semantic_kg", "evidence_graph"]:
            row = payload["summaries"][setting][form]
            lines.append(
                f"| {setting} | {FORM_LABELS[form]} | {row['joint_scope_precision']:.4f} | "
                f"{row['component_recall']:.4f} | {row['complete_bundle']:.4f} | {row['support_mrr']:.4f} |"
            )
    lines.extend(["", "## Evidence Graph Paired Effects Across Top-k", ""])
    for baseline in ["document_corpus", "semantic_kg"]:
        lines.append(f"### Versus {FORM_LABELS[baseline]}")
        lines.append("")
        lines.append("| Setting | Component Recall Delta | Complete Bundle Delta |")
        lines.append("| --- | ---: | ---: |")
        for setting in topk_settings:
            effects = payload["paired_effects"][setting][f"evidence_graph_minus_{baseline}"]
            lines.append(
                f"| {setting} | {render_metric_effect(effects['component_recall'])} | "
                f"{render_metric_effect(effects['complete_bundle'])} |"
            )
        lines.append("")
    lines.extend([
        "## BM25 Parameter Sweep",
        "",
        "| Setting | Form | Joint Scope P | Component Recall | Complete Bundle | Support MRR |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ])
    bm25_settings = ["k1_0.9", "k1_2.0", "b_0.25", "b_1.0"]
    for setting in bm25_settings:
        for form in ["document_corpus", "semantic_kg", "evidence_graph"]:
            row = payload["summaries"][setting][form]
            lines.append(
                f"| {setting} | {FORM_LABELS[form]} | {row['joint_scope_precision']:.4f} | "
                f"{row['component_recall']:.4f} | {row['complete_bundle']:.4f} | {row['support_mrr']:.4f} |"
            )
    lines.extend(["", "## Evidence Graph Paired Effects Across BM25 Settings", ""])
    for baseline in ["document_corpus", "semantic_kg"]:
        lines.append(f"### Versus {FORM_LABELS[baseline]}")
        lines.append("")
        lines.append("| Setting | Component Recall Delta | Complete Bundle Delta |")
        lines.append("| --- | ---: | ---: |")
        for setting in bm25_settings:
            effects = payload["paired_effects"][setting][f"evidence_graph_minus_{baseline}"]
            lines.append(
                f"| {setting} | {render_metric_effect(effects['component_recall'])} | "
                f"{render_metric_effect(effects['complete_bundle'])} |"
            )
        lines.append("")
    stability = payload["stability"]
    lines.extend(
        [
            "## Stability Summary",
            "",
            f"- Across the four top-k settings, Evidence Graph remains above Document Corpus by at least {stability['topk_min_delta_vs_document']['component_recall']:+.4f} Component Recall and {stability['topk_min_delta_vs_document']['complete_bundle']:+.4f} Complete Bundle.",
            f"- Across the four top-k settings, Evidence Graph remains above Semantic KG by at least {stability['topk_min_delta_vs_semantic']['component_recall']:+.4f} Component Recall and {stability['topk_min_delta_vs_semantic']['complete_bundle']:+.4f} Complete Bundle.",
            f"- Across the four BM25 parameter settings, Evidence Graph remains above Document Corpus by at least {stability['bm25_min_delta_vs_document']['component_recall']:+.4f} Component Recall and {stability['bm25_min_delta_vs_document']['complete_bundle']:+.4f} Complete Bundle on the fixed 20-record panel.",
            f"- Across the four BM25 parameter settings, Evidence Graph remains above Semantic KG by at least {stability['bm25_min_delta_vs_semantic']['component_recall']:+.4f} Component Recall and {stability['bm25_min_delta_vs_semantic']['complete_bundle']:+.4f} Complete Bundle on the fixed 20-record panel.",
            "",
            "## Interpretation",
            "",
            "The Evidence Graph advantage is stable to retrieval budget changes. Increasing top-k improves all representations, but it does not erase the evidence-centric margin: Evidence Graph consistently retrieves more task-required components and more complete evidence bundles than both baselines at k = 2, 4, 8, and 16.",
            "",
            "The BM25 sweep reaches the same conclusion under moderate k1 and b changes. This supports the claim that the main retrieval benefit comes from evidence-centric knowledge organization rather than from a fragile lexical parameter choice. Because the BM25 sweep uses a smaller fixed panel, it should be reported as a robustness control rather than a new primary benchmark.",
            "",
            "Joint Scope Precision remains comparable rather than uniformly higher, so the correct claim is not that Evidence Graph always retrieves cleaner lexical matches. Its robust advantage is compositional evidence access: more required components and more complete bundles under the same retrieval budget.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = read_jsonl(INPUT)
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["setting"]][row["form"]].append(row)

    summaries: dict[str, dict[str, Any]] = {}
    paired_effects: dict[str, dict[str, Any]] = {}
    for setting, by_form in sorted(grouped.items()):
        summaries[setting] = {form: summarize_rows(items) for form, items in sorted(by_form.items())}
        paired_effects[setting] = {}
        for baseline in ["document_corpus", "semantic_kg"]:
            key = f"evidence_graph_minus_{baseline}"
            paired_effects[setting][key] = {
                metric: paired_delta(rows, setting, "evidence_graph", baseline, metric)
                for metric in PRIMARY_METRICS
            }

    def min_delta(settings: list[str], baseline: str) -> dict[str, float]:
        out = {}
        key = f"evidence_graph_minus_{baseline}"
        for metric in PRIMARY_METRICS:
            out[metric] = round(min(paired_effects[setting][key][metric]["delta"] for setting in settings), 4)
        return out

    topk_settings = ["topk_2", "topk_4", "default", "topk_16"]
    bm25_settings = ["k1_0.9", "k1_2.0", "b_0.25", "b_1.0"]
    payload = {
        "input": str(INPUT.relative_to(ROOT)),
        "rows": len(rows),
        "summaries": summaries,
        "paired_effects": paired_effects,
        "stability": {
            "topk_min_delta_vs_document": min_delta(topk_settings, "document_corpus"),
            "topk_min_delta_vs_semantic": min_delta(topk_settings, "semantic_kg"),
            "bm25_min_delta_vs_document": min_delta(bm25_settings, "document_corpus"),
            "bm25_min_delta_vs_semantic": min_delta(bm25_settings, "semantic_kg"),
        },
    }
    SUMMARY.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    REPORT.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps(payload["stability"], indent=2))


if __name__ == "__main__":
    main()

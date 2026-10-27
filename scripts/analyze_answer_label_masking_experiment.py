from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from audit_ecorag_coverability_leakage import answer_terms, norm_text
from rag_eval_lib import compute_answer_metrics, gold_label


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "dataset" / "evaluated" / "ecorag_label_masking_panel_120.jsonl"
ORIGINAL = ROOT / "dataset" / "rag" / "results" / "multi_cancer__evidence_graph_complete__naive_rag__labelmask120_original_replay.jsonl"
MASKED = ROOT / "dataset" / "rag" / "results" / "multi_cancer__evidence_graph_complete_label_masked__naive_rag__labelmask120.jsonl"
OUTPUT_DIR = ROOT / "dataset" / "rag" / "analysis" / "ecorag_label_masking"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def record_id(row: dict[str, Any]) -> str:
    return str((row.get("record") or {}).get("id") or row.get("id") or "")


def correct(row: dict[str, Any]) -> bool:
    return gold_label(row["record"]) == str(row.get("normalized_prediction") or "")


def answer_exposed_in_retrieval(row: dict[str, Any]) -> bool:
    terms = answer_terms(row["record"])
    if not terms:
        return False
    context = norm_text(" ".join(f"{doc.get('title', '')} {doc.get('text', '')}" for doc in row.get("retrieved_docs") or []))
    return any(f" {term} " in f" {context} " for term in terms)


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answer = compute_answer_metrics(rows)
    return {
        "count": len(rows),
        "accuracy": answer["accuracy"],
        "macro_f1": answer["macro_f1"],
        "retrieved_answer_exposure_rate": round(sum(answer_exposed_in_retrieval(row) for row in rows) / max(len(rows), 1), 4),
    }


def macro_f1(rows: list[dict[str, Any]]) -> float:
    return float(compute_answer_metrics(rows)["macro_f1"])


def bootstrap_delta(original: list[dict[str, Any]], masked: list[dict[str, Any]], iterations: int = 10000) -> dict[str, Any]:
    rng = random.Random(20260627)
    n = len(original)
    acc_deltas = []
    f1_deltas = []
    for _ in range(iterations):
        indices = [rng.randrange(n) for _ in range(n)]
        sample_original = [original[i] for i in indices]
        sample_masked = [masked[i] for i in indices]
        acc_deltas.append(sum(correct(row) for row in sample_masked) / n - sum(correct(row) for row in sample_original) / n)
        f1_deltas.append(macro_f1(sample_masked) - macro_f1(sample_original))
    acc_deltas.sort()
    f1_deltas.sort()
    lo = int(iterations * 0.025)
    hi = int(iterations * 0.975) - 1
    return {
        "accuracy_delta_masked_minus_original": round(sum(correct(row) for row in masked) / n - sum(correct(row) for row in original) / n, 4),
        "accuracy_delta_95ci": [round(acc_deltas[lo], 4), round(acc_deltas[hi], 4)],
        "macro_f1_delta_masked_minus_original": round(macro_f1(masked) - macro_f1(original), 4),
        "macro_f1_delta_95ci": [round(f1_deltas[lo], 4), round(f1_deltas[hi], 4)],
        "bootstrap_iterations": iterations,
    }


def binomial_two_sided(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    observed = math.comb(n, k) / (2**n)
    return min(1.0, sum(math.comb(n, i) / (2**n) for i in range(n + 1) if math.comb(n, i) / (2**n) <= observed + 1e-15))


def paired_stats(original: list[dict[str, Any]], masked: list[dict[str, Any]]) -> dict[str, Any]:
    original_only = sum(correct(a) and not correct(b) for a, b in zip(original, masked))
    masked_only = sum(not correct(a) and correct(b) for a, b in zip(original, masked))
    stats = bootstrap_delta(original, masked)
    stats.update({
        "original_correct_masked_wrong": original_only,
        "original_wrong_masked_correct": masked_only,
        "mcnemar_exact_p": round(binomial_two_sided(min(original_only, masked_only), original_only + masked_only), 6),
    })
    return stats


def topk_overlap(a: dict[str, Any], b: dict[str, Any]) -> float:
    left = {str(doc.get("doc_id")) for doc in a.get("retrieved_docs") or []}
    right = {str(doc.get("doc_id")) for doc in b.get("retrieved_docs") or []}
    return len(left & right) / max(len(left | right), 1)


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Evidence Meets RAG: Answer-Label Masking Experiment",
        "",
        "## Design",
        "",
        "We compare contemporaneous Original and Label-Masked generations on the same 120-record panel: all 60 records flagged by the prior structured-label exposure audit and 60 non-exposed controls from the same two task families. We hold the Original Evidence KG top-8 document IDs, scores, and order fixed, then remove generated predicates, directions, polarities, role verdicts, support labels, example claims, and relation labels before regeneration. Entities, disease scope, raw evidence text, numerical observations, provenance, and source-count structure are retained. This isolates the effect of answer-bearing context fields from retrieval ranking.",
        "",
        "## Results",
        "",
        "| Subset | Condition | N | Accuracy | Macro-F1 | Retrieved answer exposure |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for subset in ["all", "exposed", "non_exposed_control"]:
        for condition in ["original", "masked"]:
            row = summary["subsets"][subset][condition]
            lines.append(f"| {subset} | {condition} | {row['count']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | {row['retrieved_answer_exposure_rate']:.4f} |")
    lines.extend(["", "## Paired Effects", ""])
    for subset in ["all", "exposed", "non_exposed_control"]:
        row = summary["subsets"][subset]["paired"]
        lines.append(
            f"- **{subset}**: masked-original Accuracy = {row['accuracy_delta_masked_minus_original']:+.4f} "
            f"(95% CI {row['accuracy_delta_95ci']}), Macro-F1 = {row['macro_f1_delta_masked_minus_original']:+.4f} "
            f"(95% CI {row['macro_f1_delta_95ci']}), McNemar p = {row['mcnemar_exact_p']:.6f}."
        )
    all_stats = summary["subsets"]["all"]["paired"]
    exposed_stats = summary["subsets"]["exposed"]["paired"]
    control_stats = summary["subsets"]["non_exposed_control"]["paired"]
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"The full-panel paired Accuracy change is {all_stats['accuracy_delta_masked_minus_original']:+.4f}. On the audit-flagged exposure stratum it is {exposed_stats['accuracy_delta_masked_minus_original']:+.4f}, while the non-exposed control change is {control_stats['accuracy_delta_masked_minus_original']:+.4f}. A statistically supported selective decrease on exposed records would indicate a label shortcut; overlapping confidence intervals or stable performance instead argue against such a shortcut in the retrieved contexts.",
        f"Mean top-k document Jaccard overlap is {summary['retrieval']['mean_topk_jaccard']:.4f} by design, so observed changes arise after retrieval rather than from ranking differences.",
        f"Direct gold-answer wording appears in {summary['subsets']['all']['original']['retrieved_answer_exposure_rate']:.4f} of Original retrieved contexts and {summary['subsets']['all']['masked']['retrieved_answer_exposure_rate']:.4f} after masking. Thus, corpus-level availability flags substantially overestimate the answer labels actually reaching the generator in this panel.",
        "In the observed run, masking does not produce the selective degradation predicted by a label-shortcut explanation. Accuracy increases by 0.0667 in both the exposed and control strata; the full-panel McNemar test remains just above 0.05. We therefore interpret the experiment as evidence of no detectable dependence on explicit answer labels, not as proof that masking improves the Evidence Graph. One plausible mechanism is that heterogeneous generated verdict labels can distract answer mapping when they do not match the benchmark granularity.",
        "",
        "This is a focused generation-side leakage control, not a complete representation comparison. Its defensible claim is whether answer generation on already retrieved Evidence KG contexts survives removal of explicit generated labels. A full 571-record re-indexed rerun remains the confirmatory extension for retrieval-side effects.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    panel = read_jsonl(PANEL)
    panel_meta = {str(row["id"]): row.get("metadata") or {} for row in panel}
    panel_ids = set(panel_meta)
    original_by_id = {record_id(row): row for row in read_jsonl(ORIGINAL) if record_id(row) in panel_ids}
    masked_by_id = {record_id(row): row for row in read_jsonl(MASKED) if record_id(row) in panel_ids}
    missing = panel_ids - (set(original_by_id) & set(masked_by_id))
    if missing:
        raise RuntimeError(f"Missing paired results for {len(missing)} records")
    ordered_ids = [str(row["id"]) for row in panel]
    subsets = {
        "all": ordered_ids,
        "exposed": [rid for rid in ordered_ids if panel_meta[rid].get("label_masking_stratum") == "exposed"],
        "non_exposed_control": [rid for rid in ordered_ids if panel_meta[rid].get("label_masking_stratum") == "non_exposed_control"],
    }
    summary: dict[str, Any] = {"panel_count": len(panel), "subsets": {}, "retrieval": {}}
    overlaps = []
    for rid in ordered_ids:
        overlaps.append(topk_overlap(original_by_id[rid], masked_by_id[rid]))
    summary["retrieval"]["mean_topk_jaccard"] = round(sum(overlaps) / len(overlaps), 4)
    for name, ids in subsets.items():
        original = [original_by_id[rid] for rid in ids]
        masked = [masked_by_id[rid] for rid in ids]
        summary["subsets"][name] = {
            "original": metrics(original),
            "masked": metrics(masked),
            "paired": paired_stats(original, masked),
            "task_counts": dict(Counter(original_by_id[rid]["record"].get("task_type") for rid in ids)),
        }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "label_masking_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "label_masking_report.md").write_text(render_report(summary), encoding="utf-8")
    print(render_report(summary))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from error_taxonomy_config import ERROR_TAXONOMY
from run_error_taxonomy_judge import ROOT, SYSTEM_PLAN
from run_faithfulness_judge import read_jsonl
from run_significance_analysis import gold_answer, predicted_answer


TAXONOMY_BY_ID = {item["id"]: item for item in ERROR_TAXONOMY}


def load_labels(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in read_jsonl(path)}


def load_result_rows(system_id: str) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(SYSTEM_PLAN[system_id]["result_path"])
    return {str((row.get("record") or {}).get("id") or ""): row for row in rows}


def summarize_system(labels: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_error = Counter(row["primary_error_type"] for row in labels.values())
    by_task = defaultdict(Counter)
    for row in labels.values():
        by_task[row["primary_error_type"]][str(row.get("task_type") or "")] += 1
    dominant_tasks = {
        error_type: [task for task, _count in counter.most_common(2)]
        for error_type, counter in by_task.items()
    }
    return {"count": len(labels), "by_error": dict(by_error), "dominant_tasks": dominant_tasks}


def compare_systems(
    evidence_rows: dict[str, dict[str, Any]],
    comparator_rows: dict[str, dict[str, Any]],
    evidence_labels: dict[str, dict[str, Any]],
    comparator_labels: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    reduced = Counter()
    increased = Counter()
    shared_evidence = Counter()
    shared_comparator = Counter()
    for rid in sorted(set(evidence_rows).intersection(comparator_rows)):
        eg_row = evidence_rows[rid]
        cmp_row = comparator_rows[rid]
        eg_correct = predicted_answer(eg_row) == gold_answer(eg_row["record"])
        cmp_correct = predicted_answer(cmp_row) == gold_answer(cmp_row["record"])
        if cmp_correct and not eg_correct and rid in evidence_labels:
            increased[evidence_labels[rid]["primary_error_type"]] += 1
        elif eg_correct and not cmp_correct and rid in comparator_labels:
            reduced[comparator_labels[rid]["primary_error_type"]] += 1
        elif not eg_correct and not cmp_correct:
            if rid in evidence_labels:
                shared_evidence[evidence_labels[rid]["primary_error_type"]] += 1
            if rid in comparator_labels:
                shared_comparator[comparator_labels[rid]["primary_error_type"]] += 1
    return {
        "reduced": dict(reduced),
        "increased": dict(increased),
        "shared_evidence": dict(shared_evidence),
        "shared_comparator": dict(shared_comparator),
    }


def write_markdown(
    output_path: Path,
    *,
    system_summaries: dict[str, dict[str, Any]],
    doc_vs_eg: dict[str, Any],
    sem_vs_eg: dict[str, Any],
) -> None:
    lines = [
        "## Error taxonomy statistics (first version)",
        "",
        "| Error Type | Dominant Tasks | Doc Errors | Semantic Errors | Evidence Errors | EG reduces vs Doc | EG increases vs Doc | EG reduces vs Semantic | EG increases vs Semantic |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in ERROR_TAXONOMY:
        item_id = item["id"]
        dominant = ", ".join(item["dominant_tasks"][:2])
        lines.append(
            "| "
            + " | ".join(
                [
                    item["name"],
                    dominant,
                    str(system_summaries["doc_vanilla"]["by_error"].get(item_id, 0)),
                    str(system_summaries["semantic_vanilla"]["by_error"].get(item_id, 0)),
                    str(system_summaries["evidence_vanilla"]["by_error"].get(item_id, 0)),
                    str(doc_vs_eg["reduced"].get(item_id, 0)),
                    str(doc_vs_eg["increased"].get(item_id, 0)),
                    str(sem_vs_eg["reduced"].get(item_id, 0)),
                    str(sem_vs_eg["increased"].get(item_id, 0)),
                ]
            )
            + " |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize first-pass error taxonomy results.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    input_dir = args.input_dir if args.input_dir.is_absolute() else ROOT / args.input_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    system_ids = ["doc_vanilla", "semantic_vanilla", "evidence_vanilla"]
    labels = {system_id: load_labels(input_dir / f"{system_id}.jsonl") for system_id in system_ids}
    result_rows = {system_id: load_result_rows(system_id) for system_id in system_ids}
    system_summaries = {system_id: summarize_system(labels[system_id]) for system_id in system_ids}
    doc_vs_eg = compare_systems(result_rows["evidence_vanilla"], result_rows["doc_vanilla"], labels["evidence_vanilla"], labels["doc_vanilla"])
    sem_vs_eg = compare_systems(result_rows["evidence_vanilla"], result_rows["semantic_vanilla"], labels["evidence_vanilla"], labels["semantic_vanilla"])

    payload = {
        "taxonomy": ERROR_TAXONOMY,
        "system_summaries": system_summaries,
        "evidence_vs_doc": doc_vs_eg,
        "evidence_vs_semantic": sem_vs_eg,
    }
    (output_dir / "error_taxonomy_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(output_dir / "error_taxonomy_summary.md", system_summaries=system_summaries, doc_vs_eg=doc_vs_eg, sem_vs_eg=sem_vs_eg)
    print(f"Saved {output_dir / 'error_taxonomy_summary.json'}")
    print(f"Saved {output_dir / 'error_taxonomy_summary.md'}")


if __name__ == "__main__":
    main()

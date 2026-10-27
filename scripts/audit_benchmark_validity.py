#!/usr/bin/env python3
"""Audit the 571-item benchmark and prepare a blinded human-review panel."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "dataset/evaluated/external_reasoning_benchmark_formal_v3_571.jsonl"
OUT_DIR = ROOT / "dataset/evaluated/rebuilt/benchmark_validity_audit"
REPORT_JSON = OUT_DIR / "automatic_audit.json"
REPORT_MD = OUT_DIR / "automatic_audit.md"
PANEL_JSONL = OUT_DIR / "blinded_review_panel_120.jsonl"
REVIEW_CSV = OUT_DIR / "blinded_review_form_120.csv"
REVIEWER_PACKET = OUT_DIR / "reviewer_packet_blinded_120.csv"

INTERNAL_KB_SOURCES = {"tcga", "pubmed"}


def stable_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_rows() -> list[dict]:
    return [json.loads(line) for line in BENCHMARK.read_text().splitlines() if line.strip()]


def source_names(row: dict) -> list[str]:
    annotation = row.get("benchmark_annotation") or {}
    names = annotation.get("data_sources") or (row.get("metadata") or {}).get("data_sources") or []
    if names:
        return [str(name).strip() for name in names if str(name).strip()]
    raw = str((row.get("provenance") or {}).get("source_dataset") or "")
    return [part.strip() for part in re.split(r"[/,]", raw) if part.strip()]


def normalized_template(row: dict) -> str:
    question = str(row.get("question") or "").lower()
    entities = row.get("entities") or {}
    replacements = []
    for value in entities.values():
        if isinstance(value, str) and value.strip():
            replacements.append(value.strip().lower())
    focus = str(row.get("oncology_focus") or "").strip().lower()
    if focus:
        replacements.append(focus)
    for value in sorted(set(replacements), key=len, reverse=True):
        question = re.sub(rf"\b{re.escape(value)}\b", "<entity>", question)
    return re.sub(r"\s+", " ", question).strip()


def norm_answer(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    return text.rstrip(". ;")


def answer_parts(value: object) -> list[str]:
    return [norm_answer(part) for part in str(value or "").split(";") if norm_answer(part)]


def allocate_panel(rows: list[dict], n: int = 120) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("task_type"))].append(row)

    total = len(rows)
    quotas = {key: max(3, round(n * len(items) / total)) for key, items in groups.items()}
    while sum(quotas.values()) > n:
        key = max((k for k in quotas if quotas[k] > 3), key=lambda k: quotas[k])
        quotas[key] -= 1
    while sum(quotas.values()) < n:
        key = max(groups, key=lambda k: len(groups[k]) - quotas[k])
        quotas[key] += 1

    panel = []
    for task, items in sorted(groups.items()):
        ranked = sorted(
            items,
            key=lambda row: stable_key(
                f"benchmark-validity-v1|{task}|{row.get('id')}|"
                f"{(row.get('metadata') or {}).get('priority_cancer_family')}|{row.get('answer')}"
            ),
        )
        panel.extend(ranked[: quotas[task]])
    return sorted(panel, key=lambda row: stable_key(str(row.get("id"))))


def main() -> None:
    rows = read_rows()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ids = [str(row.get("id") or "") for row in rows]
    questions = [str(row.get("question") or "") for row in rows]
    templates = [normalized_template(row) for row in rows]

    consistency = Counter()
    for row in rows:
        gold = row.get("gold_standard") or {}
        metadata = row.get("metadata") or {}
        answer = norm_answer(row.get("answer"))
        allowed = {
            norm_answer(value)
            for value in (
                list(metadata.get("eval_answer_options") or [])
                + list(gold.get("acceptable_answers") or [])
                + [gold.get("normalized_answer"), gold.get("canonical_answer"), gold.get("class_label")]
            )
            if norm_answer(value)
        }
        parts = answer_parts(row.get("answer"))
        entities = row.get("entities") or {}
        q_lower = str(row.get("question") or "").lower()
        names = source_names(row)
        consistency["gold_supports_record_answer"] += int(bool(answer) and (answer in allowed or (parts and all(part in allowed for part in parts))))
        consistency["answer_allowed"] += int(not allowed or answer in allowed or (parts and all(part in allowed for part in parts)))
        consistency["machine_evaluable"] += int(bool(row.get("is_machine_evaluable")))
        consistency["source_record_present"] += int(bool((row.get("provenance") or {}).get("source_record_id")))
        consistency["source_dataset_present"] += int(bool(names))
        consistency["entity_anchor_present"] += int(any(str(v).lower() in q_lower for v in entities.values() if isinstance(v, str) and v))
        consistency["declared_record_source_separation"] += int(not ({name.lower() for name in names} & INTERNAL_KB_SOURCES))

    template_counts = Counter(templates)
    source_counts = Counter(name for row in rows for name in source_names(row))
    report = {
        "benchmark": str(BENCHMARK.relative_to(ROOT)),
        "sha256": hashlib.sha256(BENCHMARK.read_bytes()).hexdigest(),
        "n": len(rows),
        "unique_ids": len(set(ids)),
        "unique_questions": len(set(questions)),
        "conflicting_exact_questions": sum(
            1
            for question in set(questions)
            if len({str(row.get("answer")) for row in rows if row.get("question") == question}) > 1
        ),
        "task_distribution": dict(Counter(str(row.get("task_type")) for row in rows)),
        "cancer_distribution": dict(Counter(str((row.get("metadata") or {}).get("priority_cancer_family")) for row in rows)),
        "answer_type_distribution": dict(Counter(str(row.get("answer_type")) for row in rows)),
        "gold_source_distribution": dict(source_counts),
        "automatic_consistency": {key: {"count": value, "rate": value / len(rows)} for key, value in consistency.items()},
        "template_audit": {
            "unique_normalized_templates": len(template_counts),
            "largest_template_size": max(template_counts.values()),
            "templates_with_multiple_items": sum(value > 1 for value in template_counts.values()),
            "top_templates": template_counts.most_common(12),
        },
        "interpretation_limits": [
            "Declared source separation is record-level, not proof of semantic independence from literature used by curated databases.",
            "Automatic consistency checks verify construction integrity, not biomedical correctness.",
            "The blinded panel is an internal review instrument until completed by independent domain experts.",
        ],
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    panel = allocate_panel(rows)
    with PANEL_JSONL.open("w") as handle:
        for index, row in enumerate(panel, 1):
            blinded = {
                "review_id": f"BR-{index:03d}",
                "question": row.get("question"),
                "answer_type": row.get("answer_type"),
                "allowed_values": (row.get("metadata") or {}).get("eval_answer_options") or [],
                "gold_answer_for_adjudicator_only": row.get("answer"),
                "source_evidence_for_adjudicator_only": row.get("evidence_text"),
                "source_record_for_adjudicator_only": row.get("provenance"),
                "stratum": {
                    "task_type": row.get("task_type"),
                    "cancer_family": (row.get("metadata") or {}).get("priority_cancer_family"),
                },
            }
            handle.write(json.dumps(blinded, ensure_ascii=False) + "\n")

    fields = [
        "review_id", "scientifically_well_formed_0_1", "gold_supported_0_1",
        "answer_identifiable_0_1", "requires_evidence_integration_0_1",
        "representation_neutral_0_1", "leakage_risk_0_1", "confidence_1_5", "notes",
    ]
    with REVIEW_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(1, len(panel) + 1):
            writer.writerow({"review_id": f"BR-{index:03d}"})

    packet_fields = ["review_id", "question", "answer_type", "allowed_values", "task_type", "cancer_family"]
    with REVIEWER_PACKET.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=packet_fields)
        writer.writeheader()
        for index, row in enumerate(panel, 1):
            writer.writerow({
                "review_id": f"BR-{index:03d}",
                "question": row.get("question"),
                "answer_type": row.get("answer_type"),
                "allowed_values": " | ".join((row.get("metadata") or {}).get("eval_answer_options") or []),
                "task_type": row.get("task_type"),
                "cancer_family": (row.get("metadata") or {}).get("priority_cancer_family"),
            })

    c = report["automatic_consistency"]
    md = f"""# Benchmark Validity Automatic Audit

## Scope

This report audits construction integrity for the frozen 571-item benchmark. It does not claim independent human validation.

## Reproducibility

- Benchmark SHA-256: `{report['sha256']}`
- Records: {len(rows)}
- Unique ids/questions: {report['unique_ids']}/{report['unique_questions']}
- Conflicting exact questions: {report['conflicting_exact_questions']}

## Integrity Checks

- Record answer is supported by the gold answer/alias set: {c['gold_supports_record_answer']['count']}/{len(rows)} ({c['gold_supports_record_answer']['rate']:.1%})
- Answer satisfies declared answer set: {c['answer_allowed']['count']}/{len(rows)} ({c['answer_allowed']['rate']:.1%})
- Machine evaluable: {c['machine_evaluable']['count']}/{len(rows)} ({c['machine_evaluable']['rate']:.1%})
- Source record present: {c['source_record_present']['count']}/{len(rows)} ({c['source_record_present']['rate']:.1%})
- Source dataset present: {c['source_dataset_present']['count']}/{len(rows)} ({c['source_dataset_present']['rate']:.1%})
- Entity anchor appears in question: {c['entity_anchor_present']['count']}/{len(rows)} ({c['entity_anchor_present']['rate']:.1%})
- Declared gold source differs from TCGA/PubMed: {c['declared_record_source_separation']['count']}/{len(rows)} ({c['declared_record_source_separation']['rate']:.1%})

## Template Audit

- Unique normalized templates: {report['template_audit']['unique_normalized_templates']}
- Largest template family: {report['template_audit']['largest_template_size']} items
- Multi-item template families: {report['template_audit']['templates_with_multiple_items']}

## Blinded Review Instrument

A deterministic, task-stratified 120-item panel and blank two-reviewer form were generated. Until completed by independent domain experts, this is an internal pilot instrument and must not be reported as human validation.

## Limits

- Record-level source separation does not prove semantic independence.
- Construction consistency does not establish biomedical correctness.
- Template diversity does not establish clinical representativeness.
"""
    REPORT_MD.write_text(md)


if __name__ == "__main__":
    main()

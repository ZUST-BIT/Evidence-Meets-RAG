#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import csv
import gzip
import io
import json
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_external_mechanism_benchmark as bem  # noqa: E402


HISTORICAL_COUNTS_PATH = ROOT / "dataset" / "evaluated" / "construction" / "historical_seed_task_counts.json"
HQ85_PATH = ROOT / "dataset" / "evaluated" / "construction" / "formal_seed_73.jsonl"
REFERENCE_PATH = ROOT / "dataset" / "evaluated" / "external_reasoning_benchmark_formal_v3_571.jsonl"
OUTPUT_PATH = ROOT / "dataset" / "evaluated" / "rebuilt" / "external_reasoning_benchmark_formal_v3_571.jsonl"
SUMMARY_PATH = ROOT / "dataset" / "evaluated" / "rebuilt" / "construction_summary.json"
RAW_DIR = ROOT / "dataset" / "raw"
CLINVAR_DIR = ROOT / "dataset" / "raw" / "clinvar"
CLINVAR_FILTERED_PATH = CLINVAR_DIR / "clinvar_somatic_cancer_submission_subset.jsonl"

SUPPORTED_LABELS_PRECISION = [
    "supports_sensitivity",
    "supports_resistance",
    "does_not_support_sensitivity",
    "does_not_support_resistance",
    "supports_adverse_outcome",
    "supports_uncertain_significance",
]

BUNDLE_V2_LABELS = [
    "causal_relation+pathway_context+target_association",
    "causal_relation+target_association",
    "pathway_context+target_association",
    "target_association_only",
]

FINAL_TARGETS = {
    "mechanism_trace_intermediate_gene": 100,
    "cross_source_corroboration_status": 100,
    "precision_oncology_evidence": 100,
    "mechanism_bioasq_total": 19,
    "cross_source_support_bundle": 81,
    "evidence_signal_interpretation_total": 100,
    "hierarchical_evidence_aggregation_total": 93,
}

REMOVED_TASKS = {
    "driver_role_assignment",
    "pathway_class_membership",
    "target_disease_association_strength",
    "cgc_support_strength",
    "intogen_support_strength",
    "cancer_cell_line_annotation",
    "precision_oncology_assertion",
}

EVIDENCE_SIGNAL_LABELS = [
    "diagnostic_positive",
    "diagnostic_negative",
    "prognostic_poor_outcome",
    "prognostic_better_outcome",
    "oncogenic_oncogenicity",
]

ASSERTION_SIGNIFICANCE_LABELS = [
    "positive",
    "poor_outcome",
    "better_outcome",
    "oncogenic",
    "likely_oncogenic",
    "uncertain_significance",
    "resistance",
    "sensitivityresponse",
]

ASSERTION_AMP_LABELS = [
    "tier_i_level_a",
    "tier_i_level_b",
    "tier_ii_level_c",
    "tier_ii_level_d",
    "tier_iii",
]

HIER_GROUP_LABELS = [
    "consistent_supports_sensitivity",
    "consistent_supports_resistance",
    "consistent_does_not_support_sensitivity",
    "consistent_does_not_support_resistance",
    "conflicting_verdicts",
]

CLINVAR_AGG_LABELS = [
    "consistent_somatic_tier",
    "conflicting_interpretations",
    "complementary_somatic_and_oncogenic",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def family_of(row: dict[str, Any]) -> str:
    return str((row.get("metadata") or {}).get("priority_cancer_family") or "unknown")


def sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        metadata = row.get("metadata") or {}
        return (
            -safe_int(metadata.get("causal_suitability_score"), 0),
            family_of(row),
            str(row.get("answer") or ""),
            str(row.get("id") or ""),
        )

    return sorted(rows, key=key)


def round_robin_sample(
    rows: list[dict[str, Any]],
    target_n: int,
    group_key: Callable[[dict[str, Any]], Any],
) -> list[dict[str, Any]]:
    buckets: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in sort_rows(rows):
        buckets[group_key(row)].append(row)
    ordered_keys = sorted(buckets, key=lambda value: str(value))

    picked: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    while ordered_keys and len(picked) < target_n:
        next_keys: list[Any] = []
        for key in ordered_keys:
            bucket = buckets[key]
            while bucket and bucket[0]["id"] in seen_ids:
                bucket.pop(0)
            if not bucket:
                continue
            row = bucket.pop(0)
            picked.append(row)
            seen_ids.add(row["id"])
            if bucket:
                next_keys.append(key)
            if len(picked) >= target_n:
                break
        ordered_keys = next_keys
    return picked


def attach_selection_metadata(
    rows: list[dict[str, Any]],
    subset_profile: str,
    action: str,
) -> list[dict[str, Any]]:
    for row in rows:
        metadata = row.setdefault("metadata", {})
        metadata["subset_origin"] = "formal_reasoning_v2"
        metadata["subset_profile"] = subset_profile
        metadata["diagnosis_action"] = action
    return rows


def evidence_level_rank(row: dict[str, Any]) -> int:
    level = str((row.get("metadata") or {}).get("civic_evidence_level") or "").upper()
    order = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}
    return order.get(level, 0)


def dedupe_precision_candidates_by_question(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("question") or "")].append(row)

    kept: list[dict[str, Any]] = []
    dropped_conflicts = 0
    dropped_same_answer_duplicates = 0
    for question, items in grouped.items():
        labels = {str(item.get("answer") or "") for item in items}
        if len(labels) > 1:
            dropped_conflicts += len(items)
            continue
        chosen = sorted(
            items,
            key=lambda item: (
                -evidence_level_rank(item),
                -len(str(item.get("evidence_text") or "")),
                str(item.get("id") or ""),
            ),
        )[0]
        kept.append(chosen)
        dropped_same_answer_duplicates += max(0, len(items) - 1)
    return kept, {
        "dropped_conflicting_rows": dropped_conflicts,
        "dropped_same_answer_duplicate_rows": dropped_same_answer_duplicates,
    }


def precision_label_from_text(answer_text: str) -> str | None:
    text = str(answer_text or "").strip().lower()
    if not text:
        return None
    if "does not support" in text and "sensitivityresponse" in text:
        return "does_not_support_sensitivity"
    if "does not support" in text and "resistance" in text:
        return "does_not_support_resistance"
    if "supports sensitivityresponse" in text:
        return "supports_sensitivity"
    if "supports resistance" in text:
        return "supports_resistance"
    if "supports poor_outcome" in text:
        return "supports_adverse_outcome"
    if "supports uncertain_significance" in text:
        return "supports_uncertain_significance"
    return None


def transform_existing_precision_record(row: dict[str, Any]) -> dict[str, Any]:
    transformed = copy.deepcopy(row)
    label = precision_label_from_text(transformed.get("answer"))
    if label is None:
        raise ValueError(f"Unable to map precision oncology answer: {transformed.get('id')}")
    metadata = transformed.setdefault("metadata", {})
    metadata["verbatim_answer"] = transformed.get("answer")
    metadata["precision_oncology_schema"] = "civic_evidence_label_v2"
    metadata["eval_answer_options"] = list(SUPPORTED_LABELS_PRECISION)
    metadata["qa_evidence_function"] = "reasoning"
    metadata["source_dataset"] = "CIViC"
    metadata["source_family"] = "precision_oncology"
    transformed["answer"] = label
    transformed["answer_type"] = "categorical"
    gold = bem.build_gold_standard(
        "categorical",
        canonical_answer=label,
        acceptable_answers=[label],
        class_label=label,
        evaluable=True,
    )
    transformed["gold_standard"] = gold
    return transformed


def build_expanded_precision_records() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(RAW_DIR / "civic" / "evidence_items.jsonl", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("status") != "ACCEPTED":
                continue
            if item.get("variantOrigin") != "SOMATIC":
                continue
            if item.get("evidenceType") != "PREDICTIVE":
                continue
            disease = item.get("disease") or {}
            disease_name = str(disease.get("name") or "").strip()
            family = bem.infer_priority_family(disease_name)
            if family not in bem.SUPPORTED_FAMILIES:
                continue
            significance = str(item.get("significance") or "").strip().upper()
            direction = str(item.get("evidenceDirection") or "").strip().upper()
            if significance not in {"SENSITIVITYRESPONSE", "RESISTANCE"}:
                continue
            if direction not in {"SUPPORTS", "DOES_NOT_SUPPORT"}:
                continue
            level = str(item.get("evidenceLevel") or "").strip().upper()
            if level not in {"A", "B"}:
                continue
            molecular_profile = item.get("molecularProfile") or {}
            molecular_name = str(molecular_profile.get("name") or "").strip()
            therapies = item.get("therapies") or []
            therapy_names = [str(entry.get("name") or "").strip() for entry in therapies if str(entry.get("name") or "").strip()]
            if not molecular_name or not therapy_names:
                continue

            label = (
                ("supports" if direction == "SUPPORTS" else "does_not_support")
                + ("_sensitivity" if significance == "SENSITIVITYRESPONSE" else "_resistance")
            )
            question = (
                f"In CIViC, what evidence verdict is supported for {molecular_name} with "
                f"{', '.join(therapy_names[:2])} in {disease_name}?"
            )
            evidence_text = str(item.get("description") or "").strip()
            gene_symbols = bem.extract_gene_symbols(question, evidence_text, molecular_name)
            record = {
                "id": f"precision-oncology-evidence-v2-civic-{item.get('id')}",
                "source": "civic",
                "task_type": "precision_oncology_evidence",
                "question": question,
                "answer": label,
                "answer_type": "categorical",
                "oncology_focus": bem.family_label(family),
                "evidence_text": evidence_text,
                "entities": {
                    "disease": disease_name,
                    "gene_symbols": gene_symbols,
                    "molecular_profile": molecular_name,
                    "therapy_names": therapy_names,
                },
                "provenance": {
                    "source_record_id": item.get("id"),
                    "source_dataset": "CIViC",
                    "source_link": item.get("link"),
                },
                "metadata": {
                    "priority_cancer_family": family,
                    "eval_answer_options": list(SUPPORTED_LABELS_PRECISION),
                    "qa_evidence_function": "reasoning",
                    "scenario_tags": ["molecular", "target"],
                    "precision_oncology_schema": "civic_evidence_label_v2",
                    "civic_evidence_level": level,
                    "civic_significance": significance.lower(),
                    "civic_direction": direction.lower(),
                    "civic_evidence_type": str(item.get("evidenceType") or "").lower(),
                },
            }
            rows.append(
                bem.finalize_record(
                    bem.attach_gold_standard(
                        record,
                        bem.build_gold_standard(
                            "categorical",
                            canonical_answer=label,
                            acceptable_answers=[label],
                            class_label=label,
                            evaluable=True,
                        ),
                    ),
                    source_dataset="CIViC",
                    source_family="precision_oncology",
                )
            )
    return rows


def build_bundle_v2_records() -> list[dict[str, Any]]:
    signor_df, _ = bem.materialize_signor_raw()
    pathway_rows, _ = bem.materialize_pathwaycommons_raw()
    intogen_df, _ = bem.materialize_intogen_raw()
    open_targets_payload, _ = bem.materialize_open_targets_raw()
    support_index = bem.build_external_support_index(intogen_df, open_targets_payload)

    signor_genes = set()
    for row in signor_df.to_dict(orient="records"):
        signor_genes.add(str(row.get("ENTITYA") or "").strip().upper())
        signor_genes.add(str(row.get("ENTITYB") or "").strip().upper())
    pathway_genes = {gene for row in pathway_rows for gene in row["genes"]}

    records: list[dict[str, Any]] = []
    for (family, gene), layers in sorted(support_index.items()):
        if family not in bem.SUPPORTED_FAMILIES:
            continue
        if not any(layer.startswith("open_targets_") for layer in layers):
            continue
        has_causal = gene in signor_genes
        has_pathway = gene in pathway_genes
        if has_causal and has_pathway:
            answer = "causal_relation+pathway_context+target_association"
        elif has_causal:
            answer = "causal_relation+target_association"
        elif has_pathway:
            answer = "pathway_context+target_association"
        else:
            answer = "target_association_only"

        present_layers = ["target_association"]
        if has_causal:
            present_layers.insert(0, "causal_relation")
        if has_pathway:
            if "causal_relation" in present_layers:
                present_layers.insert(1, "pathway_context")
            else:
                present_layers.insert(0, "pathway_context")
        record = {
            "id": f"cross-source-support-bundle-v2-{family}-{gene}",
            "source": "external_multisource",
            "task_type": "cross_source_support_bundle",
            "question": (
                f"Which external evidence bundle supports {gene} in {bem.family_label(family)} "
                f"when driver-role labels are excluded and only target-association, causal-relation, "
                f"and pathway-context signals are counted?"
            ),
            "answer": answer,
            "answer_type": "categorical",
            "oncology_focus": bem.family_label(family),
            "evidence_text": bem.trim_text(
                f"External support layers for {gene} in {bem.family_label(family)} include "
                f"{', '.join(present_layers)}."
            ),
            "entities": {
                "disease": bem.family_label(family),
                "gene_symbol": gene,
            },
            "provenance": {
                "source_record_id": f"{family}:{gene}",
                "source_dataset": "IntOGen/Open Targets/SIGNOR/Pathway Commons",
            },
            "metadata": {
                "priority_cancer_family": family,
                "external_support_layers": present_layers,
                "scenario_tags": ["mechanism", "molecular", "target"],
                "qa_evidence_function": "reasoning",
                "eval_answer_options": list(BUNDLE_V2_LABELS),
                "bundle_schema": "support_bundle_v2_no_driver",
            },
        }
        records.append(
            bem.finalize_record(
                bem.attach_gold_standard(
                    record,
                    bem.build_gold_standard(
                        "categorical",
                        canonical_answer=answer,
                        acceptable_answers=[answer],
                        class_label=answer,
                        evaluable=True,
                    ),
                ),
                source_dataset="External Multisource",
                source_family="cross_source_support",
            )
        )
    return records


def build_evidence_signal_interpretation_records() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(RAW_DIR / "civic" / "evidence_items.jsonl", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("status") != "ACCEPTED":
                continue
            if item.get("variantOrigin") != "SOMATIC":
                continue
            evidence_type = str(item.get("evidenceType") or "").strip().upper()
            if evidence_type not in {"DIAGNOSTIC", "PROGNOSTIC", "ONCOGENIC"}:
                continue
            disease = item.get("disease") or {}
            disease_name = str(disease.get("name") or "").strip()
            family = bem.infer_priority_family(disease_name)
            if family not in bem.SUPPORTED_FAMILIES:
                continue
            molecular_profile = item.get("molecularProfile") or {}
            molecular_name = str(molecular_profile.get("name") or "").strip()
            if not molecular_name:
                continue
            significance = str(item.get("significance") or "").strip().upper()
            label = None
            if evidence_type == "DIAGNOSTIC" and significance in {"POSITIVE", "NEGATIVE"}:
                label = f"diagnostic_{significance.lower()}"
            elif evidence_type == "PROGNOSTIC" and significance in {"POOR_OUTCOME", "BETTER_OUTCOME"}:
                label = f"prognostic_{significance.lower()}"
            elif evidence_type == "ONCOGENIC" and significance == "ONCOGENICITY":
                label = "oncogenic_oncogenicity"
            if label is None:
                continue
            evidence_text = str(item.get("description") or "").strip()
            if not evidence_text:
                continue
            gene_symbols = bem.extract_gene_symbols(molecular_name, evidence_text)
            record = {
                "id": f"civic-signal-interpretation-{item.get('id')}",
                "source": "civic",
                "task_type": "evidence_signal_interpretation",
                "question": f"In CIViC, what evidence signal is supported for {molecular_name} in {disease_name}?",
                "answer": label,
                "answer_type": "categorical",
                "oncology_focus": bem.family_label(family),
                "evidence_text": evidence_text,
                "entities": {
                    "disease": disease_name,
                    "gene_symbols": gene_symbols,
                    "molecular_profile": molecular_name,
                },
                "provenance": {
                    "source_record_id": item.get("id"),
                    "source_dataset": "CIViC",
                    "source_link": item.get("link"),
                },
                "metadata": {
                    "priority_cancer_family": family,
                    "eval_answer_options": list(EVIDENCE_SIGNAL_LABELS),
                    "qa_evidence_function": "reasoning",
                    "scenario_tags": ["molecular"],
                    "civic_evidence_type": evidence_type.lower(),
                    "civic_significance": significance.lower(),
                    "signal_schema": "civic_signal_interpretation_v1",
                },
            }
            rows.append(
                bem.finalize_record(
                    bem.attach_gold_standard(
                        record,
                        bem.build_gold_standard(
                            "categorical",
                            canonical_answer=label,
                            acceptable_answers=[label],
                            class_label=label,
                            evaluable=True,
                        ),
                    ),
                    source_dataset="CIViC",
                    source_family="evidence_signal_interpretation",
                )
            )
    return rows


def build_hierarchical_assertion_records() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(RAW_DIR / "civic" / "assertions.jsonl", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            disease = item.get("disease") or {}
            disease_name = str(disease.get("name") or "").strip()
            family = bem.infer_priority_family(disease_name)
            if family not in bem.SUPPORTED_FAMILIES:
                continue
            evidence_items_count = safe_int(item.get("evidenceItemsCount"))
            if evidence_items_count < 2:
                continue
            molecular_name = str((item.get("molecularProfile") or {}).get("name") or "").strip()
            if not molecular_name:
                continue
            therapies = [str(entry.get("name") or "").strip() for entry in (item.get("therapies") or []) if str(entry.get("name") or "").strip()]
            assertion_type = str(item.get("assertionType") or "").strip().upper()
            significance = str(item.get("significance") or "").strip().upper()
            amp_level = str(item.get("ampLevel") or "").strip().upper()
            evidence_text = str(item.get("description") or "").strip()
            if not evidence_text:
                continue

            if assertion_type == "PREDICTIVE" and amp_level:
                answer = amp_level.lower()
                question = (
                    f"In CIViC, what aggregated AMP/ASCO/CAP tier is assigned to {molecular_name} "
                    f"with {', '.join(therapies[:2]) or 'the referenced therapy'} in {disease_name}?"
                )
                allowed = ASSERTION_AMP_LABELS
                subtask = "assertion_amp_tier"
            elif assertion_type in {"DIAGNOSTIC", "PROGNOSTIC", "ONCOGENIC"} and significance:
                answer = significance.lower()
                if answer not in ASSERTION_SIGNIFICANCE_LABELS:
                    continue
                question = f"In CIViC, what aggregated assertion significance is assigned to {molecular_name} in {disease_name}?"
                allowed = ASSERTION_SIGNIFICANCE_LABELS
                subtask = "assertion_significance"
            else:
                continue
            gene_symbols = bem.extract_gene_symbols(question, evidence_text, molecular_name)
            record = {
                "id": f"civic-hier-assertion-{item.get('id')}",
                "source": "civic",
                "task_type": "hierarchical_evidence_aggregation",
                "question": question,
                "answer": answer,
                "answer_type": "categorical",
                "oncology_focus": bem.family_label(family),
                "evidence_text": bem.trim_text(
                    f"Aggregated from {evidence_items_count} CIViC evidence items. {evidence_text}"
                ),
                "entities": {
                    "disease": disease_name,
                    "gene_symbols": gene_symbols,
                    "molecular_profile": molecular_name,
                    "therapy_names": therapies,
                },
                "provenance": {
                    "source_record_id": item.get("id"),
                    "source_dataset": "CIViC",
                    "source_link": item.get("link"),
                },
                "metadata": {
                    "priority_cancer_family": family,
                    "eval_answer_options": list(allowed),
                    "qa_evidence_function": "reasoning",
                    "scenario_tags": ["molecular", "target"],
                    "aggregation_schema": "civic_assertion_aggregation_v1",
                    "aggregation_subtask": subtask,
                    "aggregation_evidence_count": evidence_items_count,
                    "civic_assertion_type": assertion_type.lower(),
                },
            }
            rows.append(
                bem.finalize_record(
                    bem.attach_gold_standard(
                        record,
                        bem.build_gold_standard(
                            "categorical",
                            canonical_answer=answer,
                            acceptable_answers=[answer],
                            class_label=answer,
                            evaluable=True,
                        ),
                    ),
                    source_dataset="CIViC",
                    source_family="hierarchical_aggregation",
                )
            )
    return rows


def build_hierarchical_group_records() -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, tuple[str, ...]], list[dict[str, Any]]] = defaultdict(list)
    with open(RAW_DIR / "civic" / "evidence_items.jsonl", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("status") != "ACCEPTED" or item.get("variantOrigin") != "SOMATIC":
                continue
            if str(item.get("evidenceType") or "").strip().upper() != "PREDICTIVE":
                continue
            disease_name = str((item.get("disease") or {}).get("name") or "").strip()
            family = bem.infer_priority_family(disease_name)
            if family not in bem.SUPPORTED_FAMILIES:
                continue
            molecular_name = str((item.get("molecularProfile") or {}).get("name") or "").strip()
            therapies = tuple(sorted(str(entry.get("name") or "").strip() for entry in (item.get("therapies") or []) if str(entry.get("name") or "").strip()))
            if not molecular_name or not therapies:
                continue
            significance = str(item.get("significance") or "").strip().upper()
            direction = str(item.get("evidenceDirection") or "").strip().upper()
            level = str(item.get("evidenceLevel") or "").strip().upper()
            if significance not in {"SENSITIVITYRESPONSE", "RESISTANCE"}:
                continue
            if direction not in {"SUPPORTS", "DOES_NOT_SUPPORT"}:
                continue
            if level not in {"A", "B", "C"}:
                continue
            grouped[(family, disease_name, molecular_name, therapies)].append(
                {
                    "id": item.get("id"),
                    "level": level,
                    "direction": direction,
                    "significance": significance,
                    "description": str(item.get("description") or "").strip(),
                }
            )

    rows: list[dict[str, Any]] = []
    for (family, disease_name, molecular_name, therapies), items in grouped.items():
        levels = {item["level"] for item in items}
        verdicts = {(item["direction"], item["significance"]) for item in items}
        if len(items) < 2 or (len(levels) < 2 and len(verdicts) < 2):
            continue
        if len(verdicts) == 1:
            direction, significance = next(iter(verdicts))
            answer = (
                f"consistent_{'supports' if direction == 'SUPPORTS' else 'does_not_support'}_"
                f"{'sensitivity' if significance == 'SENSITIVITYRESPONSE' else 'resistance'}"
            )
        else:
            answer = "conflicting_verdicts"
        evidence_summaries = []
        for item in sorted(items, key=lambda entry: (entry["level"], entry["id"]))[:4]:
            label = f"{item['direction'].lower()} {item['significance'].lower()} level {item['level']}"
            text = item["description"][:180]
            evidence_summaries.append(f"{label}: {text}")
        question = (
            f"Across multiple CIViC evidence items, what aggregate evidence pattern best describes "
            f"{molecular_name} with {', '.join(therapies[:2])} in {disease_name}?"
        )
        gene_symbols = bem.extract_gene_symbols(question, molecular_name, " ".join(evidence_summaries))
        record = {
            "id": f"civic-hier-group-{family}-{re.sub(r'[^A-Za-z0-9]+', '-', molecular_name)[:40]}-{int(hashlib.sha256(json.dumps([family, disease_name, molecular_name, therapies], ensure_ascii=False).encode()).hexdigest()[:12], 16) % 10**8}",
            "source": "civic",
            "task_type": "hierarchical_evidence_aggregation",
            "question": question,
            "answer": answer,
            "answer_type": "categorical",
            "oncology_focus": bem.family_label(family),
            "evidence_text": bem.trim_text(
                f"Grouped from {len(items)} CIViC evidence items across levels {', '.join(sorted(levels))}. "
                + " | ".join(evidence_summaries)
            ),
            "entities": {
                "disease": disease_name,
                "gene_symbols": gene_symbols,
                "molecular_profile": molecular_name,
                "therapy_names": list(therapies),
            },
            "provenance": {
                "source_record_id": ",".join(str(item["id"]) for item in items),
                "source_dataset": "CIViC",
            },
            "metadata": {
                "priority_cancer_family": family,
                "eval_answer_options": list(HIER_GROUP_LABELS),
                "qa_evidence_function": "reasoning",
                "scenario_tags": ["molecular", "target"],
                "aggregation_schema": "civic_grouped_evidence_v1",
                "aggregation_evidence_count": len(items),
                "aggregation_evidence_levels": sorted(levels),
            },
        }
        rows.append(
            bem.finalize_record(
                bem.attach_gold_standard(
                    record,
                    bem.build_gold_standard(
                        "categorical",
                        canonical_answer=answer,
                        acceptable_answers=[answer],
                        class_label=answer,
                        evaluable=True,
                    ),
                ),
                source_dataset="CIViC",
                source_family="hierarchical_aggregation",
            )
        )
    return rows


def materialize_clinvar_somatic_subset() -> list[dict[str, Any]]:
    bem.ensure_dir(CLINVAR_DIR)
    if CLINVAR_FILTERED_PATH.exists():
        return read_jsonl(CLINVAR_FILTERED_PATH)
    url = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/submission_summary.txt.gz"
    response = urllib.request.urlopen(url)
    stream = gzip.GzipFile(fileobj=response)
    text = io.TextIOWrapper(stream, encoding="utf-8", errors="ignore")
    header: list[str] | None = None
    kept: list[dict[str, Any]] = []
    for line in text:
        if line.startswith("##"):
            continue
        if line.startswith("#VariationID\t"):
            header = line.lstrip("#").rstrip("\n").split("\t")
            continue
        if header is None:
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) != len(header):
            continue
        row = dict(zip(header, parts))
        family = bem.infer_priority_family(row.get("SubmittedPhenotypeInfo", ""), row.get("ReportedPhenotypeInfo", ""))
        if family not in bem.SUPPORTED_FAMILIES:
            continue
        sci = str(row.get("SomaticClinicalImpact") or "").strip()
        onc = str(row.get("Oncogenicity") or "").strip()
        if sci in {"", "-"} and onc in {"", "-"}:
            continue
        kept.append(
            {
                "family": family,
                "variation_id": str(row.get("VariationID") or "").strip(),
                "gene": str(row.get("SubmittedGeneSymbol") or "").strip() or "-",
                "submitted_phenotype": str(row.get("SubmittedPhenotypeInfo") or "").strip(),
                "reported_phenotype": str(row.get("ReportedPhenotypeInfo") or "").strip(),
                "review_status": str(row.get("ReviewStatus") or "").strip(),
                "somatic_clinical_impact": sci,
                "oncogenicity": onc,
                "description": str(row.get("Description") or "").strip(),
            }
        )
    write_jsonl(CLINVAR_FILTERED_PATH, kept)
    return kept


def build_clinvar_hierarchical_records() -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in materialize_clinvar_somatic_subset():
        grouped[(row["family"], row["gene"], row["variation_id"])].append(row)

    rows: list[dict[str, Any]] = []
    for (family, gene, variation_id), items in grouped.items():
        if len(items) < 2:
            continue
        sci_values = {item["somatic_clinical_impact"] for item in items if item["somatic_clinical_impact"] not in {"", "-"}}
        onc_values = {item["oncogenicity"] for item in items if item["oncogenicity"] not in {"", "-"}}
        if len(sci_values) >= 2 or len(onc_values) >= 2:
            answer = "conflicting_interpretations"
        elif len(sci_values) == 1 and len(onc_values) == 1:
            answer = "complementary_somatic_and_oncogenic"
        elif len(sci_values) == 1 or len(onc_values) == 1:
            answer = "consistent_somatic_tier"
        else:
            continue
        descriptions = [item["description"] for item in items if item["description"]]
        evidence_text = bem.trim_text(
            f"Aggregated from {len(items)} ClinVar submissions for variation {variation_id} in {gene}. "
            f"Somatic clinical impact values={', '.join(sorted(sci_values)) if sci_values else 'none'}; "
            f"oncogenicity values={', '.join(sorted(onc_values)) if onc_values else 'none'}; "
            f"review statuses={', '.join(sorted({item['review_status'] for item in items if item['review_status']}))}. "
            + (" | ".join(descriptions[:3]) if descriptions else "")
        )
        question = (
            f"Across ClinVar somatic submissions, what aggregated interpretation pattern best describes "
            f"variation {variation_id} in {gene} for {bem.family_label(family)}?"
        )
        record = {
            "id": f"clinvar-hier-{family}-{gene}-{variation_id}",
            "source": "clinvar",
            "task_type": "hierarchical_evidence_aggregation",
            "question": question,
            "answer": answer,
            "answer_type": "categorical",
            "oncology_focus": bem.family_label(family),
            "evidence_text": evidence_text,
            "entities": {
                "disease": bem.family_label(family),
                "gene_symbol": gene,
                "variation_id": variation_id,
            },
            "provenance": {
                "source_record_id": variation_id,
                "source_dataset": "ClinVar",
                "source_path": "dataset/raw/clinvar/" + CLINVAR_FILTERED_PATH.name,
            },
            "metadata": {
                "priority_cancer_family": family,
                "eval_answer_options": list(CLINVAR_AGG_LABELS),
                "qa_evidence_function": "reasoning",
                "scenario_tags": ["molecular"],
                "aggregation_schema": "clinvar_submission_aggregation_v1",
                "aggregation_evidence_count": len(items),
                "aggregation_somatic_clinical_impact_values": sorted(sci_values),
                "aggregation_oncogenicity_values": sorted(onc_values),
            },
        }
        rows.append(
            bem.finalize_record(
                bem.attach_gold_standard(
                    record,
                    bem.build_gold_standard(
                        "categorical",
                        canonical_answer=answer,
                        acceptable_answers=[answer],
                        class_label=answer,
                        evaluable=True,
                    ),
                ),
                source_dataset="ClinVar",
                source_family="hierarchical_aggregation",
            )
        )
    return rows


def sample_by_class_quota(
    rows: list[dict[str, Any]],
    class_quotas: dict[str, int],
    class_key: Callable[[dict[str, Any]], str],
    inner_group_key: Callable[[dict[str, Any]], Any],
) -> list[dict[str, Any]]:
    picked: list[dict[str, Any]] = []
    for label, quota in class_quotas.items():
        subset = [row for row in rows if class_key(row) == label]
        picked.extend(round_robin_sample(subset, quota, inner_group_key))
    return picked


def count_by(rows: list[dict[str, Any]], key_fn: Callable[[dict[str, Any]], str]) -> dict[str, int]:
    return dict(Counter(key_fn(row) for row in rows))


def reasoning_annotation_for_row(row: dict[str, Any]) -> tuple[str, list[str]]:
    task_type = str(row.get("task_type") or "")
    if task_type == "mechanism_trace_intermediate_gene":
        return "multi_hop_evidence_reasoning", ["multi_hop_evidence_reasoning"]
    if task_type == "cross_source_corroboration_status":
        return "cross_source_corroboration", ["cross_source_corroboration", "multi_evidence_logical_synthesis"]
    if task_type == "cross_source_support_bundle":
        return "multi_evidence_logical_synthesis", ["cross_source_corroboration", "multi_evidence_logical_synthesis"]
    if task_type == "hierarchical_evidence_aggregation":
        return "hierarchical_evidence_aggregation", [
            "evidence_signal_interpretation",
            "hierarchical_evidence_aggregation",
        ]
    if task_type == "evidence_signal_interpretation":
        return "evidence_signal_interpretation", ["evidence_signal_interpretation"]
    if task_type == "precision_oncology_evidence":
        return "precision_oncology_evidence_judgment", [
            "evidence_signal_interpretation",
            "hierarchical_evidence_aggregation",
            "precision_oncology_evidence_judgment",
        ]
    if task_type == "mechanism_bioasq_list":
        return "hierarchical_evidence_aggregation", [
            "evidence_signal_interpretation",
            "hierarchical_evidence_aggregation",
        ]
    if task_type in {"mechanism_bioasq_factoid", "mechanism_bioasq_yesno"}:
        return "evidence_signal_interpretation", ["evidence_signal_interpretation"]
    return "evidence_signal_interpretation", ["evidence_signal_interpretation"]


def split_data_sources(row: dict[str, Any]) -> list[str]:
    provenance = row.get("provenance") or {}
    metadata = row.get("metadata") or {}
    raw = str(provenance.get("source_dataset") or metadata.get("source_dataset") or "").strip()
    if not raw:
        return []
    parts = [part.strip() for part in re.split(r"[\\/]+", raw) if part.strip()]
    return parts or [raw]


def answer_constraints_for_row(row: dict[str, Any]) -> dict[str, Any]:
    answer_type = str(row.get("answer_type") or "")
    metadata = row.get("metadata") or {}
    gold = row.get("gold_standard") or {}
    allowed_values = list(metadata.get("eval_answer_options") or gold.get("acceptable_answers") or [])
    constraint: dict[str, Any] = {
        "answer_type": answer_type,
        "allowed_values": allowed_values,
    }
    if answer_type == "boolean":
        constraint["rule"] = "Respond with yes or no only."
        constraint["evaluation_mode"] = "exact_match"
    elif answer_type == "categorical":
        constraint["rule"] = "Respond with one label from the allowed_values set."
        constraint["evaluation_mode"] = "exact_match"
    elif answer_type == "entity":
        acceptable_answers = list(gold.get("acceptable_answers") or ([row.get("answer")] if row.get("answer") else []))
        constraint["allowed_values"] = acceptable_answers
        constraint["canonical_answer"] = gold.get("canonical_answer") or row.get("answer")
        constraint["rule"] = "Respond with the canonical entity answer or an allowed alias."
        constraint["evaluation_mode"] = "entity_alias_match"
    else:
        constraint["rule"] = "Unsupported answer type for this benchmark."
        constraint["evaluation_mode"] = "unknown"
    return constraint


def annotate_row(row: dict[str, Any]) -> dict[str, Any]:
    primary_reasoning_type, reasoning_type_tags = reasoning_annotation_for_row(row)
    data_sources = split_data_sources(row)
    annotation = {
        "benchmark_version": "formal_v3",
        "reasoning_type": primary_reasoning_type,
        "reasoning_type_tags": reasoning_type_tags,
        "data_source": data_sources[0] if len(data_sources) == 1 else "multi_source",
        "data_sources": data_sources,
        "task_type": row.get("task_type"),
        "answer_type": row.get("answer_type"),
        "answer_constraints": answer_constraints_for_row(row),
    }
    row["benchmark_annotation"] = annotation
    metadata = row.setdefault("metadata", {})
    metadata["reasoning_type"] = primary_reasoning_type
    metadata["reasoning_type_tags"] = reasoning_type_tags
    metadata["data_sources"] = data_sources
    metadata["answer_constraints"] = annotation["answer_constraints"]
    return row


def annotate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [annotate_row(row) for row in rows]


def count_reasoning_type_tags(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        for tag in (row.get("benchmark_annotation") or {}).get("reasoning_type_tags") or []:
            counter[str(tag)] += 1
    return dict(counter)


def sample_by_family_quota(
    rows: list[dict[str, Any]],
    quotas: dict[str, int],
    inner_group_key: Callable[[dict[str, Any]], Any],
) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[family_of(row)].append(row)
    picked: list[dict[str, Any]] = []
    for family, quota in quotas.items():
        picked.extend(round_robin_sample(by_family.get(family, []), quota, inner_group_key))
    return picked


def main() -> None:
    global RAW_DIR, CLINVAR_DIR, CLINVAR_FILTERED_PATH, OUTPUT_PATH, SUMMARY_PATH
    parser = argparse.ArgumentParser(description="Rebuild the formal 571-question benchmark from pinned raw snapshots and 73 retained construction seeds. The historical script name is retained for provenance.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--seed", type=Path, default=HQ85_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--reference", type=Path, default=REFERENCE_PATH, help="Frozen benchmark used only to preserve historical grouped-record IDs.")
    parser.add_argument("--overrides", type=Path, default=ROOT / "dataset/evaluated/construction/record_overrides.json", help="Explicit archived construction annotations retained for exact paper-version reconstruction.")
    parser.add_argument("--allow-downloads", action="store_true", help="Permit upstream acquisition where an input cache is absent. Fresh live data may differ from the paper snapshot.")
    args = parser.parse_args()
    RAW_DIR = args.raw_dir.resolve()
    bem.RAW_DIR = RAW_DIR
    CLINVAR_DIR = RAW_DIR / "clinvar"
    CLINVAR_FILTERED_PATH = CLINVAR_DIR / "clinvar_somatic_cancer_submission_subset.jsonl"
    OUTPUT_PATH, SUMMARY_PATH = args.output, args.summary
    if OUTPUT_PATH.resolve() == REFERENCE_PATH.resolve():
        parser.error("Choose a separate --output; the frozen paper benchmark is preserved.")
    required = ["civic/evidence_items.jsonl", "civic/assertions.jsonl", "signor/signor_latest.tsv", "pathwaycommons/pc-hgnc.gmt.gz", "intogen/IntOGen-Drivers-20200201.zip", "intogen/IntOGen-Cohorts-20200201.zip", "opentargets/disease.parquet", "opentargets/association_overall_direct_supported.parquet", "opentargets/association_by_datasource_direct_supported.parquet", "opentargets/target_supported.parquet", "clinvar/clinvar_somatic_cancer_submission_subset.jsonl"]
    missing = [name for name in required if not (RAW_DIR / name).exists()]
    if not list((RAW_DIR / "bioasq/test").glob("*.json")):
        missing.append("bioasq/test/*_golden.json")
    if missing and not args.allow_downloads:
        parser.error("Missing pinned inputs: " + ", ".join(missing) + ". Restore source snapshots first; see docs/data.md.")
    if not args.seed.exists():
        parser.error("Missing construction seeds: " + str(args.seed))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    historical_counts = json.loads(HISTORICAL_COUNTS_PATH.read_text())
    hq85 = read_jsonl(args.seed)
    assert len(hq85) == 73 and len({row["id"] for row in hq85}) == 73, "Expected 73 retained construction seeds"


    hq_mech_trace = [copy.deepcopy(row) for row in hq85 if row.get("task_type") == "mechanism_trace_intermediate_gene"]
    hq_corrob = [copy.deepcopy(row) for row in hq85 if row.get("task_type") == "cross_source_corroboration_status"]
    hq_precision = [transform_existing_precision_record(row) for row in hq85 if row.get("task_type") == "precision_oncology_evidence"]

    bioasq_all = bem.normalize_bioasq()
    bioasq_all = [row for row in bioasq_all if row.get("task_type", "").startswith("mechanism_bioasq_")]

    signor_df, _ = bem.materialize_signor_raw()
    pathway_rows, _ = bem.materialize_pathwaycommons_raw()
    intogen_df, _ = bem.materialize_intogen_raw()
    open_targets_payload, _ = bem.materialize_open_targets_raw()
    support_index = bem.build_external_support_index(intogen_df, open_targets_payload)

    indirect_candidates = [
        row
        for row in bem.normalize_external_indirect_causal(support_index, pathway_rows, signor_df)
        if safe_int((row.get("metadata") or {}).get("causal_suitability_score")) >= 10
    ]
    trace_candidates = bem.normalize_mechanism_trace_intermediate(indirect_candidates)
    corrob_candidates = bem.normalize_cross_source_corroboration(support_index, signor_df, pathway_rows)
    precision_candidates = build_expanded_precision_records()
    bundle_candidates = build_bundle_v2_records()
    signal_candidates = build_evidence_signal_interpretation_records()
    hier_assertion_candidates = build_hierarchical_assertion_records()
    hier_group_candidates = build_hierarchical_group_records()
    hier_clinvar_candidates = build_clinvar_hierarchical_records()

    mech_existing_ids = {row["id"] for row in hq_mech_trace}
    corrob_existing_ids = {row["id"] for row in hq_corrob}
    precision_existing_ids = {row["id"] for row in hq_precision}
    bioasq_existing_ids = {row["id"] for row in bioasq_all}
    del bioasq_existing_ids  # included for symmetry; all BioASQ rows are kept.

    trace_extra = [row for row in trace_candidates if row["id"] not in mech_existing_ids]
    trace_target_by_family = {
        "aml": 10,
        "brca": 9,
        "crc": 9,
        "gastroesoph": 9,
        "glioma": 9,
        "hcc": 9,
        "hnsc": 9,
        "nsclc": 9,
        "ovarian": 9,
        "paad": 9,
        "rcc": 9,
    }
    existing_trace_family = Counter(family_of(row) for row in hq_mech_trace)
    trace_extra = sample_by_family_quota(
        trace_extra,
        {
            family: max(0, target - existing_trace_family.get(family, 0))
            for family, target in trace_target_by_family.items()
        },
        inner_group_key=lambda row: str(row.get("answer") or ""),
    )
    mechanism_trace_rows = attach_selection_metadata(
        hq_mech_trace + trace_extra,
        "mechanism_trace_keep23_expand100",
        "retain_and_expand",
    )

    corrob_extra = [row for row in corrob_candidates if row["id"] not in corrob_existing_ids]
    corrob_extra = round_robin_sample(
        corrob_extra,
        FINAL_TARGETS["cross_source_corroboration_status"] - len(hq_corrob),
        group_key=lambda row: (family_of(row), str(row.get("answer") or "")),
    )
    corrob_rows = attach_selection_metadata(
        hq_corrob + corrob_extra,
        "cross_source_corroboration_keep23_expand100",
        "retain_and_expand",
    )

    precision_seed_questions = {str(row.get("question") or "") for row in hq_precision}
    precision_candidates, precision_dedupe_stats = dedupe_precision_candidates_by_question(precision_candidates)
    precision_extra = [
        row
        for row in precision_candidates
        if row["id"] not in precision_existing_ids and str(row.get("question") or "") not in precision_seed_questions
    ]
    precision_extra = round_robin_sample(
        precision_extra,
        FINAL_TARGETS["precision_oncology_evidence"] - len(hq_precision),
        group_key=lambda row: (family_of(row), str(row.get("answer") or "")),
    )
    precision_rows = attach_selection_metadata(
        hq_precision + precision_extra,
        "precision_oncology_relabel27_expand100",
        "retain_relabel_and_expand",
    )

    bioasq_rows = attach_selection_metadata(
        bioasq_all,
        "mechanism_bioasq_keep6_expand19",
        "retain_and_expand",
    )

    signal_seed_rows = [row for row in bioasq_rows if row.get("task_type") in {"mechanism_bioasq_factoid", "mechanism_bioasq_yesno"}]
    signal_seed_questions = {str(row.get("question") or "") for row in signal_seed_rows}
    signal_candidate_questions: set[str] = set()
    signal_candidates_deduped: list[dict[str, Any]] = []
    for row in signal_candidates:
        question = str(row.get("question") or "")
        if question in signal_seed_questions or question in signal_candidate_questions:
            continue
        signal_candidate_questions.add(question)
        signal_candidates_deduped.append(row)
    signal_family_quota = {
        "aml": 15,
        "brca": 12,
        "crc": 14,
        "glioma": 10,
        "nsclc": 10,
        "rcc": 8,
        "hnsc": 7,
        "ovarian": 5,
        "paad": 5,
        "hcc": 2,
        "gastroesoph": 0,
    }
    signal_extra = sample_by_family_quota(
        signal_candidates_deduped,
        signal_family_quota,
        inner_group_key=lambda row: str(row.get("answer") or ""),
    )
    signal_rows = attach_selection_metadata(signal_extra, "evidence_signal_interpretation_expand100", "expand")

    bundle_rows = sample_by_class_quota(
        bundle_candidates,
        {
            "causal_relation+pathway_context+target_association": 21,
            "causal_relation+target_association": 20,
            "pathway_context+target_association": 20,
            "target_association_only": 20,
        },
        class_key=lambda row: str(row.get("answer") or ""),
        inner_group_key=lambda row: (family_of(row), str(row.get("answer") or "")),
    )
    bundle_rows = attach_selection_metadata(
        bundle_rows,
        "cross_source_support_bundle_rebuild81_v2_no_driver",
        "relabel_and_rebuild",
    )

    hier_assertion_rows = attach_selection_metadata(
        hier_assertion_candidates,
        "hierarchical_aggregation_local_assertions",
        "expand",
    )
    hier_group_rows = attach_selection_metadata(
        hier_group_candidates,
        "hierarchical_aggregation_local_groups",
        "expand",
    )
    hier_clinvar_rows = attach_selection_metadata(
        sample_by_family_quota(
            hier_clinvar_candidates,
            {"glioma": 25, "aml": 2, "crc": 1},
            inner_group_key=lambda row: str(row.get("answer") or ""),
        ),
        "hierarchical_aggregation_external_clinvar",
        "expand",
    )
    hierarchical_rows = attach_selection_metadata(
        hier_assertion_rows + hier_group_rows + hier_clinvar_rows,
        "hierarchical_aggregation_total",
        "expand",
    )

    final_rows = (
        mechanism_trace_rows
        + corrob_rows
        + precision_rows
        + bioasq_rows
        + bundle_rows
        + signal_rows
        + hierarchical_rows
    )
    # Legacy grouped CIViC IDs used Python's process-randomized hash. Content
    # construction is unchanged; reuse only IDs for exact question/label matches
    # in the released benchmark. Unseen groups use the stable SHA-256 ID above.
    if args.reference.exists():
        released = {(r["question"], r["answer"]): r["id"] for r in read_jsonl(args.reference)
                    if r["id"].startswith("civic-hier-group-")}
        for row in final_rows:
            if row["id"].startswith("civic-hier-group-"):
                row["id"] = released.get((row["question"], row["answer"]), row["id"])
    final_rows = annotate_rows(final_rows)
    overrides = json.loads(args.overrides.read_text())
    by_id = {row["id"]: row for row in final_rows}
    for item in overrides["records"]:
        row = by_id[item["id"]]
        for field, value in item["fields"].items():
            if field == "metadata.external_support_layers":
                row["metadata"]["external_support_layers"] = value
            elif field == "evidence_text":
                row[field] = value
            else:
                raise ValueError("Unsupported archived annotation field: " + field)
    final_rows = sorted(final_rows, key=lambda row: (str(row.get("task_type") or ""), str(row.get("id") or "")))
    write_jsonl(OUTPUT_PATH, final_rows)

    source_counts_499 = Counter(historical_counts)
    summary = {
        "input_seed_benchmark": args.seed.name,
        "archived_annotation_overrides": {"file": args.overrides.name, "records": len(overrides["records"])},
        "output_benchmark": OUTPUT_PATH.name,
        "formal_policy": {
            "retained_and_expanded": {
                "mechanism_trace_intermediate_gene": {
                    "seed_count": 23,
                    "final_count": len(mechanism_trace_rows),
                    "rule": "保留并大幅扩充",
                },
                "cross_source_corroboration_status": {
                    "seed_count": 23,
                    "final_count": len(corrob_rows),
                    "rule": "保留并大幅扩充",
                },
                "precision_oncology_evidence": {
                    "seed_count": 27,
                    "final_count": len(precision_rows),
                    "rule": "保留但改评分方式并扩充",
                    "new_label_schema": SUPPORTED_LABELS_PRECISION,
                    "candidate_dedupe": precision_dedupe_stats,
                },
                "mechanism_bioasq_*": {
                    "seed_count": 6,
                    "final_count": len(bioasq_rows),
                    "rule": "保留并扩充到本地可直接访问上限",
                },
                "evidence_signal_interpretation": {
                    "seed_count": len(signal_seed_rows),
                    "final_count": len(signal_seed_rows) + len(signal_rows),
                    "rule": "优先扩充，使用本地CIViC非predictive证据项补到100",
                },
                "hierarchical_evidence_aggregation": {
                    "seed_count": 7,
                    "final_count": 7 + len(hierarchical_rows),
                    "rule": "分别利用本地和外部知识源扩充",
                    "local_sources": ["CIViC assertions", "CIViC grouped evidence"],
                    "external_sources": ["ClinVar somatic submissions"],
                },
                "cross_source_support_bundle": {
                    "seed_count": 81,
                    "final_count": len(bundle_rows),
                    "rule": "重构答案标签，移除driver维度并重建",
                    "new_label_schema": BUNDLE_V2_LABELS,
                },
            },
            "removed_task_types": {
                task: {
                    "seed_count": source_counts_499.get(task, 0),
                    "rule": "按诊断结论剔除",
                }
                for task in sorted(REMOVED_TASKS)
            },
            "removed_total_from_499": sum(source_counts_499.get(task, 0) for task in REMOVED_TASKS),
        },
        "final_stats": {
            "total_samples": len(final_rows),
            "by_task_type": count_by(final_rows, lambda row: str(row.get("task_type") or "")),
            "by_answer_type": count_by(final_rows, lambda row: str(row.get("answer_type") or "")),
            "by_source_dataset": count_by(final_rows, lambda row: str((row.get("metadata") or {}).get("source_dataset") or "")),
            "by_priority_cancer_family": count_by(final_rows, family_of),
            "by_subset_profile": count_by(final_rows, lambda row: str((row.get("metadata") or {}).get("subset_profile") or "")),
            "by_reasoning_type": count_by(final_rows, lambda row: str((row.get("benchmark_annotation") or {}).get("reasoning_type") or "")),
            "by_reasoning_type_tag": count_reasoning_type_tags(final_rows),
        },
        "task_level_breakdown": {
            "mechanism_trace_intermediate_gene": {
                "count": len(mechanism_trace_rows),
                "source_datasets": count_by(mechanism_trace_rows, lambda row: str((row.get("metadata") or {}).get("source_dataset") or "")),
                "by_family": count_by(mechanism_trace_rows, family_of),
            },
            "cross_source_corroboration_status": {
                "count": len(corrob_rows),
                "source_datasets": count_by(corrob_rows, lambda row: str((row.get("metadata") or {}).get("source_dataset") or "")),
                "by_answer": count_by(corrob_rows, lambda row: str(row.get("answer") or "")),
                "by_family": count_by(corrob_rows, family_of),
            },
            "precision_oncology_evidence": {
                "count": len(precision_rows),
                "source_datasets": count_by(precision_rows, lambda row: str((row.get("metadata") or {}).get("source_dataset") or "")),
                "by_answer": count_by(precision_rows, lambda row: str(row.get("answer") or "")),
                "by_family": count_by(precision_rows, family_of),
            },
            "evidence_signal_interpretation": {
                "count": len(signal_seed_rows) + len(signal_rows),
                "source_datasets": count_by(signal_seed_rows + signal_rows, lambda row: str((row.get("metadata") or {}).get("source_dataset") or "")),
                "by_answer": count_by(signal_seed_rows + signal_rows, lambda row: str(row.get("answer") or "")),
                "by_family": count_by(signal_seed_rows + signal_rows, family_of),
            },
            "mechanism_bioasq": {
                "count": len(bioasq_rows),
                "source_datasets": count_by(bioasq_rows, lambda row: str((row.get("metadata") or {}).get("source_dataset") or "")),
                "by_task_type": count_by(bioasq_rows, lambda row: str(row.get("task_type") or "")),
                "by_family": count_by(bioasq_rows, family_of),
            },
            "hierarchical_evidence_aggregation": {
                "count": 7 + len(hierarchical_rows),
                "source_datasets": count_by([row for row in bioasq_rows if row.get("task_type") == "mechanism_bioasq_list"] + hierarchical_rows, lambda row: str((row.get("metadata") or {}).get("source_dataset") or "")),
                "by_answer": count_by([row for row in bioasq_rows if row.get("task_type") == "mechanism_bioasq_list"] + hierarchical_rows, lambda row: str(row.get("answer") or "")),
                "by_family": count_by([row for row in bioasq_rows if row.get("task_type") == "mechanism_bioasq_list"] + hierarchical_rows, family_of),
            },
            "cross_source_support_bundle": {
                "count": len(bundle_rows),
                "source_datasets": count_by(bundle_rows, lambda row: str((row.get("metadata") or {}).get("source_dataset") or "")),
                "by_answer": count_by(bundle_rows, lambda row: str(row.get("answer") or "")),
                "by_family": count_by(bundle_rows, family_of),
            },
        },
        "reasoning_type_coverage": count_reasoning_type_tags(final_rows),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(final_rows)} rows -> {OUTPUT_PATH}")
    print(f"Saved summary -> {SUMMARY_PATH}")


if __name__ == "__main__":
    main()

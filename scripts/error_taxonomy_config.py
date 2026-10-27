from __future__ import annotations

from typing import Any


ERROR_TAXONOMY: list[dict[str, Any]] = [
    {
        "id": "sparse_missing_evidence",
        "name": "Sparse / Missing Evidence",
        "definition": "The retrieved context misses key facts needed to justify the gold answer.",
        "dominant_tasks": [
            "precision_oncology_evidence",
            "hierarchical_evidence_aggregation",
            "mechanism_bioasq_factoid",
            "mechanism_bioasq_list",
            "mechanism_bioasq_yesno",
        ],
    },
    {
        "id": "bundle_incompleteness",
        "name": "Bundle Incompleteness",
        "definition": "The answer requires multiple evidence components, but only a partial bundle is retrieved or used.",
        "dominant_tasks": [
            "cross_source_support_bundle",
            "cross_source_corroboration_status",
        ],
    },
    {
        "id": "conflict_resolution_failure",
        "name": "Conflict Resolution Failure",
        "definition": "The context contains mixed or layered evidence, but the model fails to reconcile support, opposition, or evidence tiers correctly.",
        "dominant_tasks": [
            "hierarchical_evidence_aggregation",
            "evidence_signal_interpretation",
            "precision_oncology_evidence",
        ],
    },
    {
        "id": "multi_hop_reasoning_failure",
        "name": "Multi-hop Reasoning Failure",
        "definition": "The model fails to connect intermediate entities or path steps required for a multi-hop conclusion.",
        "dominant_tasks": [
            "mechanism_trace_intermediate_gene",
        ],
    },
    {
        "id": "label_granularity_mapping_failure",
        "name": "Label / Granularity Mapping Failure",
        "definition": "Relevant evidence is present, but the model maps it to the wrong output label, level, or answer format.",
        "dominant_tasks": [
            "hierarchical_evidence_aggregation",
            "precision_oncology_evidence",
            "evidence_signal_interpretation",
            "mechanism_bioasq_factoid",
            "mechanism_bioasq_list",
            "mechanism_bioasq_yesno",
        ],
    },
    {
        "id": "scope_distractor_mismatch",
        "name": "Scope / Distractor Mismatch",
        "definition": "The model is led by off-disease, off-entity, or otherwise irrelevant evidence instead of the intended scope.",
        "dominant_tasks": [
            "cross_source_support_bundle",
            "cross_source_corroboration_status",
            "mechanism_trace_intermediate_gene",
            "precision_oncology_evidence",
        ],
    },
]


TASK_CANDIDATES: dict[str, list[str]] = {
    "cross_source_corroboration_status": [
        "bundle_incompleteness",
        "scope_distractor_mismatch",
        "sparse_missing_evidence",
        "label_granularity_mapping_failure",
    ],
    "cross_source_support_bundle": [
        "bundle_incompleteness",
        "scope_distractor_mismatch",
        "sparse_missing_evidence",
        "label_granularity_mapping_failure",
    ],
    "evidence_signal_interpretation": [
        "conflict_resolution_failure",
        "label_granularity_mapping_failure",
        "scope_distractor_mismatch",
        "sparse_missing_evidence",
    ],
    "hierarchical_evidence_aggregation": [
        "conflict_resolution_failure",
        "label_granularity_mapping_failure",
        "scope_distractor_mismatch",
        "sparse_missing_evidence",
    ],
    "mechanism_trace_intermediate_gene": [
        "multi_hop_reasoning_failure",
        "scope_distractor_mismatch",
        "sparse_missing_evidence",
        "label_granularity_mapping_failure",
    ],
    "precision_oncology_evidence": [
        "conflict_resolution_failure",
        "label_granularity_mapping_failure",
        "scope_distractor_mismatch",
        "sparse_missing_evidence",
    ],
    "mechanism_bioasq_factoid": [
        "sparse_missing_evidence",
        "label_granularity_mapping_failure",
        "scope_distractor_mismatch",
    ],
    "mechanism_bioasq_list": [
        "sparse_missing_evidence",
        "label_granularity_mapping_failure",
        "scope_distractor_mismatch",
    ],
    "mechanism_bioasq_yesno": [
        "sparse_missing_evidence",
        "label_granularity_mapping_failure",
        "scope_distractor_mismatch",
    ],
}


def taxonomy_by_id() -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in ERROR_TAXONOMY}


def all_taxonomy_ids() -> list[str]:
    return [item["id"] for item in ERROR_TAXONOMY]

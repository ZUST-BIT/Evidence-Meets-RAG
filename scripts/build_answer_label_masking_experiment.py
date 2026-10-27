from __future__ import annotations

import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "dataset" / "rag" / "knowledge_sources" / "multi_cancer"
SOURCE_DOCS = SOURCE_DIR / "evidence_graph_complete_docs.jsonl"
SOURCE_EDGES = SOURCE_DIR / "evidence_graph_complete_edges.jsonl"
MASKED_DOCS = SOURCE_DIR / "evidence_graph_complete_label_masked_docs.jsonl"
MASKED_EDGES = SOURCE_DIR / "evidence_graph_complete_label_masked_edges.jsonl"
AUDIT_ROWS = ROOT / "dataset" / "rag" / "analysis" / "ecorag_coverability_leakage" / "ecorag_coverability_leakage_records.jsonl"
BENCHMARK = ROOT / "dataset" / "evaluated" / "external_reasoning_benchmark_formal_v3_571.jsonl"
PANEL = ROOT / "dataset" / "evaluated" / "ecorag_label_masking_panel_120.jsonl"
REPORT = ROOT / "dataset" / "rag" / "analysis" / "ecorag_label_masking" / "build_report.json"

SUMMARY_TYPES = {"target_support_summary", "mechanism_role_summary"}
RISKY_KEYS = {
    "predicate",
    "direction",
    "polarity",
    "evidence_direction",
    "outcome",
    "support",
    "support_flags",
    "summary",
    "evidence_category",
    "predicate_counts",
    "association_inventory",
    "bundle_components",
    "role_signal_counts",
    "predominant_role",
    "mechanism_bucket_counts",
    "source_mechanism_counts",
    "semantic_label",
    "semantic_labels",
    "linked_semantic_labels",
    "linked_evidence_predicates",
    "linked_evidence_claims",
    "example_claims",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def first(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, dict):
        return str(value.get("label") or value.get("id") or "")
    if isinstance(value, list):
        return ", ".join(str(item) for item in value[:8])
    return str(value)


def sanitize_mapping(value: Any, removed: Counter[str]) -> Any:
    if isinstance(value, list):
        return [sanitize_mapping(item, removed) for item in value]
    if not isinstance(value, dict):
        return value
    cleaned = {}
    for key, item in value.items():
        normalized = str(key).lower()
        if normalized in RISKY_KEYS or normalized.startswith("supports_"):
            removed[normalized] += 1
            continue
        cleaned[key] = sanitize_mapping(item, removed)
    return cleaned


def render_evidence(doc: dict[str, Any], node: dict[str, Any]) -> tuple[str, str]:
    subject = first(node.get("subject"))
    object_label = first(node.get("object"))
    disease = first(node.get("disease")) or first(doc.get("disease_tags"))
    title_parts = [part for part in [subject, object_label, disease] if part]
    title = "Evidence record: " + " | ".join(title_parts)
    lines = [
        f"Subject: {subject}" if subject else "",
        f"Object: {object_label}" if object_label else "",
        f"Disease: {disease}" if disease else "",
        f"Feature layer: {first(node.get('feature_layer'))}" if node.get("feature_layer") else "",
        f"Aggregation level: {first(node.get('aggregation_level'))}" if node.get("aggregation_level") else "",
        f"Metric: {first(node.get('metric'))}" if node.get("metric") else "",
        f"Value: {first(node.get('value'))}" if node.get("value") is not None else "",
        f"Unit: {first(node.get('unit'))}" if node.get("unit") else "",
        f"Comparison: {json.dumps(node.get('comparison'), ensure_ascii=False)}" if node.get("comparison") else "",
        f"Statistics: {json.dumps(node.get('statistics'), ensure_ascii=False)}" if node.get("statistics") else "",
        f"Raw text: {first(node.get('raw_text'))}" if node.get("raw_text") else "",
        f"Citation: {json.dumps(node.get('citation'), ensure_ascii=False)}" if node.get("citation") else "",
        f"Provenance: {json.dumps(node.get('provenance'), ensure_ascii=False)}" if node.get("provenance") else "",
        f"Source context: {json.dumps(node.get('source_context'), ensure_ascii=False)}" if node.get("source_context") else "",
    ]
    return title, "\n".join(line for line in lines if line)


def render_summary(doc: dict[str, Any], metadata: dict[str, Any]) -> tuple[str, str]:
    gene = first(metadata.get("gene_symbol"))
    disease = first(metadata.get("disease")) or first(doc.get("disease_tags"))
    title = f"Evidence inventory for {gene} in {disease}".strip()
    lines = [
        f"Gene: {gene}" if gene else "",
        f"Disease: {disease}" if disease else "",
        f"Member evidence count: {first(metadata.get('member_count'))}",
        f"Source family counts: {json.dumps(metadata.get('source_family_counts') or {}, ensure_ascii=False)}",
        f"Feature layer counts: {json.dumps(metadata.get('feature_layer_counts') or {}, ensure_ascii=False)}",
        f"Evidence level counts: {json.dumps(metadata.get('aggregation_level_counts') or {}, ensure_ascii=False)}",
    ]
    return title, "\n".join(line for line in lines if line)


def scrub_observation_text(text: str, removed: Counter[str]) -> str:
    output = []
    skip_claim_items = False
    for line in str(text or "").splitlines():
        lower = line.strip().lower()
        if lower.startswith("example linked semantic labels:") or lower.startswith("example linked evidence predicates:"):
            removed[lower.split(":", 1)[0]] += 1
            skip_claim_items = False
            continue
        if lower.startswith("example linked evidence claims:"):
            removed["example linked evidence claims"] += 1
            skip_claim_items = True
            continue
        if skip_claim_items and lower.startswith("-"):
            removed["example linked evidence claim item"] += 1
            continue
        skip_claim_items = False
        output.append(line)
    return "\n".join(output)


def mask_doc(doc: dict[str, Any], removed: Counter[str]) -> tuple[dict[str, Any], bool]:
    updated = deepcopy(doc)
    metadata = updated.get("metadata") or {}
    node = metadata.get("raw_node") if isinstance(metadata.get("raw_node"), dict) else {}
    kind = str(metadata.get("node_type") or "")
    changed = False
    if kind == "evidence":
        updated["title"], updated["text"] = render_evidence(updated, node)
        changed = True
    elif kind in SUMMARY_TYPES:
        updated["title"], updated["text"] = render_summary(updated, metadata)
        changed = True
    elif kind == "observation":
        scrubbed = scrub_observation_text(updated.get("text") or "", removed)
        changed = scrubbed != updated.get("text")
        updated["text"] = scrubbed
    updated["knowledge_source"] = "evidence_graph_complete_label_masked"
    updated["metadata"] = sanitize_mapping(metadata, removed)
    if metadata.get("relation_hits"):
        updated["metadata"]["relation_hits"] = ["RELATED_TO"]
    updated["metadata"]["masking_variant"] = "global_answer_label_mask_v1"
    return updated, changed


def build_panel() -> dict[str, Any]:
    audit = [row for row in read_jsonl(AUDIT_ROWS) if row.get("knowledge_source") == "evidence_kg"]
    exposed_ids = {
        str(row["record_id"])
        for row in audit
        if row.get("answer_exposure_risk") == "structured_or_label_exposure"
    }
    records = read_jsonl(BENCHMARK)
    by_id = {str(row["id"]): row for row in records}
    exposed = [by_id[rid] for rid in sorted(exposed_ids)]
    candidates = [
        row for row in records
        if str(row["id"]) not in exposed_ids
        and row.get("task_type") in {"evidence_signal_interpretation", "hierarchical_evidence_aggregation"}
    ]
    candidates.sort(key=lambda row: (
        0 if row.get("task_type") == "evidence_signal_interpretation" else 1,
        str((row.get("metadata") or {}).get("priority_cancer_family") or ""),
        str(row.get("id")),
    ))
    controls = candidates[: len(exposed)]
    panel = []
    for stratum, rows in [("exposed", exposed), ("non_exposed_control", controls)]:
        for row in rows:
            item = deepcopy(row)
            item.setdefault("metadata", {})["label_masking_stratum"] = stratum
            panel.append(item)
    panel.sort(key=lambda row: (str((row.get("metadata") or {}).get("label_masking_stratum")), str(row.get("id"))))
    write_jsonl(PANEL, panel)
    return {
        "panel_count": len(panel),
        "stratum_counts": dict(Counter((row.get("metadata") or {}).get("label_masking_stratum") for row in panel)),
        "task_counts": dict(Counter(row.get("task_type") for row in panel)),
    }


def main() -> None:
    panel_summary = build_panel()
    removed: Counter[str] = Counter()
    doc_count = changed_count = 0
    MASKED_DOCS.parent.mkdir(parents=True, exist_ok=True)
    with SOURCE_DOCS.open(encoding="utf-8") as source, MASKED_DOCS.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            masked, changed = mask_doc(json.loads(line), removed)
            target.write(json.dumps(masked, ensure_ascii=False) + "\n")
            doc_count += 1
            changed_count += int(changed)
            if doc_count % 50000 == 0:
                print(f"[mask] {doc_count} documents", flush=True)
    edge_count = collapsed_edge_count = 0
    with SOURCE_EDGES.open(encoding="utf-8") as source, MASKED_EDGES.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            edge = json.loads(line)
            if edge.get("relation") not in {"HAS_CHUNK", "NEXT_CHUNK"}:
                edge["relation"] = "RELATED_TO"
                collapsed_edge_count += 1
            edge["metadata"] = sanitize_mapping(edge.get("metadata") or {}, removed)
            edge["metadata"]["masking_variant"] = "global_answer_label_mask_v1"
            target.write(json.dumps(edge, ensure_ascii=False) + "\n")
            edge_count += 1
            if edge_count % 500000 == 0:
                print(f"[mask] {edge_count} edges", flush=True)
    report = {
        "policy": "global_answer_label_mask_v1",
        "source_docs": str(SOURCE_DOCS.relative_to(ROOT)),
        "masked_docs": str(MASKED_DOCS.relative_to(ROOT)),
        "edges_note": "All edge endpoints are retained; answer-bearing relation labels are collapsed to RELATED_TO, while HAS_CHUNK and NEXT_CHUNK are preserved.",
        "doc_count": doc_count,
        "changed_doc_count": changed_count,
        "edge_count": edge_count,
        "collapsed_edge_count": collapsed_edge_count,
        "removed_field_or_line_counts": dict(removed.most_common()),
        **panel_summary,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

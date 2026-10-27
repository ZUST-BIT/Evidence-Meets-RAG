from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "dataset" / "rag" / "knowledge_sources" / "multi_cancer"
SOURCE_DOCS_PATH = SOURCE_DIR / "evidence_graph_complete_docs.jsonl"
SOURCE_EDGES_PATH = SOURCE_DIR / "evidence_graph_complete_edges.jsonl"
REPORT_PATH = SOURCE_DIR / "evidence_graph_complete_ablation_suite_report.json"

SUMMARY_NODE_TYPES = {"target_support_summary", "mechanism_role_summary"}
STRUCTURED_LABELS_TO_DROP = {
    "Disease aliases",
    "Linked entities",
    "Linked entity IDs",
    "Disease",
    "Subject",
    "Predicate",
    "Object",
    "Outcome",
    "Comparison",
    "Statistics",
    "Confidence",
    "Support flags",
    "Citation",
    "Provenance",
    "Source context",
}
STRUCTURED_LABELS_TO_KEEP_VALUE_ONLY = {
    "Claim",
    "Summary",
    "Raw text",
    "Feature layer",
    "Evidence category",
    "Aggregation level",
    "Direction",
    "Polarity",
    "Claim scope",
    "Metric",
    "Value",
    "Unit",
}

ABLATIONS = {
    "evidence_graph_complete_no_structure": {
        "description": "Remove graph topology and flatten node text into plain prose blocks.",
    },
    "evidence_graph_complete_no_evidence_text": {
        "description": "Preserve graph topology but strip narrative evidence text from evidence-centric docs.",
    },
    "evidence_graph_complete_no_cross_source_links": {
        "description": "Remove cross-source links and cross-source summary/aggregate docs while preserving within-source evidence.",
    },
    "evidence_graph_complete_no_cross_layer_aggregation": {
        "description": "Remove aggregate/summary layers that synthesize lower-level evidence across layers.",
    },
    "evidence_graph_complete_no_relation_types": {
        "description": "Preserve connectivity while collapsing edge semantics to a generic relation type.",
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def raw_node(doc: dict[str, Any]) -> dict[str, Any]:
    return ((doc.get("metadata") or {}).get("raw_node") or {})


def node_type(doc: dict[str, Any]) -> str:
    return str((doc.get("metadata") or {}).get("node_type") or "")


def raw_source_context(doc: dict[str, Any]) -> dict[str, Any]:
    return raw_node(doc).get("source_context") or {}


def doc_non_integrated_source_families(doc: dict[str, Any]) -> set[str]:
    families: set[str] = set()
    node = raw_node(doc)
    meta = doc.get("metadata") or {}
    source_family = node.get("source_family") or (node.get("provenance") or {}).get("source_family")
    if source_family and source_family != "integrated":
        families.add(str(source_family))
    member_families = raw_source_context(doc).get("member_source_families") or []
    for family in member_families:
        if family and family != "integrated":
            families.add(str(family))
    for family in (meta.get("source_family_counts") or {}).keys():
        if family and family != "integrated":
            families.add(str(family))
    return families


def doc_has_cross_source_content(doc: dict[str, Any]) -> bool:
    families = doc_non_integrated_source_families(doc)
    if len(families) > 1:
        return True
    meta = doc.get("metadata") or {}
    association_inventory = meta.get("association_inventory") or {}
    if association_inventory.get("cross_source_support"):
        return True
    return False


def is_summary_doc(doc: dict[str, Any]) -> bool:
    return node_type(doc) in SUMMARY_NODE_TYPES


def is_aggregate_doc(doc: dict[str, Any]) -> bool:
    node = raw_node(doc)
    if not node:
        return False
    if str(node.get("source_family") or "") == "integrated":
        return True
    if str(node.get("aggregation_level") or "") in {"same_claim_aggregate", "cross_omics_mechanism_aggregate"}:
        return True
    source_context = raw_source_context(doc)
    if source_context.get("aggregation_scope"):
        return True
    return str(doc.get("doc_id") or "").startswith(("AGGCLAIM-", "AGGMECH-"))


def normalize_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def first_non_empty(*values: Any) -> str:
    for value in values:
        text = normalize_value(value).strip()
        if text:
            return text
    return ""


def plain_text_from_structured(doc: dict[str, Any]) -> str:
    node = raw_node(doc)
    meta = doc.get("metadata") or {}
    lines: list[str] = []
    summary = first_non_empty(node.get("summary"))
    raw_text = first_non_empty(node.get("raw_text"))
    if summary:
        lines.append(summary)
    if raw_text:
        lines.append(raw_text)
    study_object = first_non_empty(raw_source_context(doc).get("study_object"))
    experimental_context = first_non_empty(raw_source_context(doc).get("experimental_context"))
    if study_object or experimental_context:
        lines.append(". ".join(part for part in [study_object, experimental_context] if part))
    metric = first_non_empty(node.get("metric"))
    value = first_non_empty(node.get("value"))
    if metric and value:
        lines.append(f"{metric} = {value}.")
    if not lines and is_summary_doc(doc):
        disease = first_non_empty(meta.get("disease"))
        gene_symbol = first_non_empty(meta.get("gene_symbol"))
        lines.append(
            " ".join(
                part
                for part in [
                    f"Summary for {gene_symbol}." if gene_symbol else "",
                    f"Disease: {disease}." if disease else "",
                ]
                if part
            ).strip()
        )
    if lines:
        deduped_lines: list[str] = []
        seen: set[str] = set()
        for line in lines:
            text = line.strip()
            if not text:
                continue
            key = " ".join(text.split()).lower()
            if key in seen:
                continue
            seen.add(key)
            deduped_lines.append(text)
        return "\n".join(deduped_lines).strip()

    fallback_lines: list[str] = []
    for raw_line in str(doc.get("text") or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" not in line:
            fallback_lines.append(line)
            continue
        label, value_part = line.split(":", 1)
        label = label.strip()
        value_part = value_part.strip()
        if label in STRUCTURED_LABELS_TO_DROP:
            continue
        if label in STRUCTURED_LABELS_TO_KEEP_VALUE_ONLY:
            if value_part:
                fallback_lines.append(value_part)
            continue
        fallback_lines.append(line)
    return "\n".join(fallback_lines).strip() or str(doc.get("title") or "")


def minimal_evidence_text(doc: dict[str, Any]) -> str:
    meta = doc.get("metadata") or {}
    node = raw_node(doc)
    if is_summary_doc(doc):
        disease = first_non_empty(meta.get("disease"))
        gene_symbol = first_non_empty(meta.get("gene_symbol"))
        member_count = first_non_empty(meta.get("member_count"))
        source_families = ", ".join(sorted((meta.get("source_family_counts") or {}).keys()))
        predicates = ", ".join(sorted((meta.get("predicate_counts") or {}).keys())[:8])
        lines = [
            first_non_empty(f"Summary node for {gene_symbol} in {disease}".strip(), str(doc.get("title") or "")),
            f"member_count = {member_count}" if member_count else "",
            f"source_families = {source_families}" if source_families else "",
            f"predicate_inventory = {predicates}" if predicates else "",
        ]
        return "\n".join(line for line in lines if line).strip()
    if node_type(doc) != "evidence":
        return str(doc.get("text") or "")

    subject = normalize_value((node.get("subject") or {}).get("label")).strip()
    predicate = normalize_value(node.get("predicate")).strip()
    object_label = normalize_value((node.get("object") or {}).get("label")).strip()
    disease = first_non_empty(node.get("disease"), doc.get("disease_tags", [None])[0] if doc.get("disease_tags") else "")
    source_families = ", ".join(sorted(doc_non_integrated_source_families(doc)))
    lines = [
        " | ".join(part for part in [subject, predicate, object_label] if part),
        f"disease = {disease}" if disease else "",
        f"feature_layer = {normalize_value(node.get('feature_layer')).strip()}" if node.get("feature_layer") else "",
        f"aggregation_level = {normalize_value(node.get('aggregation_level')).strip()}" if node.get("aggregation_level") else "",
        f"metric = {normalize_value(node.get('metric')).strip()}" if node.get("metric") else "",
        f"value = {normalize_value(node.get('value')).strip()}" if node.get("value") is not None else "",
        f"source_families = {source_families}" if source_families else "",
    ]
    return "\n".join(line for line in lines if line).strip() or str(doc.get("title") or "")


def annotate_doc(doc: dict[str, Any], knowledge_source: str, ablation_name: str) -> dict[str, Any]:
    updated = deepcopy(doc)
    updated["knowledge_source"] = knowledge_source
    updated.setdefault("metadata", {})
    updated["metadata"]["ablation_variant"] = ablation_name
    return updated


def build_no_structure(docs: list[dict[str, Any]], edges: list[dict[str, Any]], knowledge_source: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    output_docs = []
    for doc in docs:
        updated = annotate_doc(doc, knowledge_source, "no_structure")
        updated["text"] = plain_text_from_structured(updated)
        updated["parent_id"] = None
        updated["document_level"] = "flat"
        updated["metadata"]["node_type"] = "flat_document"
        output_docs.append(updated)
    return output_docs, [], {"edges_removed": len(edges)}


def build_no_evidence_text(docs: list[dict[str, Any]], edges: list[dict[str, Any]], knowledge_source: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    output_docs = []
    mutated = 0
    for doc in docs:
        updated = annotate_doc(doc, knowledge_source, "no_evidence_text")
        if node_type(updated) == "evidence" or is_summary_doc(updated):
            updated["text"] = minimal_evidence_text(updated)
            mutated += 1
        output_docs.append(updated)
    output_edges = []
    for edge in edges:
        new_edge = deepcopy(edge)
        new_edge["metadata"] = {**(new_edge.get("metadata") or {}), "ablation_variant": "no_evidence_text"}
        output_edges.append(new_edge)
    return output_docs, output_edges, {"docs_text_scrubbed": mutated}


def build_no_cross_source_links(docs: list[dict[str, Any]], edges: list[dict[str, Any]], knowledge_source: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    kept_docs = []
    removed_doc_ids: set[str] = set()
    for doc in docs:
        if doc_has_cross_source_content(doc):
            removed_doc_ids.add(str(doc["doc_id"]))
            continue
        kept_docs.append(annotate_doc(doc, knowledge_source, "no_cross_source_links"))

    doc_by_id = {str(doc["doc_id"]): doc for doc in kept_docs}
    kept_edges = []
    removed_cross_source_edges = 0
    for edge in edges:
        source_id = str(edge["source_doc_id"])
        target_id = str(edge["target_doc_id"])
        if source_id not in doc_by_id or target_id not in doc_by_id:
            continue
        source_doc = doc_by_id[source_id]
        target_doc = doc_by_id[target_id]
        source_families = doc_non_integrated_source_families(source_doc)
        target_families = doc_non_integrated_source_families(target_doc)
        if source_families and target_families and source_families != target_families:
            removed_cross_source_edges += 1
            continue
        new_edge = deepcopy(edge)
        new_edge["metadata"] = {**(new_edge.get("metadata") or {}), "ablation_variant": "no_cross_source_links"}
        kept_edges.append(new_edge)
    return kept_docs, kept_edges, {
        "docs_removed_for_cross_source_content": len(removed_doc_ids),
        "edges_removed_for_cross_source_connectivity": removed_cross_source_edges,
    }


def build_no_cross_layer_aggregation(docs: list[dict[str, Any]], edges: list[dict[str, Any]], knowledge_source: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    kept_docs = []
    removed_doc_ids: set[str] = set()
    for doc in docs:
        if is_summary_doc(doc) or is_aggregate_doc(doc):
            removed_doc_ids.add(str(doc["doc_id"]))
            continue
        kept_docs.append(annotate_doc(doc, knowledge_source, "no_cross_layer_aggregation"))
    doc_ids = {str(doc["doc_id"]) for doc in kept_docs}
    kept_edges = []
    removed_edges = 0
    for edge in edges:
        source_id = str(edge["source_doc_id"])
        target_id = str(edge["target_doc_id"])
        if source_id not in doc_ids or target_id not in doc_ids:
            removed_edges += 1
            continue
        new_edge = deepcopy(edge)
        new_edge["metadata"] = {**(new_edge.get("metadata") or {}), "ablation_variant": "no_cross_layer_aggregation"}
        kept_edges.append(new_edge)
    return kept_docs, kept_edges, {
        "docs_removed_for_aggregation": len(removed_doc_ids),
        "edges_removed_with_aggregation_nodes": removed_edges,
    }


def build_no_relation_types(docs: list[dict[str, Any]], edges: list[dict[str, Any]], knowledge_source: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    output_docs = [annotate_doc(doc, knowledge_source, "no_relation_types") for doc in docs]
    output_edges = []
    collapsed = 0
    for edge in edges:
        new_edge = deepcopy(edge)
        if new_edge.get("relation") not in {"HAS_CHUNK", "NEXT_CHUNK"}:
            if new_edge.get("relation") != "RELATED_TO":
                collapsed += 1
            new_edge["relation"] = "RELATED_TO"
        new_edge["metadata"] = {**(new_edge.get("metadata") or {}), "ablation_variant": "no_relation_types"}
        output_edges.append(new_edge)
    return output_docs, output_edges, {"edges_collapsed_to_related_to": collapsed}


BUILDERS = {
    "evidence_graph_complete_no_structure": build_no_structure,
    "evidence_graph_complete_no_evidence_text": build_no_evidence_text,
    "evidence_graph_complete_no_cross_source_links": build_no_cross_source_links,
    "evidence_graph_complete_no_cross_layer_aggregation": build_no_cross_layer_aggregation,
    "evidence_graph_complete_no_relation_types": build_no_relation_types,
}


def summarize_docs(docs: list[dict[str, Any]]) -> dict[str, Any]:
    node_counts = Counter(node_type(doc) for doc in docs)
    source_family_counts = Counter()
    for doc in docs:
        for family in doc_non_integrated_source_families(doc):
            source_family_counts[family] += 1
    return {
        "doc_count": len(docs),
        "node_type_counts": dict(sorted(node_counts.items())),
        "source_family_counts": dict(sorted(source_family_counts.items())),
    }


def summarize_edges(edges: list[dict[str, Any]]) -> dict[str, Any]:
    relation_counts = Counter(str(edge.get("relation") or "") for edge in edges)
    return {
        "edge_count": len(edges),
        "relation_counts": dict(sorted(relation_counts.items())),
    }


def main() -> None:
    docs = read_jsonl(SOURCE_DOCS_PATH)
    edges = read_jsonl(SOURCE_EDGES_PATH)
    report = {
        "generated_by": "scripts/build_evidence_graph_ablation_suite.py",
        "source_docs_path": str(SOURCE_DOCS_PATH.relative_to(ROOT)),
        "source_edges_path": str(SOURCE_EDGES_PATH.relative_to(ROOT)),
        "source_summary": {
            **summarize_docs(docs),
            **summarize_edges(edges),
        },
        "ablations": {},
    }

    for knowledge_source, config in ABLATIONS.items():
        builder = BUILDERS[knowledge_source]
        output_docs, output_edges, notes = builder(docs, edges, knowledge_source)
        docs_path = SOURCE_DIR / f"{knowledge_source}_docs.jsonl"
        edges_path = SOURCE_DIR / f"{knowledge_source}_edges.jsonl"
        write_jsonl(docs_path, output_docs)
        write_jsonl(edges_path, output_edges)
        report["ablations"][knowledge_source] = {
            "description": config["description"],
            "docs_path": str(docs_path.relative_to(ROOT)),
            "edges_path": str(edges_path.relative_to(ROOT)),
            "summary": {
                **summarize_docs(output_docs),
                **summarize_edges(output_edges),
            },
            "notes": notes,
        }

    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

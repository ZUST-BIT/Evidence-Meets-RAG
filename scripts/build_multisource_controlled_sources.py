from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from prepare_rag_knowledge_sources import (
    collect_entity_refs,
    disease_alias_values,
    doc_record,
    edge_record,
    render_evidence_claim,
    render_evidence_node_text,
    unique_preserve,
)


ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT / "kg" / "aligned" / "multisource_evidence_graph_latest.json"
OUTPUT_DIR = ROOT / "dataset" / "rag" / "knowledge_sources" / "multi_cancer"
STATS_PATH = OUTPUT_DIR / "multisource_controlled_source_stats.json"
SOURCE_NAME_PREFIX = "controlled"
SOURCE_FAMILIES = {"pubmed_literature", "tcga_linkedomics", "depmap", "integrated"}


def stable_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16].upper()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def render_triple_text(node: dict[str, Any]) -> str:
    subject = (node.get("subject") or {}).get("label") or "unknown_subject"
    predicate = str(node.get("predicate") or "related_to")
    obj = (node.get("object") or {}).get("label") or (node.get("outcome") or {}).get("label") or str(node.get("disease") or "unknown_target")
    parts = [
        f"Triple: {subject} {predicate} {obj}.",
        f"Disease: {node.get('disease')}" if node.get("disease") else "",
        f"Source family: {node.get('source_family')}",
        f"Feature layer: {node.get('feature_layer')}" if node.get("feature_layer") else "",
        f"Evidence category: {node.get('evidence_category')}" if node.get("evidence_category") else "",
        f"Summary: {node.get('summary')}" if node.get("summary") else "",
    ]
    return "\n".join(part for part in parts if part)


def render_linearized_text(node: dict[str, Any]) -> str:
    disease = node.get("disease")
    summary = node.get("summary") or ""
    raw_text = node.get("raw_text") or ""
    confidence = (node.get("confidence") or {}).get("score")
    parts = [
        f"Evidence statement: {render_evidence_claim(node)}",
        f"Disease: {disease}" if disease else "",
        f"Source family: {node.get('source_family')}",
        f"Aggregation level: {node.get('aggregation_level')}" if node.get("aggregation_level") else "",
        f"Feature layer: {node.get('feature_layer')}" if node.get("feature_layer") else "",
        f"Summary: {summary}" if summary else "",
        f"Raw evidence text: {raw_text}" if raw_text else "",
        f"Confidence score: {confidence}" if confidence is not None else "",
    ]
    return "\n".join(part for part in parts if part)


def build_filtered_graph() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    graph = read_json(GRAPH_PATH)
    evidence_nodes = [node for node in graph["nodes"] if node.get("node_type") == "evidence" and node.get("source_family") in SOURCE_FAMILIES]
    keep_ids = {node["id"] for node in evidence_nodes}
    edges = [
        edge
        for edge in graph["edges"]
        if edge.get("source") in keep_ids and edge.get("target") in keep_ids
    ]
    return evidence_nodes, edges


def build_evidence_graph() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    nodes, edges = build_filtered_graph()
    incoming_parents: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        if edge["relation"] == "DERIVED_FROM":
            incoming_parents[edge["target"]].append(edge["source"])
    docs = []
    for node in nodes:
        entity_names, entity_ids = collect_entity_refs(node)
        docs.append(
            doc_record(
                doc_id=node["id"],
                knowledge_source=f"{SOURCE_NAME_PREFIX}_evidence_graph",
                title=render_evidence_claim(node).rstrip("."),
                text=render_evidence_node_text(node),
                document_level="evidence",
                parent_id=(incoming_parents.get(node["id"]) or [None])[0],
                disease_tags=disease_alias_values(node.get("disease")),
                entity_names=entity_names,
                entity_ids=entity_ids,
                metadata={"node_type": "evidence", "source_family": node.get("source_family"), "raw_node": node},
            )
        )
    edge_rows = [
        edge_record(
            edge["source"],
            edge["target"],
            edge["relation"],
            weight=float(edge.get("support_strength") or edge.get("linking_score") or edge.get("similarity_score") or 1.0),
            metadata={"raw_edge": edge},
        )
        for edge in edges
    ]
    return docs, edge_rows, {"doc_count": len(docs), "edge_count": len(edge_rows)}


def build_triple_kg() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    nodes, _ = build_filtered_graph()
    docs = []
    for node in nodes:
        entity_names, entity_ids = collect_entity_refs(node)
        docs.append(
            doc_record(
                doc_id=f"TRIPLE-{node['id']}",
                knowledge_source=f"{SOURCE_NAME_PREFIX}_triple_kg",
                title=render_evidence_claim(node).rstrip("."),
                text=render_triple_text(node),
                document_level="triple",
                parent_id=None,
                disease_tags=disease_alias_values(node.get("disease")),
                entity_names=entity_names,
                entity_ids=entity_ids,
                metadata={"source_evidence_id": node["id"], "source_family": node.get("source_family")},
            )
        )
    return docs, [], {"doc_count": len(docs), "edge_count": 0}


def build_linearized_text() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    nodes, _ = build_filtered_graph()
    docs = []
    for node in nodes:
        entity_names, entity_ids = collect_entity_refs(node)
        docs.append(
            doc_record(
                doc_id=f"TEXT-{node['id']}",
                knowledge_source=f"{SOURCE_NAME_PREFIX}_linearized_text",
                title=render_evidence_claim(node).rstrip("."),
                text=render_linearized_text(node),
                document_level="text",
                parent_id=None,
                disease_tags=disease_alias_values(node.get("disease")),
                entity_names=entity_names,
                entity_ids=entity_ids,
                metadata={"source_evidence_id": node["id"], "source_family": node.get("source_family")},
            )
        )
    return docs, [], {"doc_count": len(docs), "edge_count": 0}


def build_structured_multiomics() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    nodes, _ = build_filtered_graph()
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        if node.get("source_family") not in {"tcga_linkedomics", "depmap"}:
            continue
        subject = node.get("subject") or {}
        subject_label = str(subject.get("normalized_label") or subject.get("label") or subject.get("id") or "").strip()
        disease = str(node.get("disease") or "").strip()
        if not subject_label or not disease:
            continue
        grouped[(disease, subject_label)].append(node)
    docs = []
    for (disease, subject_label), group_nodes in sorted(grouped.items()):
        source_counts = Counter(node.get("source_family") for node in group_nodes)
        feature_counts = Counter(node.get("feature_layer") or node.get("evidence_category") or "unknown" for node in group_nodes)
        entity_names = [subject_label]
        entity_ids = []
        top_summaries = []
        for node in sorted(group_nodes, key=lambda item: ((item.get("confidence") or {}).get("score") or 0.0, item["id"]), reverse=True)[:8]:
            names, ids = collect_entity_refs(node)
            entity_names.extend(names[:4])
            entity_ids.extend(ids[:6])
            top_summaries.append(f"- [{node.get('source_family')}] {node.get('summary') or render_evidence_claim(node)}")
        text = "\n".join(
            [
                f"Structured multi-omics profile for {subject_label} in {disease}.",
                f"Source counts: {json.dumps(dict(source_counts), ensure_ascii=False, sort_keys=True)}",
                f"Feature counts: {json.dumps(dict(feature_counts), ensure_ascii=False, sort_keys=True)}",
                "Representative evidence:",
                *top_summaries,
            ]
        )
        docs.append(
            doc_record(
                doc_id=f"OMICS-{stable_hash(f'{disease}|{subject_label}')}",
                knowledge_source=f"{SOURCE_NAME_PREFIX}_structured_multiomics",
                title=f"{subject_label} | {disease} structured multi-omics profile",
                text=text,
                document_level="profile",
                parent_id=None,
                disease_tags=disease_alias_values(disease),
                entity_names=unique_preserve(entity_names),
                entity_ids=unique_preserve(entity_ids),
                metadata={"source_counts": dict(source_counts), "feature_counts": dict(feature_counts), "evidence_ids": [node["id"] for node in group_nodes[:24]]},
            )
        )
    return docs, [], {"doc_count": len(docs), "edge_count": 0}


def main() -> None:
    builders = {
        f"{SOURCE_NAME_PREFIX}_evidence_graph": build_evidence_graph,
        f"{SOURCE_NAME_PREFIX}_triple_kg": build_triple_kg,
        f"{SOURCE_NAME_PREFIX}_linearized_text": build_linearized_text,
        f"{SOURCE_NAME_PREFIX}_structured_multiomics": build_structured_multiomics,
    }
    stats = {}
    for source_name, builder in builders.items():
        docs, edges, source_stats = builder()
        write_jsonl(OUTPUT_DIR / f"{source_name}_docs.jsonl", docs)
        write_jsonl(OUTPUT_DIR / f"{source_name}_edges.jsonl", edges)
        stats[source_name] = source_stats
    write_json(STATS_PATH, stats)


if __name__ == "__main__":
    main()

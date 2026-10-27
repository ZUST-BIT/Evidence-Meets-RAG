from __future__ import annotations

import csv
import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
from knowledge_build_paths import PROCESSED_DIR, HGNC_PATH
ALIGNED_DIR = ROOT / "kg" / "aligned"

DOID_BY_FAMILY = {
    "HCC": "DOID:684",
    "CRC": "DOID:9256",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def stable_hash(*parts: Any) -> str:
    text = "||".join("" if part is None else str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16].upper()


def norm_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(str(value).lower().replace("/", " ").replace("-", " ").split())


def build_haystack(record: dict[str, Any]) -> str:
    return " ".join(
        [
            str(record.get("oncology_focus", "")),
            str((record.get("metadata") or {}).get("oncotree_code", "")),
            str((record.get("metadata") or {}).get("oncotree_subtype", "")),
            str((record.get("metadata") or {}).get("oncotree_lineage", "")),
            str((record.get("entities") or {}).get("disease_name", "")),
            str((record.get("question") or "")),
        ]
    ).lower()


def detect_family(record: dict[str, Any]) -> str:
    hay = build_haystack(record)
    if any(
        token in hay
        for token in [
            "hepatocellular",
            " liver hepatocellular ",
            "liver hepatocellular carcinoma",
            "hepatocellular carcinoma",
            '"oncotree_code": "hcc"',
            " hcc",
            "lihc",
        ]
    ):
        return "HCC"
    if any(
        token in hay
        for token in [
            "colorectal",
            "colon adenocarcinoma",
            "colon carcinoma",
            "malignant tumor of colon",
            "colonic neoplasms",
            "rectal",
            '"oncotree_code": "coad"',
            '"oncotree_code": "read"',
            " coad",
            " read",
            " crc",
        ]
    ):
        return "CRC"
    return "OTHER"


def split_aliases(value: str | None) -> list[str]:
    if not value:
        return []
    cleaned = str(value).replace("|", ",")
    return [part.strip() for part in cleaned.split(",") if part.strip()]


def load_hgnc_by_entrez() -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    with HGNC_PATH.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            entrez_id = (row.get("entrez_id") or "").strip()
            if not entrez_id:
                continue
            aliases = []
            aliases.extend(split_aliases(row.get("alias_symbol")))
            aliases.extend(split_aliases(row.get("prev_symbol")))
            aliases = sorted({alias for alias in aliases if alias and alias != row.get("symbol")})
            mapping[entrez_id] = {
                "symbol": (row.get("symbol") or "").strip() or f"Entrez gene {entrez_id}",
                "hgnc_id": (row.get("hgnc_id") or "").strip() or None,
                "aliases": aliases,
            }
    return mapping


def unique(values: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def make_entity_ref(node: dict[str, Any], label: str) -> dict[str, Any]:
    ref = {
        "id": node["id"],
        "label": label,
        "entity_type": node["entity_type"],
    }
    if node.get("normalized_id"):
        ref["normalized_id"] = node["normalized_id"]
    if node.get("normalized_entity_type"):
        ref["normalized_entity_type"] = node["normalized_entity_type"]
    return ref


def edge(source: str, target: str, relation: str, entity_type: str, rationale: str, rule: str) -> dict[str, Any]:
    return {
        "source": source,
        "target": target,
        "relation": relation,
        "relation_family": "semantic_link",
        "entity_type": entity_type,
        "linking_score": None,
        "similarity_score": None,
        "rationale": rationale,
        "support_strength": None,
        "evidence_count": 1,
        "created_by_rule": rule,
    }


def build_disgenet_graph(hgnc_by_entrez: dict[str, dict[str, Any]], created_at: str) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = [row for row in read_jsonl(PROCESSED_DIR / "disgenet.jsonl") if detect_family(row) in {"HCC", "CRC"}]
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    entity_nodes: dict[str, dict[str, Any]] = {}
    family_counts = Counter()

    for row in rows:
        entities = row.get("entities") or {}
        metadata = row.get("metadata") or {}
        family = detect_family(row)
        family_counts[family] += 1
        gene_entrez_id = str(entities.get("gene_entrez_id") or "").strip()
        disease_cui = str(entities.get("disease_cui") or "").strip()
        disease_name = str(entities.get("disease_name") or row.get("oncology_focus") or "").strip()
        if not gene_entrez_id or not disease_name:
            continue

        hgnc = hgnc_by_entrez.get(gene_entrez_id) or {}
        gene_id = f"NCBIGene:{gene_entrez_id}"
        gene_symbol = hgnc.get("symbol") or f"Entrez gene {gene_entrez_id}"
        gene_node = entity_nodes.setdefault(
            gene_id,
            {
                "node_type": "entity",
                "id": gene_id,
                "name": gene_symbol,
                "entity_type": "Gene",
                "db_source": "NCBIGene",
                "aliases": unique([gene_symbol, f"Entrez gene {gene_entrez_id}", gene_id, hgnc.get("hgnc_id") or "", *hgnc.get("aliases", [])]),
                "normalized_id": hgnc.get("hgnc_id") or gene_id,
                "normalized_label": gene_symbol,
                "normalized_entity_type": "Gene",
                "normalized_db_source": "HGNC" if hgnc.get("hgnc_id") else "NCBIGene",
            },
        )

        disease_node_id = f"UMLS:{disease_cui}" if disease_cui else f"DISGENET-DISEASE-{stable_hash(disease_name)}"
        disease_node = entity_nodes.setdefault(
            disease_node_id,
            {
                "node_type": "entity",
                "id": disease_node_id,
                "name": disease_name,
                "entity_type": "Disease",
                "db_source": "UMLS",
                "aliases": unique([disease_name, entities.get("disease_name") or "", row.get("oncology_focus") or "", disease_cui, f"UMLS:{disease_cui}" if disease_cui else "", DOID_BY_FAMILY[family]]),
                "normalized_id": DOID_BY_FAMILY[family],
                "normalized_entity_type": "Disease",
            },
        )

        evidence_id = f"DISGENET-{stable_hash(row['id'], gene_entrez_id, disease_cui or disease_name)}"
        evidence_node = {
            "node_type": "evidence",
            "id": evidence_id,
            "source_family": "disgenet",
            "aggregation_level": "atomic",
            "evidence_type": "curated_gene_disease_association",
            "evidence_direction": "supports",
            "subject": make_entity_ref(gene_node, gene_symbol),
            "predicate": "associated_with",
            "object": make_entity_ref(disease_node, disease_name),
            "outcome": None,
            "disease": family,
            "cohort_context": {
                "disease": family,
                "cohort": "DisGeNET curated oncology subset",
                "lineage": None,
                "sample_grouping": "knowledgebase_assertion",
            },
            "comparison": {
                "comparison_type": "curated_gene_disease_assertion",
                "group_a": gene_symbol,
                "group_b": disease_name,
                "sample_grouping": "knowledgebase_assertion",
                "n_effective": 1,
            },
            "direction": "positive",
            "confidence": {
                "score": float(metadata.get("association_score") or 0.0),
                "grade": "curated_database",
                "method": "disgenet_association_score",
                "uncertainty": None,
            },
            "statistics": {
                "sample_size": 1,
                "effect_size": float(metadata.get("association_score") or 0.0),
                "effect_type": "association_score",
                "p_value": None,
                "adjusted_p_value": None,
            },
            "feature_layer": "gene_disease_association",
            "evidence_category": "knowledgebase_association",
            "source_context": {
                "source_record_id": row["id"],
                "dataset_name": "DisGeNET via TDC Harvard Dataverse mirror",
                "gene_symbol": gene_symbol,
                "gene_entrez_id": gene_entrez_id,
                "hgnc_id": hgnc.get("hgnc_id"),
                "disease_cui": disease_cui,
                "disease_name": disease_name,
                "question_anchor": row.get("question"),
                "evidence_text": row.get("evidence_text"),
            },
            "citation": {
                "doi": None,
                "pmid": None,
                "title": "DisGeNET curated gene-disease association",
                "authors": [],
                "journal": None,
                "year": None,
                "impact_factor": None,
            },
            "provenance": {
                "source_name": "DisGeNET",
                "source_family": "disgenet",
                "source_dataset": "TDC Harvard Dataverse mirror",
                "record_id": row["id"],
                "parser_version": "build_supporting_alignment_graphs_v2",
                "extracted_from": "dataset/processed/disgenet.jsonl",
            },
            "created_at": created_at,
            "updated_at": None,
            "version": "1",
            "aggregation_parent_ids": [],
        }
        nodes.append(evidence_node)
        edges.append(
            edge(
                evidence_id,
                gene_node["id"],
                "MENTIONS",
                "Gene",
                "DisGeNET association cites the gene anchor directly.",
                "disgenet:mentions_gene",
            )
        )
        edges.append(
            edge(
                evidence_id,
                disease_node["id"],
                "MENTIONS",
                "Disease",
                "DisGeNET association cites the disease anchor directly.",
                "disgenet:mentions_disease",
            )
        )

    nodes.extend(sorted(entity_nodes.values(), key=lambda item: item["id"]))
    graph = {
        "schema_version": "v2",
        "graph_metadata": {
            "graph_id": "disgenet_aligned_supporting_hcc_crc",
            "graph_name": "DisGeNET aligned supporting graph for HCC/CRC",
            "created_by": "scripts/build_supporting_alignment_graphs.py",
            "created_at": created_at,
            "source_families": ["disgenet"],
            "disease_scope": ["HCC", "CRC"],
            "build_parameters": {
                "input_path": "dataset/processed/disgenet.jsonl",
                "disease_scope": ["HCC", "CRC"],
            },
            "notes": [
                "This aligned graph materializes HCC/CRC-relevant DisGeNET gene-disease association evidence as retrieval-friendly evidence nodes.",
                "Disease refs preserve the source disease label while normalizing onto the broader HCC/CRC family for cross-source linkage.",
            ],
        },
        "nodes": nodes,
        "edges": edges,
    }
    stats = {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "evidence_nodes": len(rows),
        "entity_nodes": len(entity_nodes),
        "by_family": dict(family_counts),
    }
    return graph, stats


def build_cellline_graph(created_at: str) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = [row for row in read_jsonl(PROCESSED_DIR / "tdc_cell_line.jsonl") if detect_family(row) in {"HCC", "CRC"}]
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    entity_nodes: dict[str, dict[str, Any]] = {}
    family_counts = Counter()

    for row in rows:
        entities = row.get("entities") or {}
        metadata = row.get("metadata") or {}
        gold = row.get("gold_standard") or {}
        family = detect_family(row)
        family_counts[family] += 1
        cell_line_name = str(entities.get("cell_line_name") or "").strip()
        model_id = str(entities.get("model_id") or "").strip()
        rrid = str(entities.get("rrid") or "").strip()
        primary_disease = str((gold.get("normalized_answer") or {}).get("primary_disease") or row.get("oncology_focus") or "").strip()
        subtype = str((gold.get("normalized_answer") or {}).get("subtype") or metadata.get("oncotree_subtype") or "").strip()
        oncotree_code = str(metadata.get("oncotree_code") or "").strip()
        if not cell_line_name or not primary_disease:
            continue

        cellline_id = model_id or rrid or f"CELLLINE:{stable_hash(cell_line_name)}"
        cellline_db = "DepMap" if model_id else ("Cellosaurus" if rrid else "local")
        cellline_aliases = unique([cell_line_name, model_id, rrid])
        cellline_node = entity_nodes.setdefault(
            cellline_id,
            {
                "node_type": "entity",
                "id": cellline_id,
                "name": cell_line_name,
                "entity_type": "CellLine",
                "db_source": cellline_db,
                "aliases": cellline_aliases,
                "normalized_id": cellline_id,
                "normalized_label": cell_line_name,
                "normalized_entity_type": "CellLine",
                "normalized_db_source": cellline_db,
            },
        )

        disease_node_id = f"CELLLINE-DISEASE-{stable_hash(family, primary_disease, subtype)}"
        disease_aliases = unique([primary_disease, subtype, oncotree_code, row.get("oncology_focus") or ""])
        disease_node = entity_nodes.setdefault(
            disease_node_id,
            {
                "node_type": "entity",
                "id": disease_node_id,
                "name": primary_disease,
                "entity_type": "Disease",
                "db_source": "OncoTree",
                "aliases": disease_aliases,
                "normalized_id": DOID_BY_FAMILY[family],
                "normalized_entity_type": "Disease",
            },
        )

        evidence_id = f"CELLLINE-ANN-{stable_hash(row['id'], cell_line_name, model_id, rrid)}"
        answer_label = primary_disease if not subtype or subtype == primary_disease else f"{primary_disease} ({subtype})"
        if oncotree_code:
            answer_label = f"{answer_label} [{oncotree_code}]"
        evidence_node = {
            "node_type": "evidence",
            "id": evidence_id,
            "source_family": "cellline_annotation",
            "aggregation_level": "atomic",
            "evidence_type": "cell_line_disease_annotation",
            "evidence_direction": "supports",
            "subject": make_entity_ref(cellline_node, cell_line_name),
            "predicate": "annotated_as",
            "object": make_entity_ref(disease_node, answer_label),
            "outcome": None,
            "disease": family,
            "cohort_context": {
                "disease": family,
                "cohort": "DepMap/Cellosaurus harmonized cell-line annotations",
                "lineage": metadata.get("oncotree_lineage"),
                "sample_grouping": "cell_line_annotation",
            },
            "comparison": {
                "comparison_type": "cell_line_annotation",
                "group_a": cell_line_name,
                "group_b": answer_label,
                "sample_grouping": "cell_line_annotation",
                "n_effective": 1,
            },
            "direction": "positive",
            "confidence": {
                "score": 1.0 if metadata.get("mapping_source") == "depmap" else 0.85,
                "grade": "harmonized_mapping",
                "method": "depmap_model_or_cellosaurus_annotation",
                "uncertainty": None,
            },
            "statistics": {
                "sample_size": 1,
                "effect_size": None,
                "effect_type": None,
                "p_value": None,
                "adjusted_p_value": None,
            },
            "feature_layer": "cell_line_annotation",
            "evidence_category": "model_annotation",
            "source_context": {
                "source_record_id": row["id"],
                "dataset_name": "DepMap Model.csv + Cellosaurus fallback",
                "mapping_source": metadata.get("mapping_source"),
                "model_id": model_id or None,
                "rrid": rrid or None,
                "oncotree_code": oncotree_code or None,
                "primary_disease": primary_disease or None,
                "oncotree_lineage": metadata.get("oncotree_lineage"),
                "oncotree_subtype": subtype or None,
                "question_anchor": row.get("question"),
                "evidence_text": row.get("evidence_text"),
            },
            "citation": {
                "doi": None,
                "pmid": None,
                "title": "Harmonized DepMap/Cellosaurus cell-line disease annotation",
                "authors": [],
                "journal": None,
                "year": None,
                "impact_factor": None,
            },
            "provenance": {
                "source_name": "TDC harmonized cell-line annotations",
                "source_family": "cellline_annotation",
                "source_dataset": "DepMap Model.csv + Cellosaurus fallback",
                "record_id": row["id"],
                "parser_version": "build_supporting_alignment_graphs_v2",
                "extracted_from": "dataset/processed/tdc_cell_line.jsonl",
            },
            "created_at": created_at,
            "updated_at": None,
            "version": "1",
            "aggregation_parent_ids": [],
        }
        nodes.append(evidence_node)
        edges.append(
            edge(
                evidence_id,
                cellline_node["id"],
                "MENTIONS",
                "CellLine",
                "Annotation evidence directly names the cell line model.",
                "cellline_annotation:mentions_cellline",
            )
        )
        edges.append(
            edge(
                evidence_id,
                disease_node["id"],
                "MENTIONS",
                "Disease",
                "Annotation evidence directly names the mapped cancer type.",
                "cellline_annotation:mentions_disease",
            )
        )

    nodes.extend(sorted(entity_nodes.values(), key=lambda item: item["id"]))
    graph = {
        "schema_version": "v2",
        "graph_metadata": {
            "graph_id": "cellline_annotation_aligned_supporting_hcc_crc",
            "graph_name": "Cell-line annotation aligned supporting graph for HCC/CRC",
            "created_by": "scripts/build_supporting_alignment_graphs.py",
            "created_at": created_at,
            "source_families": ["cellline_annotation"],
            "disease_scope": ["HCC", "CRC"],
            "build_parameters": {
                "input_path": "dataset/processed/tdc_cell_line.jsonl",
                "disease_scope": ["HCC", "CRC"],
            },
            "notes": [
                "This aligned graph exposes cell-line name, DepMap model ID, RRID, and OncoTree disease annotations as evidence nodes.",
                "It is designed to strengthen anchor matching for cell-line annotation questions without changing the evidence-graph organization.",
            ],
        },
        "nodes": nodes,
        "edges": edges,
    }
    stats = {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "evidence_nodes": len(rows),
        "entity_nodes": len(entity_nodes),
        "by_family": dict(family_counts),
    }
    return graph, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the paper's supporting source graphs independently or together.")
    parser.add_argument("--sources", nargs="+", choices=["disgenet", "cellline_annotation"], default=["disgenet", "cellline_annotation"])
    parser.add_argument("--processed-root", type=Path, help="Directory containing disgenet.jsonl and/or tdc_cell_line.jsonl.")
    parser.add_argument("--hgnc-path", type=Path, help="HGNC complete-set TSV (required for DisGeNET).")
    args = parser.parse_args()
    global PROCESSED_DIR, HGNC_PATH
    if args.processed_root is not None:
        PROCESSED_DIR = args.processed_root if args.processed_root.is_absolute() else ROOT / args.processed_root
    if args.hgnc_path is not None:
        HGNC_PATH = args.hgnc_path if args.hgnc_path.is_absolute() else ROOT / args.hgnc_path
    ensure_dir(ALIGNED_DIR)
    created_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    sources = {}
    if "disgenet" in args.sources:
        disgenet_graph, disgenet_stats = build_disgenet_graph(load_hgnc_by_entrez(), created_at)
        write_json(ALIGNED_DIR / "disgenet_aligned_normalized_latest.json", disgenet_graph)
        write_json(ALIGNED_DIR / "disgenet_aligned_normalized_latest_stats.json", disgenet_stats)
        sources["disgenet"] = disgenet_stats
    if "cellline_annotation" in args.sources:
        cellline_graph, cellline_stats = build_cellline_graph(created_at)
        write_json(ALIGNED_DIR / "cellline_annotation_aligned_normalized_latest.json", cellline_graph)
        write_json(ALIGNED_DIR / "cellline_annotation_aligned_normalized_latest_stats.json", cellline_stats)
        sources["cellline_annotation"] = cellline_stats
    summary = {
        "generated_at": created_at,
        "sources": sources,
    }
    write_json(ALIGNED_DIR / "supporting_alignment_graphs_latest_stats.json", summary)


if __name__ == "__main__":
    main()

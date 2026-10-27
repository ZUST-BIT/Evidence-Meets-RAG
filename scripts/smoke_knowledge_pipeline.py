"""Exercise real graph construction and projections on small synthetic inputs.

This is a wiring/schema smoke check, not a full reconstruction or paper result.
All generated files are temporary and are removed on exit; no API is called.
"""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import tempfile
from pathlib import Path

import build_multisource_complete_graph as complete
import build_multisource_controlled_sources as controlled
import build_multisource_integrated_evidence_graph as integrate
import build_multisource_semantic_kg as semantic
import build_single_source_knowledge_forms as single
import prepare_rag_knowledge_sources as project
from paper_raw_ids import MAP_PATH, apply_paper_raw_ids


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def fixture(source: str) -> dict:
    family = "tcga_linkedomics" if source == "tcga" else "pubmed_literature"
    cohort = "COHORT-" + source.upper()
    timestamp = "2026-01-01T00:00:00Z"
    gene = {"id": "NCBIGene:25", "label": "ABL1", "entity_type": "Gene"}
    disease = {"id": "DOID:684", "label": "Hepatocellular carcinoma", "entity_type": "Disease"}
    source_file = "LinkedOmics/LIHC-TCGA/fixture.tsv" if source == "tcga" else "kg/papers-full/hcc_papers/fixture.pdf"
    evidence_id = "EVIDENCE-" + source.upper()
    node = {
        "node_type": "evidence", "id": evidence_id, "record_id": evidence_id,
        "source_family": family, "disease": disease["label"], "cohort_id": cohort,
        "subject": gene, "object": disease, "outcome": None,
        "predicate": "upregulated_in" if source == "tcga" else "promotes_mechanism",
        "direction": "increase", "polarity": "pro_tumor", "claim_scope": "synthetic",
        "hypothesis_level": "synthetic", "support": {"supports_target_discovery": True, "supports_mechanism_inference": True, "supports_drug_repositioning": False},
        "evidence_category": "omics_expression" if source == "tcga" else "mechanistic",
        "feature_layer": "expression" if source == "tcga" else "literature_mechanism",
        "aggregation_level": "atomic", "summary": "Synthetic ABL1 activity in hepatocellular carcinoma.",
        "raw_text": "Synthetic fixture only.", "metric": "synthetic_effect", "value": 1.5,
        "comparison": {"sample_grouping": "all_samples", "comparison_type": "synthetic", "comparison_label": "synthetic", "n_effective": 20},
        "statistics": {"sample_size": 20, "effect_size": 1.5},
        "confidence": {"grade": "A", "score": 0.85}, "citation": None,
        "source_context": {}, "created_at": timestamp, "version": "1", "aggregation_parent_ids": [],
        "provenance": {
            "source_family": family, "source_dataset": "synthetic", "source_release": "synthetic",
            "source_record_id": evidence_id, "source_file": source_file,
            "analysis_pipeline": "synthetic_smoke", "extractor_name": "synthetic_smoke", "extractor_version": "1",
            "extracted_at": timestamp, "study_design": "synthetic", "clinical_stage": "synthetic",
            "is_primary_observation": True, "source_priority": 1,
        },
    }
    nodes = [
        {"node_type": "entity", "id": ref["id"], "name": ref["label"], "entity_type": ref["entity_type"], "aliases": []}
        for ref in (gene, disease)
    ] + [{"node_type": "cohort", "id": cohort, "cohort_label": source + " synthetic cohort", "disease_name": disease["label"], "cohort_type": "synthetic", "n_effective": 20}, node]
    edges = [integrate.build_edge(evidence_id, ref["id"], "LINKED_TO", "semantic_anchor", "synthetic", "Synthetic fixture", None, 1.0, 1) for ref in (gene, disease)]
    edges.append(integrate.build_edge(evidence_id, cohort, "OBSERVED_IN", "semantic_anchor", "synthetic", "Synthetic fixture", None, 1.0, 1))
    return {"nodes": nodes, "edges": edges, "graph_metadata": {"graph_id": "synthetic-" + source}}


def main() -> None:
    parent = next(row for row in json.loads(MAP_PATH.read_text())["parents"] if row["source_family"] == "tcga_linkedomics")
    expected_portable = "TCGARAW-" + project.stable_hash(f"{parent['cohort_or_screen']}|{parent['feature_layer']}|{parent['source_files'][0]}")
    assert expected_portable == parent["portable_parent_id"]
    id_docs = [{"doc_id": expected_portable, "parent_id": None}, {"doc_id": expected_portable + "-BLK0001", "parent_id": expected_portable}]
    id_edges = [{"source_doc_id": expected_portable, "target_doc_id": expected_portable + "-BLK0001"}]
    id_manifest = [{"doc_id": expected_portable + "-BLK0001"}]
    apply_paper_raw_ids(id_docs, id_edges, id_manifest)
    assert id_manifest[0]["doc_id"] == parent["paper_parent_id"] + "-BLK0001"
    with tempfile.TemporaryDirectory(prefix="evidencemeetsrag-smoke-") as directory:
        root = Path(directory)
        aligned = root / "kg/aligned"
        aligned.mkdir(parents=True)
        integrate.ROOT = root
        integrate.SOURCE_PATHS = {source: aligned / (source + ".json") for source in ("tcga", "pubmed")}
        for source, path in integrate.SOURCE_PATHS.items():
            write_json(path, fixture(source))
        for name, filename in (("OUTPUT_GRAPH_PATH", "integrated.json"), ("OUTPUT_STATS_PATH", "integrated_stats.json"), ("OUTPUT_AUDIT_PATH", "integrated_audit.json"), ("OUTPUT_BRIDGE_CANDIDATES_PATH", "bridges.json")):
            setattr(integrate, name, aligned / filename)
        integrate.main()
        graph = json.loads(integrate.OUTPUT_GRAPH_PATH.read_text())
        audit = json.loads(integrate.OUTPUT_AUDIT_PATH.read_text())
        assert audit["checks"]["schema_validation"]["passed"], audit
        assert any(node.get("aggregation_level") != "atomic" for node in graph["nodes"] if node["node_type"] == "evidence")
        assert any(edge["relation"] == "DERIVED_FROM" for edge in graph["edges"])

        raw_docs = []
        manifest = []
        for source in ("tcga", "pubmed"):
            node = fixture(source)["nodes"][-1]
            identifier = "SYNTHETIC-RAW-" + source.upper()
            raw_docs.append(project.doc_record(doc_id=identifier, knowledge_source="raw_tcga_multiomics" if source == "tcga" else "raw_paper_fulltext", title="Synthetic " + source, text=node["summary"], document_level="child", parent_id=None, disease_tags=["HCC"], entity_names=["ABL1"], entity_ids=["NCBIGene:25"], metadata={"source_family": node["source_family"]}))
            manifest.append({"doc_id": identifier, "source_family": node["source_family"], "disease_family": "HCC", "source_files": [node["provenance"]["source_file"]], "cohort_or_screen": "LIHC-TCGA" if source == "tcga" else "HCC", "feature_layer": node["feature_layer"], "entity_anchors": ["ABL1"], "raw_unit_type": "table_row_block" if source == "tcga" else "fulltext_chunk", "match_type": "exact_matched", "coverage_groups": []})
        raw_dir = root / "kg/raw-multisource"
        raw_docs_path = raw_dir / "raw_multisource_docs.jsonl"
        raw_edges_path = raw_dir / "raw_multisource_edges.jsonl"
        raw_manifest_path = raw_dir / "raw_multisource_manifest.jsonl"
        write_jsonl(raw_docs_path, raw_docs)
        write_jsonl(raw_edges_path, [])
        write_jsonl(raw_manifest_path, manifest)
        complete.ROOT = root
        lineage_asset_dir = root / "configs/publication_compatibility"
        lineage_asset_dir.mkdir(parents=True)
        for name in ("observation_lineage.json.gz", "manifest.json"):
            shutil.copy2(complete.LINEAGE_PREFERENCES_PATH.parent / name, lineage_asset_dir / name)
        complete.LINEAGE_PREFERENCES_PATH = lineage_asset_dir / "observation_lineage.json.gz"
        complete.BASE_GRAPH_PATH = integrate.OUTPUT_GRAPH_PATH
        complete.RAW_DOCS_PATH = raw_docs_path
        complete.RAW_MANIFEST_PATH = raw_manifest_path
        for name, filename in (("OUTPUT_REGISTRY_PATH", "registry.json"), ("OUTPUT_REGISTRY_STATS_PATH", "registry_stats.json"), ("OUTPUT_COMPLETE_GRAPH_PATH", "complete.json"), ("OUTPUT_COMPLETE_STATS_PATH", "complete_stats.json"), ("OUTPUT_COMPLETE_AUDIT_PATH", "complete_audit.json")):
            setattr(complete, name, aligned / filename)
        complete.main()
        complete_graph = json.loads(complete.OUTPUT_COMPLETE_GRAPH_PATH.read_text())
        assert sum(node["node_type"] == "observation" for node in complete_graph["nodes"]) == 2
        assert not complete.basic_validate_graph(complete_graph)
        assert json.loads(complete.OUTPUT_COMPLETE_AUDIT_PATH.read_text())["checks"]["schema_validation"]["passed"]

        semantic.ROOT = root
        semantic.INPUT_GRAPH_PATH = complete.OUTPUT_COMPLETE_GRAPH_PATH
        semantic.OUTPUT_GRAPH_PATH = aligned / "semantic.json"
        semantic.OUTPUT_STATS_PATH = aligned / "semantic_stats.json"
        with contextlib.redirect_stdout(io.StringIO()):
            semantic.main()
        semantic_graph = json.loads(semantic.OUTPUT_GRAPH_PATH.read_text())
        assert semantic_graph["edges"] and semantic_graph["nodes"]
        assert not any(node.get("node_type") in {"evidence", "observation"} for node in semantic_graph["nodes"])

        project.ROOT = root
        project.OUTPUT_ROOT = root / "dataset/rag/knowledge_sources"
        project.MANIFEST_PATH = project.OUTPUT_ROOT / "manifest_latest.json"
        project.COMPLETE_GRAPH_PATH = complete.OUTPUT_COMPLETE_GRAPH_PATH
        project.SEMANTIC_KG_PATH = semantic.OUTPUT_GRAPH_PATH
        project.RAW_MULTISOURCE_DOCS_PATH = raw_docs_path
        project.RAW_MULTISOURCE_EDGES_PATH = raw_edges_path
        metrics = {"evidence_graph_complete": project.build_evidence_graph_complete_source("multi_cancer"), "semantic_kg": project.build_semantic_kg_source("multi_cancer"), "raw_observation_view": project.build_raw_observation_view_source("multi_cancer")}
        assert all(item["doc_count"] > 0 for item in metrics.values())
        assert metrics["raw_observation_view"]["doc_count"] == 1
        controlled.GRAPH_PATH = integrate.OUTPUT_GRAPH_PATH
        controlled.OUTPUT_DIR = project.OUTPUT_ROOT / "multi_cancer"
        controlled.STATS_PATH = controlled.OUTPUT_DIR / "controlled_stats.json"
        single.SOURCE_DIR = controlled.OUTPUT_DIR
        single.OUTPUT_DIR = project.OUTPUT_ROOT / "single_source_forms"
        with contextlib.redirect_stdout(io.StringIO()):
            controlled.main()
            single.main()
        single_report = json.loads((single.OUTPUT_DIR / "single_source_forms_report.json").read_text())
        assert all(single_report["knowledge_bases"][source + "_alignment"]["exact_match"] for source in ("tcga", "pubmed"))
        print(json.dumps({"passed": True, "fixture": "synthetic only; no paper metrics", "integrated_nodes": len(graph["nodes"]), "integrated_edges": len(graph["edges"]), "observation_nodes": 2, "semantic_nodes": len(semantic_graph["nodes"]), "semantic_edges": len(semantic_graph["edges"]), "projected_docs": {key: value["doc_count"] for key, value in metrics.items()}, "single_source_evidence_alignment": True}, indent=2))


if __name__ == "__main__":
    main()

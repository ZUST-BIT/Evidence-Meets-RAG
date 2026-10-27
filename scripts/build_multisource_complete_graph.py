from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import jsonschema
except ImportError:  # pragma: no cover
    jsonschema = None


ROOT = Path(__file__).resolve().parents[1]
ALIGNED_DIR = ROOT / "kg" / "aligned"
RAW_DIR = ROOT / "kg" / "raw-multisource"
SCHEMA_PATH = ROOT / "kg" / "schema" / "multisource_evidence_kg_schema_v2.json"

BASE_GRAPH_PATH = ALIGNED_DIR / "multisource_evidence_graph_latest.json"
RAW_DOCS_PATH = RAW_DIR / "raw_multisource_docs.jsonl"
RAW_MANIFEST_PATH = RAW_DIR / "raw_multisource_manifest.jsonl"

OUTPUT_REGISTRY_PATH = ALIGNED_DIR / "multisource_observation_registry_latest.json"
OUTPUT_REGISTRY_STATS_PATH = ALIGNED_DIR / "multisource_observation_registry_stats_latest.json"
OUTPUT_COMPLETE_GRAPH_PATH = ALIGNED_DIR / "multisource_evidence_graph_complete.json"
OUTPUT_COMPLETE_STATS_PATH = ALIGNED_DIR / "multisource_evidence_graph_complete_stats.json"
OUTPUT_COMPLETE_AUDIT_PATH = ALIGNED_DIR / "multisource_evidence_graph_complete_audit.json"
LINEAGE_PREFERENCES_PATH = ROOT / "configs" / "publication_compatibility" / "observation_lineage.json.gz"

SOURCE_PRIORITY = {
    "integrated": 0,
    "pubmed_literature": 1,
    "depmap": 2,
    "tcga_linkedomics": 3,
}

FAMILY_LABELS = {
    "HCC": "hepatocellular carcinoma",
    "CRC": "colorectal cancer",
    "NSCLC": "non-small cell lung cancer",
    "BRCA": "breast carcinoma",
    "GLIOMA": "glioma",
    "OVARIAN": "ovarian carcinoma",
    "RCC": "renal cell carcinoma",
    "AML": "acute myeloid leukemia",
    "PAAD": "pancreatic adenocarcinoma",
    "HNSC": "head and neck squamous cell carcinoma",
    "GASTROESOPH": "gastroesophageal adenocarcinoma",
}

FAMILY_ALIASES = {
    "HCC": ["HCC", "hepatocellular carcinoma", "liver hepatocellular carcinoma", "LIHC"],
    "CRC": ["CRC", "colorectal cancer", "colorectal adenocarcinoma", "COADREAD", "COAD", "READ"],
    "NSCLC": ["NSCLC", "non-small cell lung cancer", "lung adenocarcinoma", "lung squamous cell carcinoma", "LUAD", "LUSC"],
    "BRCA": ["BRCA", "breast carcinoma", "breast cancer"],
    "GLIOMA": ["GLIOMA", "glioma", "glioblastoma", "glioblastoma multiforme", "GBM"],
    "OVARIAN": ["OVARIAN", "ovarian carcinoma", "ovarian cancer", "OV"],
    "RCC": ["RCC", "renal cell carcinoma", "kidney renal clear cell carcinoma", "KIRC"],
    "AML": ["AML", "acute myeloid leukemia", "LAML"],
    "PAAD": ["PAAD", "pancreatic adenocarcinoma", "pancreatic cancer"],
    "HNSC": ["HNSC", "head and neck squamous cell carcinoma"],
    "GASTROESOPH": ["GASTROESOPH", "gastroesophageal adenocarcinoma", "esophagogastric carcinoma", "esophageal adenocarcinoma", "stomach adenocarcinoma", "gastric cancer", "STAD"],
}

CANONICAL_DISEASE_TO_FAMILY = {
    "hcc": "HCC",
    "hepatocellular carcinoma": "HCC",
    "liver hepatocellular carcinoma": "HCC",
    "liver cancer": "HCC",
    "lihc": "HCC",
    "crc": "CRC",
    "colorectal cancer": "CRC",
    "colorectal adenocarcinoma": "CRC",
    "colon adenocarcinoma": "CRC",
    "rectal adenocarcinoma": "CRC",
    "colon cancer": "CRC",
    "rectal cancer": "CRC",
    "coadread": "CRC",
    "coad": "CRC",
    "read": "CRC",
    "nsclc": "NSCLC",
    "non small cell lung cancer": "NSCLC",
    "lung adenocarcinoma": "NSCLC",
    "lung squamous cell carcinoma": "NSCLC",
    "luad": "NSCLC",
    "lusc": "NSCLC",
    "brca": "BRCA",
    "breast carcinoma": "BRCA",
    "breast cancer": "BRCA",
    "glioma": "GLIOMA",
    "glioblastoma": "GLIOMA",
    "glioblastoma multiforme": "GLIOMA",
    "gbm": "GLIOMA",
    "ovarian": "OVARIAN",
    "ovarian cancer": "OVARIAN",
    "ovarian carcinoma": "OVARIAN",
    "ov": "OVARIAN",
    "rcc": "RCC",
    "renal cell carcinoma": "RCC",
    "kidney renal clear cell carcinoma": "RCC",
    "kirc": "RCC",
    "aml": "AML",
    "acute myeloid leukemia": "AML",
    "laml": "AML",
    "paad": "PAAD",
    "pancreatic adenocarcinoma": "PAAD",
    "pancreatic cancer": "PAAD",
    "hnsc": "HNSC",
    "head and neck squamous cell carcinoma": "HNSC",
    "gastroesoph": "GASTROESOPH",
    "gastroesophageal adenocarcinoma": "GASTROESOPH",
    "esophagogastric carcinoma": "GASTROESOPH",
    "esophageal adenocarcinoma": "GASTROESOPH",
    "stomach adenocarcinoma": "GASTROESOPH",
    "gastric cancer": "GASTROESOPH",
    "stad": "GASTROESOPH",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_hash(text: str, length: int = 16) -> str:
    import hashlib

    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length].upper()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def norm_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", " ", value)).strip().lower()


def unique_preserve(values: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def family_label(family: str | None) -> str:
    return FAMILY_LABELS.get(str(family or "").upper(), str(family or ""))


def family_aliases(family: str | None) -> list[str]:
    return unique_preserve([family_label(family), *(FAMILY_ALIASES.get(str(family or "").upper(), [])), str(family or "")])


def canonical_family(value: str | None) -> str:
    normalized = norm_text(value)
    return CANONICAL_DISEASE_TO_FAMILY.get(normalized, str(value or "").upper())


def build_edge(
    source_id: str,
    target_id: str,
    relation: str,
    family: str,
    rule: str,
    rationale: str,
    support_strength: float | None = None,
    similarity_score: float | None = None,
    evidence_count: int | None = None,
) -> dict[str, Any]:
    return {
        "source": source_id,
        "target": target_id,
        "relation": relation,
        "relation_family": family,
        "entity_type": None,
        "linking_score": None,
        "similarity_score": similarity_score,
        "rationale": rationale,
        "support_strength": support_strength,
        "evidence_count": evidence_count,
        "created_by_rule": rule,
    }


def add_edge_once(edge_list: list[dict[str, Any]], seen: set[tuple[str, str, str]], edge: dict[str, Any]) -> None:
    key = (edge["source"], edge["target"], edge["relation"])
    if key in seen:
        return
    seen.add(key)
    edge_list.append(edge)


def dedupe_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for edge in edges:
        add_edge_once(deduped, seen, edge)
    return deduped


def file_match_keys(paths: list[str]) -> set[str]:
    keys = set()
    for path in paths:
        if not path:
            continue
        keys.add(norm_text(path))
        keys.add(norm_text(Path(path).name))
    return {key for key in keys if key}


def evidence_file_keys(node: dict[str, Any]) -> set[str]:
    keys = set()
    provenance = node.get("provenance") or {}
    citation = node.get("citation") or {}
    source_context = node.get("source_context") or {}
    for value in [
        provenance.get("source_file"),
        citation.get("pdf_path"),
        source_context.get("source_file"),
    ]:
        if not value:
            continue
        parts = [str(part).strip() for part in str(value).split("|")]
        keys.update(file_match_keys(parts))
    return keys


def evidence_anchor_candidates(node: dict[str, Any]) -> list[str]:
    source_context = node.get("source_context") or {}
    candidates = []
    for ref_name in ["subject", "object", "outcome"]:
        ref = node.get(ref_name) or {}
        if str(ref.get("entity_type") or "").lower() == "disease":
            continue
        for value in [ref.get("label"), ref.get("normalized_label"), ref.get("name")]:
            if value:
                candidates.append(str(value))
    for value in [
        source_context.get("gene_symbol"),
        source_context.get("oncotree_code"),
        source_context.get("oncotree_subtype"),
        source_context.get("context_gene"),
        source_context.get("gene"),
        source_context.get("drug_name"),
        source_context.get("compound_name"),
        source_context.get("therapy_name"),
        source_context.get("model_id"),
    ]:
        if value:
            candidates.append(str(value))
    return unique_preserve(candidates)


def observation_feature_match(source_family: str, feature_layer: str | None) -> list[str]:
    value = str(feature_layer or "")
    if source_family == "tcga_linkedomics":
        mapping = {
            "expression": ["expression"],
            "methylation": ["methylation"],
            "mutation": ["mutation"],
            "copy_number": ["scna"],
            "copy_number_amplification": ["scna"],
            "copy_number_deletion": ["scna"],
            "protein_abundance": ["rppa_gene", "rppa_analyte"],
            "protein_activation": ["rppa_gene", "rppa_analyte"],
        }
        return mapping.get(value, [value] if value else [])
    if source_family == "depmap":
        mapping = {
            "crispr_dependency": ["gene_effect", "gene_dependency"],
            "expression_dependency_context": ["expression"],
            "copy_number_dependency_context": ["copy_number"],
            "mutation_dependency_context": ["hotspot_mutation", "damaging_mutation"],
            "subtype_dependency_context": ["molecular_subtypes"],
            "phenotype_dependency_context": ["global_signatures"],
            "cell_state_dependency_context": ["global_signatures"],
            "gdsc_drug_response": ["drug_response"],
            "codependency": ["gene_effect", "gene_dependency"],
        }
        return mapping.get(value, [value] if value else [])
    if source_family == "pubmed_literature":
        return ["paper_fulltext"]
    return [value] if value else []


def evidence_disease_family(node: dict[str, Any]) -> str:
    source_context = node.get("source_context") or {}
    for value in [
        node.get("disease"),
        source_context.get("disease_name"),
        source_context.get("oncotree_code"),
        source_context.get("oncotree_subtype"),
        source_context.get("oncotree_lineage"),
    ]:
        family = canonical_family(value)
        if family in FAMILY_LABELS:
            return family
    return ""


def build_entity_lookup(entity_nodes: list[dict[str, Any]]) -> dict[str, set[str]]:
    lookup: dict[str, set[str]] = defaultdict(set)
    for node in entity_nodes:
        values = [node.get("name") or ""] + list(node.get("aliases") or [])
        if str(node.get("entity_type") or "").lower() == "disease":
            values.extend(family_aliases(canonical_family(node.get("name"))))
        for value in values:
            normalized = norm_text(str(value))
            if normalized:
                lookup[normalized].add(node["id"])
    return lookup


def build_cohort_lookup(cohort_nodes: list[dict[str, Any]]) -> dict[str, set[str]]:
    lookup: dict[str, set[str]] = defaultdict(set)
    for node in cohort_nodes:
        for value in [node.get("id"), node.get("cohort_label"), node.get("disease_name"), node.get("lineage")]:
            normalized = norm_text(str(value or ""))
            if normalized:
                lookup[normalized].add(node["id"])
    return lookup


def make_observation_node(manifest_row: dict[str, Any], raw_doc: dict[str, Any]) -> dict[str, Any]:
    family = str(manifest_row.get("disease_family") or "")
    source_files = [str(path) for path in manifest_row.get("source_files") or []]
    return {
        "node_type": "observation",
        "id": f"OBS-{manifest_row['doc_id']}",
        "source_family": manifest_row["source_family"],
        "disease": family_label(family),
        "disease_family": family,
        "source_doc_id": manifest_row["doc_id"],
        "parent_doc_id": raw_doc.get("parent_id"),
        "title": raw_doc.get("title"),
        "raw_unit_type": manifest_row.get("raw_unit_type") or "raw_observation",
        "feature_layer": manifest_row.get("feature_layer"),
        "source_files": source_files,
        "entity_anchors": [str(value) for value in manifest_row.get("entity_anchors") or []],
        "coverage_groups": [str(value) for value in manifest_row.get("coverage_groups") or []],
        "match_type": manifest_row.get("match_type"),
        "source_context": {
            "cohort_or_screen": manifest_row.get("cohort_or_screen"),
            "comparison_context": manifest_row.get("comparison_context"),
            "document_level": raw_doc.get("document_level"),
            "disease_tags": raw_doc.get("disease_tags") or [],
            "entity_names": raw_doc.get("entity_names") or [],
            "feature_layer": manifest_row.get("feature_layer"),
            "raw_metadata": raw_doc.get("metadata") or {},
        },
        "provenance": {
            "source_family": manifest_row["source_family"],
            "source_dataset": "raw_multisource",
            "source_release": "latest",
            "source_record_id": manifest_row["doc_id"],
            "source_file": " | ".join(source_files) if source_files else None,
            "source_url": None,
            "citation_doi": None,
            "citation_pmid": None,
            "platform": None,
            "assay": manifest_row.get("feature_layer"),
            "sequencing_or_screen_id": manifest_row.get("cohort_or_screen"),
            "batch_id": None,
            "analysis_pipeline": "observation_registry_alignment_v1",
            "extractor_name": "scripts/build_multisource_complete_graph.py",
            "extractor_version": "1.0.0",
            "extracted_at": now_iso(),
            "study_design": "raw_multisource_observation_registry",
            "clinical_stage": "raw_observation_context",
            "licence": None,
            "is_primary_observation": True,
            "source_priority": SOURCE_PRIORITY.get(manifest_row["source_family"], 99),
        },
    }


def build_observation_registry(
    base_graph: dict[str, Any],
    raw_docs: list[dict[str, Any]],
    raw_manifest_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    docs_by_id = {doc["doc_id"]: doc for doc in raw_docs}
    entity_nodes = [node for node in base_graph["nodes"] if node.get("node_type") == "entity"]
    cohort_nodes = [node for node in base_graph["nodes"] if node.get("node_type") == "cohort"]
    entity_lookup = build_entity_lookup(entity_nodes)
    cohort_lookup = build_cohort_lookup(cohort_nodes)

    observation_nodes: list[dict[str, Any]] = []
    observation_edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str]] = set()
    observation_id_by_doc_id: dict[str, str] = {}
    link_counts = Counter()

    for row in raw_manifest_rows:
        raw_doc = docs_by_id.get(row["doc_id"])
        if not raw_doc:
            continue
        observation_node = make_observation_node(row, raw_doc)
        observation_nodes.append(observation_node)
        observation_id = observation_node["id"]
        observation_id_by_doc_id[row["doc_id"]] = observation_id

        for alias in family_aliases(row.get("disease_family")):
            matched_entities = entity_lookup.get(norm_text(alias), set())
            if len(matched_entities) == 1:
                add_edge_once(
                    observation_edges,
                    seen_edges,
                    build_edge(
                        source_id=observation_id,
                        target_id=next(iter(matched_entities)),
                        relation="LINKED_TO",
                        family="observation_semantic_anchor",
                        rule="observation_registry:disease_anchor",
                        rationale="Observation node linked to the canonical disease entity implied by the raw multisource family tag.",
                        similarity_score=1.0,
                        evidence_count=1,
                    ),
                )
                link_counts["disease_entity_links"] += 1

        for anchor in row.get("entity_anchors") or []:
            matched_entities = entity_lookup.get(norm_text(str(anchor)), set())
            if len(matched_entities) == 1:
                add_edge_once(
                    observation_edges,
                    seen_edges,
                    build_edge(
                        source_id=observation_id,
                        target_id=next(iter(matched_entities)),
                        relation="LINKED_TO",
                        family="observation_semantic_anchor",
                        rule="observation_registry:entity_anchor",
                        rationale="Observation node linked to a canonical entity resolved from the raw multisource anchor list.",
                        similarity_score=0.98,
                        evidence_count=1,
                    ),
                )
                link_counts["entity_anchor_links"] += 1

        cohort_matches = cohort_lookup.get(norm_text(str(row.get("cohort_or_screen") or "")), set())
        if len(cohort_matches) == 1:
            add_edge_once(
                observation_edges,
                seen_edges,
                build_edge(
                    source_id=observation_id,
                    target_id=next(iter(cohort_matches)),
                    relation="OBSERVED_IN",
                    family="observation_semantic_anchor",
                    rule="observation_registry:cohort_anchor",
                    rationale="Observation node linked to the exact cohort or screen context named in the raw multisource manifest.",
                    similarity_score=1.0,
                    evidence_count=1,
                ),
            )
            link_counts["cohort_links"] += 1

    registry = {
        "schema_version": "2.1.0",
        "graph_metadata": {
            "graph_id": "multisource_observation_registry_latest",
            "graph_name": "Observation registry aligned to raw multisource retrieval units",
            "created_by": "scripts/build_multisource_complete_graph.py",
            "created_at": now_iso(),
            "source_families": ["pubmed_literature", "tcga_linkedomics", "depmap"],
            "disease_scope": sorted({family_label(row.get("disease_family")) for row in raw_manifest_rows if row.get("disease_family")}),
            "build_parameters": {
                "raw_docs_path": str(RAW_DOCS_PATH.relative_to(ROOT)),
                "raw_manifest_path": str(RAW_MANIFEST_PATH.relative_to(ROOT)),
                "base_graph_path": str(BASE_GRAPH_PATH.relative_to(ROOT)),
            },
            "notes": "Observation nodes are defined directly from raw_multisource retrieval units so granularity, anchor sets, and source-file coverage remain identical to the raw view.",
        },
        "nodes": sorted(observation_nodes, key=lambda item: item["id"]),
        "edges": observation_edges,
    }
    stats = {
        "generated_at": now_iso(),
        "observation_count": len(observation_nodes),
        "edge_count": len(observation_edges),
        "source_family_counts": dict(sorted(Counter(node["source_family"] for node in observation_nodes).items())),
        "raw_unit_type_counts": dict(sorted(Counter(node["raw_unit_type"] for node in observation_nodes).items())),
        "feature_layer_counts": dict(sorted(Counter(str(node.get("feature_layer") or "unknown") for node in observation_nodes).items())),
        "link_counts": dict(sorted(link_counts.items())),
    }
    return registry, observation_nodes, observation_edges, observation_id_by_doc_id, stats


def build_observation_indexes(observation_nodes: list[dict[str, Any]]) -> tuple[dict[tuple[str, str, str, str], set[str]], dict[tuple[str, str, str, str], set[str]], dict[str, str]]:
    by_source_file_anchor: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    by_source_family_anchor: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    observation_doc_map: dict[str, str] = {}
    for node in observation_nodes:
        observation_id = node["id"]
        observation_doc_map[observation_id] = node["source_doc_id"]
        source_family = node["source_family"]
        disease_family = str(node.get("disease_family") or "")
        feature_layer = str(node.get("feature_layer") or "")
        anchors = {norm_text(anchor) for anchor in node.get("entity_anchors") or [] if norm_text(anchor)}
        for source_file_key in file_match_keys(node.get("source_files") or []):
            for anchor in anchors:
                by_source_file_anchor[(source_family, disease_family, source_file_key, anchor)].add(observation_id)
        for anchor in anchors:
            by_source_family_anchor[(source_family, disease_family, feature_layer, anchor)].add(observation_id)
            by_source_family_anchor[(source_family, disease_family, "", anchor)].add(observation_id)
    return by_source_file_anchor, by_source_family_anchor, observation_doc_map


def evidence_text_candidates(node: dict[str, Any]) -> list[str]:
    candidates = []
    for value in [node.get("raw_text"), node.get("summary"), node.get("value")]:
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())
    return unique_preserve(candidates)


def normalized_tokens(value: str | None) -> set[str]:
    return {token for token in norm_text(value).split() if len(token) >= 3}


def observation_link_limit(node: dict[str, Any]) -> int:
    source_family = str(node.get("source_family") or "")
    if source_family == "tcga_linkedomics":
        return 1
    if source_family == "pubmed_literature":
        return 2
    if source_family == "depmap":
        return min(max(len(evidence_file_keys(node)), 1), 3)
    return 0


def build_raw_doc_match_cache(raw_docs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    for doc in raw_docs:
        normalized_text = norm_text(doc.get("text"))
        cache[doc["doc_id"]] = {
            "text": normalized_text,
            "tokens": {token for token in normalized_text.split() if len(token) >= 3},
        }
    return cache


def load_lineage_preferences() -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Load published tie priorities, containing IDs and order only.

    These priorities do not introduce candidates or override a higher score.
    They resolve equal-score candidates that previously depended on set order.
    """
    path = Path(os.environ.get("EVIDENCEMEETSRAG_LINEAGE_PREFERENCES", str(LINEAGE_PREFERENCES_PATH)))
    payload_bytes = path.read_bytes()
    archive_manifest = read_json(path.with_name("manifest.json"))
    expected = archive_manifest.get("observation_lineage") or {}
    if expected.get("file") != path.name or hashlib.sha256(payload_bytes).hexdigest() != expected.get("gzip_sha256"):
        raise ValueError("Publication observation-lineage gzip SHA-256 does not match manifest")
    unpacked = gzip.decompress(payload_bytes)
    if hashlib.sha256(unpacked).hexdigest() != expected.get("uncompressed_sha256"):
        raise ValueError("Publication observation-lineage payload SHA-256 does not match manifest")
    payload = json.loads(unpacked)
    choices = payload.get("choices")
    if payload.get("format_version") != 1 or not isinstance(choices, dict):
        raise ValueError("Unsupported publication observation-lineage preference format")
    count = 0
    for source, targets in choices.items():
        if not isinstance(source, str) or not isinstance(targets, list) or not all(isinstance(value, str) and value.startswith("OBS-") for value in targets):
            raise ValueError("Lineage preferences must contain evidence IDs and observation-ID lists")
        if len(targets) != len(set(targets)):
            raise ValueError("Duplicate published observation preference for " + source)
        count += len(targets)
    if payload.get("source_count") != len(choices) or payload.get("edge_count") != count:
        raise ValueError("Publication observation-lineage preference counts do not match")
    if expected.get("source_count") != len(choices) or expected.get("edge_count") != count:
        raise ValueError("Publication observation-lineage counts do not match archive manifest")
    provenance = {"sha256": hashlib.sha256(payload_bytes).hexdigest(), "source_count": len(choices), "edge_count": count, "ranking": "descending original score, published equal-score priority, observation ID", "fallback": "observation ID for unlisted candidates; current candidate filters, scores and limits are unchanged"}
    return choices, provenance


def select_observations_for_evidence(
    node: dict[str, Any],
    observation_nodes_by_id: dict[str, dict[str, Any]],
    raw_doc_match_cache: dict[str, dict[str, Any]],
    by_source_file_anchor: dict[tuple[str, str, str, str], set[str]],
    by_source_family_anchor: dict[tuple[str, str, str, str], set[str]],
    observation_doc_map: dict[str, str],
    lineage_preferences: dict[str, list[str]] | None = None,
) -> list[str]:
    source_family = str(node.get("source_family") or "")
    if source_family not in {"tcga_linkedomics", "depmap", "pubmed_literature"}:
        return []
    anchors = [norm_text(anchor) for anchor in evidence_anchor_candidates(node) if norm_text(anchor)]
    if not anchors:
        return []
    disease_family = evidence_disease_family(node)
    if disease_family not in FAMILY_LABELS:
        return []
    feature_layers = observation_feature_match(source_family, node.get("feature_layer")) or [""]
    scores: Counter[str] = Counter()
    matched_by_file = False
    for anchor in anchors:
        for source_file_key in evidence_file_keys(node):
            for observation_id in by_source_file_anchor.get((source_family, disease_family, source_file_key, anchor), set()):
                scores[observation_id] += 6
                matched_by_file = True
    if not matched_by_file:
        for anchor in anchors:
            for feature_layer in feature_layers:
                for observation_id in by_source_family_anchor.get((source_family, disease_family, feature_layer, anchor), set()):
                    scores[observation_id] += 3
            for observation_id in by_source_family_anchor.get((source_family, disease_family, "", anchor), set()):
                scores[observation_id] += 1

    evidence_texts = evidence_text_candidates(node)
    normalized_evidence_texts = [norm_text(text) for text in evidence_texts if norm_text(text)]
    evidence_token_sets = [normalized_tokens(text) for text in evidence_texts]
    if source_family == "pubmed_literature":
        for observation_id in list(scores):
            raw_doc_cache = raw_doc_match_cache.get(observation_doc_map.get(observation_id, ""), {})
            text = str(raw_doc_cache.get("text") or "")
            if any(snippet in text for snippet in normalized_evidence_texts):
                scores[observation_id] += 8
            if any(anchor in text for anchor in anchors if len(anchor) >= 3):
                scores[observation_id] += 2
            if evidence_token_sets and raw_doc_cache:
                raw_tokens = raw_doc_cache.get("tokens") or set()
                overlap = max((len(tokens & raw_tokens) for tokens in evidence_token_sets), default=0)
                scores[observation_id] += min(overlap, 6)

    published_order = (lineage_preferences or {}).get(str(node.get("id") or ""), [])
    tie_priority = {observation_id: rank for rank, observation_id in enumerate(published_order)}
    ranked_pairs = sorted(
        ((observation_id, score) for observation_id, score in scores.items() if observation_id in observation_nodes_by_id),
        key=lambda pair: (-pair[1], tie_priority.get(pair[0], len(tie_priority)), pair[0]),
    )
    if not ranked_pairs:
        return []
    max_links = observation_link_limit(node)
    if max_links <= 0:
        return []
    best_score = ranked_pairs[0][1]
    return [observation_id for observation_id, score in ranked_pairs if score >= best_score - 2][:max_links]


def build_observation_lineage_edges(
    base_graph: dict[str, Any],
    observation_nodes: list[dict[str, Any]],
    raw_docs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observation_nodes_by_id = {node["id"]: node for node in observation_nodes}
    raw_doc_match_cache = build_raw_doc_match_cache(raw_docs)
    by_source_file_anchor, by_source_family_anchor, observation_doc_map = build_observation_indexes(observation_nodes)
    lineage_preferences, preference_provenance = load_lineage_preferences()

    lineage_edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str]] = set()
    linked_counts = Counter()
    unlinked_counts = Counter()

    for node in base_graph["nodes"]:
        if node.get("node_type") != "evidence":
            continue
        source_family = str(node.get("source_family") or "")
        if source_family not in {"tcga_linkedomics", "depmap", "pubmed_literature"}:
            continue
        observation_ids = select_observations_for_evidence(
            node,
            observation_nodes_by_id,
            raw_doc_match_cache,
            by_source_file_anchor,
            by_source_family_anchor,
            observation_doc_map,
            lineage_preferences,
        )
        if not observation_ids:
            unlinked_counts[source_family] += 1
            continue
        linked_counts[source_family] += 1
        for observation_id in observation_ids:
            add_edge_once(
                lineage_edges,
                seen_edges,
                build_edge(
                    source_id=node["id"],
                    target_id=observation_id,
                    relation="DERIVED_FROM",
                    family="observation_trace",
                    rule="observation_trace:evidence_to_observation",
                    rationale="This evidence node is grounded in a raw observation unit aligned by source family, disease family, source-file coverage, and anchor overlap.",
                    similarity_score=1.0,
                    evidence_count=len(observation_ids),
                ),
            )

    return lineage_edges, {
        "generated_at": now_iso(),
        "lineage_edge_count": len(lineage_edges),
        "linked_evidence_counts": dict(sorted(linked_counts.items())),
        "unlinked_evidence_counts": dict(sorted(unlinked_counts.items())),
        "tie_breaking": preference_provenance,
    }


def build_complete_graph(
    base_graph: dict[str, Any],
    observation_nodes: list[dict[str, Any]],
    observation_edges: list[dict[str, Any]],
    observation_lineage_edges: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    nodes = list(base_graph["nodes"]) + observation_nodes
    edges = dedupe_edges(list(base_graph["edges"]) + observation_edges + observation_lineage_edges)

    complete_graph = {
        "schema_version": "2.1.0",
        "graph_metadata": {
            "graph_id": "multisource_evidence_graph_complete",
            "graph_name": "Integrated multi-source evidence graph augmented with raw observation nodes",
            "created_by": "scripts/build_multisource_complete_graph.py",
            "created_at": now_iso(),
            "source_families": ["pubmed_literature", "tcga_linkedomics", "depmap", "integrated"],
            "disease_scope": base_graph.get("graph_metadata", {}).get("disease_scope") or [],
            "build_parameters": {
                "base_graph_path": str(BASE_GRAPH_PATH.relative_to(ROOT)),
                "observation_registry_path": str(OUTPUT_REGISTRY_PATH.relative_to(ROOT)),
                "raw_docs_path": str(RAW_DOCS_PATH.relative_to(ROOT)),
                "raw_manifest_path": str(RAW_MANIFEST_PATH.relative_to(ROOT)),
            },
            "notes": "Complete graph keeps the maintained evidence graph intact and appends raw observation nodes that are definitionally aligned to raw_multisource retrieval units, integrating them through shared entity and cohort anchors.",
        },
        "nodes": sorted(nodes, key=lambda item: (item["node_type"], item["id"])),
        "edges": edges,
    }

    stats = {
        "generated_at": now_iso(),
        "graph_id": complete_graph["graph_metadata"]["graph_id"],
        "node_counts": dict(sorted(Counter(node["node_type"] for node in complete_graph["nodes"]).items())),
        "edge_relation_counts": dict(sorted(Counter(edge["relation"] for edge in complete_graph["edges"]).items())),
        "edge_family_counts": dict(sorted(Counter(edge["relation_family"] for edge in complete_graph["edges"]).items())),
        "observation_source_family_counts": dict(sorted(Counter(node["source_family"] for node in observation_nodes).items())),
        "observation_semantic_edge_counts": dict(sorted(Counter(edge["relation"] for edge in observation_edges).items())),
        "observation_lineage_edge_counts": dict(sorted(Counter(edge["relation"] for edge in observation_lineage_edges).items())),
    }
    return complete_graph, stats


def build_audit(graph: dict[str, Any]) -> dict[str, Any]:
    validation_errors = basic_validate_graph(graph)
    node_ids = {node["id"] for node in graph["nodes"]}
    orphan_edges = [edge for edge in graph["edges"] if edge["source"] not in node_ids or edge["target"] not in node_ids]
    observation_ids = {node["id"] for node in graph["nodes"] if node.get("node_type") == "observation"}
    observation_anchor_edges = [
        edge
        for edge in graph["edges"]
        if edge["source"] in observation_ids and edge["relation"] in {"LINKED_TO", "OBSERVED_IN"}
    ]
    raw_backed_evidence_ids = {
        node["id"]
        for node in graph["nodes"]
        if node.get("node_type") == "evidence" and node.get("source_family") in {"tcga_linkedomics", "depmap", "pubmed_literature"}
    }
    evidence_observation_edges = [
        edge
        for edge in graph["edges"]
        if edge["source"] in raw_backed_evidence_ids and edge["target"] in observation_ids and edge["relation"] == "DERIVED_FROM"
    ]
    evidence_with_observation = {edge["source"] for edge in evidence_observation_edges}
    linked_observation_ids = {edge["source"] for edge in observation_anchor_edges}
    unlinked_observations = [node_id for node_id in observation_ids if node_id not in linked_observation_ids]
    return {
        "generated_at": now_iso(),
        "graph_id": graph["graph_metadata"]["graph_id"],
        "checks": {
            "schema_validation": {
                "validator": "basic_validation",
                "passed": not validation_errors,
                "error_count": len(validation_errors),
                "errors": validation_errors[:50],
            },
            "orphan_edge_examples": {
                "count": len(orphan_edges),
                "examples": orphan_edges[:10],
            },
            "observation_anchor_trace": {
                "count": len(observation_anchor_edges),
                "unlinked_observation_count": len(unlinked_observations),
                "unlinked_observation_examples": unlinked_observations[:50],
            },
            "observation_lineage_trace": {
                "count": len(evidence_observation_edges),
                "linked_evidence_count": len(evidence_with_observation),
                "unlinked_evidence_count": len(raw_backed_evidence_ids - evidence_with_observation),
                "unlinked_evidence_examples": sorted(raw_backed_evidence_ids - evidence_with_observation)[:50],
            },
        },
    }


def schema_validate(graph: dict[str, Any]) -> list[str]:
    if jsonschema is None:
        return basic_validate_graph(graph)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(graph), key=lambda err: list(err.path))
    messages = []
    for error in errors[:50]:
        location = ".".join(str(part) for part in error.path)
        messages.append(f"{location}: {error.message}")
    return messages


def basic_validate_graph(graph: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    node_ids = [node["id"] for node in graph["nodes"]]
    if len(node_ids) != len(set(node_ids)):
        errors.append("duplicate node ids")
    known_node_ids = set(node_ids)
    for edge in graph["edges"]:
        if edge["source"] not in known_node_ids or edge["target"] not in known_node_ids:
            errors.append(f"orphan edge {edge}")
    return errors


def main() -> None:
    base_graph = read_json(BASE_GRAPH_PATH)
    raw_docs = read_jsonl(RAW_DOCS_PATH)
    raw_manifest_rows = read_jsonl(RAW_MANIFEST_PATH)

    registry, observation_nodes, observation_edges, observation_id_by_doc_id, registry_stats = build_observation_registry(base_graph, raw_docs, raw_manifest_rows)
    observation_lineage_edges, lineage_stats = build_observation_lineage_edges(base_graph, observation_nodes, raw_docs)
    complete_graph, complete_stats = build_complete_graph(
        base_graph,
        observation_nodes,
        observation_edges,
        observation_lineage_edges,
    )
    complete_stats["observation_lineage_stats"] = lineage_stats
    write_json(OUTPUT_REGISTRY_PATH, registry)
    write_json(OUTPUT_REGISTRY_STATS_PATH, registry_stats)
    write_json(OUTPUT_COMPLETE_GRAPH_PATH, complete_graph)
    write_json(OUTPUT_COMPLETE_STATS_PATH, complete_stats)
    complete_audit = build_audit(complete_graph)
    write_json(OUTPUT_COMPLETE_AUDIT_PATH, complete_audit)


if __name__ == "__main__":
    main()

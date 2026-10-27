from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pypdf import PdfReader
from publication_compatibility import apply_publication_order, restore_publication_runtime


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "dataset" / "rag" / "knowledge_sources"
MANIFEST_PATH = OUTPUT_ROOT / "manifest_latest.json"

GRAPH_PATH = ROOT / "kg" / "aligned" / "multisource_evidence_graph_latest.json"
COMPLETE_GRAPH_PATH = ROOT / "kg" / "aligned" / "multisource_evidence_graph_complete.json"
SEMANTIC_KG_PATH = ROOT / "kg" / "aligned" / "multisource_semantic_kg.json"
PUBMED_GRAPH_PATH = ROOT / "kg" / "aligned" / "pubmed_aligned_normalized_latest.json"
PUBMED_MULTI_CANCER_RAW_PATH = ROOT / "dataset" / "raw" / "pubmed_baseline" / "multi_cancer_pubmed_abstracts_raw.jsonl"
PRIMEKG_PATH = ROOT / "kg" / "PrimeKG" / "kg.csv"
TARKG_NODES_PATH = ROOT / "kg" / "TarKG" / "TarKG_nodes.csv"
TARKG_EDGES_PATH = ROOT / "kg" / "TarKG" / "TarKG_edges.csv"
TARKG_CROSSWALK_PATH = ROOT / "kg" / "aligned" / "tarkg_entity_crosswalk_latest.jsonl"
RAW_MULTISOURCE_DOCS_PATH = ROOT / "kg" / "raw-multisource" / "raw_multisource_docs.jsonl"
RAW_MULTISOURCE_EDGES_PATH = ROOT / "kg" / "raw-multisource" / "raw_multisource_edges.jsonl"

VIEW_CONFIG = {
    "hcc_crc": {
        "benchmark_path": ROOT / "dataset" / "evaluated" / "hcc_crc_evaluation_subset_machine_evaluable.jsonl",
        "disease_filter": {"hcc", "crc"},
        "preserve_full_graph_nodes": True,
        "supported_sources": {"evidence_graph", "evidence_graph_complete", "semantic_kg", "raw_observation_view", "pubmed_abstracts", "papers_fulltext", "primekg", "tarkg", "evidence_tarkg_hybrid"},
    },
    "multi_cancer": {
        "benchmark_path": ROOT / "dataset" / "evaluated" / "external_reasoning_benchmark_formal_v3_571.jsonl",
        "disease_filter": None,
        "preserve_full_graph_nodes": True,
        "supported_sources": {"evidence_graph", "evidence_graph_complete", "semantic_kg", "raw_observation_view", "pubmed_abstracts", "primekg", "tarkg", "evidence_tarkg_hybrid"},
    },
}

from knowledge_build_paths import PAPERS_ROOT, source_relpath

FULLTEXT_DIRS = {
    "HCC": PAPERS_ROOT / "hcc_papers",
    "CRC": PAPERS_ROOT / "crc_papers",
}
FULLTEXT_SAMPLE_SIZE = {"HCC": 500, "CRC": 500}
TRADITIONAL_DOC_MULTIPLIER = 1.05
TRADITIONAL_EDGE_MULTIPLIER = 1.05

ALL_VIEWS = list(VIEW_CONFIG)
ALL_SOURCES = ["evidence_graph", "evidence_graph_complete", "semantic_kg", "raw_observation_view", "pubmed_abstracts", "papers_fulltext", "primekg", "tarkg", "evidence_tarkg_hybrid"]
CHUNK_SIZE = 1400
CHUNK_OVERLAP = 250
ABSTRACT_CHUNK_SIZE = 1200
ABSTRACT_CHUNK_OVERLAP = 120
TRADITIONAL_TEXT_STATEMENT_LIMIT = 120

DISEASE_ALIAS_GROUPS = {
    "hcc": [
        "hepatocellular carcinoma",
        "HCC",
        "liver hepatocellular carcinoma",
        "hepatocellular cancer",
        "liver cancer",
        "LIHC",
    ],
    "crc": [
        "colorectal cancer",
        "CRC",
        "colorectal adenocarcinoma",
        "colon adenocarcinoma",
        "rectal adenocarcinoma",
        "colon cancer",
        "rectal cancer",
        "COAD",
        "READ",
        "COADREAD",
    ],
    "nsclc": [
        "non-small cell lung cancer",
        "NSCLC",
        "lung adenocarcinoma",
        "lung squamous cell carcinoma",
        "LUAD",
        "LUSC",
    ],
    "brca": [
        "breast carcinoma",
        "BRCA",
        "breast cancer",
    ],
    "glioma": [
        "glioma",
        "GLIOMA",
        "glioblastoma",
        "glioblastoma multiforme",
        "GBM",
    ],
    "ovarian": [
        "ovarian carcinoma",
        "OVARIAN",
        "ovarian cancer",
        "OV",
    ],
    "rcc": [
        "renal cell carcinoma",
        "RCC",
        "kidney renal clear cell carcinoma",
        "KIRC",
    ],
    "aml": [
        "acute myeloid leukemia",
        "AML",
        "LAML",
    ],
    "paad": [
        "pancreatic adenocarcinoma",
        "PAAD",
        "pancreatic cancer",
    ],
    "hnsc": [
        "head and neck squamous cell carcinoma",
        "HNSC",
    ],
    "gastroesoph": [
        "gastroesophageal adenocarcinoma",
        "GASTROESOPH",
        "esophagogastric carcinoma",
        "esophageal adenocarcinoma",
        "stomach adenocarcinoma",
        "gastric cancer",
        "STAD",
    ],
}


def stable_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16].upper()


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


def raw_multisource_lookup() -> dict[str, dict[str, Any]]:
    return {row["doc_id"]: row for row in read_jsonl(RAW_MULTISOURCE_DOCS_PATH)}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    runtime_report = restore_publication_runtime(path, rows)
    if runtime_report is not None:
        print(json.dumps({"publication_retrieval_runtime": runtime_report}))
    rows = apply_publication_order(path, rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def norm_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", " ", value)).strip().lower()


def normalize_bridge_namespace(namespace: str | None, db_source: str | None = None) -> str:
    db_source_map = {
        "CHEMBL": "ChEMBL",
        "DrugBank": "DrugBank",
        "NCBI": "NCBIGene",
        "MESH": "MESH",
        "PUBCHEM": "PubChemCID",
        "UniProt": "UniProtKB",
    }
    namespace_map = {
        "CHEMBL": "ChEMBL",
        "ChEMBL": "ChEMBL",
        "DrugBank": "DrugBank",
        "MESH": "MESH",
        "NCBI": "NCBIGene",
        "NCBIGene": "NCBIGene",
        "PUBCHEM": "PubChemCID",
        "PubChemCID": "PubChemCID",
        "UniProt": "UniProtKB",
        "UniProtKB": "UniProtKB",
        "crosswalk_name": "crosswalk_name",
    }
    if db_source and db_source in db_source_map:
        return db_source_map[db_source]
    return namespace_map.get(namespace or "", namespace or "")


def gene_like_tokens(value: str | None) -> set[str]:
    text = value or ""
    tokens = set(re.findall(r"\b[A-Z0-9\-]{3,12}\b", text.upper()))
    return {token for token in tokens if not token.isdigit()}


def canonicalize_disease(value: str | None) -> str:
    normalized = norm_text(value)
    if not normalized:
        return ""
    disease_aliases = {
        "hcc": {"hcc", "hepatocellular carcinoma", "liver hepatocellular carcinoma", "hepatocellular cancer", "liver cancer"},
        "crc": {"crc", "colorectal cancer", "colorectal adenocarcinoma", "colon adenocarcinoma", "rectal adenocarcinoma", "colon cancer", "rectal cancer", "coadread"},
        "nsclc": {"nsclc", "non small cell lung cancer", "non small cell lung carcinoma", "luad", "lusc"},
        "brca": {"brca", "breast cancer", "breast carcinoma"},
        "glioma": {"glioma", "glioblastoma", "gbm"},
        "ovarian": {"ovarian cancer", "ovarian carcinoma", "ov"},
        "rcc": {"rcc", "renal cell carcinoma", "kidney renal clear cell carcinoma", "kirc"},
        "aml": {"aml", "acute myeloid leukemia", "laml"},
        "paad": {"paad", "pancreatic adenocarcinoma", "pancreatic cancer"},
        "hnsc": {"hnsc", "head and neck squamous cell carcinoma"},
        "gastroesoph": {"gastroesophageal adenocarcinoma", "stomach adenocarcinoma", "gastric cancer", "stad"},
        "prad": {"prad", "prostate cancer"},
        "melanoma": {"melanoma", "skin cutaneous melanoma", "skcm"},
        "meso": {"meso", "mesothelioma"},
    }
    for canonical, aliases in disease_aliases.items():
        if normalized in aliases:
            return canonical
    return normalized


def unique_preserve(values: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def disease_alias_values(value: str | None) -> list[str]:
    normalized = norm_text(value)
    if not normalized:
        return []
    canonical = canonicalize_disease(value)
    aliases = list(DISEASE_ALIAS_GROUPS.get(canonical, []))
    aliases.append(value or "")
    return unique_preserve([alias for alias in aliases if alias])


def preferred_disease_label(value: str | None) -> str:
    aliases = disease_alias_values(value)
    return aliases[0] if aliases else (value or "")


def expand_disease_tags(*values: str | None) -> list[str]:
    tags: list[str] = []
    for value in values:
        if not value:
            continue
        tags.extend(disease_alias_values(value))
        tags.append(str(value))
    return unique_preserve([tag for tag in tags if tag])


def node_disease_values(node: dict[str, Any]) -> list[str]:
    node_type = str(node.get("node_type") or "").lower()
    values: list[str | None] = []
    if node_type == "evidence":
        source_context = node.get("source_context") or {}
        values.extend(
            [
                node.get("disease"),
                source_context.get("disease_name"),
                source_context.get("oncotree_code"),
                source_context.get("oncotree_subtype"),
                source_context.get("oncotree_lineage"),
            ]
        )
    elif node_type == "cohort":
        values.extend([node.get("disease_name"), node.get("lineage")])
    elif node_type == "observation":
        source_context = node.get("source_context") or {}
        values.extend(
            [node.get("disease"), node.get("disease_family")]
            + list(source_context.get("disease_tags") or [])
        )
    elif node_type == "entity" and str(node.get("entity_type") or "").lower() == "disease":
        values.append(node.get("name"))
        values.extend(list(node.get("aliases") or []))
    return unique_preserve([str(value) for value in values if value])


def ref_label(ref: dict[str, Any] | None) -> str:
    if not ref:
        return ""
    return str(ref.get("normalized_label") or ref.get("label") or ref.get("name") or ref.get("id") or "").strip()


def evidence_target_ref(node: dict[str, Any]) -> dict[str, Any] | None:
    outcome = node.get("outcome") or {}
    if ref_label(outcome):
        return outcome
    obj = node.get("object") or {}
    if ref_label(obj):
        return obj
    return None


def render_evidence_claim(node: dict[str, Any]) -> str:
    subject_label = ref_label(node.get("subject")) or "Unspecified subject"
    predicate_label = str(node.get("predicate") or "related_to").replace("_", " ")
    target = evidence_target_ref(node)
    target_label = ref_label(target)
    disease_label = preferred_disease_label(node.get("disease"))
    if target_label:
        if disease_label and norm_text(target_label) != norm_text(disease_label):
            return f"{subject_label} {predicate_label} {target_label} in {disease_label}."
        return f"{subject_label} {predicate_label} {target_label}."
    if disease_label:
        return f"{subject_label} {predicate_label} in {disease_label}."
    return f"{subject_label} {predicate_label}."


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def deterministic_sample(paths: list[Path], sample_size: int) -> list[Path]:
    if len(paths) <= sample_size:
        return paths
    selected = []
    last_index = len(paths) - 1
    for position in range(sample_size):
        index = round(position * last_index / max(sample_size - 1, 1))
        selected.append(paths[index])
    deduped = []
    seen = set()
    for path in selected:
        if path not in seen:
            seen.add(path)
            deduped.append(path)
    if len(deduped) < sample_size:
        for path in paths:
            if path not in seen:
                seen.add(path)
                deduped.append(path)
                if len(deduped) >= sample_size:
                    break
    return deduped[:sample_size]


def compact_json(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def view_output_dir(view_name: str) -> Path:
    return OUTPUT_ROOT / view_name


def load_existing_manifest() -> dict[str, Any]:
    if MANIFEST_PATH.exists():
        manifest = read_json(MANIFEST_PATH)
        return {
            "generated_by": "scripts/prepare_rag_knowledge_sources.py",
            "views": manifest.get("views") or {},
            "notes": manifest.get("notes")
            or [
                "Knowledge sources are prepared per evaluation view so HCC/CRC and multi-cancer use faithful corpora aligned to their own scope.",
                "Evidence-graph corpora preserve node- and edge-level content from the maintained multisource graph, including evidence, entity, and cohort nodes.",
                "HCC/CRC literature corpora include all locally available aligned PubMed article groups and all local full-text PDFs; multi-cancer can additionally use a benchmark-aware frozen PubMed abstract corpus when prepared offline.",
                "Traditional KG corpora are view-specific induced subgraphs built from PrimeKG and TarKG around benchmark diseases, therapies, and molecular seeds.",
            ],
        }
    return {
        "generated_by": "scripts/prepare_rag_knowledge_sources.py",
        "views": {},
        "notes": [
            "Knowledge sources are prepared per evaluation view so HCC/CRC and multi-cancer use faithful corpora aligned to their own scope.",
            "Evidence-graph corpora preserve node- and edge-level content from the maintained multisource graph, including evidence, entity, and cohort nodes.",
            "HCC/CRC literature corpora include all locally available aligned PubMed article groups and all local full-text PDFs; multi-cancer can additionally use a benchmark-aware frozen PubMed abstract corpus when prepared offline.",
            "Traditional KG corpora are view-specific induced subgraphs built from PrimeKG and TarKG around benchmark diseases, therapies, and molecular seeds.",
        ],
    }


def update_manifest(view_name: str, focus: dict[str, Any], updates: dict[str, Any]) -> None:
    manifest = load_existing_manifest()
    view_entry = manifest.setdefault("views", {}).setdefault(view_name, {})
    view_entry["focus_terms"] = {
        "disease_alias_count": len(focus["disease_aliases"]),
        "therapy_name_count": len(focus["therapy_names"]),
        "entity_token_count": len(focus["entity_tokens"]),
        "benchmark_sources": focus["benchmark_sources"],
    }
    view_entry.setdefault("knowledge_sources", {}).update(updates)
    write_json(MANIFEST_PATH, manifest)


def view_matches_disease(view_name: str, disease_value: str | None) -> bool:
    view_filter = VIEW_CONFIG[view_name]["disease_filter"]
    if view_filter is None:
        return True
    return canonicalize_disease(disease_value) in view_filter


def build_focus_terms(view_name: str) -> dict[str, Any]:
    path = VIEW_CONFIG[view_name]["benchmark_path"]
    disease_aliases = set()
    therapy_counter = Counter()
    entity_token_counter = Counter()
    benchmark_sources = Counter()
    for row in read_jsonl(path):
        benchmark_sources[row.get("source")] += 1
        entities = row.get("entities") or {}
        for disease_value in [entities.get("disease"), entities.get("disease_name"), row.get("oncology_focus")]:
            normalized = norm_text(disease_value)
            if normalized:
                disease_aliases.add(normalized)
        for therapy in entities.get("therapies") or []:
            normalized = norm_text(therapy)
            if normalized:
                therapy_counter[normalized] += 1
            for token in gene_like_tokens(therapy):
                entity_token_counter[token] += 1
        for value in [entities.get("molecular_profile"), row.get("question"), entities.get("cell_line_name")]:
            for token in gene_like_tokens(value):
                entity_token_counter[token] += 1
    if view_name == "hcc_crc":
        disease_aliases.update(
            {
                "hcc",
                "hepatocellular carcinoma",
                "liver hepatocellular carcinoma",
                "crc",
                "colorectal cancer",
                "colorectal adenocarcinoma",
                "colon adenocarcinoma",
                "rectal adenocarcinoma",
            }
        )
    return {
        "disease_aliases": disease_aliases,
        "therapy_names": {name for name, _ in therapy_counter.most_common()},
        "entity_tokens": set(entity_token_counter),
        "benchmark_sources": dict(sorted(benchmark_sources.items())),
    }


def doc_record(
    *,
    doc_id: str,
    knowledge_source: str,
    title: str,
    text: str,
    document_level: str,
    parent_id: str | None,
    disease_tags: list[str],
    entity_names: list[str],
    entity_ids: list[str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "doc_id": doc_id,
        "knowledge_source": knowledge_source,
        "document_level": document_level,
        "parent_id": parent_id,
        "title": title,
        "text": text,
        "disease_tags": unique_preserve(disease_tags),
        "entity_names": unique_preserve(entity_names),
        "entity_ids": unique_preserve(entity_ids),
        "metadata": metadata,
    }


def edge_record(
    source_doc_id: str,
    target_doc_id: str,
    relation: str,
    weight: float = 1.0,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "source_doc_id": source_doc_id,
        "target_doc_id": target_doc_id,
        "relation": relation,
        "weight": weight,
        "metadata": dict(metadata or {}),
    }
    return payload


def collect_entity_refs(node: dict[str, Any]) -> tuple[list[str], list[str]]:
    names = []
    ids = []
    for ref_name in ["subject", "object", "outcome"]:
        ref = node.get(ref_name) or {}
        label = ref_label(ref)
        if label:
            names.append(label)
        if ref.get("entity_type") == "Disease":
            names.extend(disease_alias_values(label))
        for alias in ref.get("aliases") or []:
            if alias:
                names.append(alias)
        if ref.get("normalized_id"):
            ids.append(ref["normalized_id"])
        if ref.get("id"):
            ids.append(ref["id"])
    source_context = node.get("source_context") or {}
    if source_context.get("gene_symbol"):
        names.append(str(source_context["gene_symbol"]))
    if source_context.get("gene_entrez_id"):
        entrez_id = str(source_context["gene_entrez_id"])
        names.append(f"Entrez gene {entrez_id}")
        ids.extend([entrez_id, f"NCBIGene:{entrez_id}"])
    if source_context.get("hgnc_id"):
        ids.append(str(source_context["hgnc_id"]))
    if source_context.get("disease_cui"):
        disease_cui = str(source_context["disease_cui"])
        ids.extend([disease_cui, f"UMLS:{disease_cui}"])
    if source_context.get("model_id"):
        ids.append(str(source_context["model_id"]))
    if source_context.get("rrid"):
        ids.append(str(source_context["rrid"]))
    if source_context.get("oncotree_code"):
        code = str(source_context["oncotree_code"])
        names.append(code)
        ids.append(code)
    if source_context.get("oncotree_subtype"):
        names.append(str(source_context["oncotree_subtype"]))
    return unique_preserve(names), unique_preserve(ids)


def render_source_anchor_line(node: dict[str, Any]) -> str:
    source_context = node.get("source_context") or {}
    anchors = []
    if source_context.get("gene_symbol") or source_context.get("gene_entrez_id") or source_context.get("hgnc_id"):
        gene_parts = []
        if source_context.get("gene_symbol"):
            gene_parts.append(f"symbol={source_context['gene_symbol']}")
        if source_context.get("gene_entrez_id"):
            gene_parts.append(f"entrez={source_context['gene_entrez_id']}")
        if source_context.get("hgnc_id"):
            gene_parts.append(f"hgnc={source_context['hgnc_id']}")
        anchors.append("Gene anchor: " + ", ".join(gene_parts))
    if source_context.get("disease_name") or source_context.get("disease_cui"):
        disease_parts = []
        if source_context.get("disease_name"):
            disease_parts.append(f"label={source_context['disease_name']}")
        if source_context.get("disease_cui"):
            disease_parts.append(f"cui={source_context['disease_cui']}")
        anchors.append("Disease anchor: " + ", ".join(disease_parts))
    if source_context.get("model_id") or source_context.get("rrid") or source_context.get("oncotree_code") or source_context.get("oncotree_subtype"):
        model_parts = []
        if source_context.get("model_id"):
            model_parts.append(f"model_id={source_context['model_id']}")
        if source_context.get("rrid"):
            model_parts.append(f"rrid={source_context['rrid']}")
        if source_context.get("oncotree_code"):
            model_parts.append(f"oncotree_code={source_context['oncotree_code']}")
        if source_context.get("oncotree_subtype"):
            model_parts.append(f"oncotree_subtype={source_context['oncotree_subtype']}")
        anchors.append("Model anchor: " + ", ".join(model_parts))
    return "\n".join(anchors)


def render_evidence_node_text(node: dict[str, Any]) -> str:
    citation = node.get("citation") or {}
    provenance = node.get("provenance") or {}
    disease_aliases = disease_alias_values(node.get("disease"))
    entity_names, entity_ids = collect_entity_refs(node)
    parts = [
        f"Claim: {render_evidence_claim(node)}",
        f"Disease aliases: {', '.join(disease_aliases)}" if len(disease_aliases) > 1 else "",
        f"Linked entities: {', '.join(entity_names[:12])}" if entity_names else "",
        f"Linked entity IDs: {', '.join(entity_ids[:12])}" if entity_ids else "",
        render_source_anchor_line(node),
        f"Disease: {node.get('disease')}",
        f"Subject: {compact_json(node.get('subject'))}",
        f"Predicate: {node.get('predicate')}",
        f"Object: {compact_json(node.get('object'))}",
        f"Outcome: {compact_json(node.get('outcome'))}",
        f"Summary: {node.get('summary')}",
        f"Raw text: {node.get('raw_text')}",
        f"Feature layer: {node.get('feature_layer')}",
        f"Evidence category: {node.get('evidence_category')}",
        f"Aggregation level: {node.get('aggregation_level')}",
        f"Direction: {node.get('direction')}",
        f"Polarity: {node.get('polarity')}",
        f"Claim scope: {node.get('claim_scope')}",
        f"Metric: {node.get('metric')}",
        f"Value: {node.get('value')}",
        f"Unit: {node.get('unit')}",
        f"Comparison: {compact_json(node.get('comparison'))}",
        f"Statistics: {compact_json(node.get('statistics'))}",
        f"Confidence: {compact_json(node.get('confidence'))}",
        f"Support flags: {compact_json(node.get('support'))}",
        f"Citation: {compact_json({'title': citation.get('title'), 'year': citation.get('year'), 'journal': citation.get('journal'), 'doi': citation.get('doi'), 'pdf_path': citation.get('pdf_path')})}",
        f"Provenance: {compact_json(provenance)}",
        f"Source context: {compact_json(node.get('source_context'))}",
    ]
    return "\n".join(part for part in parts if part and not part.endswith("None") and not part.endswith("{}"))


def predicate_phrase(predicate: str | None) -> str:
    return str(predicate or "").replace("_", " ").strip()


def compact_observation_signal_label(node: dict[str, Any]) -> str:
    predicate = str(node.get("predicate") or "")
    evidence_category = str(node.get("evidence_category") or "")
    polarity = str(node.get("polarity") or "")
    direction = str(node.get("direction") or "")
    if "favorable_prognosis" in predicate:
        return "prognostic_better_outcome"
    if "poor_prognosis" in predicate:
        return "prognostic_poor_outcome"
    if predicate in {"integrated_supports_oncogenic_mechanism_in", "promotes_mechanism"}:
        return "oncogenic_support"
    if predicate in {"integrated_supports_protective_mechanism_in", "suppresses_mechanism"}:
        return "protective_support"
    if evidence_category == "knowledgebase_association":
        return "target_association"
    if evidence_category == "drug_sensitivity":
        return "drug_sensitivity"
    if evidence_category == "clinical_prognosis":
        return f"{direction or 'clinical'}_{polarity or 'prognosis'}".strip("_")
    if polarity and direction:
        return f"{direction}_{polarity}"
    if polarity:
        return polarity
    if direction:
        return direction
    if evidence_category:
        return evidence_category
    return predicate


def observation_semantic_evidence_rank(node: dict[str, Any]) -> tuple[int, int, str]:
    source_family = str(node.get("source_family") or "")
    aggregation_level = str(node.get("aggregation_level") or "")
    predicate = str(node.get("predicate") or "")
    atomic_rank = 0 if not aggregation_level else 1
    integrated_rank = 1 if source_family == "integrated" else 0
    predicate_rank = 0 if "prognosis" in predicate else 1
    return (atomic_rank, integrated_rank + predicate_rank, predicate)


def render_observation_semantic_hints(linked_evidence_nodes: list[dict[str, Any]]) -> str:
    if not linked_evidence_nodes:
        return ""
    ranked_nodes = sorted(linked_evidence_nodes, key=observation_semantic_evidence_rank)
    signal_counter = Counter(
        compact_observation_signal_label(node)
        for node in ranked_nodes
        if compact_observation_signal_label(node)
    )
    predicate_counter = Counter(
        predicate_phrase(node.get("predicate"))
        for node in ranked_nodes
        if predicate_phrase(node.get("predicate"))
    )
    signal_labels = [label for label, _count in signal_counter.most_common(4)]
    predicate_labels = [label for label, _count in predicate_counter.most_common(4)]
    exact_claims = unique_preserve(
        [render_evidence_claim(node).rstrip(".") for node in ranked_nodes if render_evidence_claim(node)]
    )[:2]
    parts = []
    if signal_labels:
        parts.append("Example linked semantic labels: " + "; ".join(signal_labels))
    if predicate_labels:
        parts.append("Example linked evidence predicates: " + "; ".join(predicate_labels))
    if exact_claims:
        parts.append("Example linked evidence claims:\n- " + "\n- ".join(exact_claims))
    return "\n".join(parts)


def render_entity_node_text(node: dict[str, Any]) -> str:
    parts = [
        f"Name: {node.get('name')}",
        f"Entity type: {node.get('entity_type')}",
        f"DB source: {node.get('db_source')}",
        f"Aliases: {compact_json(node.get('aliases'))}",
        f"Disease aliases: {compact_json(disease_alias_values(node.get('name')))}" if str(node.get("entity_type") or "").lower() == "disease" else "",
        f"Normalized from: {compact_json(node.get('normalized_from'))}",
    ]
    return "\n".join(part for part in parts if part and not part.endswith("None") and not part.endswith("[]"))


def render_cohort_node_text(node: dict[str, Any]) -> str:
    parts = [
        f"Cohort label: {node.get('cohort_label')}",
        f"Disease: {node.get('disease_name')}",
        f"Disease aliases: {compact_json(disease_alias_values(node.get('disease_name')))}",
        f"Lineage: {node.get('lineage')}",
        f"Cohort type: {node.get('cohort_type')}",
        f"Effective N: {node.get('n_effective')}",
    ]
    return "\n".join(part for part in parts if part and not part.endswith("None"))


def render_observation_node_text(
    node: dict[str, Any],
    raw_doc: dict[str, Any] | None,
    linked_evidence_nodes: list[dict[str, Any]] | None = None,
) -> str:
    source_context = node.get("source_context") or {}
    provenance = node.get("provenance") or {}
    text_body = (raw_doc or {}).get("text") or ""
    semantic_hints = render_observation_semantic_hints(linked_evidence_nodes or [])
    parts = [
        f"Observation title: {(raw_doc or {}).get('title') or node.get('title') or node.get('source_doc_id')}",
        f"Disease: {node.get('disease')}",
        f"Disease family: {node.get('disease_family')}",
        f"Raw unit type: {node.get('raw_unit_type')}",
        f"Feature layer: {node.get('feature_layer')}",
        f"Cohort or screen: {source_context.get('cohort_or_screen')}",
        f"Source files: {compact_json(node.get('source_files'))}",
        f"Entity anchors: {compact_json(node.get('entity_anchors'))}",
        f"Coverage groups: {compact_json(node.get('coverage_groups'))}",
        f"Match type: {node.get('match_type')}",
        semantic_hints,
        f"Provenance: {compact_json(provenance)}",
        text_body,
    ]
    return "\n".join(part for part in parts if part and not part.endswith("None") and not part.endswith("{}"))


def evidence_primary_gene_symbol(node: dict[str, Any]) -> str:
    source_context = node.get("source_context") or {}
    gene_symbol = str(source_context.get("gene_symbol") or "").strip().upper()
    if gene_symbol:
        return gene_symbol
    for ref_name in ["subject", "object", "outcome"]:
        ref = node.get(ref_name) or {}
        if str(ref.get("entity_type") or "").lower() == "gene":
            label = ref_label(ref).strip().upper()
            if label:
                return label
    return ""


def build_disease_gene_groups(selected_nodes: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped_nodes: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for node in selected_nodes:
        if node.get("node_type") != "evidence":
            continue
        source_context = node.get("source_context") or {}
        disease = canonicalize_disease(
            node.get("disease")
            or source_context.get("disease_name")
            or source_context.get("oncotree_code")
            or source_context.get("oncotree_subtype")
            or source_context.get("oncotree_lineage")
        )
        gene_symbol = evidence_primary_gene_symbol(node)
        if not disease or not gene_symbol:
            continue
        grouped_nodes[(str(disease), gene_symbol)].append(node)
    return grouped_nodes


def disease_gene_cohort_contexts(nodes: list[dict[str, Any]]) -> list[str]:
    contexts = []
    for node in nodes:
        source_context = node.get("source_context") or {}
        context = (
            str(source_context.get("model_id") or "")
            or str(source_context.get("oncotree_code") or "")
            or str(source_context.get("oncotree_subtype") or "")
            or str(source_context.get("oncotree_lineage") or "")
            or str(source_context.get("disease_name") or "")
            or str(node.get("disease") or "")
        ).strip()
        if context:
            contexts.append(context)
    return unique_preserve(contexts)


def build_evidence_pathway_links(
    graph_edges: list[dict[str, Any]],
    nodes_by_id: dict[str, dict[str, Any]],
    selected_ids: set[str],
) -> dict[str, list[str]]:
    evidence_pathways: dict[str, list[str]] = defaultdict(list)
    for edge in graph_edges:
        if edge.get("relation") != "LINKED_TO":
            continue
        source_node = nodes_by_id.get(edge.get("source"))
        target_node = nodes_by_id.get(edge.get("target"))
        if not source_node or not target_node:
            continue
        pairs = [(source_node, target_node), (target_node, source_node)]
        for evidence_node, entity_node in pairs:
            if evidence_node.get("id") not in selected_ids or entity_node.get("id") not in selected_ids:
                continue
            if evidence_node.get("node_type") != "evidence":
                continue
            if entity_node.get("node_type") != "entity" or entity_node.get("entity_type") != "Pathway":
                continue
            pathway_name = str(entity_node.get("name") or "").strip()
            if pathway_name:
                evidence_pathways[evidence_node["id"]].append(pathway_name)
    return {node_id: unique_preserve(names) for node_id, names in evidence_pathways.items()}


def summarize_disease_gene_support(nodes: list[dict[str, Any]], evidence_pathways: dict[str, list[str]]) -> dict[str, Any]:
    feature_layers = Counter(str(node.get("feature_layer") or "unspecified") for node in nodes)
    evidence_categories = Counter(str(node.get("evidence_category") or "unspecified") for node in nodes)
    source_families = Counter(str((node.get("provenance") or {}).get("source_family") or "unspecified") for node in nodes)
    predicates = Counter(str(node.get("predicate") or "related_to") for node in nodes)
    aggregation_levels = Counter(str(node.get("aggregation_level") or "atomic") for node in nodes)
    cohort_contexts = disease_gene_cohort_contexts(nodes)
    pathway_names = []
    target_association_records = 0
    knowledgebase_records = 0
    dependency_records = 0
    supportive_omics_records = 0
    prognosis_records = 0
    mechanism_records = 0
    pathway_records = 0
    literature_records = 0
    role_signal_counts = Counter()
    mechanism_bucket_counts = Counter()
    source_target_counts: Counter[str] = Counter()
    source_mechanism_counts: Counter[str] = Counter()
    oncogenic_like_predicates = {
        "amplified_in",
        "upregulated_in",
        "hypomethylated_in",
        "high_expression_associated_with_poor_prognosis_in",
        "amplification_associated_with_poor_prognosis_in",
        "deletion_associated_with_favorable_prognosis_in",
        "hypermethylation_associated_with_favorable_prognosis_in",
        "integrated_supports_oncogenic_mechanism_in",
        "promotes_mechanism",
    }
    protective_like_predicates = {
        "deleted_in",
        "downregulated_in",
        "hypermethylated_in",
        "high_expression_associated_with_favorable_prognosis_in",
        "amplification_associated_with_favorable_prognosis_in",
        "deletion_associated_with_poor_prognosis_in",
        "hypermethylation_associated_with_poor_prognosis_in",
        "integrated_supports_protective_mechanism_in",
        "suppresses_mechanism",
    }
    dependency_categories = {
        "functional_dependency",
        "expression_dependency_context",
        "copy_number_dependency_context",
        "mutation_dependency_context",
        "subtype_dependency_context",
        "phenotype_dependency_context",
        "cell_state_dependency_context",
        "gene_codependency",
        "drug_sensitivity",
    }
    for node in nodes:
        source_family = str((node.get("provenance") or {}).get("source_family") or "unspecified")
        evidence_category = str(node.get("evidence_category") or "")
        feature_layer = str(node.get("feature_layer") or "")
        predicate = str(node.get("predicate") or "")
        mechanism_bucket = str(((node.get("source_context") or {}).get("mechanism_bucket")) or "")
        pathway_hits = evidence_pathways.get(str(node.get("id")), [])
        node_has_target_association = False
        node_has_mechanism = False
        if pathway_hits:
            pathway_names.extend(pathway_hits)
            pathway_records += 1
        if evidence_category == "knowledgebase_association":
            knowledgebase_records += 1
            target_association_records += 1
            source_target_counts[source_family] += 1
            node_has_target_association = True
        if evidence_category in dependency_categories or predicate in {
            "selective_dependency_in",
            "high_expression_associated_with_dependency_in",
            "copy_number_gain_associated_with_dependency_in",
            "sensitive_to",
        }:
            dependency_records += 1
            target_association_records += 1
            source_target_counts[source_family] += 1
            node_has_target_association = True
        if evidence_category in {"expression_regulation", "epigenetic_regulation", "gene_variation", "protein_abundance"} or feature_layer in {
            "expression",
            "methylation",
            "copy_number",
            "copy_number_deletion",
            "copy_number_amplification",
            "mutation",
            "protein_abundance",
        }:
            supportive_omics_records += 1
        if evidence_category == "clinical_prognosis" or "prognosis" in predicate:
            prognosis_records += 1
        if evidence_category in {"literature_mechanism", "aggregated_cross_omics_mechanism", "aggregated_same_claim"} or "mechanism" in predicate or mechanism_bucket:
            mechanism_records += 1
            source_mechanism_counts[source_family] += 1
            node_has_mechanism = True
        if source_family == "pubmed_literature" or evidence_category == "literature_mechanism":
            literature_records += 1
        if mechanism_bucket:
            mechanism_bucket_counts[mechanism_bucket] += 1
        if predicate in oncogenic_like_predicates or mechanism_bucket == "oncogenic_support":
            role_signal_counts["oncogenic_like"] += 1
        elif predicate in protective_like_predicates or mechanism_bucket == "protective_support":
            role_signal_counts["protective_like"] += 1
        elif node_has_target_association or node_has_mechanism or pathway_hits:
            role_signal_counts["context_only"] += 1
        else:
            role_signal_counts["neutral"] += 1
    source_family_count = sum(1 for name in source_families if name and name != "unspecified")
    feature_layer_count = sum(1 for name in feature_layers if name and name != "unspecified")
    cohort_context_count = len(cohort_contexts)
    member_count = len(nodes)
    cross_source_support = source_family_count >= 2 or any(
        len(set((node.get("source_context") or {}).get("member_source_families") or [])) >= 2 for node in nodes
    )
    if role_signal_counts["oncogenic_like"] and not role_signal_counts["protective_like"]:
        predominant_role = "oncogenic_like"
    elif role_signal_counts["protective_like"] and not role_signal_counts["oncogenic_like"]:
        predominant_role = "protective_like"
    elif role_signal_counts["oncogenic_like"] >= role_signal_counts["protective_like"] * 2 and role_signal_counts["oncogenic_like"] - role_signal_counts["protective_like"] >= 2:
        predominant_role = "oncogenic_like"
    elif role_signal_counts["protective_like"] >= role_signal_counts["oncogenic_like"] * 2 and role_signal_counts["protective_like"] - role_signal_counts["oncogenic_like"] >= 2:
        predominant_role = "protective_like"
    else:
        predominant_role = "mixed_or_unclear"
    return {
        "member_count": member_count,
        "feature_layers": feature_layers,
        "evidence_categories": evidence_categories,
        "source_families": source_families,
        "predicates": predicates,
        "aggregation_levels": aggregation_levels,
        "source_family_count": source_family_count,
        "feature_layer_count": feature_layer_count,
        "cohort_context_count": cohort_context_count,
        "cohort_contexts": cohort_contexts,
        "pathway_names": unique_preserve(pathway_names),
        "target_association_records": target_association_records,
        "knowledgebase_records": knowledgebase_records,
        "dependency_records": dependency_records,
        "supportive_omics_records": supportive_omics_records,
        "prognosis_records": prognosis_records,
        "mechanism_records": mechanism_records,
        "pathway_records": pathway_records,
        "literature_records": literature_records,
        "cross_source_support": cross_source_support,
        "role_signal_counts": dict(role_signal_counts),
        "predominant_role": predominant_role,
        "mechanism_bucket_counts": dict(mechanism_bucket_counts),
        "source_target_counts": dict(source_target_counts),
        "source_mechanism_counts": dict(source_mechanism_counts),
    }


def build_evidence_summary_docs(
    disease_gene_groups: dict[tuple[str, str], list[dict[str, Any]]],
    evidence_pathways: dict[str, list[str]],
    *,
    knowledge_source_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    docs = []
    edges = []
    for (disease, gene_symbol), nodes in sorted(disease_gene_groups.items()):
        profile = summarize_disease_gene_support(nodes, evidence_pathways)
        disease_label = preferred_disease_label(disease) or disease
        feature_layers = profile["feature_layers"]
        evidence_categories = profile["evidence_categories"]
        source_families = profile["source_families"]
        predicates = profile["predicates"]
        example_claims = unique_preserve([render_evidence_claim(node) for node in nodes if render_evidence_claim(node)])[:6]
        member_ids = [node["id"] for node in nodes if node.get("id")]
        if not member_ids:
            continue
        if len(member_ids) < 2 and not (
            profile["target_association_records"]
            or profile["mechanism_records"]
            or profile["pathway_records"]
        ):
            continue

        target_summary_id = f"TARGETSUM-{stable_hash(f'{disease}|{gene_symbol}|{len(member_ids)}')}"
        target_title = f"{gene_symbol} target-disease association summary in {disease_label}"
        target_text_parts = [
            f"Summary scope: target_disease_association_strength, intogen_support_strength.",
            f"Summary: {gene_symbol} has {len(nodes)} evidence records in {disease_label}.",
            "Association evidence inventory: "
            + "; ".join(
                [
                    f"direct_target_association={profile['target_association_records']}",
                    f"knowledgebase_association={profile['knowledgebase_records']}",
                    f"dependency_or_drug_sensitivity={profile['dependency_records']}",
                    f"supportive_omics={profile['supportive_omics_records']}",
                    f"clinical_prognosis={profile['prognosis_records']}",
                    f"source_families={profile['source_family_count']}",
                ]
            ),
            "Association components present: "
            + "; ".join(
                [
                    f"direct_target_association={'yes' if profile['target_association_records'] else 'no'}",
                    f"knowledgebase_association={'yes' if profile['knowledgebase_records'] else 'no'}",
                    f"dependency_vulnerability={'yes' if profile['dependency_records'] else 'no'}",
                    f"supportive_omics={'yes' if profile['supportive_omics_records'] else 'no'}",
                    f"clinical_prognosis={'yes' if profile['prognosis_records'] else 'no'}",
                    f"cross_source_support={'yes' if profile['cross_source_support'] else 'no'}",
                ]
            ),
            "Association by source family: " + ", ".join(f"{name} ({count})" for name, count in sorted(profile["source_target_counts"].items())) if profile["source_target_counts"] else "",
            "Evidence categories: " + ", ".join(f"{name} ({count})" for name, count in evidence_categories.most_common(6)),
            "Feature layers: " + ", ".join(f"{name} ({count})" for name, count in feature_layers.most_common(6)),
            "Source families: " + ", ".join(f"{name} ({count})" for name, count in source_families.most_common(6)),
            "Predicates: " + ", ".join(f"{name} ({count})" for name, count in predicates.most_common(8)),
            "Example claims:\n- " + "\n- ".join(example_claims) if example_claims else "",
        ]
        docs.append(
            doc_record(
                doc_id=target_summary_id,
                knowledge_source=knowledge_source_name,
                title=target_title,
                text="\n".join(part for part in target_text_parts if part),
                document_level="parent",
                parent_id=None,
                disease_tags=disease_alias_values(disease),
                entity_names=[gene_symbol, disease_label],
                entity_ids=unique_preserve(
                    [str((node.get("source_context") or {}).get("hgnc_id") or "") for node in nodes]
                    + [str((node.get("subject") or {}).get("normalized_id") or "") for node in nodes]
                ),
                metadata={
                    "node_type": "target_support_summary",
                    "summary_scope": "target_disease_association",
                    "task_scopes": ["target_disease_association_strength", "intogen_support_strength"],
                    "disease": disease,
                    "gene_symbol": gene_symbol,
                    "member_count": len(member_ids),
                    "derived_from_node_ids": member_ids,
                    "feature_layer_counts": dict(feature_layers),
                    "evidence_category_counts": dict(evidence_categories),
                    "source_family_counts": dict(source_families),
                    "predicate_counts": dict(predicates),
                    "association_inventory": {
                        "direct_target_association": profile["target_association_records"],
                        "knowledgebase_association": profile["knowledgebase_records"],
                        "dependency_or_drug_sensitivity": profile["dependency_records"],
                        "supportive_omics": profile["supportive_omics_records"],
                        "clinical_prognosis": profile["prognosis_records"],
                        "cross_source_support": profile["cross_source_support"],
                    },
                    "source_target_counts": profile["source_target_counts"],
                },
            )
        )

        mechanism_summary_id = f"MECHSUM-{stable_hash(f'{disease}|{gene_symbol}|{len(member_ids)}|mechanism')}"
        mechanism_title = f"{gene_symbol} driver-role and support bundle summary in {disease_label}"
        mechanism_text_parts = [
            f"Summary scope: cross_source_support_bundle, cross_source_corroboration_status, driver_role_assignment, pathway_class_membership.",
            "Bundle component inventory: "
            + "; ".join(
                [
                    f"target_association={'yes' if profile['target_association_records'] else 'no'}",
                    f"mechanism_context={'yes' if profile['mechanism_records'] else 'no'}",
                    f"pathway_context={'yes' if profile['pathway_records'] else 'no'}",
                    f"literature_mechanism={'yes' if profile['literature_records'] else 'no'}",
                    f"dependency_vulnerability={'yes' if profile['dependency_records'] else 'no'}",
                    f"cross_source_support={'yes' if profile['cross_source_support'] else 'no'}",
                ]
            ),
            "Driver-role signal inventory: "
            + "; ".join(
                [
                    f"oncogenic_like={profile['role_signal_counts'].get('oncogenic_like', 0)}",
                    f"protective_like={profile['role_signal_counts'].get('protective_like', 0)}",
                    f"context_only={profile['role_signal_counts'].get('context_only', 0)}",
                ]
            ),
            f"Predominant driver-role signal: {profile['predominant_role']}.",
            "Mechanism buckets: " + ", ".join(f"{name} ({count})" for name, count in sorted(profile["mechanism_bucket_counts"].items())) if profile["mechanism_bucket_counts"] else "",
            "Mechanism by source family: " + ", ".join(f"{name} ({count})" for name, count in sorted(profile["source_mechanism_counts"].items())) if profile["source_mechanism_counts"] else "",
            "Linked pathways in graph: " + ", ".join(profile["pathway_names"][:10]) if profile["pathway_names"] else "Linked pathways in graph: none.",
            "Example claims:\n- " + "\n- ".join(example_claims) if example_claims else "",
        ]
        docs.append(
            doc_record(
                doc_id=mechanism_summary_id,
                knowledge_source=knowledge_source_name,
                title=mechanism_title,
                text="\n".join(part for part in mechanism_text_parts if part),
                document_level="parent",
                parent_id=None,
                disease_tags=disease_alias_values(disease),
                entity_names=[gene_symbol, disease_label] + profile["pathway_names"][:8],
                entity_ids=unique_preserve(
                    [str((node.get("source_context") or {}).get("hgnc_id") or "") for node in nodes]
                    + [str((node.get("subject") or {}).get("normalized_id") or "") for node in nodes]
                ),
                metadata={
                    "node_type": "mechanism_role_summary",
                    "summary_scope": "mechanism_role_bundle",
                    "task_scopes": ["cross_source_support_bundle", "cross_source_corroboration_status", "driver_role_assignment", "pathway_class_membership"],
                    "disease": disease,
                    "gene_symbol": gene_symbol,
                    "member_count": len(member_ids),
                    "derived_from_node_ids": member_ids,
                    "bundle_components": {
                        "target_association": bool(profile["target_association_records"]),
                        "mechanism_context": bool(profile["mechanism_records"]),
                        "pathway_context": bool(profile["pathway_records"]),
                        "literature_mechanism": bool(profile["literature_records"]),
                        "dependency_vulnerability": bool(profile["dependency_records"]),
                        "cross_source_support": bool(profile["cross_source_support"]),
                    },
                    "role_signal_counts": profile["role_signal_counts"],
                    "predominant_role": profile["predominant_role"],
                    "pathway_names": profile["pathway_names"],
                    "mechanism_bucket_counts": profile["mechanism_bucket_counts"],
                    "source_mechanism_counts": profile["source_mechanism_counts"],
                },
            )
        )
        for member_id in member_ids:
            edges.append(edge_record(target_summary_id, member_id, "DERIVED_FROM", 1.0, metadata={"summary_scope": "target_disease_association"}))
            edges.append(edge_record(mechanism_summary_id, member_id, "DERIVED_FROM", 1.0, metadata={"summary_scope": "mechanism_role_bundle"}))
    return docs, edges


def build_graph_source(
    view_name: str,
    *,
    graph_path: Path,
    docs_filename: str,
    edges_filename: str,
    knowledge_source_name: str,
) -> dict[str, Any]:
    graph = read_json(graph_path)
    nodes_by_id = {node["id"]: node for node in graph["nodes"]}
    raw_lookup = raw_multisource_lookup() if any(node.get("node_type") == "observation" for node in graph["nodes"]) else {}
    selected_ids: set[str]
    if VIEW_CONFIG[view_name].get("preserve_full_graph_nodes"):
        selected_ids = set(nodes_by_id)
    elif VIEW_CONFIG[view_name]["disease_filter"] is None:
        selected_ids = set(nodes_by_id)
    else:
        anchor_ids = {
            node["id"]
            for node in graph["nodes"]
            if any(view_matches_disease(view_name, value) for value in node_disease_values(node))
        }
        selected_ids = set(anchor_ids)
        for edge in graph["edges"]:
            if edge["source"] in anchor_ids or edge["target"] in anchor_ids:
                if edge["source"] in nodes_by_id:
                    selected_ids.add(edge["source"])
                if edge["target"] in nodes_by_id:
                    selected_ids.add(edge["target"])

    incoming_parents: dict[str, list[str]] = defaultdict(list)
    edges = []
    for edge in graph["edges"]:
        if edge["source"] not in nodes_by_id or edge["target"] not in nodes_by_id:
            continue
        if edge["source"] not in selected_ids or edge["target"] not in selected_ids:
            continue
        if edge["relation"] == "DERIVED_FROM":
            incoming_parents[edge["target"]].append(edge["source"])
        edges.append(
            edge_record(
                edge["source"],
                edge["target"],
                edge["relation"],
                weight=float(edge.get("support_strength") or edge.get("linking_score") or edge.get("similarity_score") or 1.0),
                metadata={"raw_edge": edge},
            )
        )

    docs = []
    node_type_counts = Counter()
    selected_nodes = []
    selected_node_list = [node for node in nodes_by_id.values() if node["id"] in selected_ids]
    evidence_pathways = build_evidence_pathway_links(graph["edges"], nodes_by_id, selected_ids)
    disease_gene_groups = build_disease_gene_groups(selected_node_list)
    summary_docs, summary_edges = build_evidence_summary_docs(
        disease_gene_groups,
        evidence_pathways,
        knowledge_source_name=knowledge_source_name,
    )
    incoming_evidence_nodes = {
        target_id: [
            nodes_by_id[source_id]
            for source_id in source_ids
            if source_id in nodes_by_id and nodes_by_id[source_id].get("node_type") == "evidence"
        ]
        for target_id, source_ids in incoming_parents.items()
    }
    for node_id in selected_ids:
        if node_id not in nodes_by_id:
            continue
        node = nodes_by_id[node_id]
        selected_nodes.append(node)
        node_type = node.get("node_type")
        node_type_counts[node_type] += 1
        disease_tags: list[str] = []
        entity_names: list[str] = []
        entity_ids: list[str] = []
        parent_id: str | None = None
        if node_type == "evidence":
            source_context = node.get("source_context") or {}
            disease_tags = expand_disease_tags(
                node.get("disease"),
                source_context.get("disease_name"),
                source_context.get("oncotree_code"),
                source_context.get("oncotree_subtype"),
                source_context.get("oncotree_lineage"),
            )
            entity_names, entity_ids = collect_entity_refs(node)
            parent_id = (incoming_parents.get(node_id) or [None])[0]
            title = render_evidence_claim(node).rstrip(".")
            text = render_evidence_node_text(node)
            document_level = "parent" if node.get("aggregation_level") else "chunk"
        elif node_type == "observation":
            source_context = node.get("source_context") or {}
            raw_doc = raw_lookup.get(str(node.get("source_doc_id") or ""))
            disease_tags = expand_disease_tags(node.get("disease"), node.get("disease_family"), *(source_context.get("disease_tags") or []))
            entity_names = unique_preserve(
                list(node.get("entity_anchors") or [])
                + list(source_context.get("entity_names") or [])
                + [str(node.get("disease") or ""), str(node.get("disease_family") or "")]
            )
            entity_ids = []
            parent_id = None
            title = (raw_doc or {}).get("title") or node.get("title") or node.get("source_doc_id") or node_id
            text = render_observation_node_text(node, raw_doc, incoming_evidence_nodes.get(node_id))
            document_level = "node"
        elif node_type == "entity":
            if node.get("entity_type", "").lower() == "disease":
                disease_tags = expand_disease_tags(node.get("name"), *(node.get("aliases") or []))
            entity_names = [node.get("name") or ""] + list(node.get("aliases") or [])
            if node.get("entity_type", "").lower() == "disease":
                entity_names.extend(disease_alias_values(node.get("name")))
            entity_ids = [node.get("id") or ""]
            title = f"{node.get('name')} | {node.get('entity_type')}"
            text = render_entity_node_text(node)
            document_level = "node"
        else:
            disease_tags = expand_disease_tags(node.get("disease_name"), node.get("lineage"))
            title = node.get("cohort_label") or node.get("id")
            text = render_cohort_node_text(node)
            document_level = "node"
        docs.append(
            doc_record(
                doc_id=node_id,
                knowledge_source=knowledge_source_name,
                title=title,
                text=text,
                document_level=document_level,
                parent_id=parent_id,
                disease_tags=disease_tags,
                entity_names=entity_names,
                entity_ids=entity_ids,
                metadata={"node_type": node_type, "raw_node": node, "node_disease_values": node_disease_values(node)},
            )
        )

    if summary_docs:
        docs.extend(summary_docs)
        edges.extend(summary_edges)
        for doc in summary_docs:
            node_type_counts[doc["metadata"]["node_type"]] += 1

    output_dir = view_output_dir(view_name)
    docs_path = output_dir / docs_filename
    edges_path = output_dir / edges_filename
    write_jsonl(docs_path, docs)
    write_jsonl(edges_path, edges)
    return {
        "docs_path": str(docs_path.relative_to(ROOT)),
        "edges_path": str(edges_path.relative_to(ROOT)),
        "doc_count": len(docs),
        "edge_count": len(edges),
        "node_type_counts": dict(sorted(node_type_counts.items())),
    }


def build_evidence_graph_source(view_name: str) -> dict[str, Any]:
    return build_graph_source(
        view_name,
        graph_path=GRAPH_PATH,
        docs_filename="evidence_graph_docs.jsonl",
        edges_filename="evidence_graph_edges.jsonl",
        knowledge_source_name="evidence_graph",
    )


def build_evidence_graph_complete_source(view_name: str) -> dict[str, Any]:
    return build_graph_source(
        view_name,
        graph_path=COMPLETE_GRAPH_PATH,
        docs_filename="evidence_graph_complete_docs.jsonl",
        edges_filename="evidence_graph_complete_edges.jsonl",
        knowledge_source_name="evidence_graph_complete",
    )


def semantic_kg_view_allows_edge(view_name: str, edge: dict[str, Any]) -> bool:
    if view_name == "multi_cancer":
        return True
    families = {str(value).upper() for value in edge.get("disease_families") or []}
    return bool(families.intersection({"HCC", "CRC"}))


def semantic_kg_edge_statement(edge: dict[str, Any], nodes_by_id: dict[str, dict[str, Any]]) -> str:
    source_name = str((nodes_by_id.get(edge.get("source")) or {}).get("canonical_name") or edge.get("source"))
    target_name = str((nodes_by_id.get(edge.get("target")) or {}).get("canonical_name") or edge.get("target"))
    statement = str(edge.get("statement_text") or "").strip()
    if statement:
        return statement
    return f"{source_name} {edge.get('relation_type', 'related_to').lower()} {target_name}."


def build_semantic_kg_source(view_name: str) -> dict[str, Any]:
    graph = read_json(SEMANTIC_KG_PATH)
    nodes_by_id = {node["id"]: node for node in graph["nodes"]}
    selected_edges = [edge for edge in graph["edges"] if semantic_kg_view_allows_edge(view_name, edge)]
    selected_node_ids = {
        node_id
        for edge in selected_edges
        for node_id in [edge.get("source"), edge.get("target")]
        if node_id in nodes_by_id
    }
    selected_nodes = [nodes_by_id[node_id] for node_id in selected_node_ids]
    node_text_examples: dict[str, list[str]] = defaultdict(list)
    statement_counts: Counter[str] = Counter()
    node_relation_counts: dict[str, Counter[str]] = defaultdict(Counter)
    node_neighbor_names: dict[str, list[str]] = defaultdict(list)
    graph_edges = []

    for edge in selected_edges:
        source_id = str(edge.get("source") or "")
        target_id = str(edge.get("target") or "")
        if source_id not in nodes_by_id or target_id not in nodes_by_id:
            continue
        statement = semantic_kg_edge_statement(edge, nodes_by_id)
        append_statement(node_text_examples[source_id], statement)
        append_statement(node_text_examples[target_id], statement)
        statement_counts[source_id] += 1
        statement_counts[target_id] += 1
        relation_type = str(edge.get("relation_type") or "RELATED_TO")
        node_relation_counts[source_id][relation_type] += 1
        node_relation_counts[target_id][relation_type] += 1
        source_name = str((nodes_by_id.get(source_id) or {}).get("canonical_name") or source_id)
        target_name = str((nodes_by_id.get(target_id) or {}).get("canonical_name") or target_id)
        if target_name and target_name not in node_neighbor_names[source_id]:
            node_neighbor_names[source_id].append(target_name)
        if source_name and source_name not in node_neighbor_names[target_id]:
            node_neighbor_names[target_id].append(source_name)
        graph_edges.append(
            edge_record(
                source_id,
                target_id,
                relation_type,
                float(edge.get("confidence_score") or 1.0),
                metadata={"raw_edge": edge, "source_graph": "Semantic KG"},
            )
        )

    docs = []
    for node in selected_nodes:
        title = str(node.get("canonical_name") or node.get("id"))
        node_type = str(node.get("node_type") or "Entity")
        disease_families = [str(value) for value in node.get("disease_families") or []]
        source_families = [str(value) for value in node.get("source_families") or []]
        disease_tags = disease_alias_values(title) if node_type == "Disease" else disease_families
        relation_inventory = ", ".join(f"{name} ({count})" for name, count in node_relation_counts[node["id"]].most_common())
        neighbor_text = "; ".join(node_neighbor_names.get(node["id"]) or [])
        raw_node = node
        feature_relation = str(raw_node.get("feature_relation_type") or "")
        feature_group = str(raw_node.get("feature_relation_group") or "")
        support_count = raw_node.get("support_count")
        predicate_inventory = compact_json(raw_node.get("predicate_counts"))
        source_ids = list(node.get("source_ids") or [])
        evidence_examples = raw_node.get("supporting_statement_examples") or []
        text_parts = [
            f"Title: {title}",
            f"Node type: {node_type}",
            "Source graph: Semantic KG",
            f"Statement count in selected subgraph: {statement_counts[node['id']]}",
            "Source families: " + ", ".join(source_families) if source_families else "",
            "Disease families: " + ", ".join(disease_families) if disease_families else "",
            "Source IDs: " + ", ".join(source_ids) if source_ids else "",
            "Relation inventory: " + relation_inventory if relation_inventory else "",
            "Neighbor examples: " + neighbor_text if neighbor_text else "",
            f"Feature relation type: {feature_relation}" if feature_relation else "",
            f"Feature relation group: {feature_group}" if feature_group else "",
            f"Feature support count: {support_count}" if support_count not in (None, "") else "",
            f"Predicate inventory: {predicate_inventory}" if predicate_inventory else "",
            "Supporting evidence statements:\n- " + "\n- ".join(evidence_examples) if evidence_examples else "",
            "Local semantic statements:\n- " + "\n- ".join(node_text_examples.get(node["id"]) or []) if node_text_examples.get(node["id"]) else "",
        ]
        docs.append(
            doc_record(
                doc_id=node["id"],
                knowledge_source="semantic_kg",
                title=title,
                text="\n".join(part for part in text_parts if part),
                document_level="node",
                parent_id=None,
                disease_tags=disease_tags,
                entity_names=[title] + list(node.get("aliases") or []) + disease_families,
                entity_ids=list(node.get("source_ids") or []) + [node["id"]],
                metadata={
                    "node_type": node_type,
                    "source_graph": "Semantic KG",
                    "statement_count": statement_counts[node["id"]],
                    "relation_counts": dict(node_relation_counts[node["id"]]),
                    "neighbor_names": node_neighbor_names.get(node["id"]) or [],
                    "raw_node": node,
                },
            )
        )

    output_dir = view_output_dir(view_name)
    docs_path = output_dir / "semantic_kg_docs.jsonl"
    edges_path = output_dir / "semantic_kg_edges.jsonl"
    write_jsonl(docs_path, docs)
    write_jsonl(edges_path, graph_edges)
    return {
        "docs_path": str(docs_path.relative_to(ROOT)),
        "edges_path": str(edges_path.relative_to(ROOT)),
        "doc_count": len(docs),
        "edge_count": len(graph_edges),
        "selected_node_count": len(selected_nodes),
        "selected_relation_count": len(selected_edges),
        "semantic_graph_path": str(SEMANTIC_KG_PATH.relative_to(ROOT)),
    }


def build_pubmed_abstract_source_from_frozen_raw(view_name: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if view_name != "multi_cancer":
        raise ValueError("frozen raw PubMed abstracts are only available for the multi_cancer view")
    if not PUBMED_MULTI_CANCER_RAW_PATH.exists():
        raise FileNotFoundError(
            f"Expected frozen PubMed abstract raw file at {PUBMED_MULTI_CANCER_RAW_PATH}. "
            "Build it first with scripts/build_pubmed_abstract_baseline.py."
        )
    records = read_jsonl(PUBMED_MULTI_CANCER_RAW_PATH)
    docs = []
    edges = []
    pdf_index: dict[str, dict[str, Any]] = {}

    for row in records:
        pmid = str(row.get("pmid") or "").strip()
        abstract = str(row.get("abstract") or "").strip()
        title = str(row.get("title") or "").strip() or f"PubMed article {pmid}"
        if not pmid or not abstract:
            continue
        parent_id = f"PUBABS-{stable_hash(f'multi_cancer|{pmid}')}"
        entity_ids = unique_preserve(
            [f"PMID:{pmid}"] + ([f"DOI:{str(row.get('doi') or '').strip()}"] if str(row.get("doi") or "").strip() else [])
        )
        citation = {
            "pmid": pmid,
            "doi": row.get("doi"),
            "journal": row.get("journal"),
            "year": row.get("year"),
            "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        }
        parent_text = "\n".join(
            part
            for part in [
                f"Title: {title}",
                f"Journal: {row.get('journal')}",
                f"Year: {row.get('year')}",
                f"PMID: {pmid}",
                f"DOI: {row.get('doi')}",
                f"Abstract:\n{abstract}",
            ]
            if part and not part.endswith("None")
        )
        metadata = {
            "pmid": pmid,
            "doi": row.get("doi"),
            "journal": row.get("journal"),
            "year": row.get("year"),
            "retrieval_sources": row.get("retrieval_sources") or [],
            "citation": citation,
            "source_family": "pubmed_literature",
        }
        docs.append(
            doc_record(
                doc_id=parent_id,
                knowledge_source="pubmed_abstracts",
                title=title,
                text=parent_text,
                document_level="parent",
                parent_id=None,
                disease_tags=[],
                entity_names=[],
                entity_ids=entity_ids,
                metadata=metadata,
            )
        )
        pdf_index[pmid] = {
            "disease": "",
            "title": title,
            "entity_names": [],
            "entity_ids": entity_ids,
        }
        for idx, snippet in enumerate(chunk_text(abstract, size=ABSTRACT_CHUNK_SIZE, overlap=ABSTRACT_CHUNK_OVERLAP), start=1):
            child_id = f"{parent_id}-CH{idx:03d}"
            docs.append(
                doc_record(
                    doc_id=child_id,
                    knowledge_source="pubmed_abstracts",
                    title=f"{title} | abstract chunk {idx}",
                    text=snippet,
                    document_level="chunk",
                    parent_id=parent_id,
                    disease_tags=[],
                    entity_names=[],
                    entity_ids=entity_ids,
                    metadata={
                        "pmid": pmid,
                        "doi": row.get("doi"),
                        "journal": row.get("journal"),
                        "year": row.get("year"),
                        "chunk_index": idx,
                        "source_family": "pubmed_literature",
                    },
                )
            )
            edges.append(edge_record(parent_id, child_id, "HAS_CHUNK", 1.0))

    output_dir = view_output_dir(view_name)
    docs_path = output_dir / "pubmed_abstract_docs.jsonl"
    edges_path = output_dir / "pubmed_abstract_edges.jsonl"
    write_jsonl(docs_path, docs)
    write_jsonl(edges_path, edges)
    return (
        {
            "docs_path": str(docs_path.relative_to(ROOT)),
            "edges_path": str(edges_path.relative_to(ROOT)),
            "doc_count": len(docs),
            "edge_count": len(edges),
            "paper_count": len(pdf_index),
        },
        pdf_index,
    )


def build_pubmed_abstract_source(view_name: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if view_name == "multi_cancer":
        return build_pubmed_abstract_source_from_frozen_raw(view_name)
    if view_name != "hcc_crc":
        raise ValueError("pubmed_abstracts are only available for the hcc_crc or multi_cancer view")
    graph = read_json(PUBMED_GRAPH_PATH)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    pdf_index: dict[str, dict[str, Any]] = defaultdict(lambda: {"disease": "", "title": "", "entity_names": set(), "entity_ids": set()})

    for node in graph["nodes"]:
        if node.get("node_type") != "evidence" or not view_matches_disease(view_name, node.get("disease")):
            continue
        citation = node.get("citation") or {}
        key = citation.get("doi") or citation.get("title") or citation.get("pdf_path") or node["id"]
        groups[(node.get("disease") or "", key)].append(node)
        if citation.get("pdf_path"):
            entry = pdf_index[citation["pdf_path"]]
            entry["disease"] = node.get("disease") or ""
            entry["title"] = citation.get("title") or citation["pdf_path"]
            names, ids = collect_entity_refs(node)
            entry["entity_names"].update(names)
            entry["entity_ids"].update(ids)

    docs = []
    edges = []
    for (disease, key), nodes in groups.items():
        citation = nodes[0].get("citation") or {}
        parent_id = f"PUBABS-{stable_hash(f'{disease}|{key}')}"
        snippets = []
        entity_names = []
        entity_ids = []
        for node in nodes:
            snippets.append(node.get("raw_text") or node.get("summary") or "")
            names, ids = collect_entity_refs(node)
            entity_names.extend(names)
            entity_ids.extend(ids)
        combined_text = "\n".join(unique_preserve(snippets))
        title = citation.get("title") or f"{disease} literature paper"
        parent_text = "\n".join(
            part
            for part in [
                f"Title: {title}",
                f"Journal: {citation.get('journal')}",
                f"Year: {citation.get('year')}",
                f"DOI: {citation.get('doi')}",
                f"Evidence text:\n{combined_text}",
            ]
            if part and not part.endswith("None")
        )
        docs.append(
            doc_record(
                doc_id=parent_id,
                knowledge_source="pubmed_abstracts",
                title=title,
                text=parent_text,
                document_level="parent",
                parent_id=None,
                disease_tags=[disease],
                entity_names=entity_names,
                entity_ids=entity_ids,
                metadata={
                    "disease": disease,
                    "citation": citation,
                    "paper_key": key,
                    "evidence_ids": [node["id"] for node in nodes],
                },
            )
        )
        for idx, snippet in enumerate(chunk_text(combined_text, size=ABSTRACT_CHUNK_SIZE, overlap=ABSTRACT_CHUNK_OVERLAP), start=1):
            child_id = f"{parent_id}-CH{idx:03d}"
            docs.append(
                doc_record(
                    doc_id=child_id,
                    knowledge_source="pubmed_abstracts",
                    title=f"{title} | article chunk {idx}",
                    text=snippet,
                    document_level="chunk",
                    parent_id=parent_id,
                    disease_tags=[disease],
                    entity_names=entity_names,
                    entity_ids=entity_ids,
                    metadata={
                        "disease": disease,
                        "chunk_index": idx,
                        "citation": citation,
                    },
                )
            )
            edges.append(edge_record(parent_id, child_id, "HAS_CHUNK", 1.0))

    output_dir = view_output_dir(view_name)
    docs_path = output_dir / "pubmed_abstract_docs.jsonl"
    edges_path = output_dir / "pubmed_abstract_edges.jsonl"
    write_jsonl(docs_path, docs)
    write_jsonl(edges_path, edges)
    normalized_pdf_index = {
        key: {
            "disease": value["disease"],
            "title": value["title"],
            "entity_names": sorted(value["entity_names"]),
            "entity_ids": sorted(value["entity_ids"]),
        }
        for key, value in pdf_index.items()
    }
    return (
        {
            "docs_path": str(docs_path.relative_to(ROOT)),
            "edges_path": str(edges_path.relative_to(ROOT)),
            "doc_count": len(docs),
            "edge_count": len(edges),
            "paper_count": len(groups),
        },
        normalized_pdf_index,
    )


def extract_pdf_text(path: Path) -> tuple[str, int]:
    reader = PdfReader(str(path))
    parts = []
    page_count = 0
    for page in reader.pages:
        page_count += 1
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return re.sub(r"\s+", " ", "\n".join(parts)).strip(), page_count


def build_fulltext_source(view_name: str, pdf_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if view_name != "hcc_crc":
        raise ValueError("papers_fulltext are only available for the hcc_crc view")
    sampled_files = []
    sampled_counts = {}
    for disease, pdf_dir in FULLTEXT_DIRS.items():
        candidates = sorted(path for path in pdf_dir.glob("*.pdf") if path.is_file())
        selected = deterministic_sample(candidates, FULLTEXT_SAMPLE_SIZE[disease])
        sampled_files.extend(selected)
        sampled_counts[disease] = {
            "available_papers": len(candidates),
            "sampled_papers": len(selected),
        }

    docs = []
    edges = []
    processed = 0
    for pdf_path in sampled_files:
        disease = "HCC" if "hcc_papers" in str(pdf_path) else "CRC"
        matched_meta = None
        for rel_path, metadata in pdf_index.items():
            if Path(rel_path).name == pdf_path.name:
                matched_meta = metadata
                break
        title = (matched_meta or {}).get("title") or pdf_path.stem
        text, page_count = extract_pdf_text(pdf_path)
        if not text:
            continue
        parent_id = f"FULLTEXT-{stable_hash(source_relpath(pdf_path))}"
        entity_names = (matched_meta or {}).get("entity_names") or []
        entity_ids = (matched_meta or {}).get("entity_ids") or []
        docs.append(
            doc_record(
                doc_id=parent_id,
                knowledge_source="papers_fulltext",
                title=title,
                text="\n".join(
                    [
                        f"Title: {title}",
                        f"Disease: {disease}",
                        f"PDF path: {source_relpath(pdf_path)}",
                        f"Extracted pages: {page_count}",
                        text[:4000],
                    ]
                ),
                document_level="parent",
                parent_id=None,
                disease_tags=[disease],
                entity_names=entity_names,
                entity_ids=entity_ids,
                metadata={
                    "disease": disease,
                    "pdf_path": source_relpath(pdf_path),
                    "page_count": page_count,
                    "text_char_count": len(text),
                    "title_match_from_pubmed": bool(matched_meta),
                },
            )
        )
        for idx, chunk in enumerate(chunk_text(text), start=1):
            child_id = f"{parent_id}-CH{idx:04d}"
            docs.append(
                doc_record(
                    doc_id=child_id,
                    knowledge_source="papers_fulltext",
                    title=f"{title} | fulltext chunk {idx}",
                    text=chunk,
                    document_level="chunk",
                    parent_id=parent_id,
                    disease_tags=[disease],
                    entity_names=entity_names,
                    entity_ids=entity_ids,
                    metadata={
                        "disease": disease,
                        "pdf_path": source_relpath(pdf_path),
                        "chunk_index": idx,
                    },
                )
            )
            edges.append(edge_record(parent_id, child_id, "HAS_CHUNK", 1.0))
            if idx > 1:
                prev_child_id = f"{parent_id}-CH{idx - 1:04d}"
                edges.append(edge_record(prev_child_id, child_id, "NEXT_CHUNK", 0.5))
        processed += 1

    output_dir = view_output_dir(view_name)
    docs_path = output_dir / "papers_fulltext_docs.jsonl"
    edges_path = output_dir / "papers_fulltext_edges.jsonl"
    write_jsonl(docs_path, docs)
    write_jsonl(edges_path, edges)
    return {
        "docs_path": str(docs_path.relative_to(ROOT)),
        "edges_path": str(edges_path.relative_to(ROOT)),
        "doc_count": len(docs),
        "edge_count": len(edges),
        "paper_count": processed,
        "sampled_counts": sampled_counts,
    }


def traditional_match(name: str, node_type: str, focus: dict[str, Any]) -> bool:
    normalized = norm_text(name)
    if not normalized:
        return False
    if node_type.lower().startswith("disease"):
        if normalized in focus["disease_aliases"]:
            return True
        return any(alias in normalized or normalized in alias for alias in focus["disease_aliases"])
    if normalized in focus["therapy_names"]:
        return True
    return bool(gene_like_tokens(name).intersection(focus["entity_tokens"]))


def append_statement(bucket: list[str], statement: str) -> None:
    if len(bucket) < TRADITIONAL_TEXT_STATEMENT_LIMIT:
        bucket.append(statement)


def traditional_budget(view_name: str) -> tuple[int, int]:
    manifest = load_existing_manifest()
    source_meta = (((manifest.get("views") or {}).get(view_name) or {}).get("knowledge_sources") or {}).get("evidence_graph") or {}
    evidence_doc_count = int(source_meta.get("doc_count") or 0)
    evidence_edge_count = int(source_meta.get("edge_count") or 0)
    if not evidence_doc_count or not evidence_edge_count:
        fallback = {
            "hcc_crc": (80_000, 230_000),
            "multi_cancer": (230_000, 700_000),
        }
        return fallback[view_name]
    return max(int(evidence_doc_count * TRADITIONAL_DOC_MULTIPLIER), 1000), max(int(evidence_edge_count * TRADITIONAL_EDGE_MULTIPLIER), 5000)


def primekg_doc_id(row: dict[str, str], prefix: str) -> str:
    return f"PRIMEKG-{stable_hash(prefix + '|' + row[f'{prefix}_source'] + '|' + row[f'{prefix}_id'])}"


def primekg_node_key(row: dict[str, str], prefix: str) -> str:
    return f"PRIMEKG::{row[f'{prefix}_source']}::{row[f'{prefix}_id']}"


def score_traditional_node(
    scores: Counter[str],
    key: str,
    *,
    self_match: bool,
    other_match: bool,
) -> None:
    scores[key] += 1
    if self_match:
        scores[key] += 8
    if other_match:
        scores[key] += 3


def top_scored_keys(scores: Counter[str], limit: int) -> set[str]:
    return {key for key, _ in scores.most_common(limit)}


def score_primekg_nodes(focus: dict[str, Any], max_nodes: int) -> tuple[set[str], Counter[str]]:
    scores: Counter[str] = Counter()
    with PRIMEKG_PATH.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            x_match = traditional_match(row["x_name"], row["x_type"], focus)
            y_match = traditional_match(row["y_name"], row["y_type"], focus)
            if not (x_match or y_match):
                continue
            score_traditional_node(scores, primekg_node_key(row, "x"), self_match=x_match, other_match=y_match)
            score_traditional_node(scores, primekg_node_key(row, "y"), self_match=y_match, other_match=x_match)
    return top_scored_keys(scores, max_nodes), scores


def accumulate_primekg_subgraph(
    selected_nodes: set[str],
    max_edges: int,
    node_meta: dict[str, dict[str, Any]],
    node_text_examples: dict[str, list[str]],
    full_statement_counts: Counter[str],
    graph_edges: list[dict[str, Any]],
) -> int:
    edge_count = 0
    with PRIMEKG_PATH.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            x_key = primekg_node_key(row, "x")
            y_key = primekg_node_key(row, "y")
            if x_key not in selected_nodes or y_key not in selected_nodes:
                continue
            if edge_count >= max_edges:
                break
            x_doc_id = primekg_doc_id(row, "x")
            y_doc_id = primekg_doc_id(row, "y")
            node_meta.setdefault(
                x_doc_id,
                {
                    "title": row["x_name"],
                    "node_type": row["x_type"],
                    "entity_names": [row["x_name"]],
                    "entity_ids": [row["x_id"]],
                    "disease_tags": [row["x_name"]] if row["x_type"].lower() == "disease" else [],
                    "source_graph": "PrimeKG",
                    "raw_node": {"id": row["x_id"], "source": row["x_source"], "name": row["x_name"], "type": row["x_type"]},
                },
            )
            node_meta.setdefault(
                y_doc_id,
                {
                    "title": row["y_name"],
                    "node_type": row["y_type"],
                    "entity_names": [row["y_name"]],
                    "entity_ids": [row["y_id"]],
                    "disease_tags": [row["y_name"]] if row["y_type"].lower() == "disease" else [],
                    "source_graph": "PrimeKG",
                    "raw_node": {"id": row["y_id"], "source": row["y_source"], "name": row["y_name"], "type": row["y_type"]},
                },
            )
            statement = f"{row['x_name']} ({row['x_type']}) {row['display_relation']} {row['y_name']} ({row['y_type']})."
            append_statement(node_text_examples[x_doc_id], statement)
            append_statement(node_text_examples[y_doc_id], statement)
            full_statement_counts[x_doc_id] += 1
            full_statement_counts[y_doc_id] += 1
            graph_edges.append(
                edge_record(
                    x_doc_id,
                    y_doc_id,
                    row["relation"],
                    1.0,
                    metadata={"source_graph": "PrimeKG", "raw_edge": row},
                )
            )
            edge_count += 1
    return edge_count


def load_tarkg_nodes() -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    with TARKG_NODES_PATH.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            data = {
                "name": row["name"],
                "kind": row["kind"],
                "dbid": row["dbid"],
                "db_source": row["db_source"],
                "unify_id": row["unify_id"],
            }
            lookup[row["unify_id"]] = data
            lookup[row["dbid"]] = data
    return lookup


def tarkg_node_key(meta: dict[str, str]) -> str:
    return f"TARKG::{meta['db_source']}::{meta['dbid']}"


def score_tarkg_nodes(focus: dict[str, Any], lookup: dict[str, dict[str, str]], max_nodes: int) -> tuple[set[str], Counter[str]]:
    scores: Counter[str] = Counter()
    with TARKG_EDGES_PATH.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            left = lookup.get(row["node1"], {"name": row["node1"], "kind": row["node1_type"], "dbid": row["node1"], "db_source": "TarKG", "unify_id": row["node1"]})
            right = lookup.get(row["node2"], {"name": row["node2"], "kind": row["node2_type"], "dbid": row["node2"], "db_source": "TarKG", "unify_id": row["node2"]})
            left_match = traditional_match(left["name"], left["kind"], focus)
            right_match = traditional_match(right["name"], right["kind"], focus)
            if not (left_match or right_match):
                continue
            score_traditional_node(scores, tarkg_node_key(left), self_match=left_match, other_match=right_match)
            score_traditional_node(scores, tarkg_node_key(right), self_match=right_match, other_match=left_match)
    return top_scored_keys(scores, max_nodes), scores


def tarkg_doc_id(meta: dict[str, str]) -> str:
    return f"TARKG-{stable_hash(meta['db_source'] + '|' + meta['dbid'])}"


def accumulate_tarkg_subgraph(
    selected_nodes: set[str],
    max_edges: int,
    lookup: dict[str, dict[str, str]],
    node_meta: dict[str, dict[str, Any]],
    node_text_examples: dict[str, list[str]],
    full_statement_counts: Counter[str],
    graph_edges: list[dict[str, Any]],
) -> int:
    edge_count = 0
    with TARKG_EDGES_PATH.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if edge_count >= max_edges:
                break
            left = lookup.get(row["node1"], {"name": row["node1"], "kind": row["node1_type"], "dbid": row["node1"], "db_source": "TarKG", "unify_id": row["node1"]})
            right = lookup.get(row["node2"], {"name": row["node2"], "kind": row["node2_type"], "dbid": row["node2"], "db_source": "TarKG", "unify_id": row["node2"]})
            left_key = tarkg_node_key(left)
            right_key = tarkg_node_key(right)
            if left_key not in selected_nodes or right_key not in selected_nodes:
                continue
            x_doc_id = tarkg_doc_id(left)
            y_doc_id = tarkg_doc_id(right)
            node_meta.setdefault(
                x_doc_id,
                {
                    "title": left["name"],
                    "node_type": left["kind"],
                    "entity_names": [left["name"]],
                    "entity_ids": [left["dbid"]],
                    "disease_tags": [left["name"]] if left["kind"].lower() == "disease" else [],
                    "source_graph": "TarKG",
                    "raw_node": left,
                },
            )
            node_meta.setdefault(
                y_doc_id,
                {
                    "title": right["name"],
                    "node_type": right["kind"],
                    "entity_names": [right["name"]],
                    "entity_ids": [right["dbid"]],
                    "disease_tags": [right["name"]] if right["kind"].lower() == "disease" else [],
                    "source_graph": "TarKG",
                    "raw_node": right,
                },
            )
            statement = f"{left['name']} ({left['kind']}) {row['relation']} {right['name']} ({right['kind']})."
            append_statement(node_text_examples[x_doc_id], statement)
            append_statement(node_text_examples[y_doc_id], statement)
            full_statement_counts[x_doc_id] += 1
            full_statement_counts[y_doc_id] += 1
            graph_edges.append(
                edge_record(
                    x_doc_id,
                    y_doc_id,
                    row["relation"],
                    1.0,
                    metadata={"source_graph": "TarKG", "raw_edge": row},
                )
            )
            edge_count += 1
    return edge_count


def build_primekg_source(view_name: str, focus: dict[str, Any]) -> dict[str, Any]:
    max_docs, max_edges = traditional_budget(view_name)
    node_meta: dict[str, dict[str, Any]] = {}
    node_text_examples: dict[str, list[str]] = defaultdict(list)
    full_statement_counts: Counter[str] = Counter()
    graph_edges: list[dict[str, Any]] = []

    prime_selected_nodes, prime_scores = score_primekg_nodes(focus, max_docs)
    prime_edge_count = accumulate_primekg_subgraph(prime_selected_nodes, max_edges, node_meta, node_text_examples, full_statement_counts, graph_edges)

    docs = []
    for doc_id, meta in node_meta.items():
        statement_text = "\n".join(node_text_examples.get(doc_id) or [])
        text_parts = [
            f"Title: {meta['title']}",
            f"Node type: {meta['node_type']}",
            f"Source graph: {meta['source_graph']}",
            f"Statement count in selected subgraph: {full_statement_counts[doc_id]}",
            statement_text,
        ]
        docs.append(
            doc_record(
                doc_id=doc_id,
                knowledge_source="traditional_kg",
                title=meta["title"],
                text="\n".join(part for part in text_parts if part),
                document_level="node",
                parent_id=None,
                disease_tags=meta["disease_tags"],
                entity_names=meta["entity_names"],
                entity_ids=meta["entity_ids"],
                metadata={
                    "node_type": meta["node_type"],
                    "source_graph": meta["source_graph"],
                    "statement_count": full_statement_counts[doc_id],
                    "raw_node": meta["raw_node"],
                },
            )
        )

    output_dir = view_output_dir(view_name)
    docs_path = output_dir / "primekg_docs.jsonl"
    edges_path = output_dir / "primekg_edges.jsonl"
    write_jsonl(docs_path, docs)
    write_jsonl(edges_path, graph_edges)
    return {
        "docs_path": str(docs_path.relative_to(ROOT)),
        "edges_path": str(edges_path.relative_to(ROOT)),
        "doc_count": len(docs),
        "edge_count": len(graph_edges),
        "target_doc_budget": max_docs,
        "target_edge_budget": max_edges,
        "selected_primekg_nodes": len(prime_selected_nodes),
        "primekg_selected_nodes": len(prime_selected_nodes),
        "primekg_edge_count": prime_edge_count,
        "primekg_top_score": max(prime_scores.values()) if prime_scores else 0,
    }


def build_tarkg_source(view_name: str, focus: dict[str, Any]) -> dict[str, Any]:
    max_docs, max_edges = traditional_budget(view_name)
    node_meta: dict[str, dict[str, Any]] = {}
    node_text_examples: dict[str, list[str]] = defaultdict(list)
    full_statement_counts: Counter[str] = Counter()
    graph_edges: list[dict[str, Any]] = []

    tarkg_lookup = load_tarkg_nodes()
    tarkg_selected_nodes, tarkg_scores = score_tarkg_nodes(focus, tarkg_lookup, max_docs)
    tarkg_edge_count = accumulate_tarkg_subgraph(tarkg_selected_nodes, max_edges, tarkg_lookup, node_meta, node_text_examples, full_statement_counts, graph_edges)

    docs = []
    for doc_id, meta in node_meta.items():
        statement_text = "\n".join(node_text_examples.get(doc_id) or [])
        text_parts = [
            f"Title: {meta['title']}",
            f"Node type: {meta['node_type']}",
            f"Source graph: {meta['source_graph']}",
            f"Statement count in selected subgraph: {full_statement_counts[doc_id]}",
            statement_text,
        ]
        docs.append(
            doc_record(
                doc_id=doc_id,
                knowledge_source="tarkg",
                title=meta["title"],
                text="\n".join(part for part in text_parts if part),
                document_level="node",
                parent_id=None,
                disease_tags=meta["disease_tags"],
                entity_names=meta["entity_names"],
                entity_ids=meta["entity_ids"],
                metadata={
                    "node_type": meta["node_type"],
                    "source_graph": meta["source_graph"],
                    "statement_count": full_statement_counts[doc_id],
                    "raw_node": meta["raw_node"],
                },
            )
        )

    output_dir = view_output_dir(view_name)
    docs_path = output_dir / "tarkg_docs.jsonl"
    edges_path = output_dir / "tarkg_edges.jsonl"
    write_jsonl(docs_path, docs)
    write_jsonl(edges_path, graph_edges)
    return {
        "docs_path": str(docs_path.relative_to(ROOT)),
        "edges_path": str(edges_path.relative_to(ROOT)),
        "doc_count": len(docs),
        "edge_count": len(graph_edges),
        "target_doc_budget": max_docs,
        "target_edge_budget": max_edges,
        "selected_tarkg_nodes": len(tarkg_selected_nodes),
        "tarkg_selected_nodes": len(tarkg_selected_nodes),
        "tarkg_edge_count": tarkg_edge_count,
        "tarkg_top_score": max(tarkg_scores.values()) if tarkg_scores else 0,
    }


def load_source_rows(view_name: str, source_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = load_existing_manifest()
    source_meta = (((manifest.get("views") or {}).get(view_name) or {}).get("knowledge_sources") or {}).get(source_name) or {}
    docs_path = ROOT / source_meta["docs_path"]
    edges_path = ROOT / source_meta["edges_path"]
    return read_jsonl(docs_path), read_jsonl(edges_path)


def raw_view_allows_doc(view_name: str, doc: dict[str, Any]) -> bool:
    if view_name == "multi_cancer":
        return doc.get("knowledge_source") != "raw_paper_fulltext"
    tags = {norm_text(tag) for tag in doc.get("disease_tags") or []}
    return any(tag in {"hcc", "hepatocellular carcinoma", "crc", "colorectal cancer", "colorectal adenocarcinoma"} for tag in tags)


def build_raw_observation_view_source(view_name: str) -> dict[str, Any]:
    raw_docs = read_jsonl(RAW_MULTISOURCE_DOCS_PATH)
    raw_edges = read_jsonl(RAW_MULTISOURCE_EDGES_PATH)
    selected_docs = [doc for doc in raw_docs if raw_view_allows_doc(view_name, doc)]
    selected_ids = {doc["doc_id"] for doc in selected_docs}
    selected_edges = [
        edge
        for edge in raw_edges
        if edge.get("source_doc_id") in selected_ids and edge.get("target_doc_id") in selected_ids
    ]
    output_dir = view_output_dir(view_name)
    docs_path = output_dir / "raw_observation_view_docs.jsonl"
    edges_path = output_dir / "raw_observation_view_edges.jsonl"
    write_jsonl(docs_path, selected_docs)
    write_jsonl(edges_path, selected_edges)
    return {
        "docs_path": str(docs_path.relative_to(ROOT)),
        "edges_path": str(edges_path.relative_to(ROOT)),
        "doc_count": len(selected_docs),
        "edge_count": len(selected_edges),
    }


def evidence_entity_alignment_keys(doc: dict[str, Any]) -> set[tuple[str, str]]:
    metadata = doc.get("metadata") or {}
    raw_node = metadata.get("raw_node") or {}
    if metadata.get("node_type") != "entity":
        return set()
    entity_id = raw_node.get("id") or doc.get("doc_id")
    if ":" not in str(entity_id):
        return set()
    namespace, value = str(entity_id).split(":", 1)
    supported = {"DrugBank", "ChEMBL", "NCBIGene", "MESH", "PubChemCID"}
    if namespace not in supported:
        return set()
    return {(namespace, value)}


def tarkg_alignment_keys(doc: dict[str, Any]) -> set[tuple[str, str]]:
    raw_node = ((doc.get("metadata") or {}).get("raw_node") or {})
    source = raw_node.get("db_source")
    dbid = raw_node.get("dbid")
    if not source or not dbid:
        return set()
    ns_map = {
        "DrugBank": "DrugBank",
        "CHEMBL": "ChEMBL",
        "NCBI": "NCBIGene",
        "MESH": "MESH",
        "PUBCHEM": "PubChemCID",
    }
    namespace = ns_map.get(source)
    if not namespace:
        return set()
    return {(namespace, str(dbid))}


def build_evidence_tarkg_hybrid_source(view_name: str) -> dict[str, Any]:
    evidence_docs, evidence_edges = load_source_rows(view_name, "evidence_graph")
    tarkg_docs, tarkg_edges = load_source_rows(view_name, "tarkg")
    crosswalk_rows = read_jsonl(TARKG_CROSSWALK_PATH)
    evidence_doc_by_entity_id = {}
    for doc in evidence_docs:
        metadata = doc.get("metadata") or {}
        raw_node = metadata.get("raw_node") or {}
        if metadata.get("node_type") == "entity" and raw_node.get("id"):
            evidence_doc_by_entity_id[raw_node["id"]] = doc["doc_id"]
    tarkg_doc_by_unify_id = {}
    for doc in tarkg_docs:
        raw_node = ((doc.get("metadata") or {}).get("raw_node") or {})
        if raw_node.get("unify_id"):
            tarkg_doc_by_unify_id[raw_node["unify_id"]] = doc["doc_id"]

    bridge_edges = []
    matched_pairs = set()
    matched_entity_docs = set()
    matched_tarkg_docs = set()
    namespace_counts = Counter()
    entity_type_counts = Counter()
    resolution_counts = Counter()
    for row in crosswalk_rows:
        evidence_doc_id = evidence_doc_by_entity_id.get(row["evidence_entity_id"])
        tarkg_doc_id = tarkg_doc_by_unify_id.get(row["tarkg_unify_id"])
        if not evidence_doc_id or not tarkg_doc_id:
            continue
        namespace = normalize_bridge_namespace(row.get("mapping_namespace"), row.get("tarkg_db_source"))
        pair = (evidence_doc_id, tarkg_doc_id)
        if pair in matched_pairs:
            continue
        matched_pairs.add(pair)
        matched_entity_docs.add(evidence_doc_id)
        matched_tarkg_docs.add(tarkg_doc_id)
        namespace_counts[namespace] += 1
        entity_type_counts[row["evidence_entity_type"]] += 1
        resolution_counts[row["resolution_method"]] += 1
        bridge_edges.append(
            edge_record(
                evidence_doc_id,
                tarkg_doc_id,
                "CROSS_KG_ENTITY_EQUIVALENT",
                1.0,
                metadata={
                    "mapping_namespace": namespace,
                    "alignment_namespace": namespace,
                    "resolution_method": row["resolution_method"],
                    "evidence_entity_type": row["evidence_entity_type"],
                    "tarkg_db_source": row["tarkg_db_source"],
                    "tarkg_dbid": row["tarkg_dbid"],
                    "tarkg_unify_id": row["tarkg_unify_id"],
                    "source_graph": "TarKG",
                    "hybrid_source": "evidence_tarkg_hybrid",
                },
            )
        )

    docs = [dict(doc) for doc in evidence_docs + tarkg_docs]
    docs_by_id = {doc["doc_id"]: doc for doc in docs}
    edges = evidence_edges + tarkg_edges + bridge_edges
    for row in crosswalk_rows:
        evidence_doc_id = evidence_doc_by_entity_id.get(row["evidence_entity_id"])
        tarkg_doc_id = tarkg_doc_by_unify_id.get(row["tarkg_unify_id"])
        if not evidence_doc_id or not tarkg_doc_id:
            continue
        namespace = normalize_bridge_namespace(row.get("mapping_namespace"), row.get("tarkg_db_source"))
        evidence_doc = docs_by_id.get(evidence_doc_id)
        tarkg_doc = docs_by_id.get(tarkg_doc_id)
        if not evidence_doc or not tarkg_doc:
            continue
        evidence_doc["entity_names"] = unique_preserve(evidence_doc.get("entity_names", []) + [tarkg_doc.get("title") or ""])
        evidence_doc["entity_ids"] = unique_preserve(evidence_doc.get("entity_ids", []) + [row["tarkg_dbid"], row["tarkg_unify_id"]])
        if row.get("evidence_entity_type") == "Disease":
            evidence_doc["disease_tags"] = unique_preserve(evidence_doc.get("disease_tags", []) + disease_alias_values(tarkg_doc.get("title")))
        bridge_sentence = (
            f"Aligned TarKG entity: {tarkg_doc.get('title')} "
            f"({namespace} match; TarKG {row['tarkg_db_source']}:{row['tarkg_dbid']})."
        )
        if bridge_sentence not in evidence_doc.get("text", ""):
            evidence_doc["text"] = f"{evidence_doc.get('text', '')}\n{bridge_sentence}".strip()

        evidence_name = ((evidence_doc.get("metadata") or {}).get("raw_node") or {}).get("name") or evidence_doc.get("title") or row["evidence_entity_id"]
        tarkg_doc["entity_names"] = unique_preserve(tarkg_doc.get("entity_names", []) + [evidence_name])
        bridge_back_sentence = (
            f"Aligned evidence-graph entity: {evidence_name} "
            f"({namespace} match; evidence entity {row['evidence_entity_id']})."
        )
        if bridge_back_sentence not in tarkg_doc.get("text", ""):
            tarkg_doc["text"] = f"{tarkg_doc.get('text', '')}\n{bridge_back_sentence}".strip()
    output_dir = view_output_dir(view_name)
    docs_path = output_dir / "evidence_tarkg_hybrid_docs.jsonl"
    edges_path = output_dir / "evidence_tarkg_hybrid_edges.jsonl"
    write_jsonl(docs_path, docs)
    write_jsonl(edges_path, edges)
    return {
        "docs_path": str(docs_path.relative_to(ROOT)),
        "edges_path": str(edges_path.relative_to(ROOT)),
        "doc_count": len(docs),
        "edge_count": len(edges),
        "bridge_edge_count": len(bridge_edges),
        "matched_evidence_entity_docs": len(matched_entity_docs),
        "matched_tarkg_docs": len(matched_tarkg_docs),
        "alignment_namespace_counts": dict(sorted(namespace_counts.items())),
        "alignment_entity_type_counts": dict(sorted(entity_type_counts.items())),
        "resolution_method_counts": dict(sorted(resolution_counts.items())),
        "chosen_traditional_kg": "tarkg",
        "crosswalk_path": str(TARKG_CROSSWALK_PATH.relative_to(ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare faithful view-specific corpora for RAG experiments.")
    parser.add_argument("--views", nargs="+", choices=ALL_VIEWS, default=["multi_cancer"], help="Specific evaluation views to prepare.")
    parser.add_argument("--sources", nargs="+", choices=ALL_SOURCES, default=["raw_observation_view", "semantic_kg", "evidence_graph_complete"], help="Specific knowledge sources to prepare.")
    parser.add_argument("--benchmark", type=Path, help="Benchmark supplying multi_cancer focus metadata; defaults to the released 571 questions.")
    args = parser.parse_args()
    if args.benchmark is not None:
        VIEW_CONFIG["multi_cancer"]["benchmark_path"] = args.benchmark if args.benchmark.is_absolute() else ROOT / args.benchmark

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for view_name in args.views:
        focus = build_focus_terms(view_name)
        pdf_index: dict[str, dict[str, Any]] = {}
        requested_sources = [source for source in args.sources if source in VIEW_CONFIG[view_name]["supported_sources"]]
        if "evidence_graph" in requested_sources:
            update_manifest(view_name, focus, {"evidence_graph": build_evidence_graph_source(view_name)})
        if "evidence_graph_complete" in requested_sources:
            update_manifest(view_name, focus, {"evidence_graph_complete": build_evidence_graph_complete_source(view_name)})
        if "semantic_kg" in requested_sources:
            update_manifest(view_name, focus, {"semantic_kg": build_semantic_kg_source(view_name)})
        if "raw_observation_view" in requested_sources:
            update_manifest(view_name, focus, {"raw_observation_view": build_raw_observation_view_source(view_name)})
        if "pubmed_abstracts" in requested_sources or "papers_fulltext" in requested_sources:
            pubmed_manifest, pdf_index = build_pubmed_abstract_source(view_name)
            if "pubmed_abstracts" in requested_sources:
                update_manifest(view_name, focus, {"pubmed_abstracts": pubmed_manifest})
        if "papers_fulltext" in requested_sources:
            update_manifest(view_name, focus, {"papers_fulltext": build_fulltext_source(view_name, pdf_index)})
        if "primekg" in requested_sources:
            update_manifest(view_name, focus, {"primekg": build_primekg_source(view_name, focus)})
        if "tarkg" in requested_sources:
            update_manifest(view_name, focus, {"tarkg": build_tarkg_source(view_name, focus)})
        if "evidence_tarkg_hybrid" in requested_sources:
            update_manifest(view_name, focus, {"evidence_tarkg_hybrid": build_evidence_tarkg_hybrid_source(view_name)})


if __name__ == "__main__":
    main()

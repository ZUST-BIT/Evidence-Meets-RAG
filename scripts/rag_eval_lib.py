from __future__ import annotations

import json
import math
import os
import pickle
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

RAG_PROFILE = "tuned"
STAGE_CONFIG = {
    "retrieval": "full",
    "context": "structured",
    "generation": "grounded",
}
CORPUS_CACHE_VERSION = 1
DENSE_RAG_INDEX_VERSION = 1
DENSE_RAG_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DENSE_RAG_DEFAULT_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DENSE_RAG_DEFAULT_MAX_SEQ_LENGTH = 256
CACHEABLE_KNOWLEDGE_SOURCES = {
    "evidence_graph",
    "evidence_graph_complete",
    "evidence_graph_complete_label_masked",
    "semantic_kg",
    "raw_observation_view",
}


def set_rag_profile(profile: str) -> None:
    global RAG_PROFILE
    if profile not in {"tuned", "pure"}:
        raise ValueError(f"unsupported rag profile: {profile}")
    RAG_PROFILE = profile


def set_stage_config(retrieval: str = "full", context: str = "structured", generation: str = "grounded") -> None:
    if retrieval not in {"base", "full"}:
        raise ValueError(f"unsupported retrieval stage: {retrieval}")
    if context not in {"flat", "structured"}:
        raise ValueError(f"unsupported context stage: {context}")
    if generation not in {"plain", "grounded"}:
        raise ValueError(f"unsupported generation stage: {generation}")
    STAGE_CONFIG["retrieval"] = retrieval
    STAGE_CONFIG["context"] = context
    STAGE_CONFIG["generation"] = generation


def is_pure_rag_profile() -> bool:
    return RAG_PROFILE == "pure"


def retrieval_stage_full() -> bool:
    return STAGE_CONFIG["retrieval"] == "full"


def context_stage_structured() -> bool:
    return STAGE_CONFIG["context"] == "structured"


def generation_stage_grounded() -> bool:
    return STAGE_CONFIG["generation"] == "grounded"


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, bool):
        value = "true" if value else "false"
    else:
        value = str(value)
    if not value:
        return ""
    value = value.lower()
    value = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def tokenize(value: str | None) -> list[str]:
    normalized = norm_text(value)
    return [token for token in normalized.split() if len(token) >= 2]


def canonicalize_disease(value: str | None) -> str:
    normalized = norm_text(value)
    if not normalized:
        return ""
    disease_aliases = {
        "hcc": {"hcc", "hepatocellular carcinoma", "liver hepatocellular carcinoma", "hepatocellular cancer", "liver cancer"},
        "crc": {"crc", "colorectal cancer", "colorectal adenocarcinoma", "colon adenocarcinoma", "rectal adenocarcinoma", "colon cancer", "rectal cancer"},
        "nsclc": {"nsclc", "non small cell lung cancer", "non small cell lung carcinoma"},
        "brca": {"brca", "breast cancer", "breast carcinoma"},
        "glioma": {"glioma", "glioblastoma"},
        "ovarian": {"ovarian cancer", "ovarian carcinoma"},
        "rcc": {"rcc", "renal cell carcinoma", "kidney renal clear cell carcinoma"},
        "aml": {"aml", "acute myeloid leukemia"},
        "paad": {"paad", "pancreatic adenocarcinoma", "pancreatic cancer"},
        "hnsc": {"hnsc", "head and neck squamous cell carcinoma"},
        "gastroesoph": {"gastroesophageal adenocarcinoma", "stomach adenocarcinoma", "gastric cancer"},
        "prad": {"prad", "prostate cancer"},
        "melanoma": {"melanoma", "skin cutaneous melanoma"},
        "meso": {"meso", "mesothelioma"},
    }
    for canonical, aliases in disease_aliases.items():
        if normalized in aliases:
            return canonical
    return normalized


def disease_term_set(*values: str | None) -> set[str]:
    terms = set()
    for value in values:
        normalized = norm_text(value)
        canonical = canonicalize_disease(value)
        if normalized:
            terms.add(normalized)
        if canonical:
            terms.add(canonical)
    return terms


def unique_preserve(values: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def canonicalize_gene_symbol(value: str | None) -> str:
    return re.sub(r"[^A-Z0-9\-]+", "", str(value or "").upper())


def collect_record_gene_symbols(record: dict[str, Any]) -> list[str]:
    entities = record.get("entities") or {}
    metadata = record.get("metadata") or {}
    values = []
    values.extend(entities.get("gene_symbols") or [])
    values.extend([entities.get("gene_symbol"), entities.get("target_gene_symbol")])
    normalized_anchors = metadata.get("normalized_entity_anchors") or {}
    values.extend(normalized_anchors.get("gene_symbols") or [])
    seen = set()
    ordered = []
    for value in values:
        token = canonicalize_gene_symbol(value)
        if len(token) >= 3 and token not in seen:
            seen.add(token)
            ordered.append(token)
    return ordered


def causal_query_pair(record: dict[str, Any]) -> tuple[str, str]:
    entities = record.get("entities") or {}
    source = canonicalize_gene_symbol(entities.get("gene_symbol"))
    target = canonicalize_gene_symbol(entities.get("target_gene_symbol"))
    return source, target


def gene_symbol_lookup_variants(value: str | None) -> list[str]:
    token = canonicalize_gene_symbol(value)
    if len(token) < 3:
        return []
    variants = [token]
    # TarKG protein titles occasionally append a trailing G to the HGNC symbol
    # (for example A2M -> A2MG_HUMAN), so include a minimal reversible variant.
    if not token.endswith("G"):
        variants.append(f"{token}G")
    if token.endswith("G") and len(token) >= 4:
        variants.append(token[:-1])
    return unique_preserve(variants)


def lookup_gene_symbol_doc_ids(corpus: PreparedCorpus, value: str | None, *, knowledge_source: str | None = None) -> list[str]:
    doc_ids: list[str] = []
    for variant in gene_symbol_lookup_variants(value):
        for doc_id in corpus.gene_symbol_to_doc_ids.get(variant, []):
            if knowledge_source and corpus.doc_by_id[doc_id].knowledge_source != knowledge_source:
                continue
            doc_ids.append(doc_id)
    return unique_preserve(doc_ids)


def collect_doc_gene_symbols(doc: PreparedDoc) -> set[str]:
    symbols = set()
    for value in [doc.title]:
        for match in re.findall(r"\b[A-Z0-9\-]{3,16}\b", str(value or "").upper()):
            token = canonicalize_gene_symbol(match.replace("_HUMAN", "").replace("_MOUSE", ""))
            if len(token) >= 3:
                symbols.add(token)
                symbols.update(gene_symbol_lookup_variants(token))
    for name in doc.entity_names:
        token = canonicalize_gene_symbol(name)
        if len(token) >= 3:
            symbols.add(token)
            symbols.update(gene_symbol_lookup_variants(token))
        for match in re.findall(r"\b[A-Z0-9\-]{3,12}\b", str(name or "").upper()):
            token = canonicalize_gene_symbol(match)
            if len(token) >= 3:
                symbols.add(token)
                symbols.update(gene_symbol_lookup_variants(token))
    for value in doc.entity_ids:
        for match in re.findall(r"\b[A-Z0-9\-]{3,12}\b", str(value or "").upper()):
            token = canonicalize_gene_symbol(match)
            if len(token) >= 3:
                symbols.add(token)
                symbols.update(gene_symbol_lookup_variants(token))
    return symbols


def raw_node_metadata(doc: Any) -> dict[str, Any]:
    metadata = getattr(doc, "metadata", None) or {}
    raw_node = metadata.get("raw_node")
    return raw_node if isinstance(raw_node, dict) else {}


def doc_metadata_field(doc: Any, field: str, default: Any = "") -> Any:
    raw_node = raw_node_metadata(doc)
    if raw_node.get(field) not in (None, "", [], {}):
        return raw_node.get(field)
    metadata = getattr(doc, "metadata", None) or {}
    return metadata.get(field, default)


def ref_text(ref: dict[str, Any] | None) -> str:
    if not isinstance(ref, dict):
        return ""
    return norm_text(str(ref.get("normalized_label") or ref.get("label") or ref.get("name") or ref.get("id") or ""))


def doc_evidence_target_text(doc: Any) -> str:
    raw_node = raw_node_metadata(doc)
    return ref_text(raw_node.get("outcome")) or ref_text(raw_node.get("object"))


def doc_support_flags(doc: Any) -> set[str]:
    cached = getattr(doc, "_support_flags", None)
    if cached is not None:
        return cached
    text = norm_text(" ".join([str(getattr(doc, "title", "") or ""), str(getattr(doc, "text", "") or "")]))
    feature_layer = norm_text(str(doc_metadata_field(doc, "feature_layer") or ""))
    evidence_category = norm_text(str(doc_metadata_field(doc, "evidence_category") or ""))
    predicate = norm_text(str(doc_metadata_field(doc, "predicate") or ""))
    source_family = norm_text(str(doc_metadata_field(doc, "source_family") or ""))
    source_context = doc_metadata_field(doc, "source_context", {}) or {}
    group_axis = norm_text(str(source_context.get("group_axis") or ""))
    flags = set()

    if any(term in feature_layer for term in ["expression", "transcript", "rna"]) or any(term in predicate for term in ["upregulated", "downregulated", "expression"]):
        flags.add("expression")
    if "methyl" in feature_layer or "methyl" in predicate:
        flags.add("methylation")
    if any(term in feature_layer for term in ["mutation", "hotspot", "damaging"]) or any(term in predicate for term in ["mutation", "mutated", "damaging", "hotspot"]):
        flags.add("mutation")
    if any(term in feature_layer for term in ["copy", "scna", "gistic"]) or any(term in predicate for term in ["amplified", "deleted", "copy number"]):
        flags.add("copy_number")
    if any(term in feature_layer for term in ["dependency", "crispr"]) or any(term in evidence_category for term in ["dependency", "vulnerability"]) or any(term in predicate for term in ["dependency", "sensitive", "resistant"]):
        flags.add("dependency")
    if any(term in text for term in ["driver", "oncogenic", "tumor suppressor"]) or any(term in evidence_category for term in ["driver"]):
        flags.add("driver")
    if any(term in text for term in ["pathway", "signaling", "mechanism", "mapk", "pi3k", "wnt", "emt", "apoptosis", "repair", "stat", "notch", "hypoxia", "angiogenesis"]):
        flags.add("pathway")
    if any(term in predicate for term in ["activate", "inhibit", "regulat", "phosphorylation", "dephosphorylation", "degradation", "stabil", "destabil"]) or any(term in text for term in [" activates ", " inhibits ", " phosphoryl", " signaling ", " mechanistic "]):
        flags.add("directional")
    if "prognosis" in evidence_category or "survival" in group_axis or "prognosis" in predicate or "poor prognosis" in text or "favorable prognosis" in text:
        flags.add("prognosis")
    if source_family == "pubmed_literature":
        flags.add("literature")
    if any(term in group_axis for term in ["cohort_recurrence", "recurrence"]):
        flags.add("recurrent")
    if "integrated" in source_family:
        flags.add("integrated")
    setattr(doc, "_support_flags", flags)
    return flags


def doc_targets_non_disease_entity(doc: Any, record: dict[str, Any]) -> bool:
    target_text = doc_evidence_target_text(doc)
    if not target_text:
        return False
    disease_terms = disease_term_set(record.get("oncology_focus"), (record.get("entities") or {}).get("disease"), (record.get("entities") or {}).get("disease_name"))
    return target_text not in disease_terms


def doc_matches_disease(doc: Any, record: dict[str, Any]) -> bool:
    disease_terms = disease_term_set(record.get("oncology_focus"), (record.get("entities") or {}).get("disease"), (record.get("entities") or {}).get("disease_name"))
    return bool(disease_terms.intersection(getattr(doc, "_disease_terms", set()) or set(disease_term_set(*getattr(doc, "disease_tags", [])))))


def bundle_component_clues(record: dict[str, Any], retrieved: list[tuple[PreparedDoc, float]]) -> dict[str, list[str]]:
    record_gene_symbols = set(collect_record_gene_symbols(record))
    component_clues = {key: [] for key in ["driver", "target_association", "causal_relation", "pathway_context"]}
    for doc, _score in retrieved:
        disease_match = doc_matches_disease(doc, record)
        if not disease_match:
            continue
        node_type = getattr(doc, "_node_type", "") or ""
        metadata = getattr(doc, "metadata", None) or {}
        if node_type == "mechanism_role_summary":
            bundle_components = metadata.get("bundle_components") or {}
            clue = f"{doc.title} [{doc.knowledge_source}]"
            if bundle_components.get("target_association"):
                component_clues["target_association"].append(clue)
            if bundle_components.get("pathway_context"):
                component_clues["pathway_context"].append(clue)
            if bundle_components.get("mechanism_context") or bundle_components.get("literature_mechanism"):
                component_clues["causal_relation"].append(clue)
            continue
        if node_type == "target_support_summary":
            association_inventory = metadata.get("association_inventory") or {}
            if any(bool(association_inventory.get(key)) for key in ["direct_target_association", "knowledgebase_association", "dependency_or_drug_sensitivity", "supportive_omics", "clinical_prognosis"]):
                component_clues["target_association"].append(f"{doc.title} [{doc.knowledge_source}]")
            continue
        flags = doc_support_flags(doc)
        doc_gene_symbols = getattr(doc, "_gene_symbols", set()) or collect_doc_gene_symbols(doc)
        gene_match = bool(record_gene_symbols.intersection(doc_gene_symbols))
        clue = f"{doc.title} [{doc.knowledge_source}]"
        if gene_match and (getattr(doc, "_is_evidence_like", False) or getattr(doc, "document_level", "") == "parent"):
            component_clues["target_association"].append(clue)
        if gene_match and node_type not in {"mechanism_role_summary", "target_support_summary"} and doc_has_strict_driver_clue(doc):
            component_clues["driver"].append(clue)
        if "pathway" in flags:
            component_clues["pathway_context"].append(clue)
        if "directional" in flags and doc_targets_non_disease_entity(doc, record):
            component_clues["causal_relation"].append(clue)
    return {key: unique_preserve(values) for key, values in component_clues.items()}


def bundle_safe_title(doc: Any) -> str:
    node_type = getattr(doc, "_node_type", "") or ""
    if node_type == "mechanism_role_summary":
        metadata = getattr(doc, "metadata", None) or {}
        gene_symbol = metadata.get("gene_symbol") or getattr(doc, "title", "")
        disease = metadata.get("disease") or ""
        disease_label = str(disease).replace("_", " ") if disease else ""
        if gene_symbol and disease_label:
            return f"{gene_symbol} mechanism support summary in {disease_label}"
    return getattr(doc, "title", "")


def doc_has_strict_driver_clue(doc: Any) -> bool:
    predicate = norm_text(str(doc_metadata_field(doc, "predicate") or ""))
    feature_layer = norm_text(str(doc_metadata_field(doc, "feature_layer") or ""))
    evidence_category = norm_text(str(doc_metadata_field(doc, "evidence_category") or ""))
    source_context = doc_metadata_field(doc, "source_context", {}) or {}
    group_axis = norm_text(str(source_context.get("group_axis") or ""))
    mutation_like = any(term in f"{predicate} {feature_layer} {evidence_category}" for term in ["mutation", "hotspot", "damaging"])
    copy_number_like = any(term in f"{predicate} {feature_layer} {evidence_category}" for term in ["copy number", "copy_number", "amplified", "deleted", "scna"])
    recurrent_like = "recur" in predicate or "recur" in group_axis
    explicit_driver = "driver" in predicate or "driver" in evidence_category
    return mutation_like or copy_number_like or recurrent_like or explicit_driver


def bundle_safe_text(doc: Any) -> str:
    node_type = getattr(doc, "_node_type", "") or ""
    text = str(getattr(doc, "text", "") or "")
    if node_type == "mechanism_role_summary":
        kept_lines = []
        for line in text.splitlines():
            if line.startswith("Bundle component inventory:") or line.startswith("Mechanism buckets:") or line.startswith("Mechanism by source family:") or line.startswith("Linked pathways in graph:") or line.startswith("Example claims:") or line.startswith("- "):
                kept_lines.append(line)
        return "\n".join(kept_lines) or text
    return text


def support_strength_profile(record: dict[str, Any], retrieved: list[tuple[PreparedDoc, float]]) -> dict[str, int]:
    record_gene_symbols = set(collect_record_gene_symbols(record))
    strong = 0
    primary = 0
    secondary = 0
    weak_proxy = 0
    cross_disease = 0
    for doc, _score in retrieved:
        flags = doc_support_flags(doc)
        doc_gene_symbols = getattr(doc, "_gene_symbols", set()) or collect_doc_gene_symbols(doc)
        gene_match = bool(record_gene_symbols.intersection(doc_gene_symbols))
        disease_match = doc_matches_disease(doc, record)
        if gene_match and not disease_match:
            cross_disease += 1
        if not disease_match or not getattr(doc, "_is_evidence_like", False):
            continue
        if flags.intersection({"dependency", "driver", "recurrent"}):
            strong += 1
            primary += 1
        elif flags.intersection({"mutation", "copy_number"}):
            secondary += 1
        elif flags.intersection({"expression", "methylation", "prognosis"}):
            weak_proxy += 1
    return {"strong": strong, "primary": primary, "secondary": secondary, "weak_proxy": weak_proxy, "cross_disease": cross_disease}


EVIDENCE_RELATION_PRIORITIES = {
    "SUPPORTS": 1.2,
    "COMPLEMENTS": 1.0,
    "CAUSAL_CHAIN": 1.3,
    "CONTRADICTS": 0.9,
    "DERIVED_FROM": 0.8,
    "OBSERVED_IN": 0.8,
    "MECHANISM_TO_OMICS_MANIFESTATION": 1.1,
    "MECHANISM_TO_DEPENDENCY_VULNERABILITY": 1.1,
    "DRUG_MECHANISM_TO_SENSITIVITY": 1.1,
}


def evidence_relation_priority(relation: str) -> float:
    return EVIDENCE_RELATION_PRIORITIES.get(relation, causal_relation_weight(relation))


def evidence_node_priority(doc: PreparedDoc, record: dict[str, Any]) -> float:
    score = 0.0
    node_type = getattr(doc, "_node_type", "") or ""
    disease_match = doc_matches_disease(doc, record)
    support_flags = doc_support_flags(doc)
    record_gene_symbols = set(collect_record_gene_symbols(record))
    doc_gene_symbols = getattr(doc, "_gene_symbols", set()) or collect_doc_gene_symbols(doc)
    gene_match = bool(record_gene_symbols.intersection(doc_gene_symbols))

    if node_type == "observation":
        score += 1.4
    elif node_type == "evidence":
        score += 1.1
    elif getattr(doc, "_is_evidence_like", False):
        score += 0.8
    elif getattr(doc, "_is_entity_like", False):
        score -= 0.25 if disease_match else 0.8

    if disease_match:
        score += 1.3
    elif node_type in {"mechanism_role_summary", "target_support_summary"}:
        score -= 1.6
    elif getattr(doc, "_is_evidence_like", False):
        score -= 0.7
    elif node_type == "observation":
        score -= 0.4
    if gene_match:
        score += 0.9
    if getattr(doc, "_is_evidence_like", False) and support_flags:
        score += min(0.6, 0.15 * len(support_flags.intersection({"directional", "pathway", "driver", "dependency", "mutation", "copy_number", "recurrent"})))
    if doc_metadata_field(doc, "source_family") in {"integrated", "pubmed_literature"} and disease_match:
        score += 0.2
    if not disease_match and getattr(doc, "_is_entity_like", False):
        score -= 0.4
    return score


def evidence_candidate_score(corpus: PreparedCorpus, record: dict[str, Any], doc: PreparedDoc, base_score: float) -> float:
    return base_score + metadata_overlap(doc, record) + evidence_node_priority(doc, record) + doc_scope_state(doc, record)["scope_score"]


def summarize_source_context(doc: PreparedDoc) -> str:
    source_context = doc_metadata_field(doc, "source_context", {}) or {}
    parts = []
    for key in ["group_axis", "comparison_label", "cohort_label", "cohort", "effect_label", "support_label"]:
        value = source_context.get(key)
        if value not in (None, "", [], {}):
            parts.append(f"{key}={value}")
    methods = source_context.get("methods") or doc_metadata_field(doc, "methods", [])
    if isinstance(methods, list) and methods:
        parts.append("methods=" + ",".join(str(item) for item in methods[:4]))
    return "; ".join(parts[:4])


def disease_matched_docs(record: dict[str, Any], docs: list[PreparedDoc]) -> list[PreparedDoc]:
    return [doc for doc in docs if doc_matches_disease(doc, record)]


def normalized_scope_values(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if not isinstance(value, list):
        value = [value]
    seen = set()
    ordered = []
    for item in value:
        if item in (None, "", [], {}):
            continue
        normalized = norm_text(str(item).replace("_", " "))
        if normalized and normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return ordered


def record_scope_profile(record: dict[str, Any]) -> dict[str, Any]:
    cached = record.get("_scope_profile_cache")
    if isinstance(cached, dict):
        return cached
    entities = record.get("entities") or {}
    metadata = record.get("metadata") or {}
    normalized_anchors = metadata.get("normalized_entity_anchors") or {}
    disease_terms = disease_term_set(
        record.get("oncology_focus"),
        entities.get("disease"),
        entities.get("disease_name"),
        normalized_anchors.get("disease_family"),
    )
    profile_terms = normalized_scope_values(
        [
            entities.get("molecular_profile"),
            entities.get("variant"),
            entities.get("alteration"),
            normalized_anchors.get("molecular_profile"),
        ]
    )
    therapy_terms = normalized_scope_values(
        [
            *(entities.get("therapy_names") or []),
            entities.get("therapy_name"),
            entities.get("drug"),
            normalized_anchors.get("therapy_name"),
        ]
    )
    profile = {
        "disease_terms": disease_terms,
        "gene_symbols": set(collect_record_gene_symbols(record)),
        "profile_terms": profile_terms,
        "therapy_terms": therapy_terms,
    }
    record["_scope_profile_cache"] = profile
    return profile


def doc_scope_haystack(doc: PreparedDoc) -> str:
    cached = getattr(doc, "_scope_haystack", None)
    if cached is not None:
        return cached
    metadata_blob = json.dumps(doc.metadata or {}, ensure_ascii=False, sort_keys=True)
    haystack = norm_text("\n".join([doc.searchable_text, metadata_blob]))
    setattr(doc, "_scope_haystack", haystack)
    return haystack


def doc_scope_state(doc: PreparedDoc, record: dict[str, Any]) -> dict[str, Any]:
    scope = record_scope_profile(record)
    disease_match = doc_matches_disease(doc, record)
    gene_match = bool(scope["gene_symbols"].intersection(getattr(doc, "_gene_symbols", set()) or collect_doc_gene_symbols(doc)))
    haystack = doc_scope_haystack(doc)
    profile_match = any(term in haystack for term in scope["profile_terms"]) if scope["profile_terms"] else None
    therapy_match = any(term in haystack for term in scope["therapy_terms"]) if scope["therapy_terms"] else None
    node_type = getattr(doc, "_node_type", "") or doc.document_level or ""
    score = 0.0
    strict_scope_rank = 0

    if disease_match:
        strict_scope_rank += 1
    if gene_match:
        strict_scope_rank += 1
    if scope["profile_terms"] and profile_match:
        strict_scope_rank += 1
    if scope["therapy_terms"] and therapy_match:
        strict_scope_rank += 1

    if scope["profile_terms"] or scope["therapy_terms"]:
        score += 0.25 * strict_scope_rank
    if disease_match and gene_match:
        score += 0.35
    if profile_match:
        score += 0.55
    elif scope["profile_terms"] and node_type in {"evidence", "observation", "parent"}:
        score -= 0.25
    if therapy_match:
        score += 0.65
    elif scope["therapy_terms"] and node_type in {"evidence", "observation", "parent"}:
        score -= 0.35
    if (profile_match or therapy_match) and not disease_match:
        score -= 0.4

    return {
        "disease_match": disease_match,
        "gene_match": gene_match,
        "profile_match": profile_match,
        "therapy_match": therapy_match,
        "strict_scope_rank": strict_scope_rank,
        "scope_score": score,
    }


def record_gene_match(doc: PreparedDoc, record: dict[str, Any]) -> bool:
    record_gene_symbols = set(collect_record_gene_symbols(record))
    doc_gene_symbols = getattr(doc, "_gene_symbols", set()) or collect_doc_gene_symbols(doc)
    return bool(record_gene_symbols.intersection(doc_gene_symbols))


def evidence_doc_brief(doc: PreparedDoc, record: dict[str, Any]) -> str:
    flags = sorted(doc_support_flags(doc))
    context_bits = summarize_source_context(doc)
    source_family = doc_metadata_field(doc, "source_family") or doc.knowledge_source
    scope_state = doc_scope_state(doc, record)
    line = (
        f"{doc.title} | source={source_family} | disease_match={'yes' if doc_matches_disease(doc, record) else 'no'} | "
        f"gene_match={'yes' if record_gene_match(doc, record) else 'no'} | "
        f"profile_match={scope_state['profile_match'] if scope_state['profile_match'] is not None else 'n/a'} | "
        f"therapy_match={scope_state['therapy_match'] if scope_state['therapy_match'] is not None else 'n/a'} | "
        f"flags={','.join(flags[:5]) if flags else 'none'}"
    )
    if context_bits:
        line += f" | context={context_bits}"
    return line


def matched_summary_docs(record: dict[str, Any], docs: list[PreparedDoc], node_type: str) -> list[PreparedDoc]:
    return [
        doc
        for doc in docs
        if getattr(doc, "_node_type", "") == node_type and doc_matches_disease(doc, record) and record_gene_match(doc, record)
    ]


def bundle_summary_grounded_state(record: dict[str, Any], docs: list[PreparedDoc]) -> dict[str, Any]:
    mechanism_summaries = matched_summary_docs(record, docs, "mechanism_role_summary")
    target_summaries = matched_summary_docs(record, docs, "target_support_summary")

    def summary_votes(summary_docs: list[PreparedDoc], key: str) -> tuple[str, list[str]]:
        votes = []
        clues = []
        for doc in summary_docs:
            comps = (doc.metadata or {}).get("bundle_components") or {}
            if key in comps:
                votes.append(bool(comps.get(key)))
                clues.append(f"{doc.title}:{'yes' if comps.get(key) else 'no'}")
        if not votes:
            return "unknown", clues
        if any(votes) and not all(votes):
            return "ambiguous", clues
        return ("yes" if all(votes) else "no"), clues

    causal_status, causal_clues = summary_votes(mechanism_summaries, "mechanism_context")
    pathway_status, pathway_clues = summary_votes(mechanism_summaries, "pathway_context")

    target_status = "unknown"
    target_clues: list[str] = []
    if target_summaries:
        levels = []
        for doc in target_summaries:
            inventory = (doc.metadata or {}).get("association_inventory") or {}
            direct = bool(inventory.get("direct_target_association"))
            kb = bool(inventory.get("knowledgebase_association"))
            dep = bool(inventory.get("dependency_or_drug_sensitivity"))
            omics = bool(inventory.get("supportive_omics"))
            prog = bool(inventory.get("clinical_prognosis"))
            if direct or kb or dep:
                levels.append("strong")
            elif omics or prog:
                levels.append("weak_proxy_only")
            else:
                levels.append("absent")
            target_clues.append(f"{doc.title}:{levels[-1]}")
        if len(set(levels)) > 1:
            target_status = "ambiguous"
        else:
            target_status = levels[0]

    if mechanism_summaries and target_summaries and all(
        status not in {"unknown", "ambiguous"} for status in [causal_status, pathway_status, target_status]
    ):
        confidence = "high"
    elif any(status == "ambiguous" for status in [causal_status, pathway_status, target_status]):
        confidence = "ambiguous"
    else:
        confidence = "low"

    return {
        "summary_confidence": confidence,
        "mechanism_summary_count": len(mechanism_summaries),
        "target_summary_count": len(target_summaries),
        "component_state": {
            "target_association": target_status,
            "causal_relation": causal_status,
            "pathway_context": pathway_status,
        },
        "clues": {
            "target_association": target_clues,
            "causal_relation": causal_clues,
            "pathway_context": pathway_clues,
        },
    }


def target_association_facet(record: dict[str, Any], docs: list[PreparedDoc]) -> dict[str, Any]:
    summary_docs = matched_summary_docs(record, docs, "target_support_summary")
    if summary_docs:
        doc = summary_docs[0]
        inventory = (doc.metadata or {}).get("association_inventory") or {}
        direct = bool(inventory.get("direct_target_association"))
        kb = bool(inventory.get("knowledgebase_association"))
        dep = bool(inventory.get("dependency_or_drug_sensitivity"))
        omics = bool(inventory.get("supportive_omics"))
        prog = bool(inventory.get("clinical_prognosis"))
        if direct or kb or dep:
            status = "strong"
        elif omics or prog:
            status = "weak_proxy_only"
        else:
            status = "absent"
        clues = [f"summary={doc.title}", f"direct_target_association={'yes' if direct else 'no'}", f"knowledgebase_association={'yes' if kb else 'no'}", f"dependency_or_drug_sensitivity={'yes' if dep else 'no'}", f"supportive_omics={'yes' if omics else 'no'}", f"clinical_prognosis={'yes' if prog else 'no'}"]
        return {"status": status, "clues": clues}
    evidence_docs = [doc for doc in disease_matched_docs(record, docs) if getattr(doc, "_is_evidence_like", False) and record_gene_match(doc, record)]
    if evidence_docs:
        return {"status": "weak_proxy_only", "clues": [evidence_doc_brief(doc, record) for doc in evidence_docs[:2]]}
    return {"status": "absent", "clues": []}


def bundle_facets(record: dict[str, Any], docs: list[PreparedDoc]) -> dict[str, dict[str, Any]]:
    mechanism_summaries = matched_summary_docs(record, docs, "mechanism_role_summary")
    target_summaries = matched_summary_docs(record, docs, "target_support_summary")
    matched_evidence = [doc for doc in disease_matched_docs(record, docs) if getattr(doc, "_is_evidence_like", False)]
    facets: dict[str, dict[str, Any]] = {}

    association_facet = target_association_facet(record, docs)
    facets["target_association"] = association_facet

    mechanism_yes = []
    pathway_yes = []
    driver_yes = []
    for doc in mechanism_summaries:
        comps = (doc.metadata or {}).get("bundle_components") or {}
        role_counts = (doc.metadata or {}).get("role_signal_counts") or {}
        if comps.get("mechanism_context") or comps.get("literature_mechanism"):
            mechanism_yes.append(doc.title)
        if comps.get("pathway_context"):
            pathway_yes.append(doc.title)
        if any(int(role_counts.get(key) or 0) > 0 for key in ["oncogenic_like", "protective_like"]):
            driver_yes.append(doc.title)

    if not pathway_yes and not mechanism_summaries:
        for doc in matched_evidence:
            if "pathway" in doc_support_flags(doc):
                pathway_yes.append(evidence_doc_brief(doc, record))
                if len(pathway_yes) >= 2:
                    break
    if not mechanism_yes and not mechanism_summaries:
        for doc in matched_evidence:
            if "directional" in doc_support_flags(doc) and doc_targets_non_disease_entity(doc, record):
                mechanism_yes.append(evidence_doc_brief(doc, record))
                if len(mechanism_yes) >= 2:
                    break
    if not driver_yes and not mechanism_summaries:
        for doc in matched_evidence:
            if record_gene_match(doc, record) and doc_has_strict_driver_clue(doc):
                driver_yes.append(evidence_doc_brief(doc, record))
                if len(driver_yes) >= 2:
                    break

    facets["causal_relation"] = {"status": "yes" if mechanism_yes else "no", "clues": mechanism_yes[:2]}
    facets["pathway_context"] = {"status": "yes" if pathway_yes else "no", "clues": pathway_yes[:2]}
    facets["driver"] = {"status": "yes" if driver_yes else "no", "clues": driver_yes[:2]}
    corroborated = (
        association_facet["status"] != "absent"
        and (facets["causal_relation"]["status"] == "yes" or facets["pathway_context"]["status"] == "yes")
    )
    facets["cross_source_corroboration"] = {"status": "yes" if corroborated else "no", "clues": []}
    return facets


def cohort_method_facet(record: dict[str, Any], docs: list[PreparedDoc]) -> dict[str, Any]:
    methods = []
    cohort_labels = []
    matched_docs = disease_matched_docs(record, docs)
    for doc in matched_docs:
        source_context = doc_metadata_field(doc, "source_context", {}) or {}
        cohort = (
            source_context.get("cohort")
            or source_context.get("cohort_label")
            or source_context.get("cohort_or_screen")
            or source_context.get("study_object")
        )
        if cohort:
            cohort_labels.append(str(cohort))
        method_values = source_context.get("methods") or doc_metadata_field(doc, "methods", [])
        if isinstance(method_values, list):
            methods.extend(str(item) for item in method_values if item)
    methods = unique_preserve(methods)
    cohort_labels = unique_preserve(cohort_labels)
    status = "explicit_methods_present" if methods else "explicit_methods_missing"
    clues = []
    if cohort_labels:
        clues.append("cohorts=" + ", ".join(cohort_labels[:3]))
    if methods:
        clues.append("methods=" + ", ".join(methods[:6]))
    return {"status": status, "clues": clues}


def evidence_relation_inventory(corpus: PreparedCorpus, doc_ids: list[str]) -> Counter[str]:
    selected = set(doc_ids)
    counts: Counter[str] = Counter()
    seen_edges = set()
    for doc_id in doc_ids:
        for neighbor_id, relation, _weight in corpus.graph_neighbors.get(doc_id, []):
            if neighbor_id not in selected:
                continue
            edge_key = tuple(sorted((doc_id, neighbor_id)) + [relation])
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            counts[relation] += 1
    return counts


def local_evidence_relation_snippets(corpus: PreparedCorpus, doc_ids: list[str], limit: int = 4) -> list[str]:
    selected = set(doc_ids)
    snippets = []
    seen_edges = set()
    for doc_id in doc_ids:
        source_doc = corpus.doc_by_id[doc_id]
        for neighbor_id, relation, _weight in corpus.graph_neighbors.get(doc_id, []):
            if neighbor_id not in selected:
                continue
            edge_key = tuple(sorted((doc_id, neighbor_id)) + [relation])
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            target_doc = corpus.doc_by_id[neighbor_id]
            snippets.append(f"- {source_doc.title} --{relation}--> {target_doc.title}")
            if len(snippets) >= limit:
                return snippets
    return snippets


def local_relation_program(corpus: PreparedCorpus, record: dict[str, Any], docs: list[PreparedDoc], limit: int = 4) -> list[str]:
    matched_docs = disease_matched_docs(record, docs)
    if not matched_docs:
        matched_docs = docs[:]
    selected_ids = [doc.doc_id for doc in matched_docs[:8]]
    snippets = local_evidence_relation_snippets(corpus, selected_ids, limit=limit)
    source_symbol, target_symbol = causal_query_pair(record)
    if source_symbol and target_symbol:
        source_hits = []
        target_hits = []
        for doc in matched_docs:
            if source_symbol in (getattr(doc, "_gene_symbols", set()) or set()) and "directional" in doc_support_flags(doc):
                source_hits.append(f"source_anchor={doc.title}")
            if target_symbol in (getattr(doc, "_gene_symbols", set()) or set()) and "directional" in doc_support_flags(doc):
                target_hits.append(f"target_anchor={doc.title}")
        snippets.extend(source_hits[:1])
        snippets.extend(target_hits[:1])
    return unique_preserve(snippets)[:limit]


def inject_bundle_summary_candidates(
    corpus: PreparedCorpus,
    record: dict[str, Any],
    expanded_scores: dict[str, float],
    expansion_candidates: set[str],
) -> None:
    task_type = str(record.get("task_type") or "")
    if task_type != "cross_source_support_bundle":
        return
    record_gene_symbols = collect_record_gene_symbols(record)
    boosted = 0
    for gene_symbol in record_gene_symbols[:4]:
        for doc_id in lookup_gene_symbol_doc_ids(corpus, gene_symbol):
            doc = corpus.doc_by_id[doc_id]
            node_type = getattr(doc, "_node_type", "") or ""
            if node_type not in {"mechanism_role_summary", "target_support_summary"}:
                continue
            if not doc_matches_disease(doc, record):
                continue
            score = evidence_candidate_score(corpus, record, doc, expanded_scores.get(doc_id, 0.0)) + 4.0
            if node_type == "mechanism_role_summary":
                score += 1.0
            if node_type == "target_support_summary":
                score += 0.8
            expanded_scores[doc_id] = max(expanded_scores.get(doc_id, float("-inf")), score)
            expansion_candidates.add(doc_id)
            boosted += 1
            if boosted >= 8:
                return


def filter_bundle_cross_disease_summaries(
    corpus: PreparedCorpus,
    record: dict[str, Any],
    expanded_scores: dict[str, float],
    expansion_candidates: set[str],
) -> None:
    task_type = str(record.get("task_type") or "")
    if task_type != "cross_source_support_bundle":
        return
    summary_ids = [
        doc_id
        for doc_id in expansion_candidates
        if getattr(corpus.doc_by_id[doc_id], "_node_type", "") in {"mechanism_role_summary", "target_support_summary"}
    ]
    if not summary_ids:
        return
    matched_ids = [doc_id for doc_id in summary_ids if doc_matches_disease(corpus.doc_by_id[doc_id], record)]
    keep_ids = set(matched_ids)
    for doc_id in summary_ids:
        if doc_id in keep_ids:
            continue
        expansion_candidates.discard(doc_id)
        expanded_scores.pop(doc_id, None)


def doc_aggregation_level(doc: PreparedDoc) -> str:
    value = doc_metadata_field(doc, "aggregation_level")
    if value in (None, "", [], {}):
        match = re.search(r"aggregation level:\s*([^\n|]+)", str(getattr(doc, "text", "") or ""), flags=re.IGNORECASE)
        value = match.group(1).strip() if match else ""
    normalized = norm_text(value)
    return normalized.replace(" ", "_") if normalized else "atomic_or_unspecified"


def evidence_state_profile(record: dict[str, Any], docs: list[PreparedDoc]) -> dict[str, Any]:
    allowed_labels = [label for label in allowed_answer_labels(record) if len(norm_text(label.replace("_", " "))) >= 4]
    aggregation_counts: Counter[str] = Counter()
    label_hits: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    scope_locked = 0
    partial_scope = 0
    contradictory = 0
    for doc in docs:
        scope_state = doc_scope_state(doc, record)
        if scope_state["strict_scope_rank"] >= 2:
            partial_scope += 1
        if scope_state["strict_scope_rank"] >= 3 or (
            scope_state["disease_match"]
            and scope_state["gene_match"]
            and (scope_state["profile_match"] is not False)
            and (scope_state["therapy_match"] is not False)
        ):
            scope_locked += 1
        if scope_state["disease_match"] and not all(
            match is not False for match in [scope_state["profile_match"], scope_state["therapy_match"]] if match is not None
        ):
            contradictory += 1
        aggregation_counts[doc_aggregation_level(doc)] += 1
        source_counts[str(doc_metadata_field(doc, "source_family") or doc.knowledge_source or "unknown")] += 1
        haystack = doc_scope_haystack(doc)
        for label in allowed_labels:
            normalized = norm_text(label.replace("_", " "))
            if normalized and normalized in haystack:
                label_hits[label] += 1
    consensus_candidates = [label for label, _count in label_hits.most_common(3)]
    if len(consensus_candidates) > 1:
        consensus_state = "multiple_candidate_labels"
    elif consensus_candidates:
        consensus_state = consensus_candidates[0]
    else:
        consensus_state = "implicit_or_missing"
    return {
        "scope_locked_docs": scope_locked,
        "partial_scope_docs": partial_scope,
        "scope_conflict_docs": contradictory,
        "source_diversity": len(source_counts),
        "source_counts": dict(source_counts),
        "aggregation_levels": dict(aggregation_counts),
        "primary_aggregation_level": aggregation_counts.most_common(1)[0][0] if aggregation_counts else "atomic_or_unspecified",
        "consensus_candidates": consensus_candidates,
        "consensus_state": consensus_state,
    }


def build_evidence_pack_doc(
    corpus: PreparedCorpus,
    record: dict[str, Any],
    retrieved: list[tuple[PreparedDoc, float]],
) -> PreparedDoc | None:
    real_docs = [doc for doc, _score in retrieved if not ((doc.metadata or {}).get("_evidence_rag_summary_kind"))]
    if not real_docs:
        return None
    ranked_real_docs = sorted(
        real_docs,
        key=lambda doc: (
            doc_scope_state(doc, record)["strict_scope_rank"],
            doc_scope_state(doc, record)["scope_score"],
            doc_matches_disease(doc, record),
            getattr(doc, "_is_evidence_like", False),
            doc.doc_id,
        ),
        reverse=True,
    )
    top_docs = ranked_real_docs[: min(len(ranked_real_docs), 8)]
    record_gene_symbols = collect_record_gene_symbols(record)
    node_counts = Counter((getattr(doc, "_node_type", "") or doc.document_level or "unknown") for doc in top_docs)
    source_counts = Counter(str(doc_metadata_field(doc, "source_family") or doc.knowledge_source or "unknown") for doc in top_docs)
    flag_counts = Counter()
    disease_matches = 0
    evidence_like_count = 0
    observation_count = 0
    entity_like_count = 0
    evidence_lines = []
    for doc in top_docs:
        flags = sorted(doc_support_flags(doc))
        flag_counts.update(flags)
        scope_state = doc_scope_state(doc, record)
        disease_match = scope_state["disease_match"]
        if disease_match:
            disease_matches += 1
        if getattr(doc, "_is_evidence_like", False):
            evidence_like_count += 1
        if (getattr(doc, "_node_type", "") or "") == "observation":
            observation_count += 1
        if getattr(doc, "_is_entity_like", False):
            entity_like_count += 1
        source_family = doc_metadata_field(doc, "source_family") or doc.knowledge_source
        context_bits = summarize_source_context(doc)
        line = (
            f"- {doc.title} | node_type={getattr(doc, '_node_type', '') or doc.document_level} | "
            f"source={source_family} | disease_match={'yes' if disease_match else 'no'} | "
            f"scope_rank={scope_state['strict_scope_rank']} | "
            f"profile_match={scope_state['profile_match'] if scope_state['profile_match'] is not None else 'n/a'} | "
            f"therapy_match={scope_state['therapy_match'] if scope_state['therapy_match'] is not None else 'n/a'} | "
            f"flags={','.join(flags[:5]) if flags else 'none'}"
        )
        if context_bits:
            line += f" | context={context_bits}"
        evidence_lines.append(line)

    matched_docs = disease_matched_docs(record, top_docs)
    if matched_docs:
        top_docs = matched_docs + [doc for doc in top_docs if doc not in matched_docs]
        top_docs = top_docs[: min(len(top_docs), 8)]

    relation_counts = evidence_relation_inventory(corpus, [doc.doc_id for doc in top_docs])
    relation_summary = ", ".join(f"{relation}={count}" for relation, count in relation_counts.most_common(8)) or "none"
    support_summary = ", ".join(f"{flag}={count}" for flag, count in flag_counts.most_common(8)) or "none"
    source_summary = ", ".join(f"{source}={count}" for source, count in source_counts.most_common(6))
    node_summary = ", ".join(f"{node_type}={count}" for node_type, count in node_counts.most_common())

    facets = bundle_facets(record, top_docs)
    association = target_association_facet(record, top_docs)
    cohort_method = cohort_method_facet(record, top_docs)
    relation_program = local_relation_program(corpus, record, top_docs, limit=5)
    state_profile = evidence_state_profile(record, top_docs)
    scope_profile = record_scope_profile(record)

    anchors = [
        f"oncology_focus={record.get('oncology_focus')}",
        f"genes={', '.join(record_gene_symbols) if record_gene_symbols else 'none'}",
    ]
    if scope_profile["profile_terms"]:
        anchors.append(f"molecular_profile={', '.join(scope_profile['profile_terms'][:3])}")
    if scope_profile["therapy_terms"]:
        anchors.append(f"therapy={', '.join(scope_profile['therapy_terms'][:3])}")
    text_lines = [
        "Scope-Locked Evidence Pack",
        "Anchors: " + " | ".join(anchors),
        f"Node inventory: {node_summary}",
        f"Source diversity: {source_summary}",
        f"Evidence-like nodes={evidence_like_count}; observation_nodes={observation_count}; entity_like_nodes={entity_like_count}; disease_matched={disease_matches}/{len(top_docs)}",
        "Scope state:",
        f"- scope_locked_docs: {state_profile['scope_locked_docs']}/{len(top_docs)}",
        f"- partial_scope_docs: {state_profile['partial_scope_docs']}/{len(top_docs)}",
        f"- scope_conflict_docs: {state_profile['scope_conflict_docs']}",
        f"Support inventory: {support_summary}",
        f"Relation inventory: {relation_summary}",
        "Evidence components:",
        f"- target_association: {association['status']}" + (f" | clues: {' ; '.join(association['clues'][:3])}" if association["clues"] else ""),
        f"- causal_relation: {facets['causal_relation']['status']}" + (f" | clues: {' ; '.join(facets['causal_relation']['clues'][:2])}" if facets["causal_relation"]["clues"] else ""),
        f"- pathway_context: {facets['pathway_context']['status']}" + (f" | clues: {' ; '.join(facets['pathway_context']['clues'][:2])}" if facets["pathway_context"]["clues"] else ""),
        f"- driver_signal: {facets['driver']['status']}" + (f" | clues: {' ; '.join(facets['driver']['clues'][:2])}" if facets["driver"]["clues"] else ""),
        f"- cross_source_corroboration: {facets['cross_source_corroboration']['status']}",
        f"- cohort_or_method_evidence: {cohort_method['status']}" + (f" | clues: {' ; '.join(cohort_method['clues'][:2])}" if cohort_method["clues"] else ""),
        "Evidence state:",
        f"- source_state: {'multi_source' if state_profile['source_diversity'] >= 2 else 'single_source'}",
        f"- primary_aggregation_level: {state_profile['primary_aggregation_level']}",
        f"- aggregation_levels: {', '.join(f'{key}={value}' for key, value in state_profile['aggregation_levels'].items()) or 'none'}",
        f"- consensus_state: {state_profile['consensus_state']}",
        f"- consensus_candidates: {', '.join(state_profile['consensus_candidates']) if state_profile['consensus_candidates'] else 'none'}",
        "Primary evidence slices:",
        *evidence_lines[:6],
    ]
    if relation_program:
        text_lines.extend(["Relation program:", *[f"- {line}" if not line.startswith('- ') else line for line in relation_program]])
    text_lines.extend(
        [
            "Inference policy:",
            "- preserve the locked scope first: disease, gene, molecular profile, and therapy constraints should override broader cross-disease concept matches",
            "- use Evidence components and Evidence state as the main intermediate representation; do not infer unsupported components from generic support inventory alone",
            "- preserve aggregation labels and consensus candidates exactly when mapping to the final answer; do not collapse tier or verdict labels into coarser support wording",
            "- treat weak_proxy_only target association as insufficient for positive corroboration or moderate_or_higher support unless direct, knowledgebase, or dependency evidence is present",
        ]
    )
    metadata = {
        "node_type": "evidence_rag_summary",
        "_evidence_rag_summary_kind": "evidence_pack",
        "relation_hits": list(relation_counts.keys())[:10],
        "_graph_rag_path_relations": list(relation_counts.keys())[:10],
        "facet_statuses": {
            "target_association": association["status"],
            "causal_relation": facets["causal_relation"]["status"],
            "pathway_context": facets["pathway_context"]["status"],
            "driver_signal": facets["driver"]["status"],
            "cross_source_corroboration": facets["cross_source_corroboration"]["status"],
            "cohort_or_method_evidence": cohort_method["status"],
        },
        "evidence_state": state_profile,
    }
    return PreparedDoc(
        doc_id=f"evidence-rag-summary::{record.get('id')}",
        knowledge_source="evidence_rag",
        document_level="virtual",
        parent_id=None,
        title="Scope-locked Evidence Pack",
        text="\n".join(text_lines),
        disease_tags=list(dict.fromkeys([record.get("oncology_focus")] + list(record.get("entities", {}).get("disease_tags") or []))),
        entity_names=record_gene_symbols[:8],
        entity_ids=[],
        metadata=metadata,
    )


def relation_aware_task(task_type: str | None) -> bool:
    return str(task_type or "") in {"causal_relation_effect", "cross_source_support_bundle", "pathway_class_membership"}


def graph_native_retrieval_task(task_type: str | None) -> bool:
    return str(task_type or "") in {
        "causal_relation_effect",
        "cross_source_support_bundle",
        "cross_source_corroboration_status",
        "mechanism_trace_intermediate_gene",
        "pathway_class_membership",
    }


def causal_relation_weight(relation: str) -> float:
    weights = {
        "CAUSAL_CHAIN": 1.3,
        "MECHANISM_TO_OMICS_MANIFESTATION": 1.2,
        "MECHANISM_TO_DEPENDENCY_VULNERABILITY": 1.2,
        "DRUG_MECHANISM_TO_SENSITIVITY": 1.2,
        "SUPPORTS": 0.95,
        "COMPLEMENTS": 0.95,
        "CONTRADICTS": 0.9,
        "REFINES": 0.7,
        "EXTENDS": 0.7,
        "CROSS_KG_ENTITY_EQUIVALENT": 0.85,
        "activates": 1.2,
        "inhibits": 1.2,
        "affects expression": 0.95,
        "increases expression": 1.0,
        "decreases expression": 1.0,
        "positive correlate": 0.9,
        "negative correlate": 0.9,
        "regulates": 0.85,
        "signaling pathway": 0.8,
        "relationships involving regulation and pathways": 0.8,
        "transcriptional regulation": 1.0,
        "phosphorylation": 0.95,
        "dephosphorylation": 0.95,
        "ubiquitination": 0.8,
        "degradation": 0.9,
        "binding": 0.55,
        "binds": 0.55,
        "ppi": 0.5,
        "physical association": 0.45,
        "associated with": 0.3,
        "LINKED_TO": 0.35,
        "OBSERVED_IN": 0.35,
        "DERIVED_FROM": 0.35,
    }
    return weights.get(relation, 0.35)


def causal_relation_sign(relation: str) -> int | None:
    normalized = norm_text(relation)
    if any(term in normalized for term in ["activate", "up regulate", "increase", "positive correlate", "phosphorylation", "transcriptional regulation", "stabil"]):
        return 1
    if any(term in normalized for term in ["inhibit", "down regulate", "decrease", "negative correlate", "dephosphorylation", "destabil", "degradation"]):
        return -1
    if relation in {"CROSS_KG_ENTITY_EQUIVALENT", "SUPPORTS", "COMPLEMENTS", "REFINES", "EXTENDS", "CAUSAL_CHAIN"}:
        return None
    return None


def infer_causal_path_effect(relations: list[str]) -> str:
    signs = [sign for sign in (causal_relation_sign(relation) for relation in relations) if sign is not None]
    if not signs:
        return "unknown"
    product = 1
    for sign in signs:
        product *= sign
    return "activating_or_upregulating" if product > 0 else "inhibitory_or_downregulating"


def best_causal_anchor_doc_ids(
    corpus: PreparedCorpus,
    record: dict[str, Any],
    doc_ids: set[str],
    symbol: str,
    limit: int = 3,
) -> list[str]:
    ranked = []
    for doc_id in doc_ids:
        doc = corpus.doc_by_id[doc_id]
        score = metadata_overlap(doc, record)
        if symbol in getattr(doc, "_gene_symbols", set()):
            score += 2.0
        if getattr(doc, "_is_evidence_like", False):
            score += 0.6
        if getattr(doc, "_node_type", "") in {"Gene", "entity"}:
            score += 0.4
        ranked.append((doc_id, score))
    ranked.sort(key=lambda item: (item[1], item[0]), reverse=True)
    return [doc_id for doc_id, _score in ranked[:limit]]


def build_causal_candidate_pool(
    corpus: PreparedCorpus,
    record: dict[str, Any],
    start_ids: list[str],
    goal_ids: set[str],
    max_neighbors: int = 8,
    max_second_hop: int = 5,
) -> set[str]:
    pool = set(start_ids) | set(goal_ids)
    frontier = list(pool)
    for doc_id in frontier:
        ranked_neighbors = []
        for neighbor_id, relation, weight in corpus.graph_neighbors.get(doc_id, []):
            if relation in {"HAS_CHUNK", "NEXT_CHUNK"}:
                continue
            neighbor_doc = corpus.doc_by_id[neighbor_id]
            neighbor_score = causal_relation_weight(relation) * weight + 0.5 * metadata_overlap(neighbor_doc, record)
            if getattr(neighbor_doc, "_is_evidence_like", False):
                neighbor_score += 0.3
            ranked_neighbors.append((neighbor_id, relation, neighbor_score))
        ranked_neighbors.sort(key=lambda item: (item[2], item[0]), reverse=True)
        for neighbor_id, relation, _score in ranked_neighbors[:max_neighbors]:
            pool.add(neighbor_id)
            if relation in {"CROSS_KG_ENTITY_EQUIVALENT", "SUPPORTS", "CAUSAL_CHAIN", "MECHANISM_TO_OMICS_MANIFESTATION"}:
                second_neighbors = []
                for second_id, second_relation, second_weight in corpus.graph_neighbors.get(neighbor_id, []):
                    if second_relation in {"HAS_CHUNK", "NEXT_CHUNK"}:
                        continue
                    second_doc = corpus.doc_by_id[second_id]
                    second_score = causal_relation_weight(second_relation) * second_weight + 0.35 * metadata_overlap(second_doc, record)
                    second_neighbors.append((second_id, second_score))
                second_neighbors.sort(key=lambda item: (item[1], item[0]), reverse=True)
                for second_id, _score in second_neighbors[:max_second_hop]:
                    pool.add(second_id)
    return pool


def find_causal_subgraph_paths(
    corpus: PreparedCorpus,
    record: dict[str, Any],
    source_doc_ids: set[str],
    target_doc_ids: set[str],
    source_symbol: str,
    target_symbol: str,
    max_hops: int = 3,
    beam_width: int = 14,
    max_paths: int = 6,
) -> list[tuple[list[str], list[str], float]]:
    start_ids = best_causal_anchor_doc_ids(corpus, record, source_doc_ids, source_symbol)
    goal_ids = set(best_causal_anchor_doc_ids(corpus, record, target_doc_ids, target_symbol))
    if not start_ids or not goal_ids:
        return []
    candidate_pool = build_causal_candidate_pool(corpus, record, start_ids, goal_ids)
    frontiers: list[tuple[list[str], list[str], float]] = []
    for doc_id in start_ids:
        base_doc = corpus.doc_by_id[doc_id]
        frontiers.append(([doc_id], [], metadata_overlap(base_doc, record) + 1.0))
    completed: list[tuple[list[str], list[str], float]] = []
    seen_states = set()
    for _depth in range(max_hops):
        next_frontiers: list[tuple[list[str], list[str], float]] = []
        for path_doc_ids, path_relations, path_score in frontiers:
            current_id = path_doc_ids[-1]
            state_key = (current_id, len(path_relations), tuple(path_doc_ids[:2]))
            if state_key in seen_states:
                continue
            seen_states.add(state_key)
            for neighbor_id, relation, weight in corpus.graph_neighbors.get(current_id, []):
                if relation in {"HAS_CHUNK", "NEXT_CHUNK"} or neighbor_id in path_doc_ids or neighbor_id not in candidate_pool:
                    continue
                neighbor_doc = corpus.doc_by_id[neighbor_id]
                relation_bonus = causal_relation_weight(relation) * weight
                neighbor_score = path_score + relation_bonus + 0.55 * metadata_overlap(neighbor_doc, record)
                neighbor_symbols = getattr(neighbor_doc, "_gene_symbols", set())
                if source_symbol in neighbor_symbols or target_symbol in neighbor_symbols:
                    neighbor_score += 1.2
                if getattr(neighbor_doc, "_is_evidence_like", False):
                    neighbor_score += 0.5
                if getattr(neighbor_doc, "_is_entity_like", False) and not getattr(neighbor_doc, "_has_disease_tags", False):
                    neighbor_score -= 0.2
                next_doc_ids = path_doc_ids + [neighbor_id]
                next_relations = path_relations + [relation]
                if neighbor_id in goal_ids:
                    completed.append((next_doc_ids, next_relations, neighbor_score + 2.5))
                if len(next_relations) < max_hops:
                    next_frontiers.append((next_doc_ids, next_relations, neighbor_score))
        if completed and len(completed) >= max_paths:
            break
        next_frontiers.sort(key=lambda item: item[2], reverse=True)
        frontiers = next_frontiers[:beam_width]
        if not frontiers:
            break
    completed.sort(key=lambda item: item[2], reverse=True)
    return completed[:max_paths]


def render_causal_path_snippet(corpus: PreparedCorpus, path_doc_ids: list[str], path_relations: list[str]) -> str:
    segments = [corpus.doc_by_id[path_doc_ids[0]].title]
    for relation, doc_id in zip(path_relations, path_doc_ids[1:]):
        segments.append(f"--{relation}--> {corpus.doc_by_id[doc_id].title}")
    inferred = infer_causal_path_effect(path_relations)
    return f"inferred={inferred}: " + " ".join(segments)


def direct_tarkg_pair_relations(corpus: PreparedCorpus, source_symbol: str, target_symbol: str) -> list[tuple[PreparedDoc, PreparedDoc, str]]:
    if not source_symbol or not target_symbol:
        return []
    source_ids = lookup_gene_symbol_doc_ids(corpus, source_symbol, knowledge_source="tarkg")
    target_ids = set(lookup_gene_symbol_doc_ids(corpus, target_symbol, knowledge_source="tarkg"))
    relations = []
    for source_id in source_ids:
        for neighbor_id, relation, _weight in corpus.graph_neighbors.get(source_id, []):
            if neighbor_id in target_ids:
                relations.append((corpus.doc_by_id[source_id], corpus.doc_by_id[neighbor_id], relation))
    return relations


def allowed_answer_labels(record: dict[str, Any]) -> list[str]:
    metadata = record.get("metadata") or {}
    gold = record.get("gold_standard") or {}
    options = metadata.get("eval_answer_options") or []
    if options:
        return [str(option) for option in options if str(option).strip()]
    acceptable = gold.get("acceptable_answers") or []
    if gold.get("answer_format") == "entity":
        return [str(item) for item in acceptable[:8] if str(item).strip()]
    return []


def render_anchor_hint(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") or {}
    anchors = metadata.get("normalized_entity_anchors") or {}
    parts = []
    if anchors.get("disease_family"):
        parts.append(f"disease_family={anchors['disease_family']}")
    genes = anchors.get("gene_symbols") or []
    if genes:
        parts.append(f"gene_symbols={', '.join(genes[:8])}")
    if anchors.get("pathway_name"):
        parts.append(f"pathway_name={anchors['pathway_name']}")
    if anchors.get("cohort"):
        parts.append(f"cohort={anchors['cohort']}")
    return "; ".join(parts)


def short_answer_instruction(record: dict[str, Any]) -> str:
    gold = record.get("gold_standard") or {}
    answer_format = gold.get("answer_format")
    task_type = record.get("task_type")
    source = record.get("source")
    options = allowed_answer_labels(record)
    if task_type == "driver_multimethod_support":
        return "Return only one token: yes or no."
    if task_type == "causal_relation_effect":
        return (
            "Return only one exact label from this closed set: activating_or_upregulating | inhibitory_or_downregulating | unknown. "
            "Prefer activating_or_upregulating for explicit clues such as activates, up-regulates, increases expression, phosphorylation, transcriptional regulation, or positive correlate. "
            "Prefer inhibitory_or_downregulating for explicit clues such as inhibits, down-regulates, decreases expression, destabilization, degradation, or negative correlate. "
            "Use direct pair clues first. If only a multi-hop path is present, combine the signed relations along the shortest path. "
            "If the context only shows non-directional links such as binds, ppi, association, or separate single-gene evidence without a pairwise/path relation, return unknown."
        )
    if task_type in {"target_disease_association_strength", "intogen_support_strength", "cgc_support_strength"} and options:
        return (
            f"Return only one exact label from this closed set: {' | '.join(options)}. "
            "Be conservative: disease-matched expression, methylation, prognosis, or single-modality omics clues alone should not be promoted to high support. "
            "Prefer moderate or high only when the retrieved context includes stronger and more explicit support layers such as recurrent, driver-like, dependency, or multi-layer disease-matched evidence."
        )
    if task_type == "cross_source_support_bundle" and options:
        return (
            f"Return only one exact label from this closed set: {' | '.join(options)}. "
            "Decide each bundle component explicitly. Only include driver when there is direct driver-like evidence such as mutation, recurrent alteration, copy-number, or oncogenic-role support. "
            "Only include pathway_context when the retrieved context explicitly mentions pathway, signaling, or mechanism context. "
            "Do not infer components from generic gene relevance alone."
        )
    if task_type == "driver_role_assignment" and options:
        return (
            f"Return only one exact label from this closed set: {' | '.join(options)}. "
            "Prefer activating or loss_of_function only when mutation, copy-number, or explicit functional-role clues are present. "
            "Generic expression, methylation, or prognosis evidence alone should usually remain ambiguous."
        )
    if task_type in {"mechanism_bioasq_yesno", "mechanism_literature_reasoning"}:
        return "Return only one token: yes or no."
    if task_type in {
        "driver_role_assignment",
        "target_disease_association_strength",
        "intogen_support_strength",
        "cgc_support_strength",
        "causal_relation_effect",
        "pathway_class_membership",
        "cross_source_support_bundle",
    } and options:
        return f"Return only one exact label from this closed set: {' | '.join(options)}."
    if answer_format == "boolean":
        return "Return only one token: yes or no."
    if task_type == "cancer_cell_line_annotation":
        return (
            "Return only the most specific canonical cancer label or OncoTree code supported by the retrieved context. "
            "Prefer explicit subtype codes such as COAD, READ, MACR, or HCC when present. "
            "Do not return broad family IDs such as DOID:9256 unless no more specific label or code is stated."
        )
    if source in {"civic_assertion", "civic_evidence"}:
        return (
            "Return only the shortest clinical implication label supported by the retrieved context. "
            "Prefer compact labels such as SUPPORTS POOR_OUTCOME, DOES_NOT_SUPPORT SENSITIVITYRESPONSE, "
            "SUPPORTS RESISTANCE, or SUPPORTS UNCERTAIN_SIGNIFICANCE."
        )
    if answer_format == "entity":
        if options:
            return "Return only one concise entity answer matching the canonical naming in the retrieved context."
        return "Return only the entity label, without explanation."
    if answer_format == "categorical":
        return "Return only the shortest categorical label supported by the retrieved context."
    return "Return only the shortest answer supported by the retrieved context."


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


def normalize_gold_text(value: Any) -> str:
    if isinstance(value, dict):
        return norm_text(" | ".join(str(item) for item in value.values() if item not in (None, "", [], {})))
    if isinstance(value, list):
        return norm_text(" | ".join(str(item) for item in value if item not in (None, "", [], {})))
    return norm_text(value)


@dataclass
class PreparedDoc:
    doc_id: str
    knowledge_source: str
    document_level: str
    parent_id: str | None
    title: str
    text: str
    disease_tags: list[str]
    entity_names: list[str]
    entity_ids: list[str]
    metadata: dict[str, Any]

    @property
    def searchable_text(self) -> str:
        return "\n".join([self.title, self.text] + self.disease_tags + self.entity_names)


def build_virtual_relation_doc(
    *,
    doc_id: str,
    title: str,
    text: str,
    entity_names: list[str],
    disease_tags: list[str],
    relation_hits: list[str],
    knowledge_source: str = "graph_relation_path",
) -> PreparedDoc:
    return PreparedDoc(
        doc_id=doc_id,
        knowledge_source=knowledge_source,
        document_level="virtual",
        parent_id=None,
        title=title,
        text=text,
        disease_tags=disease_tags,
        entity_names=entity_names,
        entity_ids=[],
        metadata={"relation_hits": relation_hits, "_graph_rag_path_relations": relation_hits},
    )


class PreparedCorpus:
    def __init__(self, docs: list[PreparedDoc], edges: list[dict[str, Any]]) -> None:
        self.docs = docs
        self.doc_by_id = {doc.doc_id: doc for doc in docs}
        self.edges = edges
        self.parent_to_children: dict[str, list[str]] = defaultdict(list)
        self.graph_neighbors: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
        self.postings: dict[str, list[tuple[str, int]]] = defaultdict(list)
        self.doc_len: dict[str, int] = {}
        self.doc_freq: Counter[str] = Counter()
        self.doc_term_freq: dict[str, Counter[str]] = {}
        self.gene_symbol_to_doc_ids: dict[str, list[str]] = defaultdict(list)
        self.avg_doc_len = 0.0
        self._build()

    def _build(self) -> None:
        lengths = []
        for doc in self.docs:
            if doc.parent_id:
                self.parent_to_children[doc.parent_id].append(doc.doc_id)
            doc._disease_terms = set()
            for tag in doc.disease_tags:
                doc._disease_terms.update(disease_term_set(tag))
            doc._normalized_entity_names = {norm_text(name) for name in doc.entity_names if norm_text(name)}
            doc._gene_symbols = collect_doc_gene_symbols(doc)
            doc._node_type = (doc.metadata or {}).get("node_type") or ""
            doc._has_disease_tags = bool(doc._disease_terms)
            doc._is_entity_like = doc.document_level == "node" or doc._node_type in {"entity", "cohort", "Gene", "Disease", "Drug", "Phenotype", "Pathway", "Cohort", "MolecularFeature", "Compound", "Symptom", "Anatomy", "TCM_CMM"}
            doc._is_evidence_like = doc._node_type == "evidence" or doc.document_level == "parent"
            doc._support_flags = doc_support_flags(doc)
            for symbol in doc._gene_symbols:
                self.gene_symbol_to_doc_ids[symbol].append(doc.doc_id)
            tokens = tokenize(doc.searchable_text)
            term_freq = Counter(tokens)
            self.doc_term_freq[doc.doc_id] = term_freq
            self.doc_len[doc.doc_id] = len(tokens)
            lengths.append(len(tokens))
            for token, tf in term_freq.items():
                self.postings[token].append((doc.doc_id, tf))
                self.doc_freq[token] += 1
        self.avg_doc_len = sum(lengths) / max(len(lengths), 1)
        for edge in self.edges:
            source = edge["source_doc_id"]
            target = edge["target_doc_id"]
            relation = edge["relation"]
            weight = float(edge.get("weight") or 1.0)
            self.graph_neighbors[source].append((target, relation, weight))
            self.graph_neighbors[target].append((source, relation, weight))

    def bm25_scores(self, query_text: str, k1: float = 1.5, b: float = 0.75) -> dict[str, float]:
        scores: defaultdict[str, float] = defaultdict(float)
        tokens = tokenize(query_text)
        num_docs = len(self.docs)
        for token in tokens:
            df = self.doc_freq.get(token, 0)
            if df == 0:
                continue
            idf = math.log(1 + (num_docs - df + 0.5) / (df + 0.5))
            for doc_id, tf in self.postings[token]:
                dl = self.doc_len.get(doc_id, 0)
                denom = tf + k1 * (1 - b + b * dl / max(self.avg_doc_len, 1))
                scores[doc_id] += idf * ((tf * (k1 + 1)) / max(denom, 1e-9))
        return dict(scores)

    def to_cache_payload(self) -> dict[str, Any]:
        return {
            "docs": self.docs,
            "parent_to_children": dict(self.parent_to_children),
            "graph_neighbors": dict(self.graph_neighbors),
            "postings": dict(self.postings),
            "doc_len": self.doc_len,
            "doc_freq": dict(self.doc_freq),
            "gene_symbol_to_doc_ids": dict(self.gene_symbol_to_doc_ids),
            "avg_doc_len": self.avg_doc_len,
            "edge_count": len(self.edges),
        }

    @classmethod
    def from_cache_payload(cls, payload: dict[str, Any]) -> "PreparedCorpus":
        corpus = cls.__new__(cls)
        corpus.docs = payload["docs"]
        corpus.doc_by_id = {doc.doc_id: doc for doc in corpus.docs}
        corpus.edges = []
        corpus.parent_to_children = defaultdict(list, payload.get("parent_to_children") or {})
        corpus.graph_neighbors = defaultdict(list, payload.get("graph_neighbors") or {})
        corpus.postings = defaultdict(list, payload.get("postings") or {})
        corpus.doc_len = payload.get("doc_len") or {}
        corpus.doc_freq = Counter(payload.get("doc_freq") or {})
        corpus.doc_term_freq = {}
        corpus.gene_symbol_to_doc_ids = defaultdict(list, payload.get("gene_symbol_to_doc_ids") or {})
        corpus.avg_doc_len = float(payload.get("avg_doc_len") or 0.0)
        corpus._cached_edge_count = int(payload.get("edge_count") or 0)
        return corpus


def prepared_corpus_paths(base_dir: Path, view: str, knowledge_source: str) -> tuple[Path, Path]:
    manifest_path = base_dir / "manifest_latest.json"
    docs_path = base_dir / view / f"{knowledge_source}_docs.jsonl"
    edges_path = base_dir / view / f"{knowledge_source}_edges.jsonl"
    if manifest_path.exists():
        repo_root = base_dir.parents[2]
        manifest = read_json(manifest_path)
        source_meta = (((manifest.get("views") or {}).get(view) or {}).get("knowledge_sources") or {}).get(knowledge_source) or {}
        if source_meta.get("docs_path"):
            docs_path = repo_root / source_meta["docs_path"]
        if source_meta.get("edges_path"):
            edges_path = repo_root / source_meta["edges_path"]
    return docs_path, edges_path


def prepared_corpus_cache_path(docs_path: Path) -> Path:
    return docs_path.parent / f"{docs_path.stem}__corpus_cache_v{CORPUS_CACHE_VERSION}.pkl"


def corpus_source_signature(docs_path: Path, edges_path: Path) -> dict[str, Any]:
    def path_sig(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {"path": str(path), "exists": False}
        stat = path.stat()
        return {
            "path": str(path),
            "exists": True,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    return {
        "version": CORPUS_CACHE_VERSION,
        "docs": path_sig(docs_path),
        "edges": path_sig(edges_path),
    }


def write_prepared_corpus_cache(cache_path: Path, payload: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temp_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp_path, cache_path)


def read_prepared_corpus_cache(cache_path: Path) -> dict[str, Any]:
    with cache_path.open("rb") as handle:
        return pickle.load(handle)


def load_prepared_corpus(
    base_dir: Path,
    view: str,
    knowledge_source: str,
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> PreparedCorpus:
    docs_path, edges_path = prepared_corpus_paths(base_dir, view, knowledge_source)
    cache_path = prepared_corpus_cache_path(docs_path)
    signature = corpus_source_signature(docs_path, edges_path)
    cache_enabled = use_cache and knowledge_source in CACHEABLE_KNOWLEDGE_SOURCES

    if cache_enabled and cache_path.exists() and not refresh_cache:
        try:
            cached = read_prepared_corpus_cache(cache_path)
            if cached.get("signature") == signature:
                corpus = PreparedCorpus.from_cache_payload(cached["payload"])
                corpus.source_docs_path = docs_path
                corpus.source_signature = signature
                return corpus
        except Exception:
            pass

    docs = [PreparedDoc(**row) for row in read_jsonl(docs_path)]
    edges = read_jsonl(edges_path) if edges_path.exists() else []
    corpus = PreparedCorpus(docs, edges)
    corpus.source_docs_path = docs_path
    corpus.source_signature = signature
    if cache_enabled:
        write_prepared_corpus_cache(
            cache_path,
            {
                "signature": signature,
                "payload": corpus.to_cache_payload(),
            },
        )
    return corpus


_DENSE_RAG_MODEL: Any | None = None
_DENSE_RAG_MODEL_KEY: tuple[str, str | None, str, int] | None = None
_DENSE_RAG_INDEXES: dict[str, Any] = {}


def dense_rag_model_name() -> str:
    return os.environ.get("DENSE_RAG_MODEL", DENSE_RAG_DEFAULT_MODEL)


def dense_rag_revision() -> str | None:
    if "DENSE_RAG_REVISION" in os.environ:
        return os.environ["DENSE_RAG_REVISION"] or None
    return DENSE_RAG_DEFAULT_REVISION if dense_rag_model_name() == DENSE_RAG_DEFAULT_MODEL else None


def dense_rag_model() -> Any:
    global _DENSE_RAG_MODEL, _DENSE_RAG_MODEL_KEY
    model_name = dense_rag_model_name()
    revision = dense_rag_revision()
    device = os.environ.get("DENSE_RAG_DEVICE", "cpu")
    max_seq_length = int(os.environ.get("DENSE_RAG_MAX_SEQ_LENGTH", DENSE_RAG_DEFAULT_MAX_SEQ_LENGTH))
    model_key = (model_name, revision, device, max_seq_length)
    if _DENSE_RAG_MODEL is not None and _DENSE_RAG_MODEL_KEY == model_key:
        return _DENSE_RAG_MODEL
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Dense RAG requires sentence-transformers. Install requirements-dense.txt."
        ) from exc
    offline = os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    model = SentenceTransformer(model_name, revision=revision, device=device, local_files_only=offline)
    model.max_seq_length = max_seq_length
    _DENSE_RAG_MODEL = model
    _DENSE_RAG_MODEL_KEY = model_key
    return model


def dense_rag_cache_paths(corpus: PreparedCorpus) -> tuple[Path, Path]:
    docs_path = Path(corpus.source_docs_path)
    model_slug = re.sub(r"[^a-z0-9]+", "_", dense_rag_model_name().lower()).strip("_")
    stem = f"{docs_path.stem}__dense_{model_slug}_v{DENSE_RAG_INDEX_VERSION}"
    return docs_path.parent / f"{stem}.faiss", docs_path.parent / f"{stem}.json"


def dense_rag_manifest(corpus: PreparedCorpus, dimension: int | None = None) -> dict[str, Any]:
    return {
        "version": DENSE_RAG_INDEX_VERSION,
        "source_signature": corpus.source_signature,
        "model": dense_rag_model_name(),
        "revision": dense_rag_revision(),
        "max_seq_length": int(os.environ.get("DENSE_RAG_MAX_SEQ_LENGTH", DENSE_RAG_DEFAULT_MAX_SEQ_LENGTH)),
        "normalization": "l2",
        "similarity": "inner_product",
        "document_count": len(corpus.docs),
        "dimension": dimension,
    }


def load_dense_rag_index(corpus: PreparedCorpus) -> Any | None:
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("Dense RAG requires faiss-cpu. Install requirements-dense.txt.") from exc
    index_path, manifest_path = dense_rag_cache_paths(corpus)
    cache_key = str(index_path) + "::" + json.dumps(dense_rag_manifest(corpus), sort_keys=True)
    if cache_key in _DENSE_RAG_INDEXES:
        return _DENSE_RAG_INDEXES[cache_key]
    if not index_path.exists() or not manifest_path.exists():
        return None
    try:
        manifest = read_json(manifest_path)
        expected = dense_rag_manifest(corpus, manifest.get("dimension"))
        if manifest != expected:
            return None
        index = faiss.read_index(str(index_path))
        if index.ntotal != len(corpus.docs) or index.d != manifest.get("dimension"):
            return None
    except Exception:
        return None
    _DENSE_RAG_INDEXES[cache_key] = index
    return index


def ensure_dense_rag_index(corpus: PreparedCorpus, *, force: bool = False) -> Any:
    # Load the Torch-backed encoder before FAISS to avoid an OpenMP runtime
    # initialization conflict observed on macOS when the import order is reversed.
    model = dense_rag_model()
    try:
        import faiss
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Dense RAG requires NumPy and faiss-cpu. Install requirements-dense.txt.") from exc
    if not force:
        cached = load_dense_rag_index(corpus)
        if cached is not None:
            return cached

    encode_chunk_size = int(os.environ.get("DENSE_RAG_ENCODE_CHUNK", "2048"))
    batch_size = int(os.environ.get("DENSE_RAG_BATCH_SIZE", "128"))
    index = None
    total = len(corpus.docs)
    started = time.time()
    for start in range(0, total, encode_chunk_size):
        stop = min(start + encode_chunk_size, total)
        texts = [doc.searchable_text for doc in corpus.docs[start:stop]]
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        if index is None:
            index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        elapsed = max(time.time() - started, 1e-9)
        print(f"[dense-rag] encoded {stop}/{total} documents ({stop / elapsed:.1f} docs/s)", flush=True)
    if index is None:
        raise RuntimeError("cannot build a Dense RAG index for an empty corpus")

    index_path, manifest_path = dense_rag_cache_paths(corpus)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_temp = index_path.with_suffix(index_path.suffix + ".tmp")
    manifest_temp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    faiss.write_index(index, str(index_temp))
    with manifest_temp.open("w", encoding="utf-8") as handle:
        json.dump(dense_rag_manifest(corpus, index.d), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(index_temp, index_path)
    os.replace(manifest_temp, manifest_path)
    cache_key = str(index_path) + "::" + json.dumps(dense_rag_manifest(corpus), sort_keys=True)
    _DENSE_RAG_INDEXES[cache_key] = index
    return index


def build_query_text(record: dict[str, Any]) -> str:
    entities = record.get("entities") or {}
    metadata = record.get("metadata") or {}
    anchor_meta = metadata.get("normalized_entity_anchors") or {}
    gene_entrez_id = entities.get("gene_entrez_id")
    disease_cui = entities.get("disease_cui")
    model_id = entities.get("model_id")
    rrid = entities.get("rrid")
    gene_symbols = collect_record_gene_symbols(record)
    parts = [
        record.get("question"),
        record.get("oncology_focus"),
        entities.get("disease"),
        entities.get("disease_name"),
        entities.get("gene_symbol"),
        entities.get("target_gene_symbol"),
        " ".join(gene_symbols),
        entities.get("molecular_profile"),
        entities.get("pathway_name"),
        entities.get("cohort"),
        entities.get("cell_line_name"),
        gene_entrez_id,
        f"Entrez gene {gene_entrez_id}" if gene_entrez_id else None,
        f"NCBIGene:{gene_entrez_id}" if gene_entrez_id else None,
        disease_cui,
        f"UMLS:{disease_cui}" if disease_cui else None,
        model_id,
        rrid,
        anchor_meta.get("disease_family"),
        " ".join(anchor_meta.get("gene_symbols") or []),
        metadata.get("oncotree_code"),
        metadata.get("oncotree_subtype"),
        metadata.get("oncotree_lineage"),
        " ".join(entities.get("therapies") or []),
    ]
    normalized_parts = []
    for part in parts:
        if part in (None, ""):
            continue
        normalized_parts.append(str(part))
    return " ".join(normalized_parts)


def metadata_overlap(doc: PreparedDoc, record: dict[str, Any]) -> float:
    entities = record.get("entities") or {}
    metadata = record.get("metadata") or {}
    task_type = record.get("task_type") or ""
    overlaps = 0.0
    disease_terms = disease_term_set(record.get("oncology_focus"), entities.get("disease"), entities.get("disease_name"))
    doc_disease_terms = getattr(doc, "_disease_terms", set())
    if disease_terms.intersection(doc_disease_terms):
        overlaps += 2.0
        if getattr(doc, "_is_evidence_like", False):
            overlaps += 0.8
    therapy_terms = {norm_text(term) for term in entities.get("therapies") or [] if norm_text(term)}
    if therapy_terms.intersection(getattr(doc, "_normalized_entity_names", set())):
        overlaps += 2.0
    record_gene_symbols = set(collect_record_gene_symbols(record))
    matching_gene_symbols = record_gene_symbols.intersection(getattr(doc, "_gene_symbols", set()))
    if matching_gene_symbols:
        overlaps += 1.8 * len(matching_gene_symbols)
        if len(record_gene_symbols) > 1 and len(matching_gene_symbols) == len(record_gene_symbols):
            overlaps += 2.2
    molecular_profile = entities.get("molecular_profile") or ""
    profile_tokens = set(tokenize(molecular_profile)) | set(re.findall(r"\b[A-Z0-9\-]{3,12}\b", molecular_profile.upper()))
    if profile_tokens.intersection({token.upper() for name in doc.entity_names for token in re.findall(r"\b[A-Z0-9\-]{3,12}\b", name.upper())}):
        overlaps += 1.5
    id_terms = {
        norm_text(entities.get("gene_entrez_id")),
        norm_text(entities.get("disease_cui")),
        norm_text(entities.get("model_id")),
        norm_text(entities.get("rrid")),
        norm_text(metadata.get("oncotree_code")),
    }
    id_terms = {term for term in id_terms if term}
    doc_ids = {norm_text(value) for value in doc.entity_ids if norm_text(value)}
    if id_terms.intersection(doc_ids):
        overlaps += 2.5
    pathway_name = norm_text(entities.get("pathway_name") or (metadata.get("normalized_entity_anchors") or {}).get("pathway_name"))
    if pathway_name and any(pathway_name in norm_text(name) for name in doc.entity_names):
        overlaps += 1.5
    if is_pure_rag_profile():
        if getattr(doc, "_is_entity_like", False) and not getattr(doc, "_has_disease_tags", False):
            overlaps -= 0.8
        if matching_gene_symbols and not bool(disease_terms.intersection(doc_disease_terms)):
            overlaps -= 0.4
        return overlaps
    relation_aware_tasks = {"causal_relation_effect", "cross_source_support_bundle", "pathway_class_membership"}
    if task_type in relation_aware_tasks:
        if doc.knowledge_source == "tarkg":
            overlaps += 1.8
        if getattr(doc, "_is_evidence_like", False) and disease_terms.intersection(doc_disease_terms):
            overlaps += 1.0
        if getattr(doc, "_is_entity_like", False) and not getattr(doc, "_has_disease_tags", False):
            overlaps -= 1.4
    elif getattr(doc, "_is_entity_like", False) and not getattr(doc, "_has_disease_tags", False):
        overlaps -= 0.8
    support_flags = getattr(doc, "_support_flags", set()) or doc_support_flags(doc)
    disease_match = bool(disease_terms.intersection(doc_disease_terms))
    node_type = getattr(doc, "_node_type", "") or ""
    if matching_gene_symbols and not disease_match:
        if task_type in {"target_disease_association_strength", "intogen_support_strength", "cross_source_support_bundle", "driver_role_assignment"}:
            overlaps -= 1.6
        else:
            overlaps -= 0.4
    if task_type in {"target_disease_association_strength", "intogen_support_strength", "cgc_support_strength", "driver_role_assignment", "driver_multimethod_support"}:
        if getattr(doc, "_is_evidence_like", False) and disease_terms.intersection(doc_disease_terms):
            overlaps += 0.8
    if task_type in {"target_disease_association_strength", "intogen_support_strength", "cgc_support_strength"}:
        if disease_match and getattr(doc, "_is_evidence_like", False):
            if support_flags.intersection({"dependency", "driver", "recurrent"}):
                overlaps += 1.0
            elif support_flags.intersection({"mutation", "copy_number"}):
                overlaps += 0.2
            if support_flags.intersection({"expression", "methylation", "prognosis"}) and not support_flags.intersection({"mutation", "copy_number", "dependency", "driver", "recurrent"}):
                overlaps -= 0.7
        if support_flags.intersection({"prognosis"}) and not disease_match:
            overlaps -= 0.6
    if task_type == "cross_source_support_bundle":
        if disease_match and getattr(doc, "_is_evidence_like", False):
            if support_flags.intersection({"pathway", "directional"}):
                overlaps += 0.9
            if support_flags.intersection({"mutation", "driver", "recurrent"}):
                overlaps += 0.5
            if support_flags.intersection({"expression", "methylation", "prognosis"}) and not support_flags.intersection({"pathway", "directional", "mutation", "copy_number", "driver", "recurrent"}):
                overlaps -= 0.5
        if disease_match and node_type == "observation":
            overlaps += 1.0
        if disease_match and node_type == "evidence":
            overlaps += 0.35
        if node_type == "mechanism_role_summary":
            overlaps += 2.2 if disease_match and matching_gene_symbols else -4.0
        if node_type == "target_support_summary":
            overlaps += 1.8 if disease_match and matching_gene_symbols else -2.5
        if node_type in {"mechanism_role_summary", "target_support_summary"} and matching_gene_symbols and not disease_match:
            overlaps -= 3.2
        if doc_metadata_field(doc, "source_family") in {"integrated", "pubmed_literature"} and disease_match:
            overlaps += 0.4
    if task_type == "driver_role_assignment":
        if disease_match and getattr(doc, "_is_evidence_like", False):
            if support_flags.intersection({"mutation", "copy_number", "driver", "recurrent"}):
                overlaps += 1.1
            if support_flags.intersection({"expression", "methylation", "prognosis"}) and not support_flags.intersection({"mutation", "copy_number", "driver", "recurrent"}):
                overlaps -= 0.9
    return overlaps


def top_docs_from_scores(corpus: PreparedCorpus, scores: dict[str, float], limit: int) -> list[tuple[PreparedDoc, float]]:
    ranked = sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=True)
    return [(corpus.doc_by_id[doc_id], score) for doc_id, score in ranked[:limit]]


def compact_text_snippet(text: str, limit: int = 240) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(compact) <= limit:
        return compact
    return compact[: max(limit - 3, 0)] + "..."


def kg2rag_doc_signature(doc: PreparedDoc) -> set[str]:
    tokens = {norm_text(name) for name in getattr(doc, "entity_names", []) if norm_text(name)}
    tokens.update(disease_term_set(*getattr(doc, "disease_tags", [])))
    tokens.update({norm_text(symbol) for symbol in (getattr(doc, "_gene_symbols", set()) or set()) if norm_text(symbol)})
    if not tokens:
        tokens.update({token for token in tokenize(getattr(doc, "title", "") or "") if token})
    return {token for token in tokens if token}


def kg2rag_jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def build_kg2rag_summary_doc(
    record: dict[str, Any],
    organized_docs: list[tuple[PreparedDoc, float]],
    component_summaries: list[str],
) -> PreparedDoc:
    disease_tags = unique_preserve([str(record.get("oncology_focus") or "")] + [tag for doc, _score in organized_docs for tag in doc.disease_tags])[:16]
    entity_names = unique_preserve(collect_record_gene_symbols(record) + [name for doc, _score in organized_docs for name in doc.entity_names])[:20]
    evidence_lines = [
        f"- evidence_{idx}: {compact_text_snippet(doc.title, 120)} | score={score:.3f}"
        for idx, (doc, score) in enumerate(organized_docs[:4], start=1)
    ]
    text = "\n".join(
        [
            "KG2RAG organization summary",
            f"Question focus: {record.get('question')}",
            "Organized graph components:",
            *(component_summaries[:4] or ["- no component organization available"]),
            "Ordered evidence passed to answering:",
            *(evidence_lines or ["- no evidence selected"]),
        ]
    )
    return PreparedDoc(
        doc_id=f"kg2rag-summary::{record.get('id')}",
        knowledge_source="kg2rag",
        document_level="virtual",
        parent_id=None,
        title="KG2RAG graph-guided organization summary",
        text=text,
        disease_tags=disease_tags,
        entity_names=entity_names,
        entity_ids=[],
        metadata={"_kg2rag_summary_kind": "organization_summary"},
    )


def build_raptor_parent_summary_doc(
    record: dict[str, Any],
    parent_doc: PreparedDoc,
    child_docs: list[tuple[PreparedDoc, float]],
    rank: int,
) -> PreparedDoc:
    child_lines = [
        f"- child_{idx}: {compact_text_snippet(doc.title, 110)} | score={score:.3f} | scope_rank={doc_scope_state(doc, record)['strict_scope_rank']}"
        for idx, (doc, score) in enumerate(child_docs[:2], start=1)
    ]
    summary_lines = [
        f"RAPTOR parent summary {rank}",
        f"Parent: {parent_doc.title}",
        f"Parent gist: {compact_text_snippet(parent_doc.text, 220)}",
        "Representative child evidence:",
        *(child_lines or ["- no child evidence selected"]),
    ]
    disease_tags = unique_preserve(parent_doc.disease_tags + [tag for doc, _score in child_docs for tag in doc.disease_tags])
    entity_names = unique_preserve(parent_doc.entity_names + [name for doc, _score in child_docs for name in doc.entity_names])[:16]
    return PreparedDoc(
        doc_id=f"raptor-parent-summary::{record.get('id')}::{parent_doc.doc_id}",
        knowledge_source="raptor_rag",
        document_level="virtual",
        parent_id=None,
        title=f"RAPTOR parent summary: {parent_doc.title}",
        text="\n".join(summary_lines),
        disease_tags=disease_tags,
        entity_names=entity_names,
        entity_ids=[],
        metadata={
            "_raptor_summary_kind": "parent_summary",
            "source_parent_id": parent_doc.doc_id,
        },
    )


def build_raptor_root_summary_doc(
    record: dict[str, Any],
    parent_summaries: list[tuple[PreparedDoc, float]],
    selected_real_docs: list[tuple[PreparedDoc, float]],
) -> PreparedDoc:
    branch_lines = []
    for idx, (summary_doc, score) in enumerate(parent_summaries[:3], start=1):
        branch_lines.append(f"- branch_{idx}: score={score:.3f} | {compact_text_snippet(summary_doc.text, 170)}")
    evidence_lines = [
        f"- evidence_{idx}: {compact_text_snippet(doc.title, 110)} | score={score:.3f}"
        for idx, (doc, score) in enumerate(selected_real_docs[:3], start=1)
    ]
    disease_tags = unique_preserve([str(record.get("oncology_focus") or "")] + [tag for doc, _score in selected_real_docs for tag in doc.disease_tags])
    entity_names = unique_preserve(collect_record_gene_symbols(record) + [name for doc, _score in selected_real_docs for name in doc.entity_names])[:18]
    text = "\n".join(
        [
            "RAPTOR root summary",
            f"Question focus: {record.get('question')}",
            "High-level branches:",
            *(branch_lines or ["- no branch summaries"]),
            "Leaf evidence selected for answering:",
            *(evidence_lines or ["- no leaf evidence"]),
        ]
    )
    return PreparedDoc(
        doc_id=f"raptor-root-summary::{record.get('id')}",
        knowledge_source="raptor_rag",
        document_level="virtual",
        parent_id=None,
        title="RAPTOR hierarchical root summary",
        text=text,
        disease_tags=disease_tags,
        entity_names=entity_names,
        entity_ids=[],
        metadata={
            "_raptor_summary_kind": "root_summary",
        },
    )


def crag_doc_quality(doc: PreparedDoc, record: dict[str, Any], base_score: float = 0.0, *, fallback_pass: bool = False) -> dict[str, Any]:
    task_type = str(record.get("task_type") or "")
    scope_state = doc_scope_state(doc, record)
    disease_match = bool(scope_state["disease_match"])
    gene_match = bool(scope_state["gene_match"])
    strict_scope_rank = int(scope_state["strict_scope_rank"] or 0)
    node_type = getattr(doc, "_node_type", "") or doc.document_level or ""
    support_flags = getattr(doc, "_support_flags", set()) or doc_support_flags(doc)
    quality_score = metadata_overlap(doc, record) + scope_state["scope_score"] + min(base_score, 12.0) * 0.05
    reasons: list[str] = []

    if disease_match:
        quality_score += 1.0
        reasons.append("disease-matched")
    if gene_match:
        quality_score += 0.9
        reasons.append("gene-matched")
    if disease_match and gene_match:
        quality_score += 0.8
    if strict_scope_rank >= 2:
        quality_score += 0.7
        reasons.append("scope-locked")
    if getattr(doc, "_is_evidence_like", False):
        quality_score += 0.35
    if node_type == "observation" and disease_match:
        quality_score += 0.2
    if task_type in {"causal_relation_effect", "cross_source_support_bundle", "pathway_class_membership"} and support_flags.intersection({"directional", "pathway"}):
        quality_score += 0.45
        reasons.append("relation-bearing")
    if task_type in {"target_disease_association_strength", "intogen_support_strength", "cgc_support_strength", "driver_role_assignment"} and support_flags.intersection({"dependency", "driver", "recurrent"}):
        quality_score += 0.4
        reasons.append("strong-support-signal")
    if gene_match and not disease_match:
        quality_score -= 1.2
    if node_type in {"mechanism_role_summary", "target_support_summary"} and not disease_match:
        quality_score -= 1.4
        reasons.append("cross-disease-summary")
    if fallback_pass and disease_match:
        quality_score += 0.25

    if disease_match and (strict_scope_rank >= 2 or quality_score >= 4.2):
        label = "correct"
        quality_bonus = 1.0
    elif disease_match or gene_match or strict_scope_rank >= 1 or quality_score >= 2.2:
        label = "ambiguous"
        quality_bonus = 0.2
    else:
        label = "incorrect"
        quality_bonus = -0.9
    return {
        "label": label,
        "quality_score": quality_score,
        "quality_bonus": quality_bonus,
        "scope_state": scope_state,
        "reasons": reasons,
    }


def crag_retrieval_state(
    corpus: PreparedCorpus,
    record: dict[str, Any],
    ranked: list[tuple[PreparedDoc, float]],
    evaluations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    top_docs = [doc for doc, _score in ranked[:10]]
    correct = ambiguous = incorrect = 0
    disease_locked = 0
    source_families = set()
    for doc in top_docs:
        evaluation = evaluations.get(doc.doc_id) or crag_doc_quality(doc, record)
        label = evaluation["label"]
        if label == "correct":
            correct += 1
        elif label == "ambiguous":
            ambiguous += 1
        else:
            incorrect += 1
        if int((evaluation.get("scope_state") or {}).get("strict_scope_rank") or 0) >= 2:
            disease_locked += 1
        if label != "incorrect" and doc_matches_disease(doc, record):
            source_families.add(str(doc_metadata_field(doc, "source_family") or doc.knowledge_source or "unknown"))
    if correct >= 3 and disease_locked >= 2:
        decision = "accept"
        reason = "initial retrieval already has enough scope-locked support"
    elif correct >= 1 or ambiguous >= 2:
        decision = "refine"
        reason = "initial retrieval has partial support but needs corrective filtering"
    else:
        decision = "rewrite_and_retrieve"
        reason = "initial retrieval looks noisy or off-scope"
    return {
        "decision": decision,
        "reason": reason,
        "correct_count": correct,
        "ambiguous_count": ambiguous,
        "incorrect_count": incorrect,
        "scope_locked_count": disease_locked,
        "source_diversity": len(source_families),
    }


def build_crag_fallback_query(
    corpus: PreparedCorpus,
    record: dict[str, Any],
    ranked: list[tuple[PreparedDoc, float]],
    state: dict[str, Any],
) -> str:
    entities = record.get("entities") or {}
    metadata = record.get("metadata") or {}
    task_type = str(record.get("task_type") or "")
    top_entity_names: list[str] = []
    for doc, _score in ranked[:4]:
        if not doc_matches_disease(doc, record):
            continue
        top_entity_names.extend(doc.entity_names[:4])
    query_parts = [
        record.get("question"),
        record.get("oncology_focus"),
        entities.get("disease"),
        entities.get("disease_name"),
        " ".join(collect_record_gene_symbols(record)),
        entities.get("target_gene_symbol"),
        entities.get("molecular_profile"),
        entities.get("pathway_name"),
        " ".join(entities.get("therapies") or []),
        metadata.get("oncotree_code"),
        metadata.get("oncotree_subtype"),
        " ".join(unique_preserve(top_entity_names)[:8]),
    ]
    if task_type in {"causal_relation_effect", "cross_source_support_bundle", "pathway_class_membership"}:
        query_parts.append("causal mechanism pathway target association")
    elif task_type in {"cross_source_corroboration_status", "target_disease_association_strength", "intogen_support_strength", "cgc_support_strength"}:
        query_parts.append("target disease association support evidence")
    elif task_type == "hierarchical_evidence_aggregation":
        query_parts.append("tier level evidence clinical significance response resistance sensitivity")
    if state.get("decision") == "rewrite_and_retrieve":
        query_parts.append("disease matched evidence")
    return " ".join(str(part) for part in query_parts if part)


def build_crag_summary_doc(
    record: dict[str, Any],
    state: dict[str, Any],
    selected: list[tuple[PreparedDoc, float]],
    evaluations: dict[str, dict[str, Any]],
    fallback_query: str | None,
) -> PreparedDoc:
    evidence_lines = []
    for doc, score in selected[:6]:
        evaluation = evaluations.get(doc.doc_id) or {}
        evidence_lines.append(
            f"- [{evaluation.get('label', 'unknown')}] score={score:.3f} | {evidence_doc_brief(doc, record)}"
        )
    text_lines = [
        "Closed-domain CRAG retrieval diagnosis",
        f"Decision: {state.get('decision')}",
        f"Reason: {state.get('reason')}",
        (
            "Quality counts: "
            f"correct={state.get('correct_count', 0)} | "
            f"ambiguous={state.get('ambiguous_count', 0)} | "
            f"incorrect={state.get('incorrect_count', 0)} | "
            f"scope_locked={state.get('scope_locked_count', 0)} | "
            f"source_diversity={state.get('source_diversity', 0)}"
        ),
    ]
    if fallback_query:
        text_lines.append(f"Fallback query: {fallback_query}")
    text_lines.append("Refined evidence focus:")
    text_lines.extend(evidence_lines or ["- no refined evidence selected"])
    return PreparedDoc(
        doc_id=f"crag-summary::{record.get('id')}",
        knowledge_source="crag_rag",
        document_level="virtual",
        parent_id=None,
        title="Closed-domain CRAG retrieval diagnosis",
        text="\n".join(text_lines),
        disease_tags=[str(record.get("oncology_focus") or "")] if record.get("oncology_focus") else [],
        entity_names=collect_record_gene_symbols(record),
        entity_ids=[],
        metadata={
            "_crag_summary_kind": "retrieval_diagnosis",
            "crag_decision": state.get("decision"),
            "crag_reason": state.get("reason"),
            "crag_quality_counts": {
                "correct": state.get("correct_count", 0),
                "ambiguous": state.get("ambiguous_count", 0),
                "incorrect": state.get("incorrect_count", 0),
            },
        },
    )


def retrieve_crag(corpus: PreparedCorpus, record: dict[str, Any], top_k: int) -> list[tuple[PreparedDoc, float]]:
    query_text = build_query_text(record)
    base_scores = corpus.bm25_scores(query_text)
    if not base_scores:
        return []

    rerank_limit = min(max(top_k * 30, 260), 1200 if len(corpus.docs) > 100000 else 800)
    initial_candidate_ids = [
        doc_id
        for doc_id, _score in sorted(base_scores.items(), key=lambda item: (item[1], item[0]), reverse=True)[:rerank_limit]
    ]
    evaluations: dict[str, dict[str, Any]] = {}
    candidate_scores: dict[str, float] = {}
    candidate_doc_ids: set[str] = set()
    for doc_id in initial_candidate_ids:
        doc = corpus.doc_by_id[doc_id]
        evaluation = crag_doc_quality(doc, record, base_scores[doc_id])
        evaluations[doc_id] = evaluation
        candidate_scores[doc_id] = (
            base_scores[doc_id]
            + metadata_overlap(doc, record)
            + doc_scope_state(doc, record)["scope_score"]
            + evaluation["quality_bonus"]
        )
        candidate_doc_ids.add(doc_id)

    initial_ranked = top_docs_from_scores(corpus, candidate_scores, max(top_k * 2, 12))
    retrieval_state = crag_retrieval_state(corpus, record, initial_ranked, evaluations)
    fallback_query = None
    if retrieval_state["decision"] != "accept":
        fallback_query = build_crag_fallback_query(corpus, record, initial_ranked, retrieval_state)
        fallback_scores = corpus.bm25_scores(fallback_query)
        fallback_limit = min(max(top_k * 24, 220), 1000 if len(corpus.docs) > 100000 else 700)
        for doc_id, raw_score in sorted(fallback_scores.items(), key=lambda item: (item[1], item[0]), reverse=True)[:fallback_limit]:
            doc = corpus.doc_by_id[doc_id]
            evaluation = crag_doc_quality(doc, record, raw_score, fallback_pass=True)
            evaluations[doc_id] = evaluation
            corrective_boost = 0.9 if retrieval_state["decision"] == "rewrite_and_retrieve" else 0.45
            score = (
                raw_score
                + 1.15 * metadata_overlap(doc, record)
                + 1.1 * evaluation["scope_state"]["scope_score"]
                + evaluation["quality_bonus"]
                + corrective_boost
            )
            candidate_scores[doc_id] = max(candidate_scores.get(doc_id, float("-inf")), score)
            candidate_doc_ids.add(doc_id)

    label_rank = {"correct": 2, "ambiguous": 1, "incorrect": 0}
    ranked_candidates = sorted(
        (
            (
                doc_id,
                candidate_scores[doc_id],
                label_rank[(evaluations.get(doc_id) or {}).get("label", "incorrect")],
                int(((evaluations.get(doc_id) or {}).get("scope_state") or {}).get("strict_scope_rank") or 0),
                1 if doc_matches_disease(corpus.doc_by_id[doc_id], record) else 0,
            )
            for doc_id in candidate_doc_ids
        ),
        key=lambda item: (item[2], item[3], item[4], item[1], item[0]),
        reverse=True,
    )

    selected: list[tuple[PreparedDoc, float]] = []
    selected_doc_ids: set[str] = set()
    source_usage: Counter[str] = Counter()
    node_usage: Counter[str] = Counter()
    for doc_id, score, quality_rank, scope_rank, disease_rank in ranked_candidates:
        doc = corpus.doc_by_id[doc_id]
        source_family = str(doc_metadata_field(doc, "source_family") or doc.knowledge_source or "unknown")
        node_type = getattr(doc, "_node_type", "") or doc.document_level or "unknown"
        adjusted_score = score + 0.14 * quality_rank + 0.1 * scope_rank + 0.05 * disease_rank
        adjusted_score -= 0.12 * source_usage[source_family]
        adjusted_score -= 0.06 * node_usage[node_type]
        if doc_id in selected_doc_ids:
            continue
        if quality_rank == 0 and len(selected) >= max(2, top_k // 2):
            continue
        selected.append((doc, adjusted_score))
        selected_doc_ids.add(doc_id)
        source_usage[source_family] += 1
        node_usage[node_type] += 1
        if len(selected) >= top_k:
            break
    if len(selected) < top_k:
        for doc_id, score, _quality_rank, _scope_rank, _disease_rank in ranked_candidates:
            if doc_id in selected_doc_ids:
                continue
            selected.append((corpus.doc_by_id[doc_id], score))
            selected_doc_ids.add(doc_id)
            if len(selected) >= top_k:
                break
    selected.sort(key=lambda item: (item[1], item[0].doc_id), reverse=True)
    summary_doc = build_crag_summary_doc(record, retrieval_state, selected, evaluations, fallback_query)
    return [(summary_doc, (selected[0][1] if selected else 0.0) + 1.2)] + selected


def retrieve_naive(corpus: PreparedCorpus, record: dict[str, Any], top_k: int) -> list[tuple[PreparedDoc, float]]:
    return top_docs_from_scores(corpus, corpus.bm25_scores(build_query_text(record)), top_k)


def retrieve_dense(corpus: PreparedCorpus, record: dict[str, Any], top_k: int) -> list[tuple[PreparedDoc, float]]:
    import numpy as np

    index = ensure_dense_rag_index(corpus)
    query_vector = dense_rag_model().encode(
        [build_query_text(record)],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    query_vector = np.ascontiguousarray(query_vector, dtype="float32")
    scores, positions = index.search(query_vector, min(top_k, len(corpus.docs)))
    retrieved = []
    for position, score in zip(positions[0], scores[0]):
        if position < 0:
            continue
        retrieved.append((corpus.docs[int(position)], float(score)))
    return retrieved


def retrieve_hybrid(corpus: PreparedCorpus, record: dict[str, Any], top_k: int) -> list[tuple[PreparedDoc, float]]:
    base_scores = corpus.bm25_scores(build_query_text(record))
    if not base_scores:
        return []
    rerank_limit = min(max(top_k * 30, 300), 1200 if len(corpus.docs) > 100000 else 800)
    candidate_doc_ids = [
        doc_id
        for doc_id, _score in sorted(base_scores.items(), key=lambda item: (item[1], item[0]), reverse=True)[:rerank_limit]
    ]
    reranked_scores = {}
    for doc_id in candidate_doc_ids:
        reranked_scores[doc_id] = base_scores[doc_id] + metadata_overlap(corpus.doc_by_id[doc_id], record)
    return top_docs_from_scores(corpus, reranked_scores, top_k)


def retrieve_raptor(corpus: PreparedCorpus, record: dict[str, Any], top_k: int) -> list[tuple[PreparedDoc, float]]:
    parent_docs = [doc for doc in corpus.docs if doc.document_level == "parent"]
    if not parent_docs:
        return retrieve_hybrid(corpus, record, top_k)
    query_text = build_query_text(record)
    base_scores = corpus.bm25_scores(query_text)
    if not base_scores:
        return retrieve_hybrid(corpus, record, top_k)
    candidate_limit = min(max(top_k * 18, 160), 900 if len(corpus.docs) > 100000 else 500)
    candidate_doc_ids = [
        doc_id
        for doc_id, _score in sorted(base_scores.items(), key=lambda item: (item[1], item[0]), reverse=True)[:candidate_limit]
    ]
    candidate_parent_ids: set[str] = set()
    for doc_id in candidate_doc_ids:
        doc = corpus.doc_by_id[doc_id]
        if doc.document_level == "parent":
            candidate_parent_ids.add(doc_id)
        elif doc.parent_id and doc.parent_id in corpus.doc_by_id:
            candidate_parent_ids.add(doc.parent_id)
    if not candidate_parent_ids:
        return retrieve_hybrid(corpus, record, top_k)
    candidate_parent_ids = set(
        sorted(
            candidate_parent_ids,
            key=lambda parent_id: (
                base_scores.get(parent_id, 0.0) + metadata_overlap(corpus.doc_by_id[parent_id], record),
                parent_id,
            ),
            reverse=True,
        )[: min(24, len(candidate_parent_ids))]
    )
    parent_scores: defaultdict[str, float] = defaultdict(float)
    child_scores_by_parent: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for parent_id in candidate_parent_ids:
        doc = corpus.doc_by_id[parent_id]
        base_parent_score = base_scores.get(doc.doc_id, 0.0) + metadata_overlap(doc, record) + doc_scope_state(doc, record)["scope_score"]
        parent_scores[doc.doc_id] = base_parent_score
        for child_id in corpus.parent_to_children.get(doc.doc_id, []):
            child_doc = corpus.doc_by_id[child_id]
            child_score = base_scores.get(child_id, 0.0) + metadata_overlap(child_doc, record) + doc_scope_state(child_doc, record)["scope_score"]
            child_scores_by_parent[doc.doc_id].append((child_id, child_score))
        top_child_support = sorted(child_scores_by_parent[doc.doc_id], key=lambda item: (item[1], item[0]), reverse=True)[:3]
        if top_child_support:
            parent_scores[doc.doc_id] += 0.45 * top_child_support[0][1]
            if len(top_child_support) > 1:
                parent_scores[doc.doc_id] += 0.2 * sum(score for _child_id, score in top_child_support[1:])
    top_parents = sorted(parent_scores.items(), key=lambda item: (item[1], item[0]), reverse=True)[: max(2, min(4, top_k // 2 + 1))]
    parent_summary_docs: list[tuple[PreparedDoc, float]] = []
    real_candidates: list[tuple[PreparedDoc, float]] = []
    for parent_id, parent_score in top_parents:
        parent_doc = corpus.doc_by_id[parent_id]
        top_children = [
            (corpus.doc_by_id[child_id], score)
            for child_id, score in sorted(child_scores_by_parent[parent_id], key=lambda item: (item[1], item[0]), reverse=True)[:3]
        ]
        if top_children:
            real_candidates.extend(top_children)
        else:
            real_candidates.append((parent_doc, parent_score))
        if len(parent_summary_docs) < 1:
            summary_doc = build_raptor_parent_summary_doc(record, parent_doc, top_children, len(parent_summary_docs) + 1)
            parent_summary_docs.append((summary_doc, parent_score + 0.8))
    deduped = []
    seen = set()
    root_summary_budget = 1
    parent_summary_budget = len(parent_summary_docs)
    real_budget = max(top_k - root_summary_budget - parent_summary_budget, 3)
    for doc, score in sorted(real_candidates, key=lambda item: (item[1], item[0].doc_id), reverse=True):
        if doc.doc_id in seen:
            continue
        seen.add(doc.doc_id)
        deduped.append((doc, score))
        if len(deduped) >= real_budget:
            break
    root_summary_doc = build_raptor_root_summary_doc(record, parent_summary_docs, deduped)
    return [(root_summary_doc, (deduped[0][1] if deduped else 0.0) + 1.0)] + parent_summary_docs + deduped


def retrieve_kg2rag(corpus: PreparedCorpus, record: dict[str, Any], top_k: int) -> list[tuple[PreparedDoc, float]]:
    seeds = retrieve_hybrid(corpus, record, max(6, top_k // 2 + 2))
    if not seeds:
        return []

    candidate_scores = {doc.doc_id: score for doc, score in seeds}
    relation_edges: dict[tuple[str, str], dict[str, Any]] = {}
    had_graph_structure = False
    per_seed_neighbor_limit = 6
    for seed_doc, seed_score in seeds[: min(len(seeds), 8)]:
        neighbor_candidates: list[tuple[str, float, str]] = []
        for neighbor_id, relation, weight in corpus.graph_neighbors.get(seed_doc.doc_id, []):
            if relation in {"HAS_CHUNK", "NEXT_CHUNK"}:
                continue
            had_graph_structure = True
            neighbor_doc = corpus.doc_by_id[neighbor_id]
            relation_bonus = evidence_relation_priority(relation) * float(weight or 1.0)
            candidate_score = (
                seed_score * 0.52
                + relation_bonus
                + metadata_overlap(neighbor_doc, record)
                + doc_scope_state(neighbor_doc, record)["scope_score"]
            )
            if relation_aware_task(record.get("task_type")) and relation in {"SUPPORTS", "COMPLEMENTS", "CAUSAL_CHAIN", "OBSERVED_IN", "HAS_FEATURE", "FEATURE_TARGETS"}:
                candidate_score += 0.4
            neighbor_candidates.append((neighbor_id, candidate_score, relation))
            edge_key = tuple(sorted((seed_doc.doc_id, neighbor_id)))
            edge_payload = relation_edges.setdefault(edge_key, {"weight": float("-inf"), "relations": []})
            edge_payload["weight"] = max(edge_payload["weight"], relation_bonus)
            if relation not in edge_payload["relations"]:
                edge_payload["relations"].append(relation)
        for neighbor_id, candidate_score, _relation in sorted(neighbor_candidates, key=lambda item: (item[1], item[0]), reverse=True)[:per_seed_neighbor_limit]:
            candidate_scores[neighbor_id] = max(candidate_scores.get(neighbor_id, float("-inf")), candidate_score)

    if not had_graph_structure:
        return retrieve_hybrid(corpus, record, top_k)

    candidate_limit = min(max(top_k * 6, 28), 72)
    candidate_ids = [
        doc_id
        for doc_id, _score in sorted(candidate_scores.items(), key=lambda item: (item[1], item[0]), reverse=True)[:candidate_limit]
    ]
    candidate_set = set(candidate_ids)
    signatures = {doc_id: kg2rag_doc_signature(corpus.doc_by_id[doc_id]) for doc_id in candidate_ids}

    for left_id in candidate_ids:
        for neighbor_id, relation, weight in corpus.graph_neighbors.get(left_id, []):
            if relation in {"HAS_CHUNK", "NEXT_CHUNK"} or neighbor_id not in candidate_set or left_id == neighbor_id:
                continue
            edge_key = tuple(sorted((left_id, neighbor_id)))
            edge_payload = relation_edges.setdefault(edge_key, {"weight": float("-inf"), "relations": []})
            edge_payload["weight"] = max(edge_payload["weight"], evidence_relation_priority(relation) * float(weight or 1.0))
            if relation not in edge_payload["relations"]:
                edge_payload["relations"].append(relation)

    weighted_edges: list[tuple[float, str, str, list[str], float]] = []
    for index, left_id in enumerate(candidate_ids):
        left_sig = signatures[left_id]
        for right_id in candidate_ids[index + 1 :]:
            edge_key = tuple(sorted((left_id, right_id)))
            relation_info = relation_edges.get(edge_key)
            relation_weight = 0.0
            relation_labels: list[str] = []
            if relation_info:
                relation_weight = max(float(relation_info.get("weight") or 0.0), 0.0)
                relation_labels = relation_info.get("relations") or []
            overlap = kg2rag_jaccard(left_sig, signatures[right_id])
            if relation_weight <= 0 and overlap < 0.16:
                continue
            weighted_edges.append((relation_weight + overlap, left_id, right_id, relation_labels, overlap))

    if not weighted_edges:
        return top_docs_from_scores(corpus, candidate_scores, top_k)

    parent: dict[str, str] = {doc_id: doc_id for doc_id in candidate_ids}

    def find(doc_id: str) -> str:
        while parent[doc_id] != doc_id:
            parent[doc_id] = parent[parent[doc_id]]
            doc_id = parent[doc_id]
        return doc_id

    def union(left_id: str, right_id: str) -> bool:
        left_root = find(left_id)
        right_root = find(right_id)
        if left_root == right_root:
            return False
        parent[right_root] = left_root
        return True

    mst_neighbors: dict[str, list[tuple[str, float, list[str], float]]] = defaultdict(list)
    mst_edges: list[tuple[str, str, float, list[str], float]] = []
    for total_weight, left_id, right_id, relation_labels, overlap in sorted(weighted_edges, key=lambda item: (item[0], item[1], item[2]), reverse=True):
        if not union(left_id, right_id):
            continue
        mst_neighbors[left_id].append((right_id, total_weight, relation_labels, overlap))
        mst_neighbors[right_id].append((left_id, total_weight, relation_labels, overlap))
        mst_edges.append((left_id, right_id, total_weight, relation_labels, overlap))

    to_drop: set[str] = set()
    redundancy_threshold = 0.62
    for left_id, right_id, _total_weight, _relation_labels, overlap in mst_edges:
        if overlap < redundancy_threshold or left_id in to_drop or right_id in to_drop:
            continue
        loser = left_id if candidate_scores.get(left_id, float("-inf")) < candidate_scores.get(right_id, float("-inf")) else right_id
        to_drop.add(loser)

    survivors = [doc_id for doc_id in candidate_ids if doc_id not in to_drop]
    if not survivors:
        survivors = candidate_ids[:top_k]
    survivor_set = set(survivors)

    component_map: dict[str, list[str]] = defaultdict(list)
    for doc_id in survivors:
        component_map[find(doc_id)].append(doc_id)

    ordered_ids: list[str] = []
    component_summaries: list[str] = []
    component_specs: list[tuple[int, float, list[str]]] = []
    for component_ids in component_map.values():
        component_ids = sorted(component_ids, key=lambda doc_id: (candidate_scores.get(doc_id, 0.0), doc_id), reverse=True)
        size = len(component_ids)
        max_score = candidate_scores.get(component_ids[0], 0.0) if component_ids else 0.0
        component_specs.append((size, max_score, component_ids))

    component_specs.sort(key=lambda item: (-item[0], -item[1], item[2][0]))
    for component_index, (_size, _max_score, component_ids) in enumerate(component_specs, start=1):
        queue = [component_ids[0]]
        visited = {component_ids[0]}
        ordered_component: list[str] = []
        relation_types: list[str] = []
        while queue:
            current = queue.pop(0)
            ordered_component.append(current)
            neighbors = [
                (neighbor_id, weight, relation_labels)
                for neighbor_id, weight, relation_labels, _overlap in mst_neighbors.get(current, [])
                if neighbor_id in survivor_set and neighbor_id not in visited
            ]
            neighbors.sort(key=lambda item: (item[1], candidate_scores.get(item[0], 0.0), item[0]), reverse=True)
            for neighbor_id, _weight, relation_labels in neighbors:
                visited.add(neighbor_id)
                queue.append(neighbor_id)
                relation_types.extend(relation_labels)
        for doc_id in component_ids:
            if doc_id not in visited:
                ordered_component.append(doc_id)
        ordered_ids.extend(ordered_component)
        head_titles = [compact_text_snippet(corpus.doc_by_id[doc_id].title, 70) for doc_id in ordered_component[:3]]
        component_summaries.append(
            f"- component_{component_index}: size={len(ordered_component)} | relations={', '.join(unique_preserve(relation_types)[:6]) or 'weak_overlap'} | nodes={'; '.join(head_titles)}"
        )

    ordered_docs = [(corpus.doc_by_id[doc_id], candidate_scores.get(doc_id, 0.0)) for doc_id in ordered_ids if doc_id in corpus.doc_by_id]
    summary_doc = build_kg2rag_summary_doc(record, ordered_docs, component_summaries)
    real_budget = max(top_k - 1, 3)
    deduped_docs: list[tuple[PreparedDoc, float]] = []
    seen_doc_ids: set[str] = set()
    for doc, score in ordered_docs:
        if doc.doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(doc.doc_id)
        deduped_docs.append((doc, score))
        if len(deduped_docs) >= real_budget:
            break
    return [(summary_doc, (deduped_docs[0][1] if deduped_docs else 0.0) + 0.9)] + deduped_docs


def retrieve_graph(corpus: PreparedCorpus, record: dict[str, Any], top_k: int) -> list[tuple[PreparedDoc, float]]:
    task_type = record.get("task_type") or ""
    seeds = retrieve_hybrid(corpus, record, max(8 if relation_aware_task(task_type) else 5, top_k // 2))
    if is_pure_rag_profile():
        scores = {doc.doc_id: score for doc, score in seeds}
        for seed_doc, seed_score in seeds:
            for neighbor_id, relation, weight in corpus.graph_neighbors.get(seed_doc.doc_id, []):
                if relation in {"HAS_CHUNK", "NEXT_CHUNK"}:
                    continue
                candidate_score = seed_score * 0.55 + float(weight or 1.0) + metadata_overlap(corpus.doc_by_id[neighbor_id], record)
                scores[neighbor_id] = max(scores.get(neighbor_id, 0.0), candidate_score)
        return top_docs_from_scores(corpus, scores, top_k)
    if not retrieval_stage_full():
        return retrieve_hybrid(corpus, record, top_k)
    if not graph_native_retrieval_task(task_type):
        return retrieve_hybrid(corpus, record, top_k)
    scores = {doc.doc_id: score for doc, score in seeds}
    path_clues: dict[str, list[str]] = defaultdict(list)
    causal_snippets: list[str] = []
    per_seed_neighbor_limit = 4 if relation_aware_task(task_type) else 3
    source_symbol, target_symbol = causal_query_pair(record)
    target_doc_ids = set(lookup_gene_symbol_doc_ids(corpus, target_symbol)) if target_symbol else set()
    source_doc_ids = set(lookup_gene_symbol_doc_ids(corpus, source_symbol)) if source_symbol else set()
    for seed_doc, seed_score in seeds:
        neighbor_candidates: list[tuple[str, str, float, dict[str, Any], float]] = []
        for neighbor_id, relation, weight in corpus.graph_neighbors.get(seed_doc.doc_id, []):
            bonus = causal_relation_weight(relation)
            neighbor_doc = corpus.doc_by_id[neighbor_id]
            scope_state = doc_scope_state(neighbor_doc, record)
            node_type = getattr(neighbor_doc, "_node_type", "") or neighbor_doc.document_level or ""
            candidate_score = (
                seed_score * 0.55
                + weight * bonus
                + metadata_overlap(neighbor_doc, record)
                + 0.9 * scope_state["scope_score"]
            )
            if relation_aware_task(task_type) and relation in {"CROSS_KG_ENTITY_EQUIVALENT", "CAUSAL_CHAIN", "SUPPORTS", "COMPLEMENTS"}:
                candidate_score += 0.6
            if scope_state["disease_match"] and scope_state["gene_match"]:
                candidate_score += 0.55
            elif scope_state["gene_match"] and not scope_state["disease_match"]:
                candidate_score -= 0.9
            elif not scope_state["disease_match"]:
                candidate_score -= 1.1
            if getattr(neighbor_doc, "_is_entity_like", False) and not getattr(neighbor_doc, "_has_disease_tags", False):
                candidate_score -= 1.0
            if node_type in {"mechanism_role_summary", "target_support_summary"}:
                candidate_score += 0.8 if scope_state["disease_match"] and scope_state["gene_match"] else -2.2
            neighbor_candidates.append((neighbor_id, relation, candidate_score, scope_state, float(weight or 0.0)))

        accepted_neighbors: list[tuple[str, str, float]] = []
        for neighbor_id, relation, candidate_score, scope_state, weight in sorted(
            neighbor_candidates,
            key=lambda item: (
                item[3]["strict_scope_rank"],
                item[2],
                item[4],
                item[0],
            ),
            reverse=True,
        ):
            if len(accepted_neighbors) >= per_seed_neighbor_limit:
                break
            if scope_state["strict_scope_rank"] == 0 and candidate_score < seed_score + 0.35:
                continue
            if not scope_state["disease_match"] and not scope_state["gene_match"] and candidate_score < seed_score + 0.8:
                continue
            scores[neighbor_id] = max(scores.get(neighbor_id, 0.0), candidate_score)
            accepted_neighbors.append((neighbor_id, relation, candidate_score))

        for neighbor_id, relation, candidate_score in accepted_neighbors:
            if task_type == "causal_relation_effect":
                seed_symbols = getattr(seed_doc, "_gene_symbols", set())
                neighbor_symbols = getattr(corpus.doc_by_id[neighbor_id], "_gene_symbols", set())
                if seed_symbols.intersection({source_symbol, target_symbol}) and neighbor_symbols.intersection({source_symbol, target_symbol}):
                    path_clues[neighbor_id].append(f"{seed_doc.title} --{relation}--> {corpus.doc_by_id[neighbor_id].title}")
                if relation == "CROSS_KG_ENTITY_EQUIVALENT" and seed_symbols.intersection({source_symbol, target_symbol}):
                    for second_neighbor_id, second_relation, second_weight in corpus.graph_neighbors.get(neighbor_id, []):
                        second_doc = corpus.doc_by_id[second_neighbor_id]
                        second_scope_state = doc_scope_state(second_doc, record)
                        second_symbols = getattr(corpus.doc_by_id[second_neighbor_id], "_gene_symbols", set())
                        if seed_symbols.intersection({source_symbol}) and second_neighbor_id in target_doc_ids:
                            second_score = candidate_score + second_weight * causal_relation_weight(second_relation) + 2.2 + 0.7 * second_scope_state["scope_score"]
                            scores[second_neighbor_id] = max(scores.get(second_neighbor_id, 0.0), second_score)
                            path_clues[second_neighbor_id].append(
                                f"{seed_doc.title} --{relation}--> {corpus.doc_by_id[neighbor_id].title} --{second_relation}--> {corpus.doc_by_id[second_neighbor_id].title}"
                            )
                        elif seed_symbols.intersection({target_symbol}) and second_neighbor_id in source_doc_ids:
                            second_score = candidate_score + second_weight * causal_relation_weight(second_relation) + 2.2 + 0.7 * second_scope_state["scope_score"]
                            scores[second_neighbor_id] = max(scores.get(second_neighbor_id, 0.0), second_score)
                            path_clues[second_neighbor_id].append(
                                f"{seed_doc.title} --{relation}--> {corpus.doc_by_id[neighbor_id].title} --{second_relation}--> {corpus.doc_by_id[second_neighbor_id].title}"
                            )
    if task_type == "causal_relation_effect":
        pair_relations = direct_tarkg_pair_relations(corpus, source_symbol, target_symbol)
        if pair_relations:
            relation_priority = {
                "activates": 4,
                "inhibits": 4,
                "affects expression": 3,
                "increases expression": 3,
                "decreases expression": 3,
                "positive correlate": 2,
                "negative correlate": 2,
                "regulates": 2,
                "binds": 1,
                "ppi": 1,
                "physical association": 1,
                "associated with": 0,
            }
            sorted_relations = sorted(
                pair_relations,
                key=lambda item: (
                    relation_priority.get(item[2], 0),
                    item[0].title,
                    item[1].title,
                    item[2],
                ),
                reverse=True,
            )
            causal_snippets.append(
                "direct_pair: "
                + "; ".join(f"{src.title} --{relation}--> {tgt.title}" for src, tgt, relation in sorted_relations[:6])
                + f"; inferred={infer_causal_path_effect([relation for _src, _tgt, relation in sorted_relations[:3]])}"
            )
            for src_doc, tgt_doc, relation in sorted_relations[:6]:
                boost = 2.5 + causal_relation_weight(relation)
                scores[src_doc.doc_id] = max(scores.get(src_doc.doc_id, 0.0), boost)
                scores[tgt_doc.doc_id] = max(scores.get(tgt_doc.doc_id, 0.0), boost)
                path_clues[src_doc.doc_id].append(f"{src_doc.title} --{relation}--> {tgt_doc.title}")
                path_clues[tgt_doc.doc_id].append(f"{src_doc.title} --{relation}--> {tgt_doc.title}")
        causal_paths = find_causal_subgraph_paths(
            corpus,
            record,
            source_doc_ids,
            target_doc_ids,
            source_symbol,
            target_symbol,
            max_hops=3,
            beam_width=16,
            max_paths=6,
        )
        for path_doc_ids, path_relations, path_score in causal_paths:
            snippet = render_causal_path_snippet(corpus, path_doc_ids, path_relations)
            causal_snippets.append(f"path_subgraph: {snippet}")
            for doc_id in path_doc_ids:
                scores[doc_id] = max(scores.get(doc_id, 0.0), path_score)
                path_clues[doc_id].append(snippet)
    seed_budget = min(max(4, top_k - 2), len(seeds), top_k)
    expanded_ranked = top_docs_from_scores(corpus, scores, max(top_k * 2, top_k))
    ranked: list[tuple[PreparedDoc, float]] = []
    seen_ranked: set[str] = set()
    for doc, score in seeds:
        if doc.doc_id in seen_ranked:
            continue
        ranked.append((doc, scores.get(doc.doc_id, score)))
        seen_ranked.add(doc.doc_id)
        if len(ranked) >= seed_budget:
            break
    for doc, score in expanded_ranked:
        if doc.doc_id in seen_ranked:
            continue
        ranked.append((doc, score))
        seen_ranked.add(doc.doc_id)
        if len(ranked) >= top_k:
            break
    ranked = ranked[:top_k]
    for idx, (doc, _score) in enumerate(ranked):
        if path_clues.get(doc.doc_id):
            doc.metadata["_graph_rag_path_relations"] = unique_preserve(path_clues[doc.doc_id])
        else:
            doc.metadata.pop("_graph_rag_path_relations", None)
        if idx == 0 and causal_snippets:
            doc.metadata["_graph_rag_causal_snippets"] = unique_preserve(causal_snippets)[:8]
        else:
            doc.metadata.pop("_graph_rag_causal_snippets", None)
    return ranked


def retrieve_evidence(corpus: PreparedCorpus, record: dict[str, Any], top_k: int) -> list[tuple[PreparedDoc, float]]:
    query_text = build_query_text(record)
    base_scores = corpus.bm25_scores(query_text)
    if not base_scores:
        return []

    rerank_limit = min(max(top_k * 40, 360), 1400 if len(corpus.docs) > 100000 else 900)
    candidate_doc_ids = [
        doc_id
        for doc_id, _score in sorted(base_scores.items(), key=lambda item: (item[1], item[0]), reverse=True)[:rerank_limit]
    ]
    seeded_scores: dict[str, float] = {}
    for doc_id in candidate_doc_ids:
        doc = corpus.doc_by_id[doc_id]
        seeded_scores[doc_id] = evidence_candidate_score(corpus, record, doc, base_scores[doc_id])

    seed_docs = top_docs_from_scores(corpus, seeded_scores, max(top_k * 2, 12))
    expanded_scores = dict(seeded_scores)
    expansion_candidates = {doc.doc_id for doc, _score in seed_docs}
    for seed_doc, seed_score in seed_docs:
        seed_node_type = getattr(seed_doc, "_node_type", "") or ""
        for neighbor_id, relation, weight in corpus.graph_neighbors.get(seed_doc.doc_id, []):
            if relation in {"HAS_CHUNK", "NEXT_CHUNK"}:
                continue
            neighbor_doc = corpus.doc_by_id[neighbor_id]
            relation_bonus = evidence_relation_priority(relation) * float(weight or 1.0)
            candidate_score = (
                seed_score * 0.5
                + relation_bonus
                + evidence_candidate_score(corpus, record, neighbor_doc, base_scores.get(neighbor_id, 0.0) * 0.35)
            )
            if seed_node_type in {"evidence", "observation"} and getattr(neighbor_doc, "_node_type", "") in {"evidence", "observation"}:
                candidate_score += 0.25
            expanded_scores[neighbor_id] = max(expanded_scores.get(neighbor_id, float("-inf")), candidate_score)
            expansion_candidates.add(neighbor_id)

        if seed_doc.parent_id and seed_doc.parent_id in corpus.doc_by_id:
            parent_doc = corpus.doc_by_id[seed_doc.parent_id]
            parent_score = seed_score * 0.6 + evidence_candidate_score(corpus, record, parent_doc, base_scores.get(parent_doc.doc_id, 0.0) * 0.25)
            expanded_scores[parent_doc.doc_id] = max(expanded_scores.get(parent_doc.doc_id, float("-inf")), parent_score)
            expansion_candidates.add(parent_doc.doc_id)
        for child_id in corpus.parent_to_children.get(seed_doc.doc_id, [])[:6]:
            child_doc = corpus.doc_by_id[child_id]
            child_score = seed_score * 0.45 + evidence_candidate_score(corpus, record, child_doc, base_scores.get(child_id, 0.0) * 0.2)
            expanded_scores[child_id] = max(expanded_scores.get(child_id, float("-inf")), child_score)
            expansion_candidates.add(child_id)

    inject_bundle_summary_candidates(corpus, record, expanded_scores, expansion_candidates)
    filter_bundle_cross_disease_summaries(corpus, record, expanded_scores, expansion_candidates)

    ranked_candidates = sorted(
        (
            (
                doc_id,
                expanded_scores[doc_id] + doc_scope_state(corpus.doc_by_id[doc_id], record)["scope_score"],
                1 if doc_matches_disease(corpus.doc_by_id[doc_id], record) else 0,
                doc_scope_state(corpus.doc_by_id[doc_id], record)["strict_scope_rank"],
            )
            for doc_id in expansion_candidates
        ),
        key=lambda item: (item[3], item[2], item[1], item[0]),
        reverse=True,
    )
    selected: list[tuple[PreparedDoc, float]] = []
    selected_doc_ids: set[str] = set()
    source_usage: Counter[str] = Counter()
    node_usage: Counter[str] = Counter()
    for doc_id, score, _disease_match_rank, scope_rank in ranked_candidates:
        doc = corpus.doc_by_id[doc_id]
        source_family = str(doc_metadata_field(doc, "source_family") or doc.knowledge_source or "unknown")
        node_type = getattr(doc, "_node_type", "") or doc.document_level or "unknown"
        adjusted_score = score
        adjusted_score += 0.12 * scope_rank
        adjusted_score -= 0.12 * source_usage[source_family]
        adjusted_score -= 0.08 * node_usage[node_type]
        if doc_id in selected_doc_ids:
            continue
        selected.append((doc, adjusted_score))
        selected_doc_ids.add(doc_id)
        source_usage[source_family] += 1
        node_usage[node_type] += 1
        if len(selected) >= top_k:
            break

    selected.sort(key=lambda item: (item[1], item[0].doc_id), reverse=True)
    evidence_pack_doc = build_evidence_pack_doc(corpus, record, selected)
    if evidence_pack_doc is not None:
        return [(evidence_pack_doc, max(score for _doc, score in selected) + 1.5)] + selected
    return selected


RETRIEVERS = {
    "naive_rag": retrieve_naive,
    "dense_rag": retrieve_dense,
    "hybrid_rag": retrieve_hybrid,
    "raptor_rag": retrieve_raptor,
    "kg2rag": retrieve_kg2rag,
    "graph_rag": retrieve_graph,
    "crag_rag": retrieve_crag,
    "evidence_rag": retrieve_evidence,
}


def render_context(record: dict[str, Any], retrieved: list[tuple[PreparedDoc, float]]) -> str:
    blocks = []
    causal_snippets = []
    crag_summaries = []
    raptor_summaries = []
    kg2rag_summaries = []
    evidence_rag_packs = []
    task_type = record.get("task_type") or ""
    if is_pure_rag_profile() or not context_stage_structured():
        for idx, (doc, score) in enumerate(retrieved, start=1):
            relation_hits = (doc.metadata or {}).get("relation_hits") or []
            blocks.append(
                f"[{idx}] source={doc.knowledge_source} level={doc.document_level} score={score:.4f}\n"
                f"title: {doc.title}\n"
                f"diseases: {', '.join(doc.disease_tags)}\n"
                f"entities: {', '.join(doc.entity_names[:12])}\n"
                f"relation_hits: {', '.join(relation_hits[:12])}\n"
                f"text: {doc.text[:2000]}"
            )
        return "\n\n".join(blocks)
    for doc, _score in retrieved:
        if (doc.metadata or {}).get("_crag_summary_kind"):
            crag_summaries.append(str(getattr(doc, "text", "") or ""))
        if (doc.metadata or {}).get("_raptor_summary_kind"):
            raptor_summaries.append(str(getattr(doc, "text", "") or ""))
        if (doc.metadata or {}).get("_kg2rag_summary_kind"):
            kg2rag_summaries.append(str(getattr(doc, "text", "") or ""))
        if (doc.metadata or {}).get("_evidence_rag_summary_kind"):
            evidence_rag_packs.append(str(getattr(doc, "text", "") or ""))
        causal_snippets.extend((doc.metadata or {}).get("_graph_rag_causal_snippets") or [])
    causal_snippets = unique_preserve(causal_snippets)
    if crag_summaries:
        blocks.append("retrieval_diagnostic_summary:\n" + "\n\n".join(crag_summaries[:1]))
    if raptor_summaries:
        blocks.append("hierarchical_summary:\n" + "\n\n".join(raptor_summaries[:3]))
    if kg2rag_summaries:
        blocks.append("kg_guided_organization_summary:\n" + "\n\n".join(kg2rag_summaries[:1]))
    if evidence_rag_packs:
        blocks.append("evidence_pack:\n" + "\n\n".join(evidence_rag_packs[:2]))
    if causal_snippets:
        blocks.append("causal_subgraph_summary:\n" + "\n".join(f"- {snippet}" for snippet in causal_snippets[:8]))
    if task_type in {"target_disease_association_strength", "intogen_support_strength", "cgc_support_strength"}:
        profile = support_strength_profile(record, retrieved)
        provisional_hint = "low" if profile["primary"] == 0 else "moderate_or_higher"
        blocks.append(
            "support_strength_summary:\n"
            f"- strong_support_clues={profile['strong']}\n"
            f"- primary_support_clues={profile['primary']}\n"
            f"- secondary_omics_clues={profile['secondary']}\n"
            f"- weak_proxy_clues={profile['weak_proxy']}\n"
            f"- cross_disease_gene_clues={profile['cross_disease']}\n"
            f"- provisional_label_hint={provisional_hint}\n"
            "- conservative_rule: weak proxy clues or single-modality omics clues alone should not justify high support"
        )
    if evidence_rag_packs:
        first_summary = (retrieved[0][0].metadata or {}).get("facet_statuses") or {}
        if first_summary:
            blocks.append(
                "evidence_facet_statuses:\n"
                + "\n".join(f"- {key}={value}" for key, value in first_summary.items())
            )
        evidence_state = (retrieved[0][0].metadata or {}).get("evidence_state") or {}
        if evidence_state:
            blocks.append(
                "evidence_state_summary:\n"
                + "\n".join(
                    f"- {key}={value}"
                    for key, value in evidence_state.items()
                    if key not in {"source_counts", "aggregation_levels"}
                )
            )
    if task_type == "cross_source_support_bundle":
        component_clues = bundle_component_clues(record, retrieved)
        blocks.append(
            "bundle_component_summary:\n"
            + "\n".join(
                f"- {component}: {'yes' if clues else 'no'}"
                + (f" | clues: {' ; '.join(clues[:2])}" if clues else "")
                for component, clues in component_clues.items()
            )
        )
    display_rank = 0
    for doc, score in retrieved:
        if (doc.metadata or {}).get("_crag_summary_kind"):
            continue
        if (doc.metadata or {}).get("_raptor_summary_kind"):
            continue
        if (doc.metadata or {}).get("_evidence_rag_summary_kind"):
            continue
        display_rank += 1
        relation_hits = (doc.metadata or {}).get("relation_hits") or []
        graph_path_relations = (doc.metadata or {}).get("_graph_rag_path_relations") or []
        title = bundle_safe_title(doc) if task_type == "cross_source_support_bundle" else doc.title
        text = bundle_safe_text(doc) if task_type == "cross_source_support_bundle" else doc.text
        blocks.append(
            f"[{display_rank}] source={doc.knowledge_source} level={doc.document_level} score={score:.4f}\n"
            f"title: {title}\n"
            f"diseases: {', '.join(doc.disease_tags)}\n"
            f"entities: {', '.join(doc.entity_names[:12])}\n"
            f"relation_hits: {', '.join(relation_hits[:12])}\n"
            f"graph_path_clues: {' || '.join(graph_path_relations[:6])}\n"
            f"text: {text[:2000]}"
        )
    return "\n\n".join(blocks)


def first_evidence_pack_doc(retrieved: list[tuple[Any, float]]) -> Any | None:
    for doc, _score in retrieved:
        if ((getattr(doc, "metadata", None) or {}).get("_evidence_rag_summary_kind")) == "evidence_pack":
            return doc
    return None


def build_prompt(record: dict[str, Any], retrieved: list[tuple[PreparedDoc, float]]) -> str:
    gold = record.get("gold_standard") or {}
    anchor_hint = render_anchor_hint(record)
    allowed_labels = allowed_answer_labels(record)
    synthesis_instruction = (
        "Use only the retrieved context to answer. Prefer concise answers that match the machine-evaluable gold format when possible. "
        "When a gene or disease symbol is available in the context, prefer the exact canonical symbol. "
    )
    if not is_pure_rag_profile() and generation_stage_grounded():
        synthesis_instruction += (
            "When task-specific structured summaries such as retrieval_diagnostic_summary, hierarchical_summary, evidence_pack, bundle_component_summary, support_strength_summary, evidence_facet_statuses, evidence_state_summary, or causal_subgraph_summary are present, treat them as higher-priority evidence than flat document wording alone. "
            "Use the evidence_pack as the intermediate evidence-state representation: preserve scope locking, component statuses, aggregation levels, and consensus candidates before mapping to the final answer. "
            "When a retrieval_diagnostic_summary is present, use its corrected scope judgment and refined evidence focus to ignore off-disease or low-confidence retrieval artifacts. "
            "When a hierarchical_summary is present, use it to keep high-level document context aligned with the leaf evidence rather than treating chunks independently. "
            "For bundle or corroboration questions, require explicit facet support before answering yes or adding a component. "
            "For hierarchical aggregation questions, preserve the exact tier or verdict label supported by the evidence_state_summary; do not collapse it into a weaker generic support phrase. "
            "For target-association strength questions, weak_proxy_only evidence should stay low unless direct, knowledgebase, or dependency evidence is explicit. "
            "Favor disease-matched evidence and observation slices, then use entity or summary nodes as supporting context.\n\n"
        )
    else:
        synthesis_instruction += "\n\n"
    return (
        "You are evaluating biomedical QA with retrieved evidence.\n"
        f"Question: {record['question']}\n"
        f"Expected answer format: {gold.get('answer_format')}\n"
        f"{short_answer_instruction(record)}\n"
        + (f"Canonical entity anchors: {anchor_hint}\n" if anchor_hint else "")
        + (f"Allowed answer labels: {' | '.join(allowed_labels)}\n" if allowed_labels and gold.get('answer_format') != 'entity' else "")
        + synthesis_instruction
        + f"Retrieved context:\n{render_context(record, retrieved)}\n\nAnswer:"
    )


def evidence_state_seed(record: dict[str, Any], retrieved: list[tuple[Any, float]]) -> dict[str, Any]:
    pack_doc = first_evidence_pack_doc(retrieved)
    pack_metadata = (getattr(pack_doc, "metadata", None) or {}) if pack_doc is not None else {}
    pack_state = dict((pack_metadata.get("evidence_state") or {}))
    facet_statuses = dict((pack_metadata.get("facet_statuses") or {}))
    allowed_labels = allowed_answer_labels(record)
    label_profile = label_space_profile(record)
    aggregation_family = infer_aggregation_family(label_profile)
    real_docs = [doc for doc, _score in retrieved if not ((getattr(doc, "metadata", None) or {}).get("_evidence_rag_summary_kind"))]
    bundle_state = bundle_summary_grounded_state(record, real_docs) if label_profile.get("bundle_like") else {}
    mechanism_or_pathway = "yes" if facet_statuses.get("causal_relation") == "yes" or facet_statuses.get("pathway_context") == "yes" else "no"
    source_state = "multi_source" if int(pack_state.get("source_diversity") or 0) >= 2 else "single_source"
    target_support_level = str(facet_statuses.get("target_association") or "unknown")
    if bundle_state:
        target_support_level = str((bundle_state.get("component_state") or {}).get("target_association") or target_support_level)
        mechanism_or_pathway = (
            "yes"
            if (bundle_state.get("component_state") or {}).get("causal_relation") == "yes"
            or (bundle_state.get("component_state") or {}).get("pathway_context") == "yes"
            else "no"
        )
    if target_support_level == "strong" and mechanism_or_pathway == "yes" and source_state == "multi_source":
        corroboration_grade = "strong"
    elif target_support_level == "weak_proxy_only" and mechanism_or_pathway == "yes" and source_state == "multi_source":
        corroboration_grade = "qualified"
    elif target_support_level not in {"absent", "unknown"} and mechanism_or_pathway == "yes":
        corroboration_grade = "partial"
    else:
        corroboration_grade = "none"
    aggregation_candidates = [str(item) for item in (pack_state.get("consensus_candidates") or []) if str(item).strip()]
    preferred_aggregation_label = choose_preferred_aggregation_label(
        allowed_labels,
        aggregation_family,
        aggregation_candidates,
        str(pack_state.get("consensus_state") or ""),
    )
    aggregation_state = {
        "family": aggregation_family,
        "primary_level": pack_state.get("primary_aggregation_level") or "atomic_or_unspecified",
        "candidate_labels": aggregation_candidates,
        "consensus_state": pack_state.get("consensus_state") or "implicit_or_missing",
        "conflict_state": "conflicting_candidate_labels" if len(set(aggregation_candidates)) > 1 else "none",
        "supported_levels": pack_state.get("aggregation_levels") or {},
        "preferred_label": preferred_aggregation_label,
    }
    path_clues = []
    for doc, _score in retrieved:
        path_clues.extend(((getattr(doc, "metadata", None) or {}).get("_graph_rag_path_relations") or [])[:3])
    path_clues = unique_preserve([str(item) for item in path_clues if str(item).strip()])[:6]
    return {
        "scope_state": {
            "status": (
                "locked"
                if int(pack_state.get("scope_locked_docs") or 0) > 0
                else "partial"
                if int(pack_state.get("partial_scope_docs") or 0) > 0
                else "weak"
            ),
            "scope_locked_docs": int(pack_state.get("scope_locked_docs") or 0),
            "partial_scope_docs": int(pack_state.get("partial_scope_docs") or 0),
            "scope_conflict_docs": int(pack_state.get("scope_conflict_docs") or 0),
        },
        "component_state": {
            "target_association": str((bundle_state.get("component_state") or {}).get("target_association") or target_support_level),
            "causal_relation": str((bundle_state.get("component_state") or {}).get("causal_relation") or facet_statuses.get("causal_relation") or "unknown"),
            "pathway_context": str((bundle_state.get("component_state") or {}).get("pathway_context") or facet_statuses.get("pathway_context") or "unknown"),
            "driver_signal": facet_statuses.get("driver_signal") or "unknown",
            "cohort_or_method_evidence": facet_statuses.get("cohort_or_method_evidence") or "unknown",
        },
        "corroboration_state": {
            "source_state": source_state,
            "mechanism_or_pathway": mechanism_or_pathway,
            "cross_source_corroboration": "yes" if corroboration_grade in {"qualified", "strong"} else "no",
            "target_association_ready": "yes" if target_support_level in {"weak_proxy_only", "strong"} else "no",
            "target_support_level": target_support_level,
            "corroboration_grade": corroboration_grade,
        },
        "aggregation_state": aggregation_state,
        "verdict_state": {
            "preferred_label": preferred_aggregation_label or (aggregation_candidates[0] if len(aggregation_candidates) == 1 else ""),
            "candidate_labels": aggregation_candidates,
            "allowed_labels": allowed_labels,
        },
        "path_state": {
            "path_clue_count": len(path_clues),
            "path_clues": path_clues,
        },
        "bundle_state": bundle_state,
    }


def label_space_profile(record: dict[str, Any]) -> dict[str, Any]:
    gold = record.get("gold_standard") or {}
    answer_format = str(gold.get("answer_format") or "")
    labels = [str(label) for label in allowed_answer_labels(record) if str(label).strip()]
    normalized = [norm_text(label.replace("_", " ")) for label in labels]
    return {
        "answer_format": answer_format,
        "labels": labels,
        "bundle_like": any("+" in label for label in labels),
        "tier_like": any(label.startswith("tier_") for label in labels),
        "conflict_like": any("conflict" in item for item in normalized),
        "verdict_like": any(
            key in item
            for item in normalized
            for key in [
                "supports",
                "does not support",
                "sensitivity",
                "resistance",
                "oncogenic",
                "poor outcome",
                "better outcome",
                "uncertain significance",
                "positive",
            ]
        ),
        "entity_like": answer_format == "entity",
        "boolean_like": answer_format == "boolean",
    }


def normalized_label_alias(label: str) -> str:
    return norm_text(str(label or "").replace("_", " "))


def infer_aggregation_family(label_profile: dict[str, Any]) -> str:
    labels = [normalized_label_alias(label) for label in (label_profile.get("labels") or [])]
    if label_profile.get("tier_like"):
        return "tier"
    if label_profile.get("conflict_like"):
        return "conflict"
    if any(label.startswith("consistent ") or label.startswith("does not support") or label.startswith("supports ") for label in labels):
        return "support_pattern"
    if label_profile.get("verdict_like"):
        return "clinical_verdict"
    return "generic"


def choose_preferred_aggregation_label(
    allowed_labels: list[str],
    aggregation_family: str,
    candidate_labels: list[str],
    consensus_state: str,
) -> str:
    allowed_norm = {normalized_label_alias(label): label for label in allowed_labels}
    candidate_norm = [normalized_label_alias(label) for label in candidate_labels if normalized_label_alias(label)]
    unique_candidate_norm = unique_preserve(candidate_norm)
    consensus_norm = normalized_label_alias(consensus_state)

    if consensus_norm in allowed_norm:
        return consensus_norm
    if len(unique_candidate_norm) == 1 and unique_candidate_norm[0] in allowed_norm:
        return unique_candidate_norm[0]

    if aggregation_family == "conflict":
        if consensus_state == "multiple_candidate_labels" or len(unique_candidate_norm) > 1:
            for label in ["conflicting_verdicts", "conflicting_interpretations"]:
                normed = normalized_label_alias(label)
                if normed in allowed_norm:
                    return normed

    if aggregation_family == "clinical_verdict":
        verdict_groups = {
            "positive": {"positive", "oncogenic", "likely oncogenic"},
            "poor_outcome": {"poor outcome"},
            "better_outcome": {"better outcome"},
            "resistance": {"resistance"},
            "sensitivityresponse": {"sensitivityresponse", "supports sensitivityresponse", "consistent supports sensitivity"},
            "uncertain_significance": {"uncertain significance"},
            "oncogenic": {"oncogenic"},
            "likely_oncogenic": {"likely oncogenic"},
        }
        for target_label, aliases in verdict_groups.items():
            target_norm = normalized_label_alias(target_label)
            if target_norm not in allowed_norm:
                continue
            if any(candidate in aliases for candidate in unique_candidate_norm):
                return target_norm

    if aggregation_family == "support_pattern":
        if consensus_state == "multiple_candidate_labels" or len(unique_candidate_norm) > 1:
            normed = normalized_label_alias("conflicting_verdicts")
            if normed in allowed_norm:
                return normed
        if len(unique_candidate_norm) == 1 and unique_candidate_norm[0] in allowed_norm:
            return unique_candidate_norm[0]

    return ""


def infer_state_demand_profile(record: dict[str, Any], retrieved: list[tuple[Any, float]]) -> dict[str, Any]:
    question = norm_text(record.get("question"))
    label_profile = label_space_profile(record)
    seed = evidence_state_seed(record, retrieved)
    aggregation_candidates = seed.get("aggregation_state", {}).get("candidate_labels") or []
    bundle_state = seed.get("bundle_state") or {}
    weights = {
        "scope": 1.0,
        "corroboration": 0.0,
        "aggregation": 0.0,
        "path": 0.0,
        "verdict": 0.0,
        "component": 0.0,
    }

    if any(token in question for token in ["across", "corroborat", "spans both", "multiple resource", "cross source"]):
        weights["corroboration"] += 1.2
    if any(token in question for token in ["intermediate", "chain", "path", "via", "through"]):
        weights["path"] += 1.1
    if any(token in question for token in ["aggregate", "tier", "assertion", "evidence items", "multiple civic", "conflicting"]):
        weights["aggregation"] += 1.0
    if any(token in question for token in ["support bundle", "bundle", "component"]) or label_profile["bundle_like"]:
        weights["component"] += 1.0
    if label_profile["tier_like"] or label_profile["conflict_like"]:
        weights["aggregation"] += 1.1
    if label_profile["verdict_like"]:
        weights["verdict"] += 0.9
    if label_profile["entity_like"]:
        weights["path"] += 0.8
    if label_profile["boolean_like"]:
        weights["corroboration"] += 0.25

    if int(seed.get("path_state", {}).get("path_clue_count") or 0) > 0:
        weights["path"] += 0.3
    if int(seed.get("corroboration_state", {}).get("source_state") == "multi_source"):
        weights["corroboration"] += 0.2
    if aggregation_candidates:
        weights["aggregation"] += 0.35
        weights["verdict"] += 0.2
    if seed.get("aggregation_state", {}).get("conflict_state") == "conflicting_candidate_labels":
        weights["aggregation"] += 0.35

    sorted_axes = sorted(weights.items(), key=lambda item: (item[1], item[0]), reverse=True)
    primary_axis = sorted_axes[0][0]
    secondary_axis = sorted_axes[1][0]

    if label_profile["bundle_like"] and bundle_state.get("summary_confidence") == "high":
        routing_strategy = "state_guided"
    elif label_profile["bundle_like"]:
        routing_strategy = "direct_answer"
    elif weights["path"] >= max(weights["aggregation"], weights["corroboration"], weights["component"]) + 0.35:
        routing_strategy = "direct_answer"
    elif max(weights["aggregation"], weights["corroboration"], weights["component"]) >= 0.9:
        routing_strategy = "state_guided"
    else:
        routing_strategy = "direct_answer"

    direct_mapping_preferred = False
    if routing_strategy == "state_guided":
        if label_profile["boolean_like"] and weights["corroboration"] >= 0.9:
            direct_mapping_preferred = True
        elif label_profile["bundle_like"] and weights["component"] >= 0.9:
            direct_mapping_preferred = True
        elif label_profile["tier_like"] or label_profile["conflict_like"]:
            direct_mapping_preferred = True
        elif label_profile["verdict_like"] and str(seed.get("aggregation_state", {}).get("preferred_label") or "").strip():
            direct_mapping_preferred = True

    return {
        "weights": {key: round(value, 3) for key, value in weights.items()},
        "primary_axis": primary_axis,
        "secondary_axis": secondary_axis,
        "routing_strategy": routing_strategy,
        "direct_mapping_preferred": direct_mapping_preferred,
        "label_space_profile": label_profile,
        "bundle_summary_confidence": bundle_state.get("summary_confidence"),
    }


def build_evidence_state_prompt(record: dict[str, Any], retrieved: list[tuple[Any, float]], demand_profile: dict[str, Any]) -> str:
    state_seed = evidence_state_seed(record, retrieved)
    allowed_labels = allowed_answer_labels(record)
    return (
        "You are inferring a structured biomedical evidence state from retrieved evidence.\n"
        f"Question: {record['question']}\n"
        f"Task type: {record.get('task_type')}\n"
        f"Expected answer format: {(record.get('gold_standard') or {}).get('answer_format')}\n"
        + (f"Allowed answer labels: {' | '.join(allowed_labels)}\n" if allowed_labels else "")
        + "Infer a unified intermediate evidence state before answering. Preserve exact aggregation labels when supported, and be conservative for corroboration: require strong target-association plus mechanism-or-pathway support plus multi-source support before setting corroboration to yes.\n"
        + f"Adaptive routing profile:\n{json.dumps(demand_profile, ensure_ascii=False, indent=2)}\n"
        + "Return JSON only with this schema:\n"
        + json.dumps(
            {
                "scope_state": {"status": "locked|partial|weak", "scope_conflict": "none|present"},
                "component_state": {
                    "target_association": "strong|weak_proxy_only|absent|unknown",
                    "causal_relation": "yes|no|unknown",
                    "pathway_context": "yes|no|unknown",
                    "driver_signal": "yes|no|unknown",
                },
                "corroboration_state": {
                    "source_state": "multi_source|single_source|unknown",
                    "mechanism_or_pathway": "yes|no|unknown",
                    "target_support_level": "strong|weak_proxy_only|absent|unknown",
                    "corroboration_grade": "none|partial|qualified|strong",
                    "cross_source_corroboration": "yes|no|unknown",
                },
                "aggregation_state": {
                    "family": "tier|conflict|support_pattern|clinical_verdict|generic",
                    "primary_level": "string",
                    "conflict_state": "none|conflicting_candidate_labels|mixed",
                    "preferred_label": "string_or_empty",
                    "candidate_labels": ["label1", "label2"],
                },
                "verdict_state": {
                    "preferred_label": "string_or_empty",
                    "candidate_labels": ["label1", "label2"],
                },
            },
            ensure_ascii=False,
        )
        + "\n"
        + f"State seed:\n{json.dumps(state_seed, ensure_ascii=False, indent=2)}\n\n"
        + f"Retrieved context:\n{render_context(record, retrieved)}\n\nJSON:"
    )


def parse_json_object_response(text: str) -> dict[str, Any]:
    stripped = str(text or "").strip()
    if not stripped:
        return {}
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(stripped[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def inferred_evidence_state(record: dict[str, Any], retrieved: list[tuple[Any, float]], raw_state_text: str) -> dict[str, Any]:
    seed = evidence_state_seed(record, retrieved)
    parsed = parse_json_object_response(raw_state_text)
    merged = {
        "scope_state": {**seed.get("scope_state", {}), **(parsed.get("scope_state") or {})},
        "component_state": {**seed.get("component_state", {}), **(parsed.get("component_state") or {})},
        "corroboration_state": {**seed.get("corroboration_state", {}), **(parsed.get("corroboration_state") or {})},
        "aggregation_state": {**seed.get("aggregation_state", {}), **(parsed.get("aggregation_state") or {})},
        "verdict_state": {**seed.get("verdict_state", {}), **(parsed.get("verdict_state") or {})},
    }
    # Keep component / corroboration slots grounded in the Evidence Pack rather than
    # allowing the second-stage model to upgrade absent support into positive support.
    merged["component_state"] = {**seed.get("component_state", {}), **{key: value for key, value in (parsed.get("component_state") or {}).items() if key not in {"target_association", "causal_relation", "pathway_context", "driver_signal", "cohort_or_method_evidence"}}}
    merged["corroboration_state"] = {**seed.get("corroboration_state", {}), **{key: value for key, value in (parsed.get("corroboration_state") or {}).items() if key not in {"source_state", "mechanism_or_pathway", "cross_source_corroboration", "target_association_ready"}}}
    if str((seed.get("aggregation_state") or {}).get("family") or "").strip():
        merged["aggregation_state"]["family"] = seed["aggregation_state"]["family"]
    for section in ["aggregation_state", "verdict_state"]:
        labels = merged[section].get("candidate_labels") or []
        merged[section]["candidate_labels"] = [str(item) for item in labels if str(item).strip()]
        preferred = str(merged[section].get("preferred_label") or "").strip()
        merged[section]["preferred_label"] = preferred
    seed_preferred = str((seed.get("aggregation_state") or {}).get("preferred_label") or "").strip()
    if seed_preferred:
        merged["aggregation_state"]["preferred_label"] = seed_preferred
        merged["verdict_state"]["preferred_label"] = seed_preferred
    merged["corroboration_state"]["target_association_ready"] = (
        "yes" if merged["component_state"].get("target_association") in {"weak_proxy_only", "strong"} else "no"
    )
    merged["corroboration_state"]["mechanism_or_pathway"] = (
        "yes"
        if merged["component_state"].get("causal_relation") == "yes" or merged["component_state"].get("pathway_context") == "yes"
        else "no"
    )
    target_support_level = str(merged["component_state"].get("target_association") or "unknown")
    if (
        merged["corroboration_state"].get("source_state") == "multi_source"
        and merged["corroboration_state"]["mechanism_or_pathway"] == "yes"
        and target_support_level == "strong"
    ):
        corroboration_grade = "strong"
    elif (
        merged["corroboration_state"].get("source_state") == "multi_source"
        and merged["corroboration_state"]["mechanism_or_pathway"] == "yes"
        and target_support_level == "weak_proxy_only"
    ):
        corroboration_grade = "qualified"
    elif merged["corroboration_state"]["mechanism_or_pathway"] == "yes" and target_support_level not in {"absent", "unknown"}:
        corroboration_grade = "partial"
    else:
        corroboration_grade = "none"
    merged["corroboration_state"]["target_support_level"] = target_support_level
    merged["corroboration_state"]["corroboration_grade"] = corroboration_grade
    merged["corroboration_state"]["cross_source_corroboration"] = "yes" if corroboration_grade in {"qualified", "strong"} else "no"
    if not str(merged["aggregation_state"].get("preferred_label") or "").strip():
        preferred = choose_preferred_aggregation_label(
            allowed_answer_labels(record),
            str(merged["aggregation_state"].get("family") or infer_aggregation_family(label_space_profile(record))),
            [str(item) for item in (merged["aggregation_state"].get("candidate_labels") or [])],
            str(merged["aggregation_state"].get("consensus_state") or ""),
        )
        if preferred:
            merged["aggregation_state"]["preferred_label"] = preferred
            if not str(merged["verdict_state"].get("preferred_label") or "").strip():
                merged["verdict_state"]["preferred_label"] = preferred
    return merged


def map_answer_from_evidence_state(record: dict[str, Any], state: dict[str, Any], demand_profile: dict[str, Any]) -> str | None:
    label_profile = demand_profile.get("label_space_profile") or label_space_profile(record)
    allowed = {norm_text(label) for label in (label_profile.get("labels") or [])}
    component = state.get("component_state") or {}
    corroboration = state.get("corroboration_state") or {}
    aggregation = state.get("aggregation_state") or {}
    verdict = state.get("verdict_state") or {}

    if label_profile.get("boolean_like") and demand_profile.get("weights", {}).get("corroboration", 0.0) >= 0.9:
        if (
            corroboration.get("corroboration_grade") in {"qualified", "strong"}
            and corroboration.get("mechanism_or_pathway") == "yes"
            and corroboration.get("source_state") == "multi_source"
        ):
            return "yes"
        return "no"

    if label_profile.get("bundle_like"):
        components = []
        if component.get("causal_relation") == "yes":
            components.append("causal_relation")
        if component.get("pathway_context") == "yes":
            components.append("pathway_context")
        if component.get("target_association") in {"weak_proxy_only", "strong"}:
            components.append("target_association")
        if not components:
            return None
        candidate = norm_text("+".join(sorted(components)))
        if candidate in allowed:
            return candidate
        if component.get("target_association") in {"weak_proxy_only", "strong"}:
            if component.get("causal_relation") == "yes" and norm_text("causal_relation+target_association") in allowed:
                return norm_text("causal_relation+target_association")
            if component.get("pathway_context") == "yes" and norm_text("pathway_context+target_association") in allowed:
                return norm_text("pathway_context+target_association")
            if norm_text("target_association_only") in allowed:
                return norm_text("target_association_only")

    if label_profile.get("tier_like") or label_profile.get("conflict_like") or label_profile.get("verdict_like"):
        preferred = norm_text(aggregation.get("preferred_label") or verdict.get("preferred_label") or "")
        if preferred and preferred in allowed:
            return preferred
        if aggregation.get("conflict_state") == "conflicting_candidate_labels":
            if norm_text("conflicting_verdicts") in allowed:
                return norm_text("conflicting_verdicts")
            if norm_text("conflicting_interpretations") in allowed:
                return norm_text("conflicting_interpretations")

    return None


def build_answer_from_state_prompt(record: dict[str, Any], state: dict[str, Any], retrieved: list[tuple[Any, float]], demand_profile: dict[str, Any]) -> str:
    allowed_labels = allowed_answer_labels(record)
    return (
        "You are mapping a structured biomedical evidence state to the final benchmark answer.\n"
        f"Question: {record['question']}\n"
        f"Task type: {record.get('task_type')}\n"
        f"Expected answer format: {(record.get('gold_standard') or {}).get('answer_format')}\n"
        f"{short_answer_instruction(record)}\n"
        + (f"Allowed answer labels: {' | '.join(allowed_labels)}\n" if allowed_labels else "")
        + f"Adaptive routing profile:\n{json.dumps(demand_profile, ensure_ascii=False, indent=2)}\n"
        +         "Use the evidence state first. Preserve exact aggregation labels when supported. For corroboration, answer yes only when strong target association is present together with mechanism-or-pathway support and multi-source support.\n\n"
        + f"Inferred evidence state:\n{json.dumps(state, ensure_ascii=False, indent=2)}\n\n"
        + f"Compact context:\n{render_context(record, retrieved)}\n\nAnswer:"
    )


class OpenAICompatibleClient:
    def __init__(self, model: str | None = None) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = model or os.getenv("RAG_EVAL_MODEL", "gpt-5-mini")
        self.disable_proxy = os.getenv("OPENAI_DISABLE_PROXY", "").strip().lower() in {"1", "true", "yes", "on"}

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        session = requests.Session()
        if self.disable_proxy:
            session.trust_env = False
        request_payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a precise biomedical QA assistant. Return the shortest answer consistent with the requested answer format."},
                {"role": "user", "content": prompt},
            ],
        }
        # gpt-5-family endpoints reject non-default temperature; older models keep 0.
        if not self.model.lower().startswith("gpt-5"):
            request_payload["temperature"] = 0
        if self.model.lower().startswith("qwen"):
            request_payload.update({"enable_thinking": False, "max_tokens": 128})
        transient_statuses = {429, 500, 502, 503, 504}
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = session.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json=request_payload,
                    timeout=120,
                )
                if response.status_code in transient_statuses:
                    response.raise_for_status()
                response.raise_for_status()
                payload = response.json()
                return payload["choices"][0]["message"]["content"].strip()
            except requests.HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else None
                if status_code not in transient_statuses or attempt == 2:
                    raise
                last_error = exc
                time.sleep(2 * (attempt + 1))
            except requests.RequestException as exc:
                if attempt == 2:
                    raise
                last_error = exc
                time.sleep(2 * (attempt + 1))
        if last_error is not None:
            raise last_error
        raise RuntimeError("LLM generation failed without a response")


def normalize_prediction(prediction: str, record: dict[str, Any]) -> str:
    gold = record.get("gold_standard") or {}
    answer_format = gold.get("answer_format")
    task_type = record.get("task_type")
    normalized_prediction = norm_text(prediction)
    prediction_tokens = set(normalized_prediction.split())
    acceptable = [norm_text(item) for item in gold.get("acceptable_answers") or []]
    normalized_answer = normalize_gold_text(gold.get("normalized_answer"))
    canonical = normalize_gold_text(gold.get("canonical_answer"))
    class_label = normalize_gold_text(gold.get("class_label"))
    if task_type == "driver_multimethod_support":
        if any(term in normalized_prediction for term in ["yes", "multiple", "multi method", "multimethod"]):
            return "positive"
        if any(term in normalized_prediction for term in ["no", "single method", "single_method", "single"]):
            return "negative"
    if task_type in {"mechanism_bioasq_yesno", "mechanism_literature_reasoning"}:
        if normalized_prediction.startswith("yes") or normalized_prediction == "yes":
            return "yes"
        if normalized_prediction.startswith("no") or normalized_prediction == "no":
            return "no"
    if task_type in {"target_disease_association_strength", "intogen_support_strength", "cgc_support_strength"}:
        if any(term in normalized_prediction for term in ["high", "strong"]):
            return "high"
        if any(term in normalized_prediction for term in ["moderate", "medium", "intermediate"]):
            return "moderate"
        if any(term in normalized_prediction for term in ["low", "weak"]):
            return "low"
    if task_type == "driver_role_assignment":
        if any(term in normalized_prediction for term in ["loss of function", "loss_of_function", "lof"]):
            return norm_text("loss_of_function")
        if "activat" in normalized_prediction:
            return norm_text("activating")
        if "ambiguous" in normalized_prediction or "uncertain" in normalized_prediction:
            return norm_text("ambiguous")
    if task_type == "causal_relation_effect":
        if any(term in normalized_prediction for term in ["activate", "upregulat", "increase", "positive regulation", "positive correlate", "increases expression", "activating_or_upregulating"]):
            return norm_text("activating_or_upregulating")
        if any(term in normalized_prediction for term in ["inhibit", "downregulat", "decrease", "negative regulation", "negative correlate", "decreases expression", "inhibitory_or_downregulating"]):
            return norm_text("inhibitory_or_downregulating")
        if "unknown" in normalized_prediction:
            return norm_text("unknown")
    if task_type == "pathway_class_membership":
        pathway_aliases = {
            "rtk_mapk": ["rtk mapk", "mapk", "ras raf mek erk", "egfr mapk"],
            "pi3k_akt_mtor": ["pi3k akt mtor", "akt mtor", "pi3k"],
            "wnt_beta_catenin": ["wnt beta catenin", "beta catenin", "wnt"],
            "tgfb_emt": ["tgfb emt", "tgf beta emt", "emt", "smad"],
            "p53_cell_cycle": ["p53 cell cycle", "cell cycle", "p53"],
            "dna_damage_repair": ["dna damage repair", "dna repair", "homologous recombination"],
            "apoptosis": ["apoptosis", "apoptotic"],
            "jak_stat": ["jak stat", "jak", "stat"],
            "notch": ["notch"],
            "hypoxia_angiogenesis": ["hypoxia angiogenesis", "angiogenesis", "hypoxia", "vegf"],
        }
        for label, aliases in pathway_aliases.items():
            if label in normalized_prediction or any(alias in normalized_prediction for alias in aliases):
                return norm_text(label)
    if task_type == "cross_source_support_bundle":
        components = set()
        if "driver" in normalized_prediction:
            components.add("driver")
        if any(term in normalized_prediction for term in ["target association", "open targets", "target_association"]):
            components.add("target_association")
        if any(term in normalized_prediction for term in ["causal", "signor", "regulat", "effect"]):
            components.add("causal_relation")
        if any(term in normalized_prediction for term in ["pathway", "context", "pathway_context"]):
            components.add("pathway_context")
        if components:
            return norm_text("+".join(sorted(components)))
    if task_type == "cancer_cell_line_annotation":
        explicit_code_match = re.search(r"\b(MACR|COAD|READ|HCC)\b", prediction.upper())
        if explicit_code_match:
            return explicit_code_match.group(1).lower()
        if class_label and class_label in prediction_tokens:
            return class_label
        entity_alias_map = [
            ("macr", ["macr", "mucinous adenocarcinoma", "mucinous adenocarcinoma of the colon and rectum"]),
            ("coad", ["coad", "colon adenocarcinoma", "colorectal adenocarcinoma", "colorectal cancer", "colon cancer"]),
            ("read", ["read", "rectal adenocarcinoma", "rectal cancer"]),
            ("hcc", ["hcc", "hepatocellular carcinoma", "liver hepatocellular carcinoma", "hepatocellular cancer", "liver cancer"]),
        ]
        for canonical_label, aliases in entity_alias_map:
            if any(re.search(rf"\b{re.escape(alias)}\b", normalized_prediction) for alias in aliases):
                return canonical_label
        if normalized_prediction in {"doid 684", "doid684"} and class_label == "hcc":
            return "hcc"
    if answer_format == "boolean":
        if normalized_prediction == "yes" or normalized_prediction.startswith("yes "):
            return "positive"
        if normalized_prediction == "no" or normalized_prediction.startswith("no "):
            return "negative"
        negative_phrases = [
            "not associated",
            "not sensitive",
            "not synergistic",
            "no synergy",
            "does not support",
            "do not support",
            "lack of response",
            "no response",
            "resistant",
            "negative",
            "false",
            "not found",
            "unknown",
        ]
        positive_phrases = [
            "positive",
            "yes",
            "true",
            "associated",
            "sensitive",
            "synergistic",
            "supports",
            "response",
            "benefit",
        ]
        if any(token in normalized_prediction for token in negative_phrases):
            return "negative"
        if any(token in normalized_prediction for token in positive_phrases):
            return "positive"
    if normalized_answer and normalized_answer in normalized_prediction:
        return normalized_answer
    if canonical and canonical in normalized_prediction:
        return canonical
    if class_label and class_label in normalized_prediction:
        return class_label
    for answer in acceptable:
        if answer and answer in normalized_prediction:
            return normalized_answer or answer
    if answer_format == "categorical":
        direction = None
        if any(
            phrase in normalized_prediction
            for phrase in [
                "does not support",
                "do not support",
                "not support",
                "no support",
                "lack of response",
                "no response",
                "not sensitive",
            ]
        ):
            direction = "does not support"
        elif any(
            phrase in normalized_prediction
            for phrase in [
                "supports",
                "supported",
                "supportive",
                "predictive",
                "diagnostic",
                "prognostic",
                "oncogenic",
                "sensitive",
                "sensitivity",
                "resistance",
                "poor prognosis",
                "better survival",
                "uncertain significance",
                "predisposition",
                "positive",
            ]
        ):
            direction = "supports"
        significance_aliases = [
            ("poor outcome", ["poor outcome", "poor prognosis", "worse prognosis", "worse survival", "adverse prognosis", "increased mortality", "worse outcome"]),
            ("better outcome", ["better outcome", "favorable prognosis", "favourable prognosis", "better survival", "improved survival", "good prognosis"]),
            ("reduced sensitivity", ["reduced sensitivity", "decreased sensitivity", "lower sensitivity"]),
            ("resistance", ["resistance", "resistant", "lack of response", "no response", "non response", "refractory"]),
            ("sensitivityresponse", ["sensitivityresponse", "sensitivity response", "sensitivity", "sensitive", "responds", "response", "benefit", "responsive"]),
            ("uncertain significance", ["uncertain significance", "unknown significance", "uncertain", "unknown clinical significance"]),
            ("oncogenicity", ["oncogenicity", "oncogenic"]),
            ("predisposition", ["predisposition", "predisposing"]),
            ("positive", ["positive"]),
        ]
        significance = None
        for label, aliases in significance_aliases:
            if any(alias in normalized_prediction for alias in aliases):
                significance = label
                break
        if direction and significance:
            return f"{direction} {significance}"
        if significance and significance in acceptable:
            return significance
    return normalized_prediction


def gold_label(record: dict[str, Any]) -> str:
    gold = record.get("gold_standard") or {}
    answer_format = gold.get("answer_format")
    if answer_format == "boolean":
        return "positive" if bool(gold.get("binary_answer")) else "negative"
    if answer_format in {"entity", "categorical"}:
        return normalize_gold_text(gold.get("class_label") or gold.get("normalized_answer") or gold.get("canonical_answer"))
    return normalize_gold_text(gold.get("normalized_answer") or gold.get("class_label") or gold.get("canonical_answer"))


def macro_f1_from_labels(gold_labels: list[str], pred_labels: list[str]) -> float:
    classes = sorted(set(gold_labels) | set(pred_labels))
    if not classes:
        return 0.0
    f1_scores = []
    for cls in classes:
        tp = sum(1 for g, p in zip(gold_labels, pred_labels) if g == cls and p == cls)
        fp = sum(1 for g, p in zip(gold_labels, pred_labels) if g != cls and p == cls)
        fn = sum(1 for g, p in zip(gold_labels, pred_labels) if g == cls and p != cls)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_scores.append(f1)
    return sum(f1_scores) / len(f1_scores)


def compute_answer_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gold_labels = [gold_label(row["record"]) for row in rows]
    pred_labels = [row["normalized_prediction"] for row in rows]
    exact = sum(1 for g, p in zip(gold_labels, pred_labels) if g == p)
    by_task = defaultdict(list)
    by_family = defaultdict(list)
    by_source = defaultdict(list)
    by_function = defaultdict(list)
    for row in rows:
        by_task[row["record"].get("task_type")].append(row)
        family = (row["record"].get("metadata") or {}).get("priority_cancer_family") or row["record"].get("oncology_focus")
        by_family[family].append(row)
        by_source[row["record"].get("source")].append(row)
        qa_function = (row["record"].get("metadata") or {}).get("qa_evidence_function") or "unspecified"
        by_function[qa_function].append(row)

    def summarize(group_rows: list[dict[str, Any]]) -> dict[str, float]:
        gl = [gold_label(item["record"]) for item in group_rows]
        pl = [item["normalized_prediction"] for item in group_rows]
        return {
            "count": len(group_rows),
            "accuracy": round(sum(1 for g, p in zip(gl, pl) if g == p) / max(len(group_rows), 1), 4),
            "macro_f1": round(macro_f1_from_labels(gl, pl), 4),
        }

    return {
        "count": len(rows),
        "accuracy": round(exact / max(len(rows), 1), 4),
        "macro_f1": round(macro_f1_from_labels(gold_labels, pred_labels), 4),
        "by_task_type": {key: summarize(value) for key, value in sorted(by_task.items())},
        "by_priority_cancer_family": {key: summarize(value) for key, value in sorted(by_family.items())},
        "by_dataset_source": {key: summarize(value) for key, value in sorted(by_source.items())},
        "by_qa_evidence_function": {key: summarize(value) for key, value in sorted(by_function.items())},
    }


def compute_retrieval_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bridge_coverage = 0
    disease_match = 0
    retrieved_any = 0
    for row in rows:
        retrieved = row["retrieved_docs"]
        if retrieved:
            retrieved_any += 1
        question_disease_terms = disease_term_set((row["record"].get("entities") or {}).get("disease"), row["record"].get("oncology_focus"))
        if any(question_disease_terms.intersection({term for tag in doc["disease_tags"] for term in disease_term_set(tag)}) for doc in retrieved):
            disease_match += 1
        if any(doc["metadata"].get("relation_hits") for doc in retrieved):
            bridge_coverage += 1
    return {
        "retrieved_any_rate": round(retrieved_any / max(len(rows), 1), 4),
        "disease_match_rate": round(disease_match / max(len(rows), 1), 4),
        "relation_hit_rate": round(bridge_coverage / max(len(rows), 1), 4),
        "average_retrieved_docs": round(sum(len(row["retrieved_docs"]) for row in rows) / max(len(rows), 1), 2),
    }

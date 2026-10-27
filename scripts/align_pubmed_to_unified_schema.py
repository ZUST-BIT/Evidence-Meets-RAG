from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

try:
    import jsonschema
except ImportError:  # pragma: no cover - optional dependency
    jsonschema = None


ROOT = Path(__file__).resolve().parents[1]
from knowledge_build_paths import PUBMED_ROOT, source_relpath
SCHEMA_PATH = ROOT / "kg" / "schema" / "multisource_evidence_kg_schema_v2.json"
OUTPUT_DIR = ROOT / "kg" / "aligned"

INPUT_DISEASES = ["hcc", "crc"]
DISEASE_DIR_TO_CANONICAL = {
    "hcc": "HCC",
    "crc": "CRC",
}

STUDY_DESIGN_MAP = {
    "cohort": "cohort",
    "in_vivo": "in_vivo",
    "in_vitro": "in_vitro",
    "case_control": "case_control",
    "meta_analysis": "meta_analysis",
    "rct": "clinical_trial",
    "computational": "derived_inference",
    "unknown": "derived_inference",
}

CLINICAL_STAGE_MAP = {
    "clinical": "clinical",
    "preclinical": "preclinical",
    "phase_i": "clinical",
    "phase_ii": "clinical",
    "phase_iii": "clinical",
    "phase_iv": "clinical",
}

EDGE_RELATION_MAP = {
    "LINKED_TO": "LINKED_TO",
    "SUPPORTS": "SUPPORTS",
    "CONTRADICTS": "CONTRADICTS",
    "REFINES": "REFINES",
    "EXTENDS": "EXTENDS",
    "REPLICATES": "REPLICATES",
    "CAUSAL_CHAIN": "CAUSAL_CHAIN",
}

EVIDENCE_SLOT_MAP = {
    "SUPPORTS": "supports",
    "CONTRADICTS": "contradicts",
    "REFINES": "refines",
    "EXTENDS": "extends",
    "REPLICATES": "replicates",
    "CAUSAL_CHAIN": "causal_chain",
}

ALLOWED_EDGE_RELATIONS = {
    "LINKED_TO",
    "SUPPORTS",
    "CONTRADICTS",
    "CAUSAL_CHAIN",
    "COMPLEMENTS",
    "REDUNDANT_WITH",
    "REFINES",
    "EXTENDS",
    "REPLICATES",
    "DERIVED_FROM",
    "OBSERVED_IN",
}

ALLOWED_ENTITY_TYPES = {
    "Gene",
    "Protein",
    "Disease",
    "Drug",
    "Pathway",
    "CellType",
    "CellLine",
    "Phenotype",
    "Cohort",
    "Tissue",
    "Other",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def stable_hash(*parts: str) -> str:
    joined = "||".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16].upper()


def relpath(path: Path) -> str:
    return source_relpath(path)


def normalize_entity_type(value: str | None) -> str:
    if value in ALLOWED_ENTITY_TYPES:
        return value
    return "Other"


def normalize_entity_mention(entity: dict[str, Any], role: str | None = None) -> dict[str, Any]:
    normalized_id = coalesce(entity.get("tarkg_id"), entity.get("dbid"))
    normalized_name = coalesce(entity.get("tarkg_name"), entity.get("name"))
    return {
        "name": entity.get("name") or "unknown_entity",
        "entity_type": normalize_entity_type(entity.get("entity_type")),
        "normalized_id": normalized_id,
        "normalized_name": normalized_name,
        "db_source": entity.get("db_source"),
        "dbid": entity.get("dbid"),
        "linking_score": safe_float(entity.get("linking_score")),
        "role": role,
    }


def infer_entity_roles(core_entities: list[dict[str, Any]], target_disease: str) -> list[dict[str, Any]]:
    subject_priority = {"Gene", "Protein", "Drug", "Pathway", "Disease", "Phenotype", "Other"}
    subject_index = None
    for idx, entity in enumerate(core_entities):
        entity_type = normalize_entity_type(entity.get("entity_type"))
        if entity_type in subject_priority:
            if subject_index is None:
                subject_index = idx
            if entity_type in {"Gene", "Protein", "Drug", "Pathway"}:
                subject_index = idx
                break

    normalized: list[dict[str, Any]] = []
    object_assigned = False
    for idx, entity in enumerate(core_entities):
        entity_type = normalize_entity_type(entity.get("entity_type"))
        role: str | None
        if subject_index is not None and idx == subject_index:
            role = "subject"
        elif not object_assigned and entity_type in {"Disease", "Pathway", "Drug", "Phenotype"}:
            role = "object" if entity_type != "Phenotype" else "outcome"
            object_assigned = True
        else:
            role = "supporting_entity"
        normalized.append(normalize_entity_mention(entity, role))

    if not any(item["entity_type"] == "Disease" and item["name"] == target_disease for item in normalized):
        normalized.append(
            {
                "name": target_disease,
                "entity_type": "Disease",
                "normalized_id": target_disease,
                "normalized_name": target_disease,
                "db_source": "custom_disease_family",
                "dbid": target_disease,
                "linking_score": 1.0,
                "role": "object",
            }
        )
    return normalized


def normalize_study_design(value: str | None) -> str:
    return STUDY_DESIGN_MAP.get((value or "").lower(), "derived_inference")


def normalize_clinical_stage(value: str | None) -> str:
    return CLINICAL_STAGE_MAP.get((value or "").lower(), "unknown")


def infer_predicate_and_direction(text: str) -> tuple[str, str]:
    lowered = text.lower()
    negative_terms = [
        "contradict",
        "fails to suppress",
        "failed to suppress",
        "downregulat",
        "decrease",
        "reduce",
        "suppress",
        "inhibit",
        "blockade",
        "knockdown",
        "silenc",
        "loss of",
    ]
    positive_terms = [
        "upregulat",
        "increase",
        "promot",
        "induc",
        "activat",
        "enhanc",
        "driv",
        "accelerat",
        "stimulat",
        "associated with",
        "supports",
    ]
    if any(term in lowered for term in ["contradict", "opposing outcome"]):
        return "contradicts_mechanistic_claim", "negative"
    if any(term in lowered for term in negative_terms):
        return "suppresses_mechanism", "decrease"
    if any(term in lowered for term in positive_terms):
        return "promotes_mechanism", "increase"
    return "supports_mechanistic_claim", "unknown"


def infer_polarity(text: str) -> str | None:
    lowered = text.lower()
    if any(term in lowered for term in ["poor outcome", "worse survival", "tumor progression", "metast", "proliferat", "oncogen", "hepatocarcinogenesis", "carcinogenesis"]):
        return "pro_tumor"
    if any(term in lowered for term in ["better survival", "improved survival", "protect", "tumor suppress", "anti-tumor", "antitumor", "regression", "sensit"]):
        return "anti_tumor"
    if "resistan" in lowered and any(term in lowered for term in ["drug", "therapy", "treatment", "inhibitor", "blockade", "chemotherapy"]):
        return "resistance"
    return "uncertain"


def infer_claim_scope(entities: list[dict[str, Any]]) -> str | None:
    entity_types = {entity["entity_type"] for entity in entities}
    if "Drug" in entity_types and {"Gene", "Protein"} & entity_types:
        return "gene_drug"
    if {"Gene", "Protein"} & entity_types and "Disease" in entity_types:
        return "gene_disease"
    if "Pathway" in entity_types:
        return "pathway"
    if "Phenotype" in entity_types:
        return "clinical_outcome"
    return "mechanism_statement"


def infer_hypothesis_level(text: str, clinical_stage: str, entities: list[dict[str, Any]]) -> str | None:
    lowered = text.lower()
    entity_types = {entity["entity_type"] for entity in entities}
    if clinical_stage == "clinical":
        return "clinical"
    if "Drug" in entity_types or any(term in lowered for term in ["therapy", "therapeutic", "inhibitor", "blockade"]):
        return "therapeutic"
    if any(term in lowered for term in ["pathway", "signaling", "signal transduction", "metabolic"]):
        return "pathway"
    if any(term in lowered for term in ["migration", "invasion", "apoptosis", "proliferation", "immune", "cell cycle"]):
        return "cellular"
    return "molecular"


def infer_support_flags(entities: list[dict[str, Any]], text: str) -> tuple[bool, bool, bool]:
    entity_types = {entity["entity_type"] for entity in entities}
    lowered = text.lower()
    supports_target_discovery = bool({"Gene", "Protein", "Drug", "Pathway"} & entity_types)
    supports_mechanism_inference = True
    supports_drug_repositioning = "Drug" in entity_types or any(term in lowered for term in ["therapy", "therapeutic", "drug", "inhibitor", "blockade", "combination"])
    return supports_target_discovery, supports_mechanism_inference, supports_drug_repositioning


def build_cohort_label(node: dict[str, Any], disease_name: str) -> str:
    study_object = (node.get("pico") or {}).get("study_object")
    comparison = (node.get("pico") or {}).get("comparison")
    if study_object and comparison:
        return f"{disease_name} | {study_object} | {comparison}"
    if study_object:
        return f"{disease_name} | {study_object}"
    return disease_name


def build_cohort_id(node: dict[str, Any], disease_name: str) -> str:
    pico = node.get("pico") or {}
    return f"COHORT-{stable_hash(disease_name, str(pico.get('study_object')), str(pico.get('comparison')), str(node.get('clinical_stage')))}"


def build_statistics(node: dict[str, Any]) -> dict[str, Any]:
    stats = dict(node.get("statistics") or {})
    return {
        "p_value": safe_float(stats.get("p_value")),
        "q_value": safe_float(stats.get("q_value")),
        "fold_change": safe_float(stats.get("fold_change")),
        "confidence_interval": stats.get("confidence_interval"),
        "sample_size": int(stats["sample_size"]) if safe_float(stats.get("sample_size")) is not None else None,
        "effect_size": safe_float(stats.get("effect_size")),
        "hazard_ratio": safe_float(stats.get("hazard_ratio")),
        "mutation_frequency": safe_float(stats.get("mutation_frequency")),
        "ceres_score": safe_float(stats.get("ceres_score")),
        "codependency_coefficient": safe_float(stats.get("codependency_coefficient")),
        "synergy_score": safe_float(stats.get("synergy_score")),
        "statistical_method": stats.get("statistical_method"),
    }


def build_score(node: dict[str, Any]) -> dict[str, Any]:
    score = dict(node.get("score") or {})
    belief = safe_float(score.get("ds_belief"))
    plausibility = safe_float(score.get("ds_plausibility"))
    ds_ignorance = None
    if belief is not None and plausibility is not None:
        ds_ignorance = max(0.0, round(plausibility - belief, 6))
    return {
        "study_type_score": safe_float(score.get("study_type_score")),
        "impact_score": safe_float(score.get("impact_score")),
        "stat_significance_score": safe_float(score.get("stat_significance_score")),
        "sample_size_score": safe_float(score.get("sample_size_score")),
        "llm_confidence": safe_float(score.get("llm_confidence")),
        "composite_score": safe_float(score.get("composite_score")),
        "grade_level": score.get("grade_level"),
        "ds_belief": belief,
        "ds_plausibility": plausibility,
        "ds_ignorance": ds_ignorance,
    }


def summarize_relation_slots(evidence_relations: list[dict[str, Any]], linked_entities: list[str]) -> tuple[dict[str, list[str]], list[str], float | None]:
    slots = {
        "supports": [],
        "contradicts": [],
        "causal_chain": [],
        "complements": [],
        "redundant_with": [],
        "same_entity_context": [],
        "same_cohort_context": [],
        "refines": [],
        "extends": [],
        "replicates": [],
        "linked_entities": sorted({entity for entity in linked_entities if entity}),
    }
    relation_strings: list[str] = []
    contradiction_count = 0
    total_relation_count = 0
    for relation in evidence_relations:
        relation_type = relation.get("relation_type")
        target_id = relation.get("target_id")
        if not relation_type or not target_id:
            continue
        relation_strings.append(f"{relation_type}:{target_id}")
        slot_name = EVIDENCE_SLOT_MAP.get(relation_type)
        if slot_name:
            slots[slot_name].append(target_id)
        if relation_type == "CONTRADICTS":
            contradiction_count += 1
        total_relation_count += 1
    contradiction_risk = None
    if total_relation_count:
        contradiction_risk = contradiction_count / total_relation_count
    for key in slots:
        slots[key] = sorted(set(slots[key]))
    return slots, sorted(set(relation_strings)), contradiction_risk


def build_provenance(node: dict[str, Any], disease_dir: str, disease_name: str) -> dict[str, Any]:
    source = node.get("source") or {}
    source_file = PUBMED_ROOT / disease_dir / "evidence_nodes.json"
    doi = source.get("doi")
    source_url = f"https://doi.org/{doi}" if doi else None
    extracted_at = coalesce(node.get("updated_at"), node.get("created_at"), now_iso())
    return {
        "source_family": "pubmed_literature",
        "source_dataset": "pubmed-evidence-kg",
        "source_release": "v1_pre_alignment",
        "source_record_id": node["evidence_id"],
        "source_file": relpath(source_file),
        "source_url": source_url,
        "citation_doi": doi,
        "citation_pmid": source.get("pmid"),
        "platform": None,
        "assay": None,
        "sequencing_or_screen_id": None,
        "batch_id": None,
        "analysis_pipeline": "pubmed_schema_alignment_v2",
        "extractor_name": "pubmed-evidence-kg",
        "extractor_version": str(node.get("version")),
        "extracted_at": extracted_at,
        "study_design": normalize_study_design(node.get("study_design")),
        "clinical_stage": normalize_clinical_stage(node.get("clinical_stage")),
        "licence": None,
        "is_primary_observation": True,
        "source_priority": 1 if disease_name in {"HCC", "CRC"} else 2,
    }


def compact_entity_ref(entity: dict[str, Any] | None) -> dict[str, Any] | None:
    if not entity:
        return None
    label = coalesce(entity.get("normalized_name"), entity.get("name"))
    if not label:
        return None
    return {
        "id": coalesce(entity.get("normalized_id"), entity.get("dbid")),
        "label": label,
        "entity_type": normalize_entity_type(entity.get("entity_type")),
    }


def build_citation(node: dict[str, Any]) -> dict[str, Any]:
    source = node.get("source") or {}
    citation_count = source.get("citation_count")
    if safe_float(citation_count) is not None:
        citation_count = int(float(citation_count))
    else:
        citation_count = None
    return {
        "doi": source.get("doi"),
        "pmid": source.get("pmid"),
        "title": source.get("title"),
        "authors": list(source.get("authors") or []),
        "journal": source.get("journal"),
        "year": source.get("year"),
        "impact_factor": safe_float(source.get("impact_factor")),
        "jcr_quartile": source.get("jcr_quartile") or source.get("quartile"),
        "citation_count": citation_count,
        "pdf_path": source.get("pdf_path"),
    }


def build_comparison(node: dict[str, Any], sample_size: int | None) -> dict[str, Any]:
    pico = node.get("pico") or {}
    return {
        "sample_grouping": pico.get("comparison"),
        "comparison_type": "paper_statement",
        "comparison_label": pico.get("comparison"),
        "group_a": pico.get("intervention"),
        "group_b": pico.get("comparison"),
        "n_total": sample_size,
        "n_effective": sample_size,
        "n_group_a": None,
        "n_group_b": None,
    }


def build_support(proposition: dict[str, Any]) -> dict[str, bool]:
    return {
        "supports_target_discovery": bool(proposition.get("supports_target_discovery")),
        "supports_mechanism_inference": bool(proposition.get("supports_mechanism_inference")),
        "supports_drug_repositioning": bool(proposition.get("supports_drug_repositioning")),
    }


def build_compact_confidence(node: dict[str, Any], contradiction_risk: float | None) -> dict[str, Any]:
    score = build_score(node)
    data_completeness = estimate_data_completeness(node)
    grade = score.get("grade_level") or "C"
    if grade not in {"A", "B", "C", "D", "E"}:
        grade = "C"
    return {
        "grade": grade,
        "score": coalesce(score.get("composite_score"), score.get("llm_confidence"), 0.5),
        "study_quality_score": score.get("study_type_score"),
        "statistical_support_score": score.get("stat_significance_score"),
        "source_reliability_score": score.get("impact_score"),
        "sample_size_score": score.get("sample_size_score"),
        "extraction_confidence": safe_float(node.get("extraction_confidence")),
        "ds_belief": score.get("ds_belief"),
        "ds_plausibility": score.get("ds_plausibility"),
        "ds_ignorance": score.get("ds_ignorance"),
        "contradiction_risk": contradiction_risk,
        "data_completeness": data_completeness,
        "review_status": normalize_review_status(node.get("review_status")),
    }


def build_cohort_context(node: dict[str, Any], disease_name: str) -> dict[str, Any]:
    pico = node.get("pico") or {}
    stats = build_statistics(node)
    sample_size = stats["sample_size"]
    return {
        "disease_name": disease_name,
        "disease_family": disease_name,
        "disease_identifiers": [{"db_source": "custom_disease_family", "dbid": disease_name, "label": disease_name, "score": 1.0}],
        "cohort_id": build_cohort_id(node, disease_name),
        "cohort_label": build_cohort_label(node, disease_name),
        "cohort_type": "paper_cohort",
        "lineage": None,
        "tissue": None,
        "sample_origin": pico.get("study_object"),
        "sample_grouping": pico.get("comparison"),
        "comparison_type": "paper_statement",
        "comparison_label": pico.get("comparison"),
        "case_definition": pico.get("intervention"),
        "control_definition": pico.get("comparison"),
        "n_total": sample_size,
        "n_effective": sample_size,
        "n_cases": None,
        "n_controls": None,
        "n_samples": sample_size,
        "n_patients": sample_size if normalize_clinical_stage(node.get("clinical_stage")) == "clinical" else None,
        "n_cell_lines": None,
        "effective_unit": "paper_statement",
        "platform_summary": source_platform_summary(node),
        "batch_summary": None,
        "context_summary": node.get("experimental_context") or pico.get("study_object"),
    }


def source_platform_summary(node: dict[str, Any]) -> str | None:
    study_design = normalize_study_design(node.get("study_design"))
    if study_design == "in_vivo":
        return "animal_or_in_vivo_literature_evidence"
    if study_design == "in_vitro":
        return "cellular_or_in_vitro_literature_evidence"
    if study_design == "clinical_trial":
        return "clinical_trial_literature_evidence"
    if study_design == "meta_analysis":
        return "meta_analysis_literature_evidence"
    if study_design == "cohort":
        return "clinical_cohort_literature_evidence"
    return "literature_statement"


def build_proposition(node: dict[str, Any], core_entities: list[dict[str, Any]], disease_name: str) -> dict[str, Any]:
    text_parts = [
        disease_name,
        node.get("bio_mechanism"),
        node.get("phenotype"),
        node.get("source_text"),
        (node.get("pico") or {}).get("intervention"),
        (node.get("pico") or {}).get("comparison"),
    ]
    joined_text = " | ".join(part for part in text_parts if part)
    predicate, direction = infer_predicate_and_direction(joined_text)
    polarity = infer_polarity(joined_text)
    supports_target_discovery, supports_mechanism_inference, supports_drug_repositioning = infer_support_flags(core_entities, joined_text)

    subject_entities = [entity for entity in core_entities if entity["role"] == "subject"] or core_entities[:1]
    object_entities = [entity for entity in core_entities if entity["role"] == "object"]
    outcome_entities = [entity for entity in core_entities if entity["role"] == "outcome"]

    return {
        "claim_id": f"CLAIM-{node['evidence_id']}",
        "subject_entities": subject_entities,
        "predicate": predicate,
        "object_entities": object_entities,
        "outcome_entities": outcome_entities,
        "direction": direction,
        "polarity": polarity,
        "claim_scope": infer_claim_scope(core_entities),
        "summary": joined_text,
        "normalized_claim_text": joined_text,
        "hypothesis_level": infer_hypothesis_level(joined_text, normalize_clinical_stage(node.get("clinical_stage")), core_entities),
        "supports_target_discovery": supports_target_discovery,
        "supports_mechanism_inference": supports_mechanism_inference,
        "supports_drug_repositioning": supports_drug_repositioning,
    }


def build_evidence_payload(node: dict[str, Any], proposition: dict[str, Any]) -> dict[str, Any]:
    pico = node.get("pico") or {}
    stats = build_statistics(node)
    subject = proposition["subject_entities"][0] if proposition["subject_entities"] else {"name": None, "normalized_id": None}
    object_entity = proposition["object_entities"][0] if proposition["object_entities"] else {"name": None, "normalized_id": None}
    metric = "mechanistic_statement"
    observation_value = node.get("phenotype") or node.get("bio_mechanism") or proposition["summary"]
    return {
        "evidence_category": "literature_mechanism",
        "aggregation_level": "paper_statement",
        "atomic_observations": [
            {
                "observation_id": f"OBS-{node['evidence_id']}",
                "feature_layer": "literature_claim",
                "subject_label": subject.get("name"),
                "subject_id": subject.get("normalized_id"),
                "object_label": object_entity.get("name"),
                "object_id": object_entity.get("normalized_id"),
                "metric": metric,
                "value": observation_value,
                "unit": None,
                "direction": proposition["direction"] if proposition["direction"] in {"increase", "decrease", "positive", "negative", "mixed", "present", "absent", "unknown"} else "unknown",
                "comparator": pico.get("comparison"),
                "group_a": pico.get("intervention"),
                "group_b": pico.get("comparison"),
                "supporting_n": stats["sample_size"],
                "p_value": stats["p_value"],
                "q_value": stats["q_value"],
                "effect_size": stats["effect_size"],
                "hazard_ratio": stats["hazard_ratio"],
                "ci_lower": None,
                "ci_upper": None,
                "mutation_frequency": stats["mutation_frequency"],
                "dependency_score": None,
                "viability_effect": None,
                "codependency_score": stats["codependency_coefficient"],
                "synergy_score": stats["synergy_score"],
                "text_evidence": node.get("source_text"),
                "is_primary": True,
            }
        ],
        "summary_statistics": stats,
        "platform_specific_fields": {
            "study_object": pico.get("study_object"),
            "intervention": pico.get("intervention"),
            "comparison": pico.get("comparison"),
            "outcome_metrics": pico.get("outcome_metrics") or [],
        },
        "data_completeness": estimate_data_completeness(node),
    }


def estimate_data_completeness(node: dict[str, Any]) -> float:
    score = 0
    possible = 8
    source = node.get("source") or {}
    pico = node.get("pico") or {}
    stats = node.get("statistics") or {}
    if source.get("doi"):
        score += 1
    if source.get("title"):
        score += 1
    if node.get("source_text"):
        score += 1
    if node.get("core_entities"):
        score += 1
    if pico.get("study_object"):
        score += 1
    if pico.get("intervention"):
        score += 1
    if any(stats.get(key) is not None for key in ["p_value", "fold_change", "effect_size", "sample_size"]):
        score += 1
    if node.get("bio_mechanism") or node.get("phenotype"):
        score += 1
    return round(score / possible, 4)


def build_uncertainty(node: dict[str, Any], contradiction_risk: float | None) -> dict[str, Any]:
    score = build_score(node)
    grade = score.get("grade_level") or "C"
    if grade not in {"A", "B", "C", "D", "E"}:
        grade = "C"
    confidence_score = coalesce(score.get("composite_score"), score.get("llm_confidence"), 0.5)
    return {
        "evidence_grade": grade,
        "grade_framework": "hybrid_ebm_ds",
        "confidence_score": float(confidence_score),
        "study_quality_score": score.get("study_type_score"),
        "sample_size_score": score.get("sample_size_score"),
        "statistical_support_score": score.get("stat_significance_score"),
        "source_reliability_score": score.get("impact_score"),
        "llm_confidence": score.get("llm_confidence"),
        "ds_mass_support": score.get("ds_belief"),
        "ds_mass_against": None,
        "ds_mass_uncertainty": score.get("ds_ignorance"),
        "ds_belief": score.get("ds_belief"),
        "ds_plausibility": score.get("ds_plausibility"),
        "contradiction_risk": contradiction_risk,
        "missingness_notes": None,
        "review_status": normalize_review_status(node.get("review_status")),
    }


def normalize_review_status(value: str | None) -> str:
    value = (value or "").lower()
    if value in {"pending", "validated", "rejected", "needs_manual_review"}:
        return value
    return "pending"


def align_evidence_node(node: dict[str, Any], disease_dir: str) -> tuple[dict[str, Any], dict[str, Any]]:
    disease_name = DISEASE_DIR_TO_CANONICAL[disease_dir]
    core_entities = infer_entity_roles(node.get("core_entities") or [], disease_name)
    _, _, contradiction_risk = summarize_relation_slots(node.get("evidence_relations") or [], node.get("linked_tarkg_entities") or [])
    proposition = build_proposition(node, core_entities, disease_name)
    provenance = build_provenance(node, disease_dir, disease_name)
    cohort_context = build_cohort_context(node, disease_name)
    stats = build_statistics(node)
    subject = compact_entity_ref(proposition["subject_entities"][0] if proposition["subject_entities"] else None)
    object_entity = compact_entity_ref(proposition["object_entities"][0] if proposition["object_entities"] else None)
    outcome_entity = compact_entity_ref(proposition["outcome_entities"][0] if proposition["outcome_entities"] else None)
    observation_value = node.get("phenotype") or node.get("bio_mechanism") or proposition["summary"]
    confidence = build_compact_confidence(node, contradiction_risk)
    aligned = {
        "node_type": "evidence",
        "id": node["evidence_id"],
        "record_id": node["evidence_id"],
        "source_family": "pubmed_literature",
        "disease": disease_name,
        "cohort_id": cohort_context["cohort_id"],
        "subject": subject or {"id": None, "label": disease_name, "entity_type": "Disease"},
        "object": object_entity,
        "outcome": outcome_entity,
        "predicate": proposition["predicate"],
        "direction": proposition["direction"],
        "polarity": proposition["polarity"],
        "claim_scope": proposition["claim_scope"],
        "hypothesis_level": proposition["hypothesis_level"],
        "support": build_support(proposition),
        "evidence_category": "literature_mechanism",
        "feature_layer": "literature_claim",
        "aggregation_level": "paper_statement",
        "summary": proposition["summary"],
        "raw_text": node.get("source_text"),
        "metric": "mechanistic_statement",
        "value": observation_value,
        "unit": None,
        "comparison": build_comparison(node, stats["sample_size"]),
        "statistics": stats,
        "confidence": confidence,
        "citation": build_citation(node),
        "source_context": {
            "study_object": (node.get("pico") or {}).get("study_object"),
            "outcome_metrics": list((node.get("pico") or {}).get("outcome_metrics") or []),
            "experimental_context": node.get("experimental_context"),
        },
        "reviewer_comments": node.get("reviewer_comments"),
        "created_at": coalesce(node.get("created_at"), now_iso()),
        "updated_at": node.get("updated_at"),
        "version": node.get("version"),
        "provenance": provenance,
        "aggregation_parent_ids": list(node.get("merged_from") or []),
    }

    cohort_node = {
        "node_type": "cohort",
        "id": cohort_context["cohort_id"],
        "cohort_label": cohort_context["cohort_label"],
        "disease_name": cohort_context["disease_name"],
        "lineage": cohort_context["lineage"],
        "cohort_type": cohort_context["cohort_type"],
        "n_effective": cohort_context["n_effective"],
    }

    return aligned, cohort_node


def merge_entity_nodes(entity_nodes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    merged: dict[str, dict[str, Any]] = {}
    aliases: dict[str, set[str]] = defaultdict(set)
    normalized_from: dict[str, set[str]] = defaultdict(set)

    for entity in entity_nodes:
        entity_id = entity["id"]
        if entity_id not in merged:
            merged[entity_id] = {
                "node_type": "entity",
                "id": entity_id,
                "name": entity.get("name") or entity_id,
                "entity_type": normalize_entity_type(entity.get("entity_type")),
                "db_source": entity.get("db_source"),
                "aliases": [],
                "normalized_from": [],
            }
        aliases[entity_id].add(entity.get("name") or entity_id)
        normalized_from[entity_id].add(entity_id)

    for entity_id, node in merged.items():
        node["aliases"] = sorted(aliases[entity_id])
        node["normalized_from"] = sorted(normalized_from[entity_id])
    return sorted(merged.values(), key=lambda item: item["id"]), {key: sorted(value) for key, value in aliases.items()}


def align_graph_edges(graph_edges: list[dict[str, Any]], known_node_ids: set[str]) -> tuple[list[dict[str, Any]], Counter[str], list[dict[str, Any]]]:
    aligned_edges: list[dict[str, Any]] = []
    relation_counts: Counter[str] = Counter()
    issues: list[dict[str, Any]] = []

    for edge in graph_edges:
        original_relation = edge.get("relation")
        mapped_relation = EDGE_RELATION_MAP.get(original_relation)
        if mapped_relation is None:
            issues.append({"type": "unmapped_relation", "edge": edge})
            continue
        if edge.get("source") not in known_node_ids or edge.get("target") not in known_node_ids:
            issues.append({"type": "orphan_input_edge", "edge": edge})
            continue
        aligned_edge = {
            "source": edge["source"],
            "target": edge["target"],
            "relation": mapped_relation,
            "relation_family": "semantic_anchor" if mapped_relation == "LINKED_TO" else "evidence_logic",
            "entity_type": edge.get("entity_type"),
            "linking_score": safe_float(edge.get("linking_score")),
            "similarity_score": safe_float(edge.get("similarity_score")),
            "rationale": edge.get("rationale"),
            "support_strength": safe_float(edge.get("support_strength")),
            "evidence_count": edge.get("evidence_count"),
            "created_by_rule": f"pubmed_alignment:{original_relation}->{mapped_relation}",
        }
        aligned_edges.append(aligned_edge)
        relation_counts[f"{original_relation}->{mapped_relation}"] += 1

    return aligned_edges, relation_counts, issues


def build_alignment_edges(evidence_nodes: list[dict[str, Any]], cohort_nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cohort_by_id = {node["id"]: node for node in cohort_nodes}
    edges: list[dict[str, Any]] = []
    for evidence in evidence_nodes:
        cohort_id = evidence["cohort_id"]
        if cohort_id in cohort_by_id:
            edges.append(
                {
                    "source": evidence["id"],
                    "target": cohort_id,
                    "relation": "OBSERVED_IN",
                    "relation_family": "semantic_anchor",
                    "entity_type": "Cohort",
                    "linking_score": None,
                    "similarity_score": None,
                    "rationale": "Evidence node observed in aligned paper-level cohort context.",
                    "support_strength": None,
                    "evidence_count": 1,
                    "created_by_rule": "pubmed_alignment:evidence_observed_in_cohort",
                }
            )
    return edges


def compute_completeness(evidence_nodes: list[dict[str, Any]]) -> dict[str, float]:
    total = len(evidence_nodes)
    fields = {
        "doi": lambda n: bool((n.get("citation") or {}).get("doi")),
        "title": lambda n: bool((n.get("citation") or {}).get("title")),
        "authors": lambda n: bool((n.get("citation") or {}).get("authors")),
        "journal": lambda n: bool((n.get("citation") or {}).get("journal")),
        "year": lambda n: (n.get("citation") or {}).get("year") is not None,
        "pdf_path": lambda n: bool((n.get("citation") or {}).get("pdf_path")),
        "raw_text": lambda n: bool(n.get("raw_text")),
        "summary": lambda n: bool(n.get("summary")),
        "study_object": lambda n: bool((n.get("source_context") or {}).get("study_object")),
        "subject": lambda n: bool((n.get("subject") or {}).get("label")),
        "sample_size": lambda n: n["statistics"].get("sample_size") is not None,
        "any_statistic": lambda n: any(n["statistics"].get(key) is not None for key in ["p_value", "fold_change", "effect_size", "sample_size", "hazard_ratio"]),
    }
    return {
        field: round(sum(1 for node in evidence_nodes if check(node)) / total, 4) if total else 0.0
        for field, check in fields.items()
    }


def summarize_numeric(values: list[float | None]) -> dict[str, float | int | None]:
    clean = [value for value in values if value is not None]
    if not clean:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(clean),
        "mean": round(mean(clean), 4),
        "min": round(min(clean), 4),
        "max": round(max(clean), 4),
    }


def build_stats(graph: dict[str, Any], relation_mapping_counts: Counter[str]) -> dict[str, Any]:
    evidence_nodes = [node for node in graph["nodes"] if node["node_type"] == "evidence"]
    cohort_nodes = [node for node in graph["nodes"] if node["node_type"] == "cohort"]
    entity_nodes = [node for node in graph["nodes"] if node["node_type"] == "entity"]

    return {
        "graph_id": graph["graph_metadata"]["graph_id"],
        "disease_scope": graph["graph_metadata"]["disease_scope"],
        "node_counts": {
            "total": len(graph["nodes"]),
            "evidence": len(evidence_nodes),
            "entity": len(entity_nodes),
            "cohort": len(cohort_nodes),
        },
        "edge_counts": {
            "total": len(graph["edges"]),
            "by_relation": dict(sorted(Counter(edge["relation"] for edge in graph["edges"]).items())),
            "input_relation_mapping": dict(sorted(relation_mapping_counts.items())),
        },
        "evidence_distributions": {
            "by_disease": dict(sorted(Counter(node["disease"] for node in evidence_nodes).items())),
            "by_study_design": dict(sorted(Counter(node["provenance"]["study_design"] for node in evidence_nodes).items())),
            "by_clinical_stage": dict(sorted(Counter(node["provenance"]["clinical_stage"] for node in evidence_nodes).items())),
            "by_grade": dict(sorted(Counter(node["confidence"]["grade"] for node in evidence_nodes).items())),
            "by_predicate": dict(sorted(Counter(node["predicate"] for node in evidence_nodes).items())),
            "by_direction": dict(sorted(Counter(node["direction"] for node in evidence_nodes).items())),
            "by_claim_scope": dict(sorted(Counter(node["claim_scope"] for node in evidence_nodes).items())),
        },
        "completeness": compute_completeness(evidence_nodes),
        "numeric_quality": {
            "extraction_confidence": summarize_numeric([node["confidence"].get("extraction_confidence") for node in evidence_nodes]),
            "confidence_score": summarize_numeric([node["confidence"].get("score") for node in evidence_nodes]),
            "contradiction_risk": summarize_numeric([node["confidence"].get("contradiction_risk") for node in evidence_nodes]),
            "data_completeness": summarize_numeric([node["confidence"].get("data_completeness") for node in evidence_nodes]),
        },
    }


def basic_validation(graph: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if graph.get("schema_version") != "2.0.0":
        errors.append("schema_version must be 2.0.0")
    if "graph_metadata" not in graph or "nodes" not in graph or "edges" not in graph:
        errors.append("top-level graph keys are incomplete")
        return errors
    node_ids = [node.get("id") for node in graph["nodes"]]
    if len(node_ids) != len(set(node_ids)):
        errors.append("duplicate node ids detected")
    known_node_ids = set(node_ids)
    for edge in graph["edges"]:
        if edge.get("relation") not in ALLOWED_EDGE_RELATIONS:
            errors.append(f"invalid edge relation: {edge.get('relation')}")
        if edge.get("source") not in known_node_ids or edge.get("target") not in known_node_ids:
            errors.append(f"orphan edge detected: {edge}")
    return errors


def schema_validate(graph: dict[str, Any]) -> list[str]:
    if jsonschema is None:
        return basic_validation(graph)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(graph), key=lambda err: list(err.path))
    messages = []
    for error in errors[:50]:
        location = ".".join(str(part) for part in error.path)
        messages.append(f"{location}: {error.message}")
    return messages


def build_alignment_mapping() -> dict[str, Any]:
    return {
        "schema_version": "2.0.0",
        "input": "kg/pubmed-evidence-kg/{hcc,crc}/evidence_nodes.json + evidence_graph.json",
        "output": "kg/aligned/pubmed_aligned_latest.json",
        "field_mapping": {
            "source.*": "evidence.citation + evidence.provenance",
            "pico.*": "evidence.comparison + evidence.source_context",
            "core_entities": "flattened into evidence.subject/object/outcome; remaining entity links stay on graph edges",
            "target_disease": "evidence.disease + cohort node disease_name",
            "bio_mechanism/phenotype/source_text": "evidence.summary + evidence.value + evidence.raw_text",
            "statistics": "evidence.statistics",
            "score/uncertainty": "evidence.confidence",
            "study_design/clinical_stage": "normalized into evidence.provenance enums compatible with compact schema",
        },
        "relation_mapping": EDGE_RELATION_MAP,
        "notes": [
            "PubMed evidence nodes are preserved as first-class evidence nodes, but repeated legacy payload fields are folded into compact claim, comparison, confidence, citation, and provenance sections.",
            "PubMed evidence-logic relations preserve the original EvidenceNet semantics, including CAUSAL_CHAIN as a first-class evidence relation.",
            "Standalone proposition nodes remain removed; paper-level cohort nodes remain explicit.",
        ],
    }


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    created_at = now_iso()

    all_evidence_nodes: list[dict[str, Any]] = []
    all_cohort_nodes_raw: list[dict[str, Any]] = []
    all_entity_nodes_raw: list[dict[str, Any]] = []
    input_graph_edges: list[dict[str, Any]] = []
    audit_issues: list[dict[str, Any]] = []

    source_manifest: dict[str, Any] = {}
    raw_evidence_counts: dict[str, int] = {}

    for disease_dir in INPUT_DISEASES:
        evidence_path = PUBMED_ROOT / disease_dir / "evidence_nodes.json"
        graph_path = PUBMED_ROOT / disease_dir / "evidence_graph.json"

        nodes = json.loads(evidence_path.read_text(encoding="utf-8"))
        graph = json.loads(graph_path.read_text(encoding="utf-8"))["graph"]

        source_manifest[disease_dir] = {
            "evidence_nodes_path": relpath(evidence_path),
            "evidence_graph_path": relpath(graph_path),
            "raw_evidence_count": len(nodes),
            "raw_graph_node_count": len(graph.get("nodes", [])),
            "raw_graph_edge_count": len(graph.get("edges", [])),
        }
        raw_evidence_counts[DISEASE_DIR_TO_CANONICAL[disease_dir]] = len(nodes)

        for node in nodes.values():
            aligned, cohort_node = align_evidence_node(node, disease_dir)
            all_evidence_nodes.append(aligned)
            all_cohort_nodes_raw.append(cohort_node)

        for node in graph.get("nodes", []):
            if node.get("node_type") == "entity":
                all_entity_nodes_raw.append(node)
        input_graph_edges.extend(graph.get("edges", []))

    entity_nodes, entity_aliases = merge_entity_nodes(all_entity_nodes_raw)

    unique_cohort_nodes = sorted({node["id"]: node for node in all_cohort_nodes_raw}.values(), key=lambda item: item["id"])

    known_input_node_ids = {node["id"] for node in all_evidence_nodes} | {node["id"] for node in entity_nodes}
    mapped_input_edges, relation_mapping_counts, edge_issues = align_graph_edges(input_graph_edges, known_input_node_ids)
    audit_issues.extend(edge_issues)
    generated_edges = build_alignment_edges(all_evidence_nodes, unique_cohort_nodes)

    combined_nodes = sorted(
        all_evidence_nodes + entity_nodes + unique_cohort_nodes,
        key=lambda item: (item["node_type"], item["id"]),
    )
    combined_edges = mapped_input_edges + generated_edges

    graph = {
        "schema_version": "2.0.0",
        "graph_metadata": {
            "graph_id": "pubmed_aligned_latest",
            "graph_name": "Aligned PubMed evidence graph for compact unified multisource KG schema",
            "created_by": "scripts/align_pubmed_to_unified_schema.py",
            "created_at": created_at,
            "source_families": ["pubmed_literature"],
            "disease_scope": [DISEASE_DIR_TO_CANONICAL[d] for d in INPUT_DISEASES],
            "build_parameters": {
                "input_diseases": INPUT_DISEASES,
                "relation_mapping": EDGE_RELATION_MAP,
                "study_design_mapping": STUDY_DESIGN_MAP,
                "clinical_stage_mapping": CLINICAL_STAGE_MAP,
                "source_manifest": source_manifest,
            },
            "notes": "PubMed evidence KG aligned into the compact unified multisource schema with flattened claim semantics and retained graph edges.",
        },
        "nodes": combined_nodes,
        "edges": combined_edges,
    }

    validation_errors = schema_validate(graph)
    node_ids = {node["id"] for node in combined_nodes}
    orphan_edges = [
        edge
        for edge in combined_edges
        if edge["source"] not in node_ids or edge["target"] not in node_ids
    ]

    audit = {
        "graph_id": graph["graph_metadata"]["graph_id"],
        "timestamp": created_at,
        "input_sources": source_manifest,
        "raw_evidence_counts": raw_evidence_counts,
        "checks": {
            "schema_validation": {
                "validator": "jsonschema" if jsonschema is not None else "basic_validation",
                "passed": not validation_errors,
                "error_count": len(validation_errors),
                "errors": validation_errors,
            },
            "orphan_edges": {
                "count": len(orphan_edges),
                "examples": orphan_edges[:10],
            },
            "alignment_issues": {
                "count": len(audit_issues),
                "by_type": dict(sorted(Counter(issue["type"] for issue in audit_issues).items())),
                "examples": audit_issues[:20],
            },
            "entity_alias_merge": {
                "merged_entity_count": len(entity_nodes),
                "entity_ids_with_multiple_aliases": sum(1 for aliases in entity_aliases.values() if len(aliases) > 1),
            },
        },
    }

    stats = build_stats(graph, relation_mapping_counts)
    mapping = build_alignment_mapping()

    (OUTPUT_DIR / "pubmed_aligned_latest.json").write_text(json.dumps(graph, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "pubmed_aligned_latest_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "pubmed_aligned_latest_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "pubmed_alignment_mapping_latest.json").write_text(json.dumps(mapping, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

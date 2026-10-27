from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

try:
    import jsonschema
except ImportError:  # pragma: no cover
    jsonschema = None


ROOT = Path(__file__).resolve().parents[1]
ALIGNED_DIR = ROOT / "kg" / "aligned"
SCHEMA_PATH = ROOT / "kg" / "schema" / "multisource_evidence_kg_schema_v2.json"

SOURCE_PATHS = {
    "pubmed": ALIGNED_DIR / "pubmed_aligned_normalized_latest.json",
    "tcga": ALIGNED_DIR / "tcga_atomic_compact_normalized_latest.json",
    "depmap": ALIGNED_DIR / "depmap_atomic_compact_normalized_latest.json",
    "disgenet": ALIGNED_DIR / "disgenet_aligned_normalized_latest.json",
    "cellline_annotation": ALIGNED_DIR / "cellline_annotation_aligned_normalized_latest.json",
}

OUTPUT_GRAPH_PATH = ALIGNED_DIR / "multisource_evidence_graph_latest.json"
OUTPUT_STATS_PATH = ALIGNED_DIR / "multisource_evidence_graph_stats_latest.json"
OUTPUT_AUDIT_PATH = ALIGNED_DIR / "multisource_evidence_graph_audit_latest.json"
OUTPUT_BRIDGE_CANDIDATES_PATH = ALIGNED_DIR / "multisource_bridge_candidates_latest.json"

SOURCE_PRIORITY = {
    "integrated": 0,
    "pubmed": 1,
    "disgenet": 1,
    "depmap": 2,
    "cellline_annotation": 2,
    "tcga": 3,
}

TCGA_ACTIVITY_STATE = {
    "expression_regulation": {"positive": "activated_program", "negative": "suppressed_program"},
    "copy_number_regulation": {"positive": "activated_program", "negative": "suppressed_program"},
    "methylation_regulation": {"positive": "suppressed_program", "negative": "activated_program"},
    "protein_activation_regulation": {"positive": "activated_program", "negative": "suppressed_program"},
}

TCGA_PROGNOSIS_STATE = {
    "expression_prognosis": {"positive": "favorable_outcome", "negative": "poor_outcome"},
    "methylation_prognosis": {"positive": "favorable_outcome", "negative": "poor_outcome"},
    "copy_number_amplification_prognosis": {"positive": "favorable_outcome", "negative": "poor_outcome"},
    "copy_number_deletion_prognosis": {"positive": "favorable_outcome", "negative": "poor_outcome"},
    "mutation_prognosis": {"positive": "favorable_outcome", "negative": "poor_outcome"},
    "protein_abundance_prognosis": {"positive": "favorable_outcome", "negative": "poor_outcome"},
}

TCGA_RELATION_GROUP = {
    "upregulated_in": ("expression_regulation", "positive"),
    "downregulated_in": ("expression_regulation", "negative"),
    "hypermethylated_in": ("methylation_regulation", "positive"),
    "hypomethylated_in": ("methylation_regulation", "negative"),
    "amplified_in": ("copy_number_regulation", "positive"),
    "deleted_in": ("copy_number_regulation", "negative"),
    "hyperphosphorylated_in": ("protein_activation_regulation", "positive"),
    "hypophosphorylated_in": ("protein_activation_regulation", "negative"),
    "high_expression_associated_with_poor_prognosis_in": ("expression_prognosis", "negative"),
    "high_expression_associated_with_favorable_prognosis_in": ("expression_prognosis", "positive"),
    "hypermethylation_associated_with_poor_prognosis_in": ("methylation_prognosis", "negative"),
    "hypermethylation_associated_with_favorable_prognosis_in": ("methylation_prognosis", "positive"),
    "amplification_associated_with_poor_prognosis_in": ("copy_number_amplification_prognosis", "negative"),
    "amplification_associated_with_favorable_prognosis_in": ("copy_number_amplification_prognosis", "positive"),
    "deletion_associated_with_poor_prognosis_in": ("copy_number_deletion_prognosis", "negative"),
    "deletion_associated_with_favorable_prognosis_in": ("copy_number_deletion_prognosis", "positive"),
    "mutation_associated_with_poor_prognosis_in": ("mutation_prognosis", "negative"),
    "mutation_associated_with_favorable_prognosis_in": ("mutation_prognosis", "positive"),
    "high_protein_abundance_associated_with_poor_prognosis_in": ("protein_abundance_prognosis", "negative"),
    "high_protein_abundance_associated_with_favorable_prognosis_in": ("protein_abundance_prognosis", "positive"),
    "mutated_in": ("mutation_occurrence", "positive"),
}

DEPMAP_SPECIFICITY = {
    "dependency_in": 0,
    "selective_dependency_in": 1,
    "high_expression_associated_with_dependency_in": 2,
    "copy_number_gain_associated_with_dependency_in": 2,
    "mutation_associated_with_dependency_in": 2,
    "subtype_associated_with_dependency_in": 2,
    "phenotype_associated_with_dependency_in": 2,
    "metastatic_state_associated_with_dependency_in": 2,
}

ALLOWED_EDGE_RELATIONS = {
    "LINKED_TO",
    "OBSERVED_IN",
    "SUPPORTS",
    "CONTRADICTS",
    "REFINES",
    "EXTENDS",
    "REPLICATES",
    "CAUSAL_CHAIN",
    "COMPLEMENTS",
    "MECHANISM_TO_OMICS_MANIFESTATION",
    "MECHANISM_TO_DEPENDENCY_VULNERABILITY",
    "DRUG_MECHANISM_TO_SENSITIVITY",
    "DERIVED_FROM",
}

BRIDGE_RELATIONS = {
    "MECHANISM_TO_OMICS_MANIFESTATION",
    "MECHANISM_TO_DEPENDENCY_VULNERABILITY",
    "DRUG_MECHANISM_TO_SENSITIVITY",
}


@dataclass
class Descriptor:
    evidence_id: str
    source_name: str
    source_family: str
    subject_ref: dict[str, Any]
    object_ref: dict[str, Any] | None
    outcome_ref: dict[str, Any] | None
    disease_key: str | None
    disease_label: str
    anchor_key: str | None
    anchor_label: str
    anchor_type: str
    target_key: str | None
    relation_group: str
    state: str | None
    mechanism_bucket: str | None
    specificity: int
    predicate: str
    feature_layer: str
    evidence_category: str
    confidence_score: float
    sample_size: int
    created_at: str
    supports_target_discovery: bool
    supports_mechanism_inference: bool
    supports_drug_repositioning: bool


def stable_hash(text: str, length: int = 16) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length].upper()


def now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def canonical_ref(ref: dict[str, Any] | None) -> dict[str, Any] | None:
    if not ref:
        return None
    return {
        "id": ref.get("normalized_id") or ref.get("id"),
        "label": ref.get("normalized_label") or ref.get("label") or ref.get("name") or ref.get("id"),
        "entity_type": ref.get("normalized_entity_type") or ref.get("entity_type"),
    }


def canonical_entity_id(node: dict[str, Any]) -> str:
    return node.get("normalized_id") or node["id"]


def merge_entity_nodes(source_graphs: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    merged: dict[str, dict[str, Any]] = {}
    aliases: dict[str, set[str]] = defaultdict(set)
    normalized_from: dict[str, set[str]] = defaultdict(set)
    original_to_unified: dict[str, str] = {}

    for graph in source_graphs.values():
        for node in graph["nodes"]:
            if node.get("node_type") != "entity":
                continue
            entity_id = canonical_entity_id(node)
            original_to_unified[node["id"]] = entity_id
            if entity_id not in merged:
                merged[entity_id] = {
                    "node_type": "entity",
                    "id": entity_id,
                    "name": node.get("normalized_label") or node.get("name") or entity_id,
                    "entity_type": node.get("normalized_entity_type") or node.get("entity_type"),
                    "db_source": node.get("normalized_db_source") or node.get("db_source"),
                    "aliases": [],
                    "normalized_from": [],
                }
            aliases[entity_id].add(node.get("name") or entity_id)
            for alias in node.get("aliases") or []:
                if alias:
                    aliases[entity_id].add(alias)
            normalized_from[entity_id].add(node["id"])
            for source_id in node.get("normalized_from") or []:
                normalized_from[entity_id].add(source_id)

    entity_nodes = []
    for entity_id, node in merged.items():
        node["aliases"] = sorted(aliases[entity_id])
        node["normalized_from"] = sorted(normalized_from[entity_id])
        entity_nodes.append(node)
    entity_nodes.sort(key=lambda item: item["id"])
    return entity_nodes, original_to_unified


def build_integrated_cohorts(descriptors: list[Descriptor]) -> list[dict[str, Any]]:
    cohorts: dict[str, dict[str, Any]] = {}
    for descriptor in descriptors:
        if not descriptor.disease_key:
            continue
        cohort_id = f"COHORT-INTEGRATED-{stable_hash(descriptor.disease_key)}"
        cohorts.setdefault(
            cohort_id,
            {
                "node_type": "cohort",
                "id": cohort_id,
                "cohort_label": f"Integrated evidence context for {descriptor.disease_label}",
                "disease_name": descriptor.disease_label,
                "lineage": None,
                "cohort_type": "integrated_evidence_context",
                "n_effective": None,
            },
        )
    return sorted(cohorts.values(), key=lambda item: item["id"])


def evidence_rank(node: dict[str, Any]) -> tuple[float, int, str]:
    confidence = float((node.get("confidence") or {}).get("score") or 0.0)
    sample_size = int((node.get("statistics") or {}).get("sample_size") or (node.get("comparison") or {}).get("n_effective") or 0)
    return (confidence, sample_size, node["id"])


def node_disease_ref(node: dict[str, Any]) -> dict[str, Any] | None:
    for ref_name in ["subject", "outcome", "object"]:
        ref = node.get(ref_name) or {}
        if ref.get("entity_type") == "Disease":
            return ref
    return None


def descriptor_from_evidence(node: dict[str, Any], source_name: str) -> Descriptor | None:
    if node.get("node_type") != "evidence":
        return None
    subject = canonical_ref(node.get("subject"))
    obj = canonical_ref(node.get("object"))
    outcome = canonical_ref(node.get("outcome"))
    disease_ref = node_disease_ref({"subject": subject, "object": obj, "outcome": outcome})
    disease_key = disease_ref["id"] if disease_ref else None
    disease_label = disease_ref["label"] if disease_ref else node.get("disease", "unknown")
    predicate = node["predicate"]
    feature_layer = node.get("feature_layer") or "unknown"
    evidence_category = node.get("evidence_category") or "unknown"
    confidence_score = float((node.get("confidence") or {}).get("score") or 0.0)
    sample_size = int((node.get("statistics") or {}).get("sample_size") or (node.get("comparison") or {}).get("n_effective") or 0)
    specificity = 0
    relation_group = predicate
    state: str | None = None
    mechanism_bucket: str | None = None
    anchor_ref = subject
    target_key = None

    if source_name == "tcga":
        relation_group, sign = TCGA_RELATION_GROUP.get(predicate, (predicate, node.get("direction") or "unknown"))
        sample_grouping = (node.get("comparison") or {}).get("sample_grouping")
        specificity = 1 if sample_grouping in {"tumor_purity_quantiles", "pathologic_stage"} else 0
        if relation_group in TCGA_ACTIVITY_STATE:
            state = TCGA_ACTIVITY_STATE[relation_group].get(sign)
        elif relation_group in TCGA_PROGNOSIS_STATE:
            state = TCGA_PROGNOSIS_STATE[relation_group].get(sign)
        elif relation_group == "mutation_occurrence":
            state = "genomic_alteration"
        mechanism_bucket = {
            "activated_program": "oncogenic_support",
            "poor_outcome": "oncogenic_support",
            "dependency_vulnerability": "oncogenic_support",
            "suppressed_program": "protective_support",
            "favorable_outcome": "protective_support",
            "therapeutic_benefit": "therapeutic_benefit",
        }.get(state)
    elif source_name == "depmap":
        if predicate in DEPMAP_SPECIFICITY:
            relation_group = "dependency_core"
            state = "dependency_vulnerability"
            specificity = DEPMAP_SPECIFICITY[predicate]
        elif predicate == "sensitive_to":
            relation_group = "drug_sensitivity"
            state = "therapeutic_benefit"
            anchor_ref = obj
            target_key = anchor_ref["id"] if anchor_ref else None
        elif predicate == "codependent_with":
            relation_group = "gene_codependency"
            state = "codependency"
            target_key = obj["id"] if obj else None
        mechanism_bucket = {
            "dependency_vulnerability": "oncogenic_support",
            "therapeutic_benefit": "therapeutic_benefit",
        }.get(state)
    else:
        subject_type = subject.get("entity_type") if subject else None
        if predicate == "promotes_mechanism" and subject_type in {"Gene", "Protein"}:
            relation_group = "literature_gene_mechanism"
            state = "activated_program"
        elif predicate == "suppresses_mechanism" and subject_type in {"Gene", "Protein"}:
            relation_group = "literature_gene_mechanism"
            state = "suppressed_program"
        elif predicate == "suppresses_mechanism" and subject_type == "Drug":
            relation_group = "literature_drug_mechanism"
            state = "therapeutic_benefit"
            target_key = subject["id"]
        elif predicate == "promotes_mechanism" and subject_type == "Drug":
            relation_group = "literature_drug_mechanism"
            state = "therapeutic_risk"
            target_key = subject["id"]
        elif predicate == "supports_mechanistic_claim" and subject_type in {"Gene", "Protein"}:
            relation_group = "literature_gene_mechanism"
            state = "generic_mechanism"
        mechanism_bucket = {
            "activated_program": "oncogenic_support",
            "suppressed_program": "protective_support",
            "therapeutic_benefit": "therapeutic_benefit",
        }.get(state)

    if anchor_ref is None or disease_key is None:
        return None

    anchor_key = anchor_ref["id"]
    if target_key is None and obj and obj.get("entity_type") != "Disease":
        target_key = obj["id"]
    if target_key is None and outcome and outcome.get("entity_type") != "Disease":
        target_key = outcome["id"]

    return Descriptor(
        evidence_id=node["id"],
        source_name=source_name,
        source_family=node["source_family"],
        subject_ref=subject,
        object_ref=obj,
        outcome_ref=outcome,
        disease_key=disease_key,
        disease_label=disease_label,
        anchor_key=anchor_key,
        anchor_label=anchor_ref["label"],
        anchor_type=anchor_ref["entity_type"],
        target_key=target_key,
        relation_group=relation_group,
        state=state,
        mechanism_bucket=mechanism_bucket,
        specificity=specificity,
        predicate=predicate,
        feature_layer=feature_layer,
        evidence_category=evidence_category,
        confidence_score=confidence_score,
        sample_size=sample_size,
        created_at=node["created_at"],
        supports_target_discovery=bool((node.get("support") or {}).get("supports_target_discovery")),
        supports_mechanism_inference=bool((node.get("support") or {}).get("supports_mechanism_inference")),
        supports_drug_repositioning=bool((node.get("support") or {}).get("supports_drug_repositioning")),
    )


def relation_priority(source_name: str) -> int:
    return SOURCE_PRIORITY.get(source_name, 99)


def descriptor_rank(descriptor: Descriptor) -> tuple[int, float, int, str]:
    return (-relation_priority(descriptor.source_name), descriptor.confidence_score, descriptor.sample_size, descriptor.evidence_id)


def build_edge(
    source_id: str,
    target_id: str,
    relation: str,
    family: str,
    rule: str,
    rationale: str,
    support_strength: float | None,
    similarity_score: float | None,
    evidence_count: int | None,
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


def convert_source_graph(
    graph: dict[str, Any],
    source_name: str,
    original_to_unified_entity: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[Descriptor]]:
    evidence_nodes: list[dict[str, Any]] = []
    cohort_nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    descriptors: list[Descriptor] = []
    for node in graph["nodes"]:
        if node.get("node_type") == "cohort":
            cohort_nodes.append(node)
        elif node.get("node_type") == "evidence":
            converted = dict(node)
            converted["subject"] = canonical_ref(node.get("subject"))
            converted["object"] = canonical_ref(node.get("object"))
            converted["outcome"] = canonical_ref(node.get("outcome"))
            evidence_nodes.append(converted)
            descriptor = descriptor_from_evidence(converted, source_name)
            if descriptor is not None:
                descriptors.append(descriptor)
    for edge in graph["edges"]:
        converted = dict(edge)
        if converted["relation"] == "LINKED_TO":
            converted["target"] = original_to_unified_entity.get(converted["target"], converted["target"])
        edges.append(converted)
    return evidence_nodes, cohort_nodes, edges, descriptors


def comparable_family(state: str | None) -> str | None:
    if state in {"activated_program", "suppressed_program"}:
        return "activity"
    if state in {"poor_outcome", "favorable_outcome"}:
        return "prognosis"
    if state in {"therapeutic_benefit", "therapeutic_risk"}:
        return "therapy"
    return None


def descriptor_disease_bucket(descriptor: Descriptor) -> str:
    return descriptor.disease_label or descriptor.disease_key or "unknown"


def bridge_support_strength(*scores: float, bonus: float = 0.0) -> float:
    valid_scores = [score for score in scores if score is not None]
    if not valid_scores:
        return round(min(0.99, 0.5 + bonus), 4)
    return round(min(0.99, min(valid_scores) + bonus), 4)


def is_pubmed_gene_mechanism(descriptor: Descriptor) -> bool:
    return descriptor.source_family == "pubmed_literature" and descriptor.anchor_type in {"Gene", "Protein"} and descriptor.relation_group == "literature_gene_mechanism"


def is_pubmed_drug_mechanism(descriptor: Descriptor) -> bool:
    return descriptor.source_family == "pubmed_literature" and descriptor.anchor_type == "Drug" and descriptor.relation_group == "literature_drug_mechanism"


def is_tcga_structured_manifestation(descriptor: Descriptor) -> bool:
    return descriptor.source_family == "tcga_linkedomics" and descriptor.state in {
        "activated_program",
        "suppressed_program",
        "poor_outcome",
        "favorable_outcome",
        "genomic_alteration",
    }


def is_depmap_dependency(descriptor: Descriptor) -> bool:
    return descriptor.source_family == "depmap" and descriptor.state == "dependency_vulnerability"


def is_depmap_drug_sensitivity(descriptor: Descriptor) -> bool:
    return descriptor.source_family == "depmap" and descriptor.relation_group == "drug_sensitivity" and descriptor.state == "therapeutic_benefit"


def is_pubmed_to_tcga_bridge_compatible(pubmed_descriptor: Descriptor, tcga_descriptor: Descriptor) -> bool:
    if pubmed_descriptor.state == "activated_program":
        return tcga_descriptor.state in {"activated_program", "poor_outcome", "genomic_alteration"}
    if pubmed_descriptor.state == "suppressed_program":
        return tcga_descriptor.state in {"suppressed_program", "favorable_outcome"}
    if pubmed_descriptor.state == "generic_mechanism":
        return pubmed_descriptor.supports_target_discovery or pubmed_descriptor.supports_mechanism_inference
    return False


def is_pubmed_to_depmap_bridge_compatible(pubmed_descriptor: Descriptor, depmap_descriptor: Descriptor) -> bool:
    if depmap_descriptor.state != "dependency_vulnerability":
        return False
    if pubmed_descriptor.state == "activated_program":
        return True
    if pubmed_descriptor.state == "generic_mechanism":
        return pubmed_descriptor.supports_target_discovery or pubmed_descriptor.supports_mechanism_inference
    return False


def build_bridge_edges(descriptors: list[Descriptor]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    bridge_rule_counts: Counter[str] = Counter()
    bridge_relation_counts: Counter[str] = Counter()
    bridge_source_pairs: Counter[str] = Counter()
    bridge_relation_counts_by_disease: dict[str, Counter[str]] = defaultdict(Counter)
    candidate_summary: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    candidate_examples: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

    omics_groups: dict[tuple[str, str], dict[str, list[Descriptor]]] = defaultdict(lambda: {"pubmed": [], "tcga": []})
    dependency_groups: dict[tuple[str, str], dict[str, list[Descriptor]]] = defaultdict(lambda: {"pubmed": [], "depmap": []})
    drug_groups: dict[tuple[str, str], dict[str, list[Descriptor]]] = defaultdict(lambda: {"pubmed": [], "depmap": []})

    for descriptor in descriptors:
        if descriptor.anchor_key is None or descriptor.disease_key is None:
            continue
        key = (descriptor.anchor_key, descriptor.disease_key)
        if is_pubmed_gene_mechanism(descriptor):
            omics_groups[key]["pubmed"].append(descriptor)
            dependency_groups[key]["pubmed"].append(descriptor)
        elif is_tcga_structured_manifestation(descriptor):
            omics_groups[key]["tcga"].append(descriptor)
        elif is_depmap_dependency(descriptor):
            dependency_groups[key]["depmap"].append(descriptor)
        if is_pubmed_drug_mechanism(descriptor):
            drug_groups[key]["pubmed"].append(descriptor)
        elif is_depmap_drug_sensitivity(descriptor):
            drug_groups[key]["depmap"].append(descriptor)

    def record_candidate(relation: str, descriptor_group: list[Descriptor], source_key: str, target_group: list[Descriptor], target_key: str) -> None:
        if not descriptor_group or not target_group:
            return
        disease = descriptor_disease_bucket(descriptor_group[0])
        candidate_summary[relation][disease]["shared_anchor_groups"] += 1
        candidate_summary[relation][disease][f"{source_key}_evidence"] += len(descriptor_group)
        candidate_summary[relation][disease][f"{target_key}_evidence"] += len(target_group)
        if len(candidate_examples[relation][disease]) < 5:
            candidate_examples[relation][disease].append(
                {
                    "disease": disease,
                    "anchor_key": descriptor_group[0].anchor_key,
                    "anchor_label": descriptor_group[0].anchor_label,
                    "source_family_counts": {
                        descriptor_group[0].source_family: len(descriptor_group),
                        target_group[0].source_family: len(target_group),
                    },
                }
            )

    for (anchor_key, disease_key), grouped in omics_groups.items():
        pubmed_group = grouped["pubmed"]
        tcga_group = grouped["tcga"]
        record_candidate("MECHANISM_TO_OMICS_MANIFESTATION", pubmed_group, "pubmed", tcga_group, "tcga")
        if not pubmed_group or not tcga_group:
            continue
        best_pubmed_by_state: dict[str, Descriptor] = {}
        for descriptor in pubmed_group:
            if descriptor.state not in {"activated_program", "suppressed_program", "generic_mechanism"}:
                continue
            current = best_pubmed_by_state.get(descriptor.state)
            if current is None or descriptor_rank(descriptor) > descriptor_rank(current):
                best_pubmed_by_state[descriptor.state] = descriptor
        best_tcga_by_signature: dict[tuple[str, str], Descriptor] = {}
        for descriptor in tcga_group:
            signature = (descriptor.feature_layer, descriptor.state or "unknown")
            current = best_tcga_by_signature.get(signature)
            if current is None or descriptor_rank(descriptor) > descriptor_rank(current):
                best_tcga_by_signature[signature] = descriptor
        for pubmed_descriptor in best_pubmed_by_state.values():
            for tcga_descriptor in best_tcga_by_signature.values():
                if not is_pubmed_to_tcga_bridge_compatible(pubmed_descriptor, tcga_descriptor):
                    continue
                bonus = 0.08 if pubmed_descriptor.state != "generic_mechanism" else 0.03
                edge = build_edge(
                    source_id=pubmed_descriptor.evidence_id,
                    target_id=tcga_descriptor.evidence_id,
                    relation="MECHANISM_TO_OMICS_MANIFESTATION",
                    family="evidence_logic",
                    rule="bridge_logic:mechanism_to_omics_manifestation",
                    rationale="PubMed mechanism evidence is operationalized by TCGA omics, alteration, or prognosis evidence for the same normalized anchor and disease.",
                    support_strength=bridge_support_strength(pubmed_descriptor.confidence_score, tcga_descriptor.confidence_score, bonus=bonus),
                    similarity_score=0.91 if pubmed_descriptor.state != "generic_mechanism" else 0.82,
                    evidence_count=2,
                )
                before_count = len(edges)
                add_edge_once(edges, seen, edge)
                if len(edges) == before_count:
                    continue
                disease = descriptor_disease_bucket(pubmed_descriptor)
                bridge_rule_counts["mechanism_to_omics_manifestation"] += 1
                bridge_relation_counts["MECHANISM_TO_OMICS_MANIFESTATION"] += 1
                bridge_relation_counts_by_disease["MECHANISM_TO_OMICS_MANIFESTATION"][disease] += 1
                bridge_source_pairs["pubmed_literature|tcga_linkedomics"] += 1

    for (anchor_key, disease_key), grouped in dependency_groups.items():
        pubmed_group = grouped["pubmed"]
        depmap_group = grouped["depmap"]
        record_candidate("MECHANISM_TO_DEPENDENCY_VULNERABILITY", pubmed_group, "pubmed", depmap_group, "depmap")
        if not pubmed_group or not depmap_group:
            continue
        best_pubmed_by_state: dict[str, Descriptor] = {}
        for descriptor in pubmed_group:
            if descriptor.state not in {"activated_program", "generic_mechanism"}:
                continue
            current = best_pubmed_by_state.get(descriptor.state)
            if current is None or descriptor_rank(descriptor) > descriptor_rank(current):
                best_pubmed_by_state[descriptor.state] = descriptor
        best_depmap_by_signature: dict[tuple[str, str], Descriptor] = {}
        for descriptor in depmap_group:
            signature = (descriptor.feature_layer, descriptor.predicate)
            current = best_depmap_by_signature.get(signature)
            if current is None or descriptor_rank(descriptor) > descriptor_rank(current):
                best_depmap_by_signature[signature] = descriptor
        for pubmed_descriptor in best_pubmed_by_state.values():
            for depmap_descriptor in best_depmap_by_signature.values():
                if not is_pubmed_to_depmap_bridge_compatible(pubmed_descriptor, depmap_descriptor):
                    continue
                bonus = 0.09 if pubmed_descriptor.state == "activated_program" else 0.04
                edge = build_edge(
                    source_id=pubmed_descriptor.evidence_id,
                    target_id=depmap_descriptor.evidence_id,
                    relation="MECHANISM_TO_DEPENDENCY_VULNERABILITY",
                    family="evidence_logic",
                    rule="bridge_logic:mechanism_to_dependency_vulnerability",
                    rationale="PubMed mechanism evidence is reinforced by DepMap lineage vulnerability for the same normalized anchor and disease.",
                    support_strength=bridge_support_strength(pubmed_descriptor.confidence_score, depmap_descriptor.confidence_score, bonus=bonus),
                    similarity_score=0.9 if pubmed_descriptor.state == "activated_program" else 0.8,
                    evidence_count=2,
                )
                before_count = len(edges)
                add_edge_once(edges, seen, edge)
                if len(edges) == before_count:
                    continue
                disease = descriptor_disease_bucket(pubmed_descriptor)
                bridge_rule_counts["mechanism_to_dependency_vulnerability"] += 1
                bridge_relation_counts["MECHANISM_TO_DEPENDENCY_VULNERABILITY"] += 1
                bridge_relation_counts_by_disease["MECHANISM_TO_DEPENDENCY_VULNERABILITY"][disease] += 1
                bridge_source_pairs["depmap|pubmed_literature"] += 1

    for (anchor_key, disease_key), grouped in drug_groups.items():
        pubmed_group = grouped["pubmed"]
        depmap_group = grouped["depmap"]
        record_candidate("DRUG_MECHANISM_TO_SENSITIVITY", pubmed_group, "pubmed", depmap_group, "depmap")
        if not pubmed_group or not depmap_group:
            continue
        pubmed_anchor = sorted(
            [descriptor for descriptor in pubmed_group if descriptor.state == "therapeutic_benefit"],
            key=descriptor_rank,
            reverse=True,
        )
        depmap_anchor = sorted(depmap_group, key=descriptor_rank, reverse=True)
        if not pubmed_anchor or not depmap_anchor:
            continue
        edge = build_edge(
            source_id=pubmed_anchor[0].evidence_id,
            target_id=depmap_anchor[0].evidence_id,
            relation="DRUG_MECHANISM_TO_SENSITIVITY",
            family="evidence_logic",
            rule="bridge_logic:drug_mechanism_to_sensitivity",
            rationale="PubMed therapeutic mechanism evidence aligns with lineage-specific drug sensitivity for the same normalized drug and disease.",
            support_strength=bridge_support_strength(pubmed_anchor[0].confidence_score, depmap_anchor[0].confidence_score, bonus=0.08),
            similarity_score=0.93,
            evidence_count=2,
        )
        before_count = len(edges)
        add_edge_once(edges, seen, edge)
        if len(edges) == before_count:
            continue
        disease = descriptor_disease_bucket(pubmed_anchor[0])
        bridge_rule_counts["drug_mechanism_to_sensitivity"] += 1
        bridge_relation_counts["DRUG_MECHANISM_TO_SENSITIVITY"] += 1
        bridge_relation_counts_by_disease["DRUG_MECHANISM_TO_SENSITIVITY"][disease] += 1
        bridge_source_pairs["depmap|pubmed_literature"] += 1

    pan_cancer_summary = {
        relation: dict(sorted(sum((counts for counts in per_disease.values()), Counter()).items()))
        for relation, per_disease in candidate_summary.items()
    }

    focus_aliases = {
        "HCC": {"HCC", "hepatocellular carcinoma"},
        "CRC": {"CRC", "colorectal cancer"},
    }
    hcc_crc_focus = {}
    for relation in sorted(candidate_summary):
        hcc_crc_focus[relation] = {}
        for focus_name, aliases in focus_aliases.items():
            merged = Counter()
            for disease, counts in candidate_summary[relation].items():
                if disease in aliases:
                    merged.update(counts)
            if merged:
                hcc_crc_focus[relation][focus_name] = dict(sorted(merged.items()))

    return edges, {
        "generated_at": now_iso(),
        "bridge_rule_counts": dict(sorted(bridge_rule_counts.items())),
        "bridge_relation_counts": dict(sorted(bridge_relation_counts.items())),
        "bridge_source_pairs": dict(sorted(bridge_source_pairs.items())),
        "bridge_relation_counts_by_disease": {
            relation: dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
            for relation, counts in bridge_relation_counts_by_disease.items()
        },
        "candidate_summary_by_relation_and_disease": {
            relation: {
                disease: dict(sorted(counts.items()))
                for disease, counts in per_disease.items()
            }
            for relation, per_disease in candidate_summary.items()
        },
        "candidate_summary_pan_cancer": pan_cancer_summary,
        "candidate_examples": {
            relation: dict(per_disease)
            for relation, per_disease in candidate_examples.items()
        },
        "hcc_crc_focus": hcc_crc_focus,
    }


def build_cross_source_logic_edges(descriptors: list[Descriptor]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    rule_counts: Counter[str] = Counter()

    support_groups: dict[tuple[Any, ...], list[Descriptor]] = defaultdict(list)
    contradiction_groups: dict[tuple[Any, ...], list[Descriptor]] = defaultdict(list)
    refine_groups: dict[tuple[Any, ...], list[Descriptor]] = defaultdict(list)
    complement_groups: dict[tuple[Any, ...], list[Descriptor]] = defaultdict(list)

    for descriptor in descriptors:
        if descriptor.state:
            support_groups[(descriptor.anchor_key, descriptor.disease_key, descriptor.state, descriptor.anchor_type)].append(descriptor)
            refine_groups[(descriptor.anchor_key, descriptor.disease_key, descriptor.state, descriptor.anchor_type)].append(descriptor)
        family = comparable_family(descriptor.state)
        if family:
            contradiction_groups[(descriptor.anchor_key, descriptor.disease_key, family, descriptor.anchor_type)].append(descriptor)
        if descriptor.mechanism_bucket:
            complement_groups[(descriptor.anchor_key, descriptor.disease_key, descriptor.mechanism_bucket, descriptor.anchor_type)].append(descriptor)

    for group_records in support_groups.values():
        best_by_source: dict[str, Descriptor] = {}
        for descriptor in group_records:
            current = best_by_source.get(descriptor.source_name)
            if current is None or descriptor_rank(descriptor) > descriptor_rank(current):
                best_by_source[descriptor.source_name] = descriptor
        anchors = sorted(best_by_source.values(), key=descriptor_rank, reverse=True)
        if len(anchors) < 2:
            continue
        anchor = anchors[0]
        for descriptor in anchors[1:]:
            edge = build_edge(
                source_id=descriptor.evidence_id,
                target_id=anchor.evidence_id,
                relation="SUPPORTS",
                family="evidence_logic",
                rule="multisource_logic:same_state_support",
                rationale="Different source families converge on the same normalized anchor, disease, and directional state.",
                support_strength=round(min(descriptor.confidence_score, anchor.confidence_score), 4),
                similarity_score=0.9,
                evidence_count=2,
            )
            add_edge_once(edges, seen, edge)
            rule_counts["same_state_support"] += 1

    for group_records in contradiction_groups.values():
        best_by_state_source: dict[tuple[str, str], Descriptor] = {}
        for descriptor in group_records:
            if descriptor.state is None:
                continue
            key = (descriptor.state, descriptor.source_name)
            current = best_by_state_source.get(key)
            if current is None or descriptor_rank(descriptor) > descriptor_rank(current):
                best_by_state_source[key] = descriptor
        states = defaultdict(list)
        for (state, _), descriptor in best_by_state_source.items():
            states[state].append(descriptor)
        if len(states) < 2:
            continue
        ordered_states = sorted(states)
        left = sorted(states[ordered_states[0]], key=descriptor_rank, reverse=True)[0]
        right = sorted(states[ordered_states[1]], key=descriptor_rank, reverse=True)[0]
        if left.source_name == right.source_name:
            continue
        source_id, target_id = sorted([left.evidence_id, right.evidence_id])
        edge = build_edge(
            source_id=source_id,
            target_id=target_id,
            relation="CONTRADICTS",
            family="evidence_logic",
            rule="multisource_logic:opposite_state_contradiction",
            rationale="Different source families support opposing directional states for the same normalized anchor and disease claim family.",
            support_strength=round(min(left.confidence_score, right.confidence_score), 4),
            similarity_score=0.86,
            evidence_count=2,
        )
        add_edge_once(edges, seen, edge)
        rule_counts["opposite_state_contradiction"] += 1

    for group_records in refine_groups.values():
        broad = [descriptor for descriptor in group_records if descriptor.specificity == 0]
        narrow = [descriptor for descriptor in group_records if descriptor.specificity > 0]
        if not broad or not narrow:
            continue
        for descriptor in sorted(narrow, key=descriptor_rank, reverse=True):
            cross_source_broad = [candidate for candidate in broad if candidate.source_name != descriptor.source_name]
            if not cross_source_broad:
                continue
            broad_anchor = sorted(cross_source_broad, key=descriptor_rank, reverse=True)[0]
            edge = build_edge(
                source_id=descriptor.evidence_id,
                target_id=broad_anchor.evidence_id,
                relation="REFINES",
                family="evidence_logic",
                rule="multisource_logic:narrow_refines_broad",
                rationale="A narrower subgroup or context-conditioned evidence unit sharpens a broader cross-source claim with the same normalized anchor and disease.",
                support_strength=round(min(descriptor.confidence_score, broad_anchor.confidence_score), 4),
                similarity_score=0.88,
                evidence_count=2,
            )
            add_edge_once(edges, seen, edge)
            rule_counts["narrow_refines_broad"] += 1

    for group_records in complement_groups.values():
        by_signature: dict[tuple[str, str, str], Descriptor] = {}
        for descriptor in group_records:
            signature = (descriptor.source_name, descriptor.relation_group, descriptor.state or "unknown")
            current = by_signature.get(signature)
            if current is None or descriptor_rank(descriptor) > descriptor_rank(current):
                by_signature[signature] = descriptor
        candidates = sorted(by_signature.values(), key=descriptor_rank, reverse=True)
        if len(candidates) < 2:
            continue
        anchor = candidates[0]
        for descriptor in candidates[1:]:
            if descriptor.state == anchor.state:
                continue
            if descriptor.source_name == anchor.source_name:
                continue
            edge = build_edge(
                source_id=descriptor.evidence_id,
                target_id=anchor.evidence_id,
                relation="COMPLEMENTS",
                family="evidence_logic",
                rule="multisource_logic:mechanism_complement",
                rationale="Different source families or assay types contribute mechanistically complementary evidence for the same normalized anchor and disease.",
                support_strength=round(min(descriptor.confidence_score, anchor.confidence_score), 4),
                similarity_score=0.8,
                evidence_count=2,
            )
            add_edge_once(edges, seen, edge)
            rule_counts["mechanism_complement"] += 1

    return edges, dict(sorted(rule_counts.items()))


def integrated_cohort_id(disease_key: str) -> str:
    return f"COHORT-INTEGRATED-{stable_hash(disease_key)}"


def aggregate_confidence(member_nodes: list[dict[str, Any]], member_sources: set[str], member_layers: set[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    member_scores = [float((node.get("confidence") or {}).get("score") or 0.0) for node in member_nodes]
    mean_score = mean(member_scores) if member_scores else 0.0
    adjusted = min(0.99, mean_score + 0.03 * max(len(member_sources) - 1, 0) + 0.02 * max(len(member_layers) - 1, 0))
    grade = "A" if adjusted >= 0.85 else "B" if adjusted >= 0.7 else "C" if adjusted >= 0.55 else "D"
    confidence = {
        "grade": grade,
        "score": round(adjusted, 4),
        "study_quality_score": round(min(1.0, mean_score + 0.1), 4),
        "statistical_support_score": round(min(1.0, mean_score + 0.05), 4),
        "source_reliability_score": round(min(1.0, 0.75 + 0.05 * len(member_sources)), 4),
        "sample_size_score": round(min(1.0, sum(int((node.get("statistics") or {}).get("sample_size") or 0) for node in member_nodes) / 500.0), 4),
        "extraction_confidence": round(adjusted, 4),
        "ds_belief": round(min(0.95, adjusted * 0.86), 4),
        "ds_plausibility": round(min(1.0, adjusted * 0.86 + 0.12), 4),
        "ds_ignorance": 0.12,
        "contradiction_risk": 0.0,
        "data_completeness": round(mean(float((node.get("confidence") or {}).get("data_completeness") or 0.8) for node in member_nodes), 4) if member_nodes else 0.8,
        "review_status": "pending",
    }
    stats = {
        "p_value": None,
        "q_value": None,
        "fold_change": None,
        "confidence_interval": None,
        "sample_size": len(member_nodes),
        "effect_size": round(mean_score, 4) if member_scores else None,
        "hazard_ratio": None,
        "mutation_frequency": None,
        "ceres_score": None,
        "codependency_coefficient": None,
        "synergy_score": None,
        "statistical_method": "integrated evidence aggregation across normalized member evidence",
    }
    return confidence, stats


def support_flags(member_nodes: list[dict[str, Any]]) -> dict[str, bool]:
    return {
        "supports_target_discovery": any((node.get("support") or {}).get("supports_target_discovery") for node in member_nodes),
        "supports_mechanism_inference": any((node.get("support") or {}).get("supports_mechanism_inference") for node in member_nodes),
        "supports_drug_repositioning": any((node.get("support") or {}).get("supports_drug_repositioning") for node in member_nodes),
    }


def aggregate_predicate(level: str, mechanism_bucket: str | None, representative_predicate: str) -> tuple[str, str, str]:
    if level == "same_claim_aggregate":
        return representative_predicate, "integrated_claim_summary", "integrated"
    mapping = {
        "oncogenic_support": ("integrated_supports_oncogenic_mechanism_in", "evidence_convergence", "integrated"),
        "protective_support": ("integrated_supports_protective_mechanism_in", "evidence_convergence", "integrated"),
        "therapeutic_benefit": ("integrated_supports_therapeutic_benefit_in", "evidence_convergence", "integrated"),
        "codependency": ("integrated_supports_codependency_mechanism_in", "evidence_convergence", "integrated"),
    }
    return mapping.get(mechanism_bucket or "", ("integrated_supports_multisource_mechanism_in", "evidence_convergence", "integrated"))


def make_aggregate_node(
    *,
    aggregate_id: str,
    level: str,
    member_nodes: list[dict[str, Any]],
    member_descriptors: list[Descriptor],
    child_ids: list[str],
    mechanism_bucket: str | None,
) -> dict[str, Any]:
    representative = sorted(member_nodes, key=evidence_rank, reverse=True)[0]
    descriptor = sorted(member_descriptors, key=lambda item: (item.confidence_score, item.sample_size, item.evidence_id), reverse=True)[0]
    member_sources = {node["source_family"] for node in member_nodes}
    member_layers = {node.get("feature_layer") or "unknown" for node in member_nodes}
    confidence, statistics = aggregate_confidence(member_nodes, member_sources, member_layers)
    predicate, metric, direction = aggregate_predicate(level, mechanism_bucket, descriptor.predicate)
    cohort_id = integrated_cohort_id(descriptor.disease_key or descriptor.disease_label)
    summary = (
        f"Integrated aggregate for {descriptor.anchor_label} in {descriptor.disease_label} "
        f"combining {len(member_nodes)} evidence units across {len(member_sources)} source families."
    )
    if level == "cross_omics_mechanism_aggregate" and mechanism_bucket:
        summary = (
            f"Cross-omics mechanism aggregate for {descriptor.anchor_label} in {descriptor.disease_label} "
            f"capturing {mechanism_bucket.replace('_', ' ')} across {len(member_layers)} layers and {len(member_sources)} sources."
        )
    return {
        "node_type": "evidence",
        "id": aggregate_id,
        "record_id": aggregate_id,
        "source_family": "integrated",
        "disease": descriptor.disease_label,
        "cohort_id": cohort_id,
        "subject": descriptor.subject_ref,
        "object": descriptor.object_ref,
        "outcome": descriptor.outcome_ref,
        "predicate": predicate,
        "direction": direction,
        "polarity": representative.get("polarity"),
        "claim_scope": representative.get("claim_scope"),
        "hypothesis_level": representative.get("hypothesis_level"),
        "support": support_flags(member_nodes),
        "evidence_category": "aggregated_same_claim" if level == "same_claim_aggregate" else "aggregated_cross_omics_mechanism",
        "feature_layer": "same_claim_aggregate" if level == "same_claim_aggregate" else "cross_omics_mechanism_aggregate",
        "aggregation_level": level,
        "summary": summary,
        "raw_text": None,
        "metric": metric,
        "value": len(member_nodes),
        "unit": "evidence_count",
        "comparison": {
            "sample_grouping": "integrated_multisource_aggregation",
            "comparison_type": level,
            "comparison_label": f"{len(member_nodes)} member evidence units",
            "group_a": "member_evidence_units",
            "group_b": None,
            "n_total": len(member_nodes),
            "n_effective": len(member_nodes),
            "n_group_a": len(member_nodes),
            "n_group_b": None,
        },
        "statistics": statistics,
        "confidence": confidence,
        "citation": None,
        "source_context": {
            "member_evidence_ids": child_ids,
            "member_source_families": sorted(member_sources),
            "member_feature_layers": sorted(member_layers),
            "member_predicates": sorted({node["predicate"] for node in member_nodes}),
            "member_count": len(member_nodes),
            "aggregation_scope": level,
            "mechanism_bucket": mechanism_bucket,
        },
        "reviewer_comments": None,
        "created_at": now_iso(),
        "updated_at": None,
        "version": "1.0.0",
        "provenance": {
            "source_family": "integrated",
            "source_dataset": "multisource_evidence_graph",
            "source_release": "latest",
            "source_record_id": aggregate_id,
            "source_file": "kg/aligned/{pubmed,tcga,depmap}_*_normalized_latest.json",
            "source_url": None,
            "citation_doi": None,
            "citation_pmid": None,
            "platform": None,
            "assay": level,
            "sequencing_or_screen_id": None,
            "batch_id": None,
            "analysis_pipeline": "multisource_integrated_graph_v1",
            "extractor_name": "scripts/build_multisource_integrated_evidence_graph.py",
            "extractor_version": "1.0.0",
            "extracted_at": now_iso(),
            "study_design": "integrated_multisource_synthesis",
            "clinical_stage": "integrated_context",
            "licence": None,
            "is_primary_observation": False,
            "source_priority": 0,
        },
        "aggregation_parent_ids": child_ids,
    }


def aggregate_descriptors_from_node(node: dict[str, Any], mechanism_bucket: str | None) -> Descriptor:
    return Descriptor(
        evidence_id=node["id"],
        source_name="integrated",
        source_family="integrated",
        subject_ref=node["subject"],
        object_ref=node.get("object"),
        outcome_ref=node.get("outcome"),
        disease_key=node_disease_ref(node)["id"] if node_disease_ref(node) else None,
        disease_label=node["disease"],
        anchor_key=node["subject"]["id"],
        anchor_label=node["subject"]["label"],
        anchor_type=node["subject"]["entity_type"],
        target_key=(node.get("object") or {}).get("id") if (node.get("object") or {}).get("entity_type") != "Disease" else (node.get("outcome") or {}).get("id"),
        relation_group=node["predicate"],
        state=None,
        mechanism_bucket=mechanism_bucket,
        specificity=0,
        predicate=node["predicate"],
        feature_layer=node["feature_layer"],
        evidence_category=node["evidence_category"],
        confidence_score=float((node.get("confidence") or {}).get("score") or 0.0),
        sample_size=int((node.get("statistics") or {}).get("sample_size") or 0),
        created_at=node["created_at"],
        supports_target_discovery=bool((node.get("support") or {}).get("supports_target_discovery")),
        supports_mechanism_inference=bool((node.get("support") or {}).get("supports_mechanism_inference")),
        supports_drug_repositioning=bool((node.get("support") or {}).get("supports_drug_repositioning")),
    )


def build_aggregate_layers(
    evidence_nodes: list[dict[str, Any]],
    descriptors: list[Descriptor],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    evidence_by_id = {node["id"]: node for node in evidence_nodes}
    same_claim_groups: dict[tuple[Any, ...], list[Descriptor]] = defaultdict(list)
    for descriptor in descriptors:
        if descriptor.state is None:
            continue
        exact_key = (
            descriptor.anchor_key,
            descriptor.disease_key,
            descriptor.relation_group,
            descriptor.state,
            descriptor.target_key,
            descriptor.predicate,
        )
        same_claim_groups[exact_key].append(descriptor)

    aggregate_nodes: list[dict[str, Any]] = []
    trace_edges: list[dict[str, Any]] = []
    same_claim_descriptor_by_id: dict[str, Descriptor] = {}
    member_to_same_claim: dict[str, str] = {}
    stats: dict[str, Any] = {"same_claim_count": 0, "cross_omics_count": 0, "trace_edge_count": 0}

    for key, group_descriptors in same_claim_groups.items():
        if len(group_descriptors) < 2:
            continue
        member_ids = sorted({descriptor.evidence_id for descriptor in group_descriptors})
        member_nodes = [evidence_by_id[member_id] for member_id in member_ids]
        aggregate_id = f"AGGCLAIM-{stable_hash('|'.join(key_part or 'null' for key_part in map(str, key)))}"
        aggregate_node = make_aggregate_node(
            aggregate_id=aggregate_id,
            level="same_claim_aggregate",
            member_nodes=member_nodes,
            member_descriptors=group_descriptors,
            child_ids=member_ids,
            mechanism_bucket=group_descriptors[0].mechanism_bucket,
        )
        aggregate_nodes.append(aggregate_node)
        same_claim_descriptor_by_id[aggregate_id] = aggregate_descriptors_from_node(aggregate_node, group_descriptors[0].mechanism_bucket)
        stats["same_claim_count"] += 1
        for member_id in member_ids:
            member_to_same_claim[member_id] = aggregate_id
            trace_edges.append(
                build_edge(
                    source_id=aggregate_id,
                    target_id=member_id,
                    relation="DERIVED_FROM",
                    family="aggregation_trace",
                    rule="aggregation_trace:same_claim_to_atomic",
                    rationale="This same-claim aggregate is derived from aligned atomic evidence members with the same claim signature.",
                    support_strength=None,
                    similarity_score=1.0,
                    evidence_count=len(member_ids),
                )
            )

    mechanism_groups: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    child_descriptor_lookup: dict[str, Descriptor] = {}
    covered_atomic = set(member_to_same_claim)
    for aggregate_id, descriptor in same_claim_descriptor_by_id.items():
        if descriptor.mechanism_bucket is None or descriptor.anchor_key is None or descriptor.disease_key is None:
            continue
        mechanism_groups[(descriptor.anchor_key, descriptor.disease_key, descriptor.mechanism_bucket, descriptor.anchor_type)].append(aggregate_id)
        child_descriptor_lookup[aggregate_id] = descriptor

    for descriptor in descriptors:
        if descriptor.mechanism_bucket is None or descriptor.anchor_key is None or descriptor.disease_key is None:
            continue
        if descriptor.evidence_id in covered_atomic:
            continue
        mechanism_groups[(descriptor.anchor_key, descriptor.disease_key, descriptor.mechanism_bucket, descriptor.anchor_type)].append(descriptor.evidence_id)
        child_descriptor_lookup[descriptor.evidence_id] = descriptor

    aggregate_lookup = {node["id"]: node for node in aggregate_nodes}
    for key, child_ids in mechanism_groups.items():
        unique_child_ids = sorted(set(child_ids))
        if len(unique_child_ids) < 2:
            continue
        child_descriptors = [child_descriptor_lookup[child_id] for child_id in unique_child_ids]
        child_sources = {descriptor.source_name for descriptor in child_descriptors}
        child_layers = {descriptor.feature_layer for descriptor in child_descriptors}
        child_groups = {descriptor.relation_group for descriptor in child_descriptors}
        if len(child_sources) < 2 and len(child_layers) < 2 and len(child_groups) < 2:
            continue

        member_nodes = [aggregate_lookup[child_id] if child_id in aggregate_lookup else evidence_by_id[child_id] for child_id in unique_child_ids]
        aggregate_id = f"AGGMECH-{stable_hash('|'.join(map(str, key)))}"
        aggregate_node = make_aggregate_node(
            aggregate_id=aggregate_id,
            level="cross_omics_mechanism_aggregate",
            member_nodes=member_nodes,
            member_descriptors=child_descriptors,
            child_ids=unique_child_ids,
            mechanism_bucket=key[2],
        )
        aggregate_nodes.append(aggregate_node)
        stats["cross_omics_count"] += 1
        for child_id in unique_child_ids:
            trace_edges.append(
                build_edge(
                    source_id=aggregate_id,
                    target_id=child_id,
                    relation="DERIVED_FROM",
                    family="aggregation_trace",
                    rule="aggregation_trace:mechanism_to_child",
                    rationale="This cross-omics mechanism aggregate is derived from lower-level claim aggregates or atomic evidence that support the same integrated mechanism bucket.",
                    support_strength=None,
                    similarity_score=0.9,
                    evidence_count=len(unique_child_ids),
                )
            )

    stats["trace_edge_count"] = len(trace_edges)
    return aggregate_nodes, trace_edges, stats


def aggregate_semantic_anchor_edges(
    aggregate_nodes: list[dict[str, Any]],
    cohort_lookup: set[str],
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for node in aggregate_nodes:
        for ref in [node["subject"], node.get("object"), node.get("outcome")]:
            if not ref or not ref.get("id"):
                continue
            edge = build_edge(
                source_id=node["id"],
                target_id=ref["id"],
                relation="LINKED_TO",
                family="semantic_anchor",
                rule="integrated_aggregate:link_entity",
                rationale="Integrated aggregate node linked to its canonical entity anchor.",
                support_strength=None,
                similarity_score=1.0,
                evidence_count=1,
            )
            add_edge_once(edges, seen, edge)
        if node["cohort_id"] in cohort_lookup:
            edge = build_edge(
                source_id=node["id"],
                target_id=node["cohort_id"],
                relation="OBSERVED_IN",
                family="semantic_anchor",
                rule="integrated_aggregate:observed_in_integrated_cohort",
                rationale="Integrated aggregate node observed in a disease-level integrated evidence context.",
                support_strength=None,
                similarity_score=None,
                evidence_count=1,
            )
            add_edge_once(edges, seen, edge)
    return edges


def basic_validate_graph(graph: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    node_ids = [node["id"] for node in graph["nodes"]]
    if len(node_ids) != len(set(node_ids)):
        errors.append("duplicate node ids")
    known_node_ids = set(node_ids)
    for edge in graph["edges"]:
        if edge["relation"] not in ALLOWED_EDGE_RELATIONS:
            errors.append(f"invalid edge relation {edge['relation']}")
        if edge["source"] not in known_node_ids or edge["target"] not in known_node_ids:
            errors.append(f"orphan edge {edge}")
    return errors


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


def build_stats(
    graph: dict[str, Any],
    cross_source_rule_counts: dict[str, int],
    bridge_payload: dict[str, Any],
    aggregate_stats: dict[str, Any],
    evidence_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    node_counts = Counter(node["node_type"] for node in graph["nodes"])
    evidence_nodes = [node for node in graph["nodes"] if node.get("node_type") == "evidence"]
    edge_relation_counts = Counter(edge["relation"] for edge in graph["edges"])
    edge_family_counts = Counter(edge["relation_family"] for edge in graph["edges"])
    evidence_source_counts = Counter(node["source_family"] for node in evidence_nodes)
    aggregate_levels = Counter(node["aggregation_level"] for node in evidence_nodes if node["source_family"] == "integrated")
    cross_source_relation_counts = Counter()
    cross_source_source_pairs = Counter()
    by_disease = defaultdict(Counter)
    for edge in graph["edges"]:
        if edge["relation_family"] != "evidence_logic":
            continue
        source_node = evidence_by_id.get(edge["source"])
        target_node = evidence_by_id.get(edge["target"])
        if not source_node or not target_node:
            continue
        disease = source_node.get("disease") or target_node.get("disease") or "unknown"
        by_disease[edge["relation"]][disease] += 1
        if source_node["source_family"] != target_node["source_family"]:
            cross_source_relation_counts[edge["relation"]] += 1
            pair = "|".join(sorted({source_node["source_family"], target_node["source_family"]}))
            cross_source_source_pairs[pair] += 1

    reasonableness = {}
    for relation, counts in by_disease.items():
        total = sum(counts.values()) or 1
        top_disease, top_count = counts.most_common(1)[0]
        reasonableness[relation] = {
            "top_disease": top_disease,
            "top_disease_share": round(top_count / total, 4),
            "assessment": "distributed" if top_count / total <= 0.55 else "concentrated_but_plausible",
        }

    return {
        "generated_at": now_iso(),
        "graph_id": graph["graph_metadata"]["graph_id"],
        "node_counts": dict(sorted(node_counts.items())),
        "edge_relation_counts": dict(sorted(edge_relation_counts.items())),
        "edge_family_counts": dict(sorted(edge_family_counts.items())),
        "evidence_source_counts": dict(sorted(evidence_source_counts.items())),
        "aggregate_level_counts": dict(sorted(aggregate_levels.items())),
        "cross_source_rule_counts": cross_source_rule_counts,
        "bridge_rule_counts": bridge_payload["bridge_rule_counts"],
        "bridge_relation_counts": bridge_payload["bridge_relation_counts"],
        "bridge_relation_counts_by_disease": bridge_payload["bridge_relation_counts_by_disease"],
        "bridge_source_pairs": bridge_payload["bridge_source_pairs"],
        "bridge_candidate_summary": bridge_payload["candidate_summary_pan_cancer"],
        "bridge_hcc_crc_focus": bridge_payload["hcc_crc_focus"],
        "cross_source_relation_counts": dict(sorted(cross_source_relation_counts.items())),
        "cross_source_source_pairs": dict(sorted(cross_source_source_pairs.items())),
        "aggregation_stats": aggregate_stats,
        "edge_node_ratio": round(len(graph["edges"]) / len(graph["nodes"]), 4),
        "relation_counts_by_disease": {
            relation: dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:20]) for relation, counts in by_disease.items()
        },
        "distribution_reasonableness": reasonableness,
    }


def build_audit(graph: dict[str, Any], evidence_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    validation_errors = schema_validate(graph)
    duplicate_edges = Counter((edge["source"], edge["target"], edge["relation"]) for edge in graph["edges"])
    duplicate_edge_examples = [key for key, count in duplicate_edges.items() if count > 1][:20]
    duplicate_edge_count = sum(1 for count in duplicate_edges.values() if count > 1)
    orphan_edges = [
        edge for edge in graph["edges"]
        if edge["source"] not in evidence_by_id and edge["source"] not in {node["id"] for node in graph["nodes"]}
    ]
    cross_source_errors = []
    bridge_relation_errors = []
    derivation_errors = []
    aggregate_ids = {node["id"] for node in graph["nodes"] if node.get("node_type") == "evidence" and node.get("source_family") == "integrated"}
    derived_targets = defaultdict(int)
    for edge in graph["edges"]:
        if edge["relation_family"] == "evidence_logic":
            source = evidence_by_id.get(edge["source"])
            target = evidence_by_id.get(edge["target"])
            if source and target and edge["relation"] in {"SUPPORTS", "CONTRADICTS", "REFINES", "COMPLEMENTS"}:
                if source["source_family"] == target["source_family"] and edge["created_by_rule"].startswith("multisource_logic:"):
                    cross_source_errors.append(edge)
            if source and target and edge["relation"] in BRIDGE_RELATIONS:
                if source["source_family"] != "pubmed_literature" or target["source_family"] not in {"tcga_linkedomics", "depmap"}:
                    bridge_relation_errors.append(edge)
        if edge["relation"] == "DERIVED_FROM":
            derived_targets[edge["source"]] += 1
    for aggregate_id in aggregate_ids:
        if derived_targets[aggregate_id] == 0:
            derivation_errors.append(aggregate_id)

    return {
        "generated_at": now_iso(),
        "graph_id": graph["graph_metadata"]["graph_id"],
        "checks": {
            "schema_validation": {
                "validator": "jsonschema" if jsonschema is not None else "basic_validation",
                "passed": not validation_errors,
                "error_count": len(validation_errors),
                "errors": validation_errors,
            },
            "duplicate_edge_keys": {
                "count": duplicate_edge_count,
                "examples": duplicate_edge_examples,
            },
            "orphan_edge_examples": {
                "count": len(orphan_edges),
                "examples": orphan_edges[:10],
            },
            "cross_source_relation_source_mismatch": {
                "count": len(cross_source_errors),
                "examples": cross_source_errors[:10],
            },
            "bridge_relation_source_mismatch": {
                "count": len(bridge_relation_errors),
                "examples": bridge_relation_errors[:10],
            },
            "aggregate_trace_missing": {
                "count": len(derivation_errors),
                "examples": derivation_errors[:20],
            },
        },
    }


def main() -> None:
    source_graphs = {name: read_json(path) for name, path in SOURCE_PATHS.items()}
    entity_nodes, original_to_unified_entity = merge_entity_nodes(source_graphs)

    all_evidence_nodes: list[dict[str, Any]] = []
    all_cohort_nodes: list[dict[str, Any]] = []
    all_edges: list[dict[str, Any]] = []
    descriptors: list[Descriptor] = []
    for source_name, graph in source_graphs.items():
        evidence_nodes, cohort_nodes, edges, local_descriptors = convert_source_graph(graph, source_name, original_to_unified_entity)
        all_evidence_nodes.extend(evidence_nodes)
        all_cohort_nodes.extend(cohort_nodes)
        all_edges.extend(edges)
        descriptors.extend(local_descriptors)

    integrated_cohorts = build_integrated_cohorts(descriptors)
    cohort_nodes = sorted({node["id"]: node for node in all_cohort_nodes + integrated_cohorts}.values(), key=lambda item: item["id"])
    cohort_lookup = {node["id"] for node in cohort_nodes}

    cross_source_edges, cross_source_rule_counts = build_cross_source_logic_edges(descriptors)
    bridge_edges, bridge_payload = build_bridge_edges(descriptors)
    aggregate_nodes, trace_edges, aggregate_stats = build_aggregate_layers(all_evidence_nodes, descriptors)
    aggregate_anchor_edges = aggregate_semantic_anchor_edges(aggregate_nodes, cohort_lookup)

    all_evidence = sorted(all_evidence_nodes + aggregate_nodes, key=lambda item: (item["source_family"], item["id"]))
    nodes = sorted(entity_nodes + cohort_nodes + all_evidence, key=lambda item: (item["node_type"], item["id"]))
    edges = dedupe_edges(all_edges + cross_source_edges + bridge_edges + trace_edges + aggregate_anchor_edges)

    graph = {
        "schema_version": "2.0.0",
        "graph_metadata": {
            "graph_id": "multisource_evidence_graph_latest",
            "graph_name": "Integrated multi-source evidence knowledge graph with cross-source logic and aggregation",
            "created_by": "scripts/build_multisource_integrated_evidence_graph.py",
            "created_at": now_iso(),
            "source_families": ["pubmed_literature", "tcga_linkedomics", "depmap", "integrated"],
            "disease_scope": sorted({descriptor.disease_label for descriptor in descriptors}),
            "build_parameters": {
                "source_graphs": {name: str(path.relative_to(ROOT)) for name, path in SOURCE_PATHS.items()},
                "cross_source_rules": cross_source_rule_counts,
                "bridge_rules": bridge_payload["bridge_rule_counts"],
                "aggregate_stats": aggregate_stats,
            },
            "notes": "Integrated graph rewires evidence to canonical entity nodes, adds cross-source logic edges, high-value PubMed-to-TCGA/DepMap bridge edges, same-claim aggregates, cross-omics mechanism aggregates, and aggregation trace edges.",
        },
        "nodes": nodes,
        "edges": edges,
    }

    evidence_by_id = {node["id"]: node for node in all_evidence}
    stats = build_stats(graph, cross_source_rule_counts, bridge_payload, aggregate_stats, evidence_by_id)
    audit = build_audit(graph, evidence_by_id)
    write_json(OUTPUT_GRAPH_PATH, graph)
    write_json(OUTPUT_STATS_PATH, stats)
    write_json(OUTPUT_AUDIT_PATH, audit)
    write_json(OUTPUT_BRIDGE_CANDIDATES_PATH, bridge_payload)


if __name__ == "__main__":
    main()

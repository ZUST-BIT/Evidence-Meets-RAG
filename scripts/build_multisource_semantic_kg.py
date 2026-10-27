from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INPUT_GRAPH_PATH = ROOT / "kg" / "aligned" / "multisource_evidence_graph_complete.json"
OUTPUT_GRAPH_PATH = ROOT / "kg" / "aligned" / "multisource_semantic_kg.json"
OUTPUT_STATS_PATH = ROOT / "kg" / "aligned" / "multisource_semantic_kg_stats.json"
OUTPUT_SCHEMA_PATH = ROOT / "kg" / "schema" / "multisource_semantic_kg_schema_v1.json"

ALLOWED_ATOMIC_SOURCES = {"tcga_linkedomics", "depmap", "pubmed_literature"}
HCC_CRC_FAMILIES = {"HCC", "CRC"}
SUPPORTED_ENTITY_TYPES = {"Gene", "Disease", "Drug", "Phenotype", "Pathway", "Cohort", "MolecularFeature"}

DEPENDENCY_PREDICATES = {
    "dependency_in",
    "selective_dependency_in",
    "high_expression_associated_with_dependency_in",
    "copy_number_gain_associated_with_dependency_in",
    "mutation_associated_with_dependency_in",
    "subtype_associated_with_dependency_in",
    "phenotype_associated_with_dependency_in",
    "metastatic_state_associated_with_dependency_in",
}
ALTERATION_PREDICATES = {
    "amplified_in",
    "deleted_in",
    "hypermethylated_in",
    "hypomethylated_in",
    "mutated_in",
    "hypophosphorylated_in",
    "hyperphosphorylated_in",
}
PROMOTE_PREDICATES = {"promotes_mechanism", "integrated_supports_oncogenic_mechanism_in"}
SUPPRESS_PREDICATES = {"suppresses_mechanism", "integrated_supports_protective_mechanism_in"}

DISEASE_ALIASES = {
    "HCC": {"hcc", "hepatocellular carcinoma", "liver hepatocellular carcinoma", "lihc", "liver cancer"},
    "CRC": {"crc", "colorectal cancer", "colorectal adenocarcinoma", "coadread", "coad", "read", "colon cancer", "rectal cancer"},
    "NSCLC": {"nsclc", "non small cell lung cancer", "luad", "lusc", "lung adenocarcinoma", "lung squamous cell carcinoma"},
    "BRCA": {"brca", "breast carcinoma", "breast cancer"},
    "GLIOMA": {"glioma", "gbm", "glioblastoma", "glioblastoma multiforme"},
    "OVARIAN": {"ovarian", "ovarian cancer", "ovarian carcinoma", "ov"},
    "RCC": {"rcc", "renal cell carcinoma", "kidney renal clear cell carcinoma", "kirc"},
    "AML": {"aml", "acute myeloid leukemia", "laml"},
    "PAAD": {"paad", "pancreatic adenocarcinoma", "pancreatic cancer"},
    "HNSC": {"hnsc", "head and neck squamous cell carcinoma"},
    "GASTROESOPH": {"gastroesoph", "gastroesophageal adenocarcinoma", "esophagogastric carcinoma", "esophageal adenocarcinoma", "stomach adenocarcinoma", "gastric cancer", "stad"},
}

DISEASE_LABELS = {
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

FEATURE_RELATION_LABELS = {
    "UPREGULATED_IN": "upregulation signal",
    "DOWNREGULATED_IN": "downregulation signal",
    "ALTERED_IN": "molecular alteration signal",
    "ASSOCIATED_WITH_POOR_OUTCOME": "poor-outcome association",
    "ASSOCIATED_WITH_FAVORABLE_OUTCOME": "favorable-outcome association",
    "DEPENDENCY_IN": "dependency signal",
    "SENSITIVE_TO": "drug-sensitivity signal",
    "PROMOTES": "mechanism-promotion signal",
    "SUPPRESSES": "mechanism-suppression signal",
    "CODEPENDENT_WITH": "codependency signal",
    "PARTICIPATES_IN": "pathway-membership signal",
}


def stable_hash(value: str, length: int = 16) -> str:
    import hashlib

    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length].upper()


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    value = str(value).strip().lower()
    if not value:
        return ""
    value = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def unique_preserve(values: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def canonical_disease_family(value: str | None) -> str:
    normalized = norm_text(value)
    for family, aliases in DISEASE_ALIASES.items():
        if normalized in aliases:
            return family
    return str(value or "").upper()


def semantic_entity_type(entity_type: str | None) -> str | None:
    value = str(entity_type or "")
    if value in {"Gene", "Protein"}:
        return "Gene"
    if value == "Disease":
        return "Disease"
    if value == "Drug":
        return "Drug"
    if value == "Pathway":
        return "Pathway"
    if value == "Cohort":
        return "Cohort"
    if value == "MolecularFeature":
        return "MolecularFeature"
    if value in {"Phenotype", "CellType", "CellLine", "Tissue", "Other"}:
        return "Phenotype"
    return None


def semantic_node_id(ref: dict[str, Any], semantic_type: str) -> str:
    preferred = str(ref.get("id") or "").strip()
    if preferred:
        return f"SKG-{stable_hash(f'{semantic_type}|{preferred}')}"
    label = str(ref.get("label") or "").strip()
    return f"SKG-{stable_hash(f'{semantic_type}|{label}')}"


def semantic_label(ref: dict[str, Any]) -> str:
    return str(ref.get("label") or "").strip()


def semantic_ref(ref: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(ref, dict):
        return None
    label = semantic_label(ref)
    semantic_type = semantic_entity_type(ref.get("entity_type"))
    if not label or semantic_type not in SUPPORTED_ENTITY_TYPES:
        return None
    return {
        "id": semantic_node_id(ref, semantic_type),
        "canonical_name": label,
        "entity_type": semantic_type,
        "source_id": str(ref.get("id") or "").strip() or None,
    }


def disease_ref_from_family(family: str) -> dict[str, Any] | None:
    family = str(family or "").upper()
    if family not in DISEASE_LABELS:
        return None
    return {
        "id": f"SKG-{stable_hash(f'Disease|{family}')}",
        "canonical_name": DISEASE_LABELS[family],
        "entity_type": "Disease",
        "source_id": f"CANCER_FAMILY:{family}",
    }


def cohort_ref_from_node(node: dict[str, Any]) -> dict[str, Any] | None:
    label = str(node.get("cohort_label") or node.get("disease_name") or node.get("id") or "").strip()
    if not label:
        return None
    source_id = str(node.get("id") or "").strip() or None
    return {
        "id": f"SKG-{stable_hash(f'Cohort|{source_id or label}')}",
        "canonical_name": label,
        "entity_type": "Cohort",
        "source_id": source_id,
    }


def node_disease_families(node: dict[str, Any]) -> list[str]:
    values = [node.get("disease"), (node.get("source_context") or {}).get("disease_name")]
    values.extend((node.get("source_context") or {}).get("primary_diseases") or [])
    values.extend((node.get("source_context") or {}).get("oncotree_codes") or [])
    families = []
    for value in values:
        family = canonical_disease_family(str(value or ""))
        if family in DISEASE_ALIASES:
            families.append(family)
    return unique_preserve(families)


def member_source_families(node: dict[str, Any]) -> list[str]:
    source_family = str(node.get("source_family") or "")
    if source_family == "integrated":
        members = [str(value) for value in ((node.get("source_context") or {}).get("member_source_families") or [])]
        return unique_preserve([value for value in members if value in ALLOWED_ATOMIC_SOURCES])
    if source_family in ALLOWED_ATOMIC_SOURCES:
        return [source_family]
    return []


def node_support_weight(node: dict[str, Any]) -> int:
    context = node.get("source_context") or {}
    if str(node.get("source_family") or "") == "integrated":
        return max(int(context.get("member_count") or 1), 1)
    return 1


def relation_payload(node: dict[str, Any]) -> tuple[str, str, dict[str, Any] | None] | None:
    predicate = str(node.get("predicate") or "")
    bucket = str(((node.get("source_context") or {}).get("mechanism_bucket")) or "")
    subject = semantic_ref(node.get("subject"))
    obj = semantic_ref(node.get("object"))
    outcome = semantic_ref(node.get("outcome"))
    if not subject:
        return None
    if predicate == "upregulated_in" and obj:
        return "UPREGULATED_IN", "omics_association", obj
    if predicate == "downregulated_in" and obj:
        return "DOWNREGULATED_IN", "omics_association", obj
    if predicate in ALTERATION_PREDICATES and obj:
        return "ALTERED_IN", "omics_association", obj
    if "poor_prognosis" in predicate and obj:
        return "ASSOCIATED_WITH_POOR_OUTCOME", "outcome_association", obj
    if "favorable_prognosis" in predicate and obj:
        return "ASSOCIATED_WITH_FAVORABLE_OUTCOME", "outcome_association", obj
    if predicate in DEPENDENCY_PREDICATES and obj:
        return "DEPENDENCY_IN", "dependency", obj
    if predicate == "sensitive_to" and obj:
        return "SENSITIVE_TO", "drug_response", obj
    if predicate == "codependent_with" and obj:
        return "CODEPENDENT_WITH", "dependency", obj
    if predicate in PROMOTE_PREDICATES:
        return "PROMOTES", "mechanism", outcome or obj
    if predicate in SUPPRESS_PREDICATES:
        return "SUPPRESSES", "mechanism", outcome or obj
    if predicate == "supports_mechanistic_claim":
        if bucket == "oncogenic_support":
            return "PROMOTES", "mechanism", outcome or obj
        if bucket == "protective_support":
            return "SUPPRESSES", "mechanism", outcome or obj
    return None


def relation_statement(source_name: str, relation_type: str, target_name: str, edge_meta: dict[str, Any]) -> str:
    disease_context = ", ".join(edge_meta["disease_families"]) if edge_meta["disease_families"] else "multi-cancer context"
    original_predicates = ", ".join(edge_meta["predicate_counts"].keys())
    if relation_type == "UPREGULATED_IN":
        return f"{source_name} is upregulated in {target_name}."
    if relation_type == "DOWNREGULATED_IN":
        return f"{source_name} is downregulated in {target_name}."
    if relation_type == "ALTERED_IN":
        return f"{source_name} is altered in {target_name}. Original signals: {original_predicates}."
    if relation_type == "ASSOCIATED_WITH_POOR_OUTCOME":
        return f"{source_name} is associated with poor outcome in {target_name}."
    if relation_type == "ASSOCIATED_WITH_FAVORABLE_OUTCOME":
        return f"{source_name} is associated with favorable outcome in {target_name}."
    if relation_type == "DEPENDENCY_IN":
        return f"{source_name} is implicated as a dependency in {target_name}."
    if relation_type == "SENSITIVE_TO":
        return f"{source_name} is sensitive to {target_name}."
    if relation_type == "CODEPENDENT_WITH":
        return f"{source_name} is codependent with {target_name} in {disease_context}."
    if relation_type == "PROMOTES":
        return f"{source_name} promotes {target_name} in {disease_context}."
    if relation_type == "SUPPRESSES":
        return f"{source_name} suppresses {target_name} in {disease_context}."
    if relation_type == "PARTICIPATES_IN":
        return f"{source_name} participates in {target_name} in {disease_context}."
    if relation_type == "HAS_FEATURE":
        return f"{source_name} has feature {target_name}."
    if relation_type == "FEATURE_TARGETS":
        return f"{source_name} targets or summarizes {target_name}."
    if relation_type == "OBSERVED_IN":
        return f"{source_name} is observed in {target_name}."
    if relation_type == "HAS_COHORT":
        return f"{source_name} has cohort context {target_name}."
    return f"{source_name} {relation_type.lower()} {target_name}."


def confidence_score(support_count: int, source_family_count: int) -> float:
    score = 0.4 + min(support_count, 6) * 0.08 + min(max(source_family_count - 1, 0), 2) * 0.12
    return round(min(score, 0.98), 4)


def semantic_pathway_targets(graph: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    nodes_by_id = {node["id"]: node for node in graph["nodes"]}
    pathway_hits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in graph["edges"]:
        source_id = edge.get("source")
        target_id = edge.get("target")
        source_node = nodes_by_id.get(source_id)
        target_node = nodes_by_id.get(target_id)
        if not source_node or not target_node:
            continue
        pairs = ((source_node, target_node), (target_node, source_node))
        for evidence_node, entity_node in pairs:
            if evidence_node.get("node_type") != "evidence":
                continue
            if entity_node.get("node_type") != "entity" or entity_node.get("entity_type") != "Pathway":
                continue
            pathway_ref = {
                "id": entity_node.get("id"),
                "label": entity_node.get("name"),
                "entity_type": "Pathway",
            }
            semantic_pathway = semantic_ref(pathway_ref)
            if semantic_pathway:
                pathway_hits[evidence_node["id"]].append(semantic_pathway)
    return {key: value for key, value in pathway_hits.items()}


def semantic_cohort_targets(graph: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    nodes_by_id = {node["id"]: node for node in graph["nodes"]}
    cohort_hits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in graph["edges"]:
        source_id = edge.get("source")
        target_id = edge.get("target")
        source_node = nodes_by_id.get(source_id)
        target_node = nodes_by_id.get(target_id)
        if not source_node or not target_node:
            continue
        pairs = ((source_node, target_node), (target_node, source_node))
        for evidence_node, cohort_node in pairs:
            if evidence_node.get("node_type") != "evidence":
                continue
            if cohort_node.get("node_type") != "cohort":
                continue
            cohort_ref = cohort_ref_from_node(cohort_node)
            if cohort_ref:
                cohort_hits[evidence_node["id"]].append(cohort_ref)
    normalized: dict[str, list[dict[str, Any]]] = {}
    for evidence_id, refs in cohort_hits.items():
        seen = set()
        ordered = []
        for ref in refs:
            if ref["id"] in seen:
                continue
            seen.add(ref["id"])
            ordered.append(ref)
        normalized[evidence_id] = ordered
    return normalized


def should_use_evidence_node(node: dict[str, Any]) -> bool:
    if node.get("node_type") != "evidence":
        return False
    source_family = str(node.get("source_family") or "")
    predicate = str(node.get("predicate") or "")
    if source_family == "integrated":
        members = set(member_source_families(node))
        return bool(members) and members.issubset(ALLOWED_ATOMIC_SOURCES)
    if source_family not in ALLOWED_ATOMIC_SOURCES:
        return False
    return predicate in DEPENDENCY_PREDICATES | {"sensitive_to", "codependent_with", "promotes_mechanism", "suppresses_mechanism"}


def register_node(
    nodes: dict[str, dict[str, Any]],
    ref: dict[str, Any],
    *,
    source_families: list[str],
    disease_families: list[str],
) -> None:
    node = nodes.setdefault(
        ref["id"],
        {
            "id": ref["id"],
            "node_type": ref["entity_type"],
            "canonical_name": ref["canonical_name"],
            "aliases": [],
            "source_ids": [],
            "source_families": [],
            "disease_families": [],
        },
    )
    if ref.get("source_id"):
        node["source_ids"] = unique_preserve(node["source_ids"] + [ref["source_id"]])
    node["source_families"] = unique_preserve(node["source_families"] + source_families)
    node["disease_families"] = unique_preserve(node["disease_families"] + disease_families)


def register_disease_context_nodes(nodes: dict[str, dict[str, Any]], disease_families: list[str], source_families: list[str]) -> list[dict[str, Any]]:
    refs = []
    for family in disease_families:
        ref = disease_ref_from_family(family)
        if not ref:
            continue
        register_node(nodes, ref, source_families=source_families, disease_families=[family])
        refs.append(ref)
    return refs


def add_semantic_edge(
    edges: list[dict[str, Any]],
    seen: set[tuple[str, str, str]],
    *,
    source: str,
    target: str,
    relation_type: str,
    relation_group: str,
    source_families: list[str],
    disease_families: list[str],
    predicate_counts: dict[str, int],
    support_count: int,
    supporting_evidence_ids: list[str],
    supporting_statement_examples: list[str],
) -> None:
    key = (source, target, relation_type)
    if key in seen:
        return
    seen.add(key)
    edges.append(
        {
            "source": source,
            "target": target,
            "relation_type": relation_type,
            "relation_group": relation_group,
            "source_families": source_families,
            "disease_families": disease_families,
            "predicate_counts": predicate_counts,
            "support_count": support_count,
            "supporting_evidence_ids": supporting_evidence_ids,
            "supporting_statement_examples": supporting_statement_examples,
        }
    )


def feature_node_ref(source_node: dict[str, Any], target_node: dict[str, Any], edge_bucket: dict[str, Any]) -> dict[str, Any]:
    relation_type = str(edge_bucket["relation_type"])
    disease_context = ", ".join(edge_bucket["disease_families"]) if edge_bucket["disease_families"] else "multi-cancer"
    label = f"{source_node['canonical_name']} {FEATURE_RELATION_LABELS.get(relation_type, relation_type.lower())} :: {target_node['canonical_name']} :: {disease_context}"
    feature_key = "|".join(["MolecularFeature", edge_bucket["source"], edge_bucket["target"], relation_type, disease_context])
    return {
        "id": f"SKG-{stable_hash(feature_key)}",
        "canonical_name": label,
        "entity_type": "MolecularFeature",
        "source_id": None,
    }


def main() -> None:
    graph = read_json(INPUT_GRAPH_PATH)
    pathway_targets = semantic_pathway_targets(graph)
    cohort_targets = semantic_cohort_targets(graph)
    nodes: dict[str, dict[str, Any]] = {}
    edge_buckets: dict[tuple[str, str, str], dict[str, Any]] = {}
    selected_evidence_count = 0
    skipped_no_relation = 0
    source_counts = Counter()

    for node in graph["nodes"]:
        if not should_use_evidence_node(node):
            continue
        payload = relation_payload(node)
        if payload is None:
            skipped_no_relation += 1
            continue
        relation_type, relation_group, target = payload
        if target is None:
            skipped_no_relation += 1
            continue
        subject = semantic_ref(node.get("subject"))
        if not subject:
            skipped_no_relation += 1
            continue
        disease_families = node_disease_families(node)
        source_families = member_source_families(node)
        if not source_families:
            skipped_no_relation += 1
            continue
        selected_evidence_count += 1
        source_counts.update(source_families)
        register_node(nodes, subject, source_families=source_families, disease_families=disease_families)
        register_node(nodes, target, source_families=source_families, disease_families=disease_families)

        support_weight = node_support_weight(node)
        edge_key = (subject["id"], target["id"], relation_type)
        bucket = edge_buckets.setdefault(
            edge_key,
            {
                "source": subject["id"],
                "target": target["id"],
                "relation_type": relation_type,
                "relation_group": relation_group,
                "source_families": [],
                "disease_families": [],
                "predicate_counts": Counter(),
                "support_count": 0,
                "supporting_evidence_ids": [],
                "supporting_statement_examples": [],
                "cohort_refs": [],
            },
        )
        bucket["source_families"] = unique_preserve(bucket["source_families"] + source_families)
        bucket["disease_families"] = unique_preserve(bucket["disease_families"] + disease_families)
        bucket["predicate_counts"][str(node.get("predicate") or "")] += support_weight
        bucket["support_count"] += support_weight
        if len(bucket["supporting_evidence_ids"]) < 16:
            bucket["supporting_evidence_ids"].append(str(node.get("id") or ""))
        statement = str(node.get("summary") or node.get("raw_text") or "").strip()
        if statement and statement not in bucket["supporting_statement_examples"] and len(bucket["supporting_statement_examples"]) < 6:
            bucket["supporting_statement_examples"].append(statement)
        cohort_candidates = list(cohort_targets.get(str(node.get("id") or ""), []))
        for member_id in (node.get("source_context") or {}).get("member_evidence_ids") or []:
            cohort_candidates.extend(cohort_targets.get(str(member_id), []))
        seen_cohort_ids = {ref["id"] for ref in bucket["cohort_refs"]}
        for cohort_ref in cohort_candidates:
            if cohort_ref["id"] in seen_cohort_ids:
                continue
            seen_cohort_ids.add(cohort_ref["id"])
            bucket["cohort_refs"].append(cohort_ref)

        if subject["entity_type"] == "Gene":
            pathway_candidates = list(pathway_targets.get(str(node.get("id") or ""), []))
            for member_id in (node.get("source_context") or {}).get("member_evidence_ids") or []:
                pathway_candidates.extend(pathway_targets.get(str(member_id), []))
            seen_pathway_ids = set()
            for pathway in pathway_candidates:
                pathway_id = str(pathway.get("id") or "")
                if not pathway_id or pathway_id in seen_pathway_ids:
                    continue
                seen_pathway_ids.add(pathway_id)
                register_node(nodes, pathway, source_families=source_families, disease_families=disease_families)
                pathway_key = (subject["id"], pathway["id"], "PARTICIPATES_IN")
                pathway_bucket = edge_buckets.setdefault(
                    pathway_key,
                    {
                        "source": subject["id"],
                        "target": pathway["id"],
                        "relation_type": "PARTICIPATES_IN",
                        "relation_group": "pathway",
                        "source_families": [],
                        "disease_families": [],
                        "predicate_counts": Counter(),
                        "support_count": 0,
                        "supporting_evidence_ids": [],
                        "supporting_statement_examples": [],
                        "cohort_refs": [],
                    },
                )
                pathway_bucket["source_families"] = unique_preserve(pathway_bucket["source_families"] + source_families)
                pathway_bucket["disease_families"] = unique_preserve(pathway_bucket["disease_families"] + disease_families)
                pathway_bucket["predicate_counts"][str(node.get("predicate") or "")] += support_weight
                pathway_bucket["support_count"] += support_weight
                if len(pathway_bucket["supporting_evidence_ids"]) < 16:
                    pathway_bucket["supporting_evidence_ids"].append(str(node.get("id") or ""))
                seen_pathway_cohort_ids = {ref["id"] for ref in pathway_bucket["cohort_refs"]}
                for cohort_ref in cohort_candidates:
                    if cohort_ref["id"] in seen_pathway_cohort_ids:
                        continue
                    seen_pathway_cohort_ids.add(cohort_ref["id"])
                    pathway_bucket["cohort_refs"].append(cohort_ref)

    edges = []
    supplemental_edges = []
    relation_counts = Counter()
    supplemental_relation_counts = Counter()
    seen_supplemental_edges: set[tuple[str, str, str]] = set()
    for bucket in edge_buckets.values():
        source_node = nodes[bucket["source"]]
        target_node = nodes[bucket["target"]]
        bucket["predicate_counts"] = dict(sorted(bucket["predicate_counts"].items()))
        bucket["confidence_score"] = confidence_score(bucket["support_count"], len(bucket["source_families"]))
        bucket["statement_text"] = relation_statement(source_node["canonical_name"], bucket["relation_type"], target_node["canonical_name"], bucket)
        edges.append(bucket)
        relation_counts[bucket["relation_type"]] += 1

        feature_ref = feature_node_ref(source_node, target_node, bucket)
        register_node(nodes, feature_ref, source_families=bucket["source_families"], disease_families=bucket["disease_families"])
        feature_node = nodes[feature_ref["id"]]
        feature_node["feature_relation_type"] = bucket["relation_type"]
        feature_node["feature_relation_group"] = bucket["relation_group"]
        feature_node["support_count"] = bucket["support_count"]
        feature_node["predicate_counts"] = bucket["predicate_counts"]
        feature_node["supporting_evidence_ids"] = bucket["supporting_evidence_ids"]
        feature_node["supporting_statement_examples"] = bucket["supporting_statement_examples"]
        feature_node["cohort_ids"] = [ref["id"] for ref in bucket["cohort_refs"]]
        add_semantic_edge(
            supplemental_edges,
            seen_supplemental_edges,
            source=source_node["id"],
            target=feature_ref["id"],
            relation_type="HAS_FEATURE",
            relation_group="feature_projection",
            source_families=bucket["source_families"],
            disease_families=bucket["disease_families"],
            predicate_counts=bucket["predicate_counts"],
            support_count=bucket["support_count"],
            supporting_evidence_ids=bucket["supporting_evidence_ids"],
            supporting_statement_examples=bucket["supporting_statement_examples"],
        )
        add_semantic_edge(
            supplemental_edges,
            seen_supplemental_edges,
            source=feature_ref["id"],
            target=target_node["id"],
            relation_type="FEATURE_TARGETS",
            relation_group="feature_projection",
            source_families=bucket["source_families"],
            disease_families=bucket["disease_families"],
            predicate_counts=bucket["predicate_counts"],
            support_count=bucket["support_count"],
            supporting_evidence_ids=bucket["supporting_evidence_ids"],
            supporting_statement_examples=bucket["supporting_statement_examples"],
        )

        disease_refs = register_disease_context_nodes(nodes, bucket["disease_families"], bucket["source_families"])
        for disease_ref in disease_refs:
            add_semantic_edge(
                supplemental_edges,
                seen_supplemental_edges,
                source=feature_ref["id"],
                target=disease_ref["id"],
                relation_type="OBSERVED_IN",
                relation_group="context",
                source_families=bucket["source_families"],
                disease_families=bucket["disease_families"],
                predicate_counts=bucket["predicate_counts"],
                support_count=bucket["support_count"],
                supporting_evidence_ids=bucket["supporting_evidence_ids"],
                supporting_statement_examples=bucket["supporting_statement_examples"],
            )
        for cohort_ref in bucket["cohort_refs"]:
            register_node(nodes, cohort_ref, source_families=bucket["source_families"], disease_families=bucket["disease_families"])
            add_semantic_edge(
                supplemental_edges,
                seen_supplemental_edges,
                source=feature_ref["id"],
                target=cohort_ref["id"],
                relation_type="OBSERVED_IN",
                relation_group="context",
                source_families=bucket["source_families"],
                disease_families=bucket["disease_families"],
                predicate_counts=bucket["predicate_counts"],
                support_count=bucket["support_count"],
                supporting_evidence_ids=bucket["supporting_evidence_ids"],
                supporting_statement_examples=bucket["supporting_statement_examples"],
            )
            add_semantic_edge(
                supplemental_edges,
                seen_supplemental_edges,
                source=source_node["id"],
                target=cohort_ref["id"],
                relation_type="OBSERVED_IN",
                relation_group="context",
                source_families=bucket["source_families"],
                disease_families=bucket["disease_families"],
                predicate_counts=bucket["predicate_counts"],
                support_count=bucket["support_count"],
                supporting_evidence_ids=bucket["supporting_evidence_ids"],
                supporting_statement_examples=bucket["supporting_statement_examples"],
            )
            for disease_ref in disease_refs:
                add_semantic_edge(
                    supplemental_edges,
                    seen_supplemental_edges,
                    source=disease_ref["id"],
                    target=cohort_ref["id"],
                    relation_type="HAS_COHORT",
                    relation_group="context",
                    source_families=bucket["source_families"],
                    disease_families=bucket["disease_families"],
                    predicate_counts=bucket["predicate_counts"],
                    support_count=bucket["support_count"],
                    supporting_evidence_ids=bucket["supporting_evidence_ids"],
                    supporting_statement_examples=bucket["supporting_statement_examples"],
                )

    for edge in supplemental_edges:
        edge["confidence_score"] = confidence_score(edge["support_count"], len(edge["source_families"]))
        source_node = nodes[edge["source"]]
        target_node = nodes[edge["target"]]
        edge["statement_text"] = relation_statement(source_node["canonical_name"], edge["relation_type"], target_node["canonical_name"], edge)
        supplemental_relation_counts[edge["relation_type"]] += 1

    for node in nodes.values():
        node["source_ids"] = unique_preserve(node["source_ids"])
        node["source_families"] = unique_preserve(node["source_families"])
        node["disease_families"] = unique_preserve(node["disease_families"])
        if "cohort_ids" in node:
            node["cohort_ids"] = unique_preserve(node["cohort_ids"])

    all_edges = edges + supplemental_edges

    payload = {
        "schema_version": "1.0.0",
        "graph_metadata": {
            "graph_id": "multisource_semantic_kg",
            "graph_name": "Same-source relation-centric Semantic KG projected from the multi-source evidence graph",
            "created_by": "scripts/build_multisource_semantic_kg.py",
            "created_from": str(INPUT_GRAPH_PATH.relative_to(ROOT)),
            "source_families": sorted(ALLOWED_ATOMIC_SOURCES),
            "disease_scope": sorted(DISEASE_ALIASES),
            "entity_types": ["Gene", "Disease", "Drug", "Phenotype", "Pathway", "Cohort", "MolecularFeature"],
            "relation_types": [
                "UPREGULATED_IN",
                "DOWNREGULATED_IN",
                "ALTERED_IN",
                "ASSOCIATED_WITH_POOR_OUTCOME",
                "ASSOCIATED_WITH_FAVORABLE_OUTCOME",
                "DEPENDENCY_IN",
                "SENSITIVE_TO",
                "PROMOTES",
                "SUPPRESSES",
                "CODEPENDENT_WITH",
                "PARTICIPATES_IN",
                "HAS_FEATURE",
                "FEATURE_TARGETS",
                "OBSERVED_IN",
                "HAS_COHORT",
            ],
            "notes": [
                "Semantic KG keeps only canonical entity nodes and aggregated semantic relations.",
                "Evidence nodes, observation nodes, and evidence-trace relations are intentionally removed to create a traditional relation-centric baseline.",
                "Integrated evidence nodes are preferred; atomic source nodes are additionally used to preserve classic cohort and dependency context.",
                "Feature nodes and cohort nodes expand the relation-centric structure while staying within a traditional KG design.",
            ],
        },
        "nodes": sorted(nodes.values(), key=lambda item: (item["node_type"], item["canonical_name"], item["id"])),
        "edges": sorted(all_edges, key=lambda item: (item["relation_type"], item["source"], item["target"])),
    }
    stats = {
        "node_count": len(payload["nodes"]),
        "edge_count": len(payload["edges"]),
        "selected_evidence_count": selected_evidence_count,
        "skipped_no_relation": skipped_no_relation,
        "entity_type_counts": dict(sorted(Counter(node["node_type"] for node in payload["nodes"]).items())),
        "relation_type_counts": dict(sorted((relation_counts + supplemental_relation_counts).items())),
        "base_relation_type_counts": dict(sorted(relation_counts.items())),
        "supplemental_relation_type_counts": dict(sorted(supplemental_relation_counts.items())),
        "source_family_support_counts": dict(sorted(source_counts.items())),
        "hcc_crc_edge_count": sum(1 for edge in payload["edges"] if set(edge["disease_families"]).intersection(HCC_CRC_FAMILIES)),
    }
    write_json(OUTPUT_GRAPH_PATH, payload)
    write_json(OUTPUT_STATS_PATH, stats)
    print(json.dumps({"graph_path": str(OUTPUT_GRAPH_PATH), "stats_path": str(OUTPUT_STATS_PATH), **stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

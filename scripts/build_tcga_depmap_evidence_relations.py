from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ALIGNED_DIR = ROOT / "kg" / "aligned"

GRAPH_TARGETS = {
    "tcga": {
        "raw": ALIGNED_DIR / "tcga_atomic_compact_latest.json",
        "normalized": ALIGNED_DIR / "tcga_atomic_compact_normalized_latest.json",
        "stats": ALIGNED_DIR / "tcga_atomic_compact_relation_stats_latest.json",
        "audit": ALIGNED_DIR / "tcga_atomic_compact_relation_audit_latest.json",
    },
    "depmap": {
        "raw": ALIGNED_DIR / "depmap_atomic_compact_latest.json",
        "normalized": ALIGNED_DIR / "depmap_atomic_compact_normalized_latest.json",
        "stats": ALIGNED_DIR / "depmap_atomic_compact_relation_stats_latest.json",
        "audit": ALIGNED_DIR / "depmap_atomic_compact_relation_audit_latest.json",
    },
}

SUMMARY_PATH = ALIGNED_DIR / "tcga_depmap_evidence_relation_summary_latest.json"

BROAD_TCGA_GROUPINGS = {"pan_cancer_background", "cohort_recurrence", "overall_survival_association"}
NARROW_TCGA_GROUPINGS = {"tumor_purity_quantiles", "pathologic_stage"}

TCGA_PREDICATE_MAP: dict[str, tuple[str, str]] = {
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
}

TCGA_REGULATION_STATE = {
    "expression_regulation": {"positive": "activated_program", "negative": "suppressed_program"},
    "copy_number_regulation": {"positive": "activated_program", "negative": "suppressed_program"},
    "methylation_regulation": {"positive": "suppressed_program", "negative": "activated_program"},
    "protein_activation_regulation": {"positive": "activated_program", "negative": "suppressed_program"},
}

TCGA_PROGNOSIS_GROUPS = {
    "expression_prognosis",
    "methylation_prognosis",
    "copy_number_amplification_prognosis",
    "copy_number_deletion_prognosis",
    "mutation_prognosis",
    "protein_abundance_prognosis",
}

DEPMAP_DEPENDENCY_SPECIFICITY = {
    "dependency_in": 0,
    "selective_dependency_in": 1,
    "high_expression_associated_with_dependency_in": 2,
    "copy_number_gain_associated_with_dependency_in": 2,
    "mutation_associated_with_dependency_in": 2,
    "subtype_associated_with_dependency_in": 2,
    "phenotype_associated_with_dependency_in": 2,
    "metastatic_state_associated_with_dependency_in": 2,
}


@dataclass
class EvidenceRecord:
    evidence_id: str
    source_name: str
    subject_key: str | None
    disease_key: str | None
    target_key: str | None
    target_label: str | None
    predicate: str
    relation_group: str
    sign: str
    sample_grouping: str | None
    specificity: int
    confidence_score: float
    sample_size: int
    evidence_category: str | None
    feature_layer: str | None
    aggregation_level: str | None
    comparison_type: str | None
    summary: str | None


def now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def entity_key(ref: dict[str, Any] | None) -> str | None:
    if not ref:
        return None
    return ref.get("normalized_id") or ref.get("id") or ref.get("label")


def disease_key(node: dict[str, Any]) -> str | None:
    subject = node.get("subject") or {}
    obj = node.get("object") or {}
    outcome = node.get("outcome") or {}
    if subject.get("entity_type") == "Disease":
        return entity_key(subject)
    if outcome.get("entity_type") == "Disease":
        return entity_key(outcome)
    if obj.get("entity_type") == "Disease":
        return entity_key(obj)
    return node.get("disease")


def target_key(node: dict[str, Any]) -> str | None:
    subject = node.get("subject") or {}
    obj = node.get("object") or {}
    outcome = node.get("outcome") or {}
    if subject.get("entity_type") == "Disease":
        return entity_key(obj) or entity_key(outcome)
    if obj.get("entity_type") != "Disease":
        return entity_key(obj)
    if outcome.get("entity_type") != "Disease":
        return entity_key(outcome)
    return None


def target_label(node: dict[str, Any]) -> str | None:
    subject = node.get("subject") or {}
    obj = node.get("object") or {}
    outcome = node.get("outcome") or {}
    if subject.get("entity_type") == "Disease":
        return obj.get("normalized_label") or obj.get("label") or outcome.get("normalized_label") or outcome.get("label")
    if obj.get("entity_type") != "Disease":
        return obj.get("normalized_label") or obj.get("label")
    if outcome.get("entity_type") != "Disease":
        return outcome.get("normalized_label") or outcome.get("label")
    return None


def make_record(node: dict[str, Any], source_name: str) -> EvidenceRecord | None:
    if node.get("node_type") != "evidence":
        return None
    subject = node.get("subject") or {}
    predicate = node.get("predicate")
    if not predicate:
        return None

    if source_name == "tcga":
        relation_group, sign = TCGA_PREDICATE_MAP.get(predicate, (predicate, node.get("direction") or "unknown"))
        sample_grouping = (node.get("comparison") or {}).get("sample_grouping")
        specificity = 1 if sample_grouping in NARROW_TCGA_GROUPINGS else 0
    else:
        sample_grouping = (node.get("comparison") or {}).get("sample_grouping")
        if predicate in DEPMAP_DEPENDENCY_SPECIFICITY:
            relation_group = "dependency_core"
            sign = "positive"
            specificity = DEPMAP_DEPENDENCY_SPECIFICITY[predicate]
        elif predicate == "sensitive_to":
            relation_group = "drug_sensitivity"
            sign = "positive"
            specificity = 0
        elif predicate == "codependent_with":
            relation_group = "gene_codependency"
            sign = "positive"
            specificity = 0
        else:
            relation_group = predicate
            sign = node.get("direction") or "unknown"
            specificity = 0

    return EvidenceRecord(
        evidence_id=node["id"],
        source_name=source_name,
        subject_key=entity_key(subject),
        disease_key=disease_key(node),
        target_key=target_key(node),
        target_label=target_label(node),
        predicate=predicate,
        relation_group=relation_group,
        sign=sign,
        sample_grouping=sample_grouping,
        specificity=specificity,
        confidence_score=float((node.get("confidence") or {}).get("score") or 0.0),
        sample_size=int(
            (node.get("statistics") or {}).get("sample_size")
            or (node.get("comparison") or {}).get("n_effective")
            or 0
        ),
        evidence_category=node.get("evidence_category"),
        feature_layer=node.get("feature_layer"),
        aggregation_level=node.get("aggregation_level"),
        comparison_type=(node.get("comparison") or {}).get("comparison_type"),
        summary=node.get("summary"),
    )


def rank_tuple(record: EvidenceRecord) -> tuple[float, int, str]:
    return (record.confidence_score, record.sample_size, record.evidence_id)


def sorted_by_rank(records: list[EvidenceRecord], *, reverse: bool = True) -> list[EvidenceRecord]:
    return sorted(records, key=rank_tuple, reverse=reverse)


def build_edge(
    source: EvidenceRecord,
    target: EvidenceRecord,
    relation: str,
    rule: str,
    rationale: str,
    similarity_score: float | None,
) -> dict[str, Any]:
    return {
        "source": source.evidence_id,
        "target": target.evidence_id,
        "relation": relation,
        "relation_family": "evidence_logic",
        "entity_type": None,
        "linking_score": None,
        "similarity_score": similarity_score,
        "rationale": rationale,
        "support_strength": round(min(source.confidence_score, target.confidence_score), 4),
        "evidence_count": 2,
        "created_by_rule": rule,
    }


def tcga_contradiction_comparable(left: EvidenceRecord, right: EvidenceRecord) -> bool:
    if left.specificity != right.specificity:
        return False
    if left.sample_grouping == right.sample_grouping:
        return True
    return (
        left.sample_grouping in BROAD_TCGA_GROUPINGS
        and right.sample_grouping in BROAD_TCGA_GROUPINGS
    )


def add_edge_once(edge_list: list[dict[str, Any]], seen: set[tuple[str, str, str]], edge: dict[str, Any]) -> None:
    key = (edge["source"], edge["target"], edge["relation"])
    if key in seen:
        return
    seen.add(key)
    edge_list.append(edge)


def support_context_bucket(record: EvidenceRecord) -> str | None:
    if record.specificity == 0:
        return "broad"
    return record.sample_grouping


def tcga_support_state(record: EvidenceRecord) -> str | None:
    if record.relation_group in TCGA_REGULATION_STATE:
        return TCGA_REGULATION_STATE[record.relation_group].get(record.sign)
    if record.relation_group in TCGA_PROGNOSIS_GROUPS:
        return "favorable_outcome" if record.sign == "positive" else "poor_outcome"
    return None


def add_tcga_cross_layer_supports(
    records: list[EvidenceRecord],
    edges: list[dict[str, Any]],
    seen: set[tuple[str, str, str]],
) -> int:
    groups: dict[tuple[Any, ...], list[EvidenceRecord]] = defaultdict(list)
    for record in records:
        state = tcga_support_state(record)
        context_bucket = support_context_bucket(record)
        if state is None or context_bucket is None:
            continue
        groups[(record.subject_key, record.disease_key, state, context_bucket)].append(record)

    added = 0
    for group_records in groups.values():
        best_by_relation_group: dict[str, EvidenceRecord] = {}
        for record in group_records:
            current = best_by_relation_group.get(record.relation_group)
            if current is None or rank_tuple(record) > rank_tuple(current):
                best_by_relation_group[record.relation_group] = record
        anchors = sorted(best_by_relation_group.values(), key=rank_tuple, reverse=True)
        if len(anchors) < 2:
            continue
        parent = anchors[0]
        for child in anchors[1:]:
            if child.relation_group == parent.relation_group:
                continue
            edge = build_edge(
                source=child,
                target=parent,
                relation="SUPPORTS",
                rule="structured_relation:tcga:cross_layer_coherent_support",
                rationale=(
                    "These TCGA evidence units come from different omics layers but converge on the same normalized "
                    "gene-disease state with directionally coherent biology."
                ),
                similarity_score=0.82,
            )
            add_edge_once(edges, seen, edge)
            added += 1
    return added


def build_tcga_edges(records: list[EvidenceRecord]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    exact_groups: dict[tuple[Any, ...], list[EvidenceRecord]] = defaultdict(list)
    contradiction_groups: dict[tuple[Any, ...], list[EvidenceRecord]] = defaultdict(list)
    refine_groups: dict[tuple[Any, ...], list[EvidenceRecord]] = defaultdict(list)

    for record in records:
        exact_groups[(record.subject_key, record.disease_key, record.target_key, record.predicate)].append(record)
        contradiction_groups[(record.subject_key, record.disease_key, record.target_key, record.relation_group)].append(record)
        refine_groups[(record.subject_key, record.disease_key, record.target_key, record.relation_group, record.sign)].append(record)

    support_count = 0
    contradiction_count = 0
    refine_count = 0

    for group_records in exact_groups.values():
        if len(group_records) < 2:
            continue
        broad_records = [record for record in group_records if record.specificity == 0]
        narrow_records = [record for record in group_records if record.specificity > 0]
        broad_anchor = sorted_by_rank(broad_records)[0] if broad_records else None

        if broad_anchor is not None:
            for record in narrow_records:
                edge = build_edge(
                    source=record,
                    target=broad_anchor,
                    relation="REFINES",
                    rule="structured_relation:tcga:subgroup_refines_general",
                    rationale=(
                        f"{record.sample_grouping or 'subgroup'} evidence narrows the same TCGA claim represented by a broader "
                        f"{broad_anchor.sample_grouping or 'cohort-level'} signal."
                    ),
                    similarity_score=0.95,
                )
                add_edge_once(edges, seen, edge)
                refine_count += 1

        support_anchor_pool = broad_records if broad_records else group_records
        support_anchor = sorted_by_rank(support_anchor_pool)[0]
        for record in group_records:
            if record.evidence_id == support_anchor.evidence_id:
                continue
            if record.specificity > support_anchor.specificity and broad_anchor is not None:
                continue
            edge = build_edge(
                source=record,
                target=support_anchor,
                relation="SUPPORTS",
                rule="structured_relation:tcga:exact_proposition_support",
                rationale="These TCGA evidence units share the same normalized subject, disease, predicate, and directional claim.",
                similarity_score=1.0,
            )
            add_edge_once(edges, seen, edge)
            support_count += 1

    for group_records in contradiction_groups.values():
        sign_buckets: dict[str, list[EvidenceRecord]] = defaultdict(list)
        for record in group_records:
            sign_buckets[record.sign].append(record)
        if len(sign_buckets) < 2:
            continue
        ordered_signs = sorted(sign_buckets.keys())
        anchors = {sign: sorted_by_rank(sign_buckets[sign])[0] for sign in ordered_signs}
        for left_index, left_sign in enumerate(ordered_signs):
            for right_sign in ordered_signs[left_index + 1 :]:
                left_anchor = anchors[left_sign]
                right_anchor = anchors[right_sign]
                if not tcga_contradiction_comparable(left_anchor, right_anchor):
                    continue
                source, target = sorted([left_anchor, right_anchor], key=rank_tuple)
                edge = build_edge(
                    source=source,
                    target=target,
                    relation="CONTRADICTS",
                    rule="structured_relation:tcga:opposite_direction_contradiction",
                    rationale="These TCGA evidence units share the same normalized subject and disease claim family but encode opposite directional outcomes.",
                    similarity_score=0.9,
                )
                add_edge_once(edges, seen, edge)
                contradiction_count += 1

    cross_layer_support_count = add_tcga_cross_layer_supports(records, edges, seen)
    support_count += cross_layer_support_count

    return edges, {
        "support_count": support_count,
        "cross_layer_support_count": cross_layer_support_count,
        "contradiction_count": contradiction_count,
        "refine_count": refine_count,
    }


def add_depmap_cross_context_supports(
    dependency_groups: dict[tuple[Any, ...], list[EvidenceRecord]],
    edges: list[dict[str, Any]],
    seen: set[tuple[str, str, str]],
) -> int:
    added = 0
    for group_records in dependency_groups.values():
        context_records = [record for record in group_records if record.specificity == 2]
        best_by_feature: dict[str, EvidenceRecord] = {}
        for record in context_records:
            feature_key = record.feature_layer or record.predicate
            current = best_by_feature.get(feature_key)
            if current is None or rank_tuple(record) > rank_tuple(current):
                best_by_feature[feature_key] = record
        anchors = sorted(best_by_feature.values(), key=rank_tuple, reverse=True)
        if len(anchors) < 2:
            continue
        parent = anchors[0]
        for child in anchors[1:]:
            edge = build_edge(
                source=child,
                target=parent,
                relation="SUPPORTS",
                rule="structured_relation:depmap:cross_context_support",
                rationale=(
                    "These DepMap context-conditioned evidence units arise from different context modalities but converge "
                    "on the same lineage-specific dependency signal for the same gene."
                ),
                similarity_score=0.8,
            )
            add_edge_once(edges, seen, edge)
            added += 1
    return added


def build_depmap_edges(records: list[EvidenceRecord]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    exact_groups: dict[tuple[Any, ...], list[EvidenceRecord]] = defaultdict(list)
    dependency_groups: dict[tuple[Any, ...], list[EvidenceRecord]] = defaultdict(list)
    contradiction_groups: dict[tuple[Any, ...], list[EvidenceRecord]] = defaultdict(list)

    for record in records:
        exact_groups[(record.subject_key, record.disease_key, record.target_key, record.predicate)].append(record)
        contradiction_groups[(record.subject_key, record.disease_key, record.target_key, record.relation_group)].append(record)
        if record.relation_group == "dependency_core":
            dependency_groups[(record.subject_key, record.disease_key)].append(record)

    support_count = 0
    contradiction_count = 0
    refine_count = 0

    for group_records in exact_groups.values():
        if len(group_records) < 2:
            continue
        support_anchor = sorted_by_rank(group_records)[0]
        for record in group_records:
            if record.evidence_id == support_anchor.evidence_id:
                continue
            edge = build_edge(
                source=record,
                target=support_anchor,
                relation="SUPPORTS",
                rule="structured_relation:depmap:exact_proposition_support",
                rationale="These DepMap evidence units share the same normalized proposition signature and reinforce the same lineage-level functional claim.",
                similarity_score=1.0,
            )
            add_edge_once(edges, seen, edge)
            support_count += 1

    for group_records in dependency_groups.values():
        by_specificity: dict[int, list[EvidenceRecord]] = defaultdict(list)
        for record in group_records:
            by_specificity[record.specificity].append(record)

        dependency_anchor = sorted_by_rank(by_specificity.get(0, []))[0] if by_specificity.get(0) else None
        selective_anchor = sorted_by_rank(by_specificity.get(1, []))[0] if by_specificity.get(1) else None

        if dependency_anchor is not None and selective_anchor is not None:
            edge = build_edge(
                source=selective_anchor,
                target=dependency_anchor,
                relation="REFINES",
                rule="structured_relation:depmap:selective_refines_recurrent_dependency",
                rationale="Selective lineage dependency evidence narrows a broader recurrent dependency claim for the same gene and disease.",
                similarity_score=0.92,
            )
            add_edge_once(edges, seen, edge)
            refine_count += 1

        if by_specificity.get(2):
            parent_anchor = selective_anchor or dependency_anchor
            if parent_anchor is not None:
                for record in by_specificity[2]:
                    edge = build_edge(
                        source=record,
                        target=parent_anchor,
                        relation="REFINES",
                        rule="structured_relation:depmap:context_refines_parent_dependency",
                        rationale="Context-conditioned DepMap dependency evidence narrows a broader lineage dependency claim for the same gene and disease.",
                        similarity_score=0.9,
                    )
                    add_edge_once(edges, seen, edge)
                    refine_count += 1

    for group_records in contradiction_groups.values():
        sign_buckets: dict[str, list[EvidenceRecord]] = defaultdict(list)
        for record in group_records:
            sign_buckets[record.sign].append(record)
        if len(sign_buckets) < 2:
            continue
        ordered_signs = sorted(sign_buckets.keys())
        anchors = {sign: sorted_by_rank(sign_buckets[sign])[0] for sign in ordered_signs}
        for left_index, left_sign in enumerate(ordered_signs):
            for right_sign in ordered_signs[left_index + 1 :]:
                left_anchor = anchors[left_sign]
                right_anchor = anchors[right_sign]
                source, target = sorted([left_anchor, right_anchor], key=rank_tuple)
                edge = build_edge(
                    source=source,
                    target=target,
                    relation="CONTRADICTS",
                    rule="structured_relation:depmap:opposite_direction_contradiction",
                    rationale="These DepMap evidence units share the same normalized claim family but encode opposite directional functional interpretations.",
                    similarity_score=0.85,
                )
                add_edge_once(edges, seen, edge)
                contradiction_count += 1

    cross_context_support_count = add_depmap_cross_context_supports(dependency_groups, edges, seen)
    support_count += cross_context_support_count

    return edges, {
        "support_count": support_count,
        "cross_context_support_count": cross_context_support_count,
        "contradiction_count": contradiction_count,
        "refine_count": refine_count,
    }


def contradiction_risk_by_evidence(edges: list[dict[str, Any]]) -> dict[str, float]:
    logic_counts: Counter[str] = Counter()
    contradiction_counts: Counter[str] = Counter()
    for edge in edges:
        if edge.get("relation_family") != "evidence_logic":
            continue
        logic_counts[edge["source"]] += 1
        logic_counts[edge["target"]] += 1
        if edge.get("relation") == "CONTRADICTS":
            contradiction_counts[edge["source"]] += 1
            contradiction_counts[edge["target"]] += 1
    risk: dict[str, float] = {}
    for evidence_id, total in logic_counts.items():
        risk[evidence_id] = round(contradiction_counts[evidence_id] / total, 4) if total else 0.0
    return risk


def enrich_graph(graph: dict[str, Any], relation_edges: list[dict[str, Any]], source_name: str) -> dict[str, Any]:
    enriched = json.loads(json.dumps(graph))
    preserved_edges = [edge for edge in enriched.get("edges", []) if edge.get("relation_family") != "evidence_logic"]
    enriched["edges"] = preserved_edges + relation_edges
    contradiction_risk = contradiction_risk_by_evidence(relation_edges)
    for node in enriched["nodes"]:
        if node.get("node_type") != "evidence":
            continue
        confidence = node.setdefault("confidence", {})
        confidence["contradiction_risk"] = contradiction_risk.get(node["id"], 0.0)
    enriched.setdefault("graph_metadata", {})
    enriched["graph_metadata"]["relation_construction"] = {
        "generated_at": now_iso(),
        "source_profile": source_name,
        "relation_subset": {
            "semantic_anchor": ["LINKED_TO", "OBSERVED_IN"],
            "evidence_logic": ["SUPPORTS", "CONTRADICTS", "REFINES"],
            "aggregation_trace": ["DERIVED_FROM"],
        },
        "selection_policy": {
            "supports": "exact proposition duplicates collapse to a representative anchor",
            "contradicts": "opposite-sign evidence collapse to representative anchors within the same normalized claim family",
            "refines": "narrower subgroup or context-conditioned evidence points to the broadest compatible parent claim",
        },
    }
    return enriched


def build_stats(source_name: str, graph: dict[str, Any], relation_edges: list[dict[str, Any]], rule_counts: dict[str, Any]) -> dict[str, Any]:
    evidence_by_id = {node["id"]: node for node in graph["nodes"] if node.get("node_type") == "evidence"}
    relation_counts = Counter(edge["relation"] for edge in relation_edges)
    by_disease = Counter()
    by_source_predicate = Counter()
    target_disease_distribution: dict[str, Counter[str]] = defaultdict(Counter)
    contradiction_risk_values = []

    for node in graph["nodes"]:
        if node.get("node_type") == "evidence":
            contradiction_risk_values.append(float((node.get("confidence") or {}).get("contradiction_risk") or 0.0))

    for edge in relation_edges:
        source_node = evidence_by_id[edge["source"]]
        target_node = evidence_by_id[edge["target"]]
        disease = source_node.get("disease") or target_node.get("disease") or "unknown"
        by_disease[(edge["relation"], disease)] += 1
        predicate_pair = f"{source_node.get('predicate')}->{target_node.get('predicate')}"
        by_source_predicate[(edge["relation"], predicate_pair)] += 1
        target_disease_distribution[edge["relation"]][disease] += 1

    reasonableness = {}
    for relation, counts in target_disease_distribution.items():
        total = sum(counts.values()) or 1
        top_disease, top_count = counts.most_common(1)[0]
        share = round(top_count / total, 4)
        reasonableness[relation] = {
            "top_disease": top_disease,
            "top_disease_share": share,
            "assessment": "concentrated_but_plausible" if share > 0.55 else "distributed",
        }

    return {
        "generated_at": now_iso(),
        "source_name": source_name,
        "evidence_count": sum(1 for node in graph["nodes"] if node.get("node_type") == "evidence"),
        "edge_count_total": len(graph.get("edges", [])),
        "semantic_anchor_edge_count": sum(1 for edge in graph["edges"] if edge.get("relation_family") == "semantic_anchor"),
        "evidence_logic_edge_count": len(relation_edges),
        "aggregation_trace_edge_count": sum(1 for edge in graph["edges"] if edge.get("relation_family") == "aggregation_trace"),
        "relation_counts": dict(sorted(relation_counts.items())),
        "rule_counts": rule_counts,
        "relation_counts_by_disease": {
            relation: dict(sorted({disease: count for (rel, disease), count in by_disease.items() if rel == relation}.items(), key=lambda item: (-item[1], item[0])))
            for relation in relation_counts
        },
        "relation_counts_by_predicate_pair": {
            relation: dict(sorted({pair: count for (rel, pair), count in by_source_predicate.items() if rel == relation}.items(), key=lambda item: (-item[1], item[0]))[:25])
            for relation in relation_counts
        },
        "contradiction_risk_summary": {
            "mean": round(mean(contradiction_risk_values), 6) if contradiction_risk_values else 0.0,
            "max": round(max(contradiction_risk_values), 6) if contradiction_risk_values else 0.0,
            "nonzero_count": sum(1 for value in contradiction_risk_values if value > 0),
        },
        "distribution_reasonableness": reasonableness,
    }


def build_audit(source_name: str, graph: dict[str, Any], relation_edges: list[dict[str, Any]], records: list[EvidenceRecord]) -> dict[str, Any]:
    node_ids = {node["id"] for node in graph["nodes"]}
    allowed_logic = {"SUPPORTS", "CONTRADICTS", "REFINES"}
    duplicates = Counter((edge["source"], edge["target"], edge["relation"]) for edge in relation_edges)
    duplicate_edges = [key for key, count in duplicates.items() if count > 1]
    orphan_edges = [edge for edge in relation_edges if edge["source"] not in node_ids or edge["target"] not in node_ids]
    invalid_relations = [edge for edge in relation_edges if edge["relation"] not in allowed_logic]
    record_by_id = {record.evidence_id: record for record in records}

    refine_errors = []
    contradiction_errors = []
    for edge in relation_edges:
        source = record_by_id[edge["source"]]
        target = record_by_id[edge["target"]]
        if edge["relation"] == "REFINES" and source.specificity <= target.specificity:
            refine_errors.append(edge)
        if edge["relation"] == "CONTRADICTS" and source.sign == target.sign:
            contradiction_errors.append(edge)

    return {
        "generated_at": now_iso(),
        "source_name": source_name,
        "checks": {
            "orphan_edges": {"count": len(orphan_edges), "examples": orphan_edges[:10]},
            "duplicate_relation_edges": {"count": len(duplicate_edges), "examples": duplicate_edges[:10]},
            "invalid_logic_relations": {"count": len(invalid_relations), "examples": invalid_relations[:10]},
            "refine_direction_errors": {"count": len(refine_errors), "examples": refine_errors[:10]},
            "contradiction_sign_errors": {"count": len(contradiction_errors), "examples": contradiction_errors[:10]},
        },
    }


def process_source(source_name: str) -> dict[str, Any]:
    normalized_graph = read_json(GRAPH_TARGETS[source_name]["normalized"])
    raw_graph = read_json(GRAPH_TARGETS[source_name]["raw"])
    records = [record for node in normalized_graph["nodes"] if (record := make_record(node, source_name)) is not None]
    if source_name == "tcga":
        relation_edges, rule_counts = build_tcga_edges(records)
    else:
        relation_edges, rule_counts = build_depmap_edges(records)

    enriched_normalized = enrich_graph(normalized_graph, relation_edges, source_name)
    enriched_raw = enrich_graph(raw_graph, relation_edges, source_name)
    write_json(GRAPH_TARGETS[source_name]["normalized"], enriched_normalized)
    write_json(GRAPH_TARGETS[source_name]["raw"], enriched_raw)

    stats = build_stats(source_name, enriched_normalized, relation_edges, rule_counts)
    audit = build_audit(source_name, enriched_normalized, relation_edges, records)
    write_json(GRAPH_TARGETS[source_name]["stats"], stats)
    write_json(GRAPH_TARGETS[source_name]["audit"], audit)
    return {
        "source_name": source_name,
        "stats_path": str(GRAPH_TARGETS[source_name]["stats"].relative_to(ROOT)),
        "audit_path": str(GRAPH_TARGETS[source_name]["audit"].relative_to(ROOT)),
        "relation_counts": stats["relation_counts"],
        "distribution_reasonableness": stats["distribution_reasonableness"],
    }


def main() -> None:
    summary = {
        "generated_at": now_iso(),
        "sources": [process_source("tcga"), process_source("depmap")],
    }
    write_json(SUMMARY_PATH, summary)


if __name__ == "__main__":
    main()

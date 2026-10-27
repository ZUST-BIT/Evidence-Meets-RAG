from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from extract_tcga_atomic_evidence import (
    LINKEDOMICS_ROOT,
    PRIORITY_COHORTS,
    build_entity_mention,
    build_pubmed_alignment_candidates,
    disease_identifier,
    ensure_dir,
    find_file,
    now_iso,
    open_text,
    relpath,
    safe_float,
    stable_hash,
    summarize_numeric,
)


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "kg" / "aligned"

MIN_GROUP_SIZE = 30

EXPR_OVERALL_MIN_DELTA = 2.5
EXPR_SUBGROUP_MIN_DELTA = 1.5
METHYL_OVERALL_MIN_DELTA = 0.25
METHYL_SUBGROUP_MIN_DELTA = 0.15
MUTATION_OVERALL_MIN_FREQ = 0.10
MUTATION_SUBGROUP_MIN_DELTA = 0.08
CNV_OVERALL_MIN_FREQ = 0.50
CNV_SUBGROUP_MIN_DELTA = 0.25
CNV_AMP_THRESHOLD = 0.30
CNV_DEL_THRESHOLD = -0.30
PROGNOSIS_MIN_GROUP_SIZE = 40
PROGNOSIS_MIN_EVENT_COUNT = 15
PROGNOSIS_MIN_EVENT_RATE_DELTA = 0.20
PROGNOSIS_MIN_MEDIAN_OS_RATIO = 1.40

RPPA_MAX_MISSING_RATE = 0.20
RPPA_GENE_OVERALL_MIN_MEDIAN_DELTA = 0.50
RPPA_GENE_OVERALL_MIN_COHEN_D = 0.80
RPPA_GENE_SUBGROUP_MIN_MEDIAN_DELTA = 0.35
RPPA_GENE_SUBGROUP_MIN_COHEN_D = 0.60
RPPA_PHOSPHO_OVERALL_MIN_MEDIAN_DELTA = 0.50
RPPA_PHOSPHO_OVERALL_MIN_COHEN_D = 0.90
RPPA_PHOSPHO_SUBGROUP_MIN_MEDIAN_DELTA = 0.40
RPPA_PHOSPHO_SUBGROUP_MIN_COHEN_D = 0.70

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


def normalize_stage(raw: str | None) -> str | None:
    if not raw or raw == "NA":
        return None
    value = raw.strip().lower().replace(" ", "")
    if value.startswith("stageiv") or value.startswith("stageiii"):
        return "late"
    if value.startswith("stageii") or value.startswith("stagei"):
        return "early"
    return None


def mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def variance_or_zero(values: list[float], center: float) -> float:
    if len(values) < 2:
        return 0.0
    return sum((value - center) ** 2 for value in values) / (len(values) - 1)


def cohen_d(group_a: list[float], group_b: list[float]) -> float:
    if not group_a or not group_b:
        return 0.0
    mean_a = mean(group_a)
    mean_b = mean(group_b)
    var_a = variance_or_zero(group_a, mean_a)
    var_b = variance_or_zero(group_b, mean_b)
    denom_df = len(group_a) + len(group_b) - 2
    if denom_df <= 0:
        return 0.0
    pooled_var = (((len(group_a) - 1) * var_a) + ((len(group_b) - 1) * var_b)) / denom_df
    if pooled_var <= 0:
        return 0.0 if mean_a == mean_b else min(abs(mean_a - mean_b) * 2.0, 3.0)
    return abs((mean_a - mean_b) / math.sqrt(pooled_var))


def completeness_ratio(observed_count: int, total_count: int) -> float:
    if total_count <= 0:
        return 0.0
    return round(observed_count / total_count, 4)


def parse_status_flag(raw: str | None) -> int | None:
    value = (raw or "").strip()
    if value in {"0", "1"}:
        return int(value)
    return None


def parse_gene_means(path: Path) -> tuple[dict[str, float], int]:
    gene_means: dict[str, float] = {}
    sample_count = 0
    with open_text(path) as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        sample_count = len(header) - 1
        for row in reader:
            if not row:
                continue
            gene = row[0].strip()
            values = [safe_float(v) for v in row[1:]]
            clean = [v for v in values if v is not None]
            if gene and clean:
                gene_means[gene] = mean(clean)
    return gene_means, sample_count


def parse_feature_medians(path: Path, include_feature=None) -> tuple[dict[str, float], int]:
    feature_medians: dict[str, float] = {}
    sample_count = 0
    with open_text(path) as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        sample_count = len(header) - 1
        for row in reader:
            if not row:
                continue
            feature = row[0].strip()
            if include_feature and not include_feature(feature):
                continue
            values = [safe_float(v) for v in row[1:]]
            clean = [v for v in values if v is not None]
            if feature and clean:
                feature_medians[feature] = median_or_none(clean)
    return feature_medians, sample_count


def is_rppa_phospho_feature(feature: str) -> bool:
    lower = feature.lower()
    return "|" in feature and "_p" in lower


def build_gene_subject(gene: str) -> dict[str, Any]:
    return {
        "subject_id": f"HGNC:{gene}",
        "subject_label": gene,
        "subject_entity_type": "Gene",
        "subject_aliases": [gene],
        "subject_db_source": "HGNC",
        "subject_normalized_from": [gene],
    }


def build_rppa_gene_subject(gene: str) -> dict[str, Any]:
    return {
        "subject_id": f"PROTEIN:{gene}",
        "subject_label": gene,
        "subject_entity_type": "Protein",
        "subject_aliases": [gene],
        "subject_db_source": "HGNC_symbol_anchor",
        "subject_normalized_from": [gene],
    }


def build_rppa_phospho_subject(feature: str) -> dict[str, Any] | None:
    if not is_rppa_phospho_feature(feature):
        return None
    gene, analyte = feature.split("|", 1)
    return {
        "subject_id": f"RPPA_ANALYTE:{stable_hash(gene, analyte)}",
        "subject_label": feature,
        "subject_entity_type": "Protein",
        "subject_aliases": [feature, gene, analyte],
        "subject_db_source": "LinkedOmics_RPPA",
        "subject_normalized_from": [feature, gene],
        "gene_anchor": gene,
        "analyte_label": analyte,
    }


def parse_clinical_groups(path: Path) -> dict[str, Any]:
    with open_text(path) as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        sample_ids = header[1:]
        row_map = {row[0]: row[1:] for row in reader if row}

    stage_groups = {"early": [], "late": []}
    for sample_id, value in zip(sample_ids, row_map.get("pathologic_stage", []), strict=False):
        group = normalize_stage(value)
        if group:
            stage_groups[group].append(sample_id)

    purity_pairs = []
    for sample_id, value in zip(sample_ids, row_map.get("Tumor_purity", []), strict=False):
        numeric = safe_float(value)
        if numeric is not None:
            purity_pairs.append((sample_id, numeric))
    purity_pairs.sort(key=lambda item: item[1])
    quartile_n = len(purity_pairs) // 4
    purity_groups = {"low": [], "high": []}
    purity_summary = {"mean": None, "q1": None, "q3": None}
    if purity_pairs:
        purity_summary["mean"] = round(mean(v for _, v in purity_pairs), 4)
    if quartile_n >= MIN_GROUP_SIZE:
        low = purity_pairs[:quartile_n]
        high = purity_pairs[-quartile_n:]
        purity_groups["low"] = [sid for sid, _ in low]
        purity_groups["high"] = [sid for sid, _ in high]
        purity_summary["q1"] = round(low[-1][1], 4)
        purity_summary["q3"] = round(high[0][1], 4)

    subgroup_defs = []
    if len(stage_groups["early"]) >= MIN_GROUP_SIZE and len(stage_groups["late"]) >= MIN_GROUP_SIZE:
        subgroup_defs.append(
            {
                "comparison_id": "stage_late_vs_early",
                "sample_grouping": "pathologic_stage",
                "comparison_type": None,
                "group_a_name": "late_stage_iii_iv",
                "group_b_name": "early_stage_i_ii",
                "group_a_ids": set(stage_groups["late"]),
                "group_b_ids": set(stage_groups["early"]),
            }
        )
    if len(purity_groups["low"]) >= MIN_GROUP_SIZE and len(purity_groups["high"]) >= MIN_GROUP_SIZE:
        subgroup_defs.append(
            {
                "comparison_id": "purity_high_vs_low",
                "sample_grouping": "tumor_purity_quantiles",
                "comparison_type": "high_vs_low",
                "group_a_name": "high_purity_q4",
                "group_b_name": "low_purity_q1",
                "group_a_ids": set(purity_groups["high"]),
                "group_b_ids": set(purity_groups["low"]),
            }
        )

    survival_records: dict[str, dict[str, float | int]] = {}
    event_values: list[int] = []
    os_values: list[float] = []
    for sample_id, os_raw, status_raw in zip(
        sample_ids,
        row_map.get("overall_survival", []),
        row_map.get("status", []),
        strict=False,
    ):
        os_days = safe_float(os_raw)
        status = parse_status_flag(status_raw)
        if os_days is None or status is None:
            continue
        survival_records[sample_id] = {"overall_survival_days": os_days, "status": status}
        event_values.append(status)
        os_values.append(os_days)

    return {
        "clinical_sample_count": len(sample_ids),
        "pathologic_stage_distribution": dict(sorted(Counter(v for v in row_map.get("pathologic_stage", []) if v and v != "NA").items())),
        "tumor_purity_summary": purity_summary,
        "survival_summary": {
            "n_with_overall_survival": len(survival_records),
            "event_count": sum(event_values),
            "median_overall_survival_days": round(median_or_none(os_values), 4) if os_values else None,
        },
        "survival_records": survival_records,
        "subgroup_definitions": subgroup_defs,
    }


def clinical_summary_for_output(clinical_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "clinical_sample_count": clinical_summary["clinical_sample_count"],
        "pathologic_stage_distribution": clinical_summary["pathologic_stage_distribution"],
        "tumor_purity_summary": clinical_summary["tumor_purity_summary"],
        "survival_summary": clinical_summary["survival_summary"],
        "subgroup_definitions": [
            {
                "comparison_id": item["comparison_id"],
                "sample_grouping": item["sample_grouping"],
                "comparison_type": item["comparison_type"],
                "group_a_name": item["group_a_name"],
                "group_b_name": item["group_b_name"],
                "group_a_size": len(item["group_a_ids"]),
                "group_b_size": len(item["group_b_ids"]),
            }
            for item in clinical_summary["subgroup_definitions"]
        ],
    }


def load_sources() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
]:
    cohort_data: dict[str, Any] = {}
    source_manifest: dict[str, Any] = {}
    expression_means: dict[str, dict[str, float]] = {}
    methylation_means: dict[str, dict[str, float]] = {}
    rppa_gene_medians: dict[str, dict[str, float]] = {}
    rppa_phospho_medians: dict[str, dict[str, float]] = {}

    for config in PRIORITY_COHORTS:
        cohort = config["cohort"]
        print(f"[load] {cohort}: locating files", flush=True)
        cohort_dir = LINKEDOMICS_ROOT / cohort
        expr_path = find_file(cohort_dir, lambda name: "RNAseq" in name and "RSEM_log2.cct.gz" in name)
        mut_path = find_file(cohort_dir, lambda name: "Mutation" in name and "Gene__Firehose_MutSig2CV.cbt" in name)
        scna_path = find_file(cohort_dir, lambda name: "SCNA" in name and "Gene__Firehose_GISTIC2.cct.gz" in name)
        methyl_path = find_file(cohort_dir, lambda name: "Methylation" in name and "__Gene__" in name and "Prepocessor.cct.gz" in name)
        clinical_path = find_file(cohort_dir, lambda name: "Clinical" in name and name.endswith(".tsi"))
        rppa_gene_path = find_file(cohort_dir, lambda name: "RPPA" in name and "__Gene__" in name)
        rppa_analyte_path = find_file(cohort_dir, lambda name: "RPPA" in name and "__Analyte__" in name)
        if not expr_path or not mut_path or not clinical_path or not methyl_path:
            raise FileNotFoundError(f"Missing required TCGA files for {cohort}")

        expr_means, expr_n = parse_gene_means(expr_path)
        print(f"[load] {cohort}: expression means ready ({len(expr_means)} genes, {expr_n} samples)", flush=True)
        methyl_means, methyl_n = parse_gene_means(methyl_path)
        print(f"[load] {cohort}: methylation means ready ({len(methyl_means)} genes, {methyl_n} samples)", flush=True)
        if rppa_gene_path:
            gene_medians, rppa_gene_n = parse_feature_medians(rppa_gene_path)
            rppa_gene_medians[cohort] = gene_medians
            print(f"[load] {cohort}: RPPA gene medians ready ({len(gene_medians)} proteins, {rppa_gene_n} samples)", flush=True)
        if rppa_analyte_path:
            phospho_medians, rppa_phospho_n = parse_feature_medians(rppa_analyte_path, include_feature=is_rppa_phospho_feature)
            rppa_phospho_medians[cohort] = phospho_medians
            print(f"[load] {cohort}: RPPA phospho medians ready ({len(phospho_medians)} analytes, {rppa_phospho_n} samples)", flush=True)
        clinical_summary = parse_clinical_groups(clinical_path)
        print(
            f"[load] {cohort}: subgroups={[item['comparison_id'] for item in clinical_summary['subgroup_definitions']]} "
            f"stages={clinical_summary['pathologic_stage_distribution']} purity={clinical_summary['tumor_purity_summary']}",
            flush=True,
        )

        cohort_data[cohort] = {
            "config": config,
            "paths": {
                "expression": expr_path,
                "mutation": mut_path,
                "scna": scna_path,
                "methylation": methyl_path,
                "clinical": clinical_path,
                "rppa_gene": rppa_gene_path,
                "rppa_analyte": rppa_analyte_path,
            },
            "clinical_summary": clinical_summary,
            "expression_n": expr_n,
            "methylation_n": methyl_n,
        }
        expression_means[cohort] = expr_means
        methylation_means[cohort] = methyl_means
        source_manifest[cohort] = {
            "family": config["family"],
            "label": config["label"],
            "expression_path": relpath(expr_path),
            "mutation_path": relpath(mut_path),
            "scna_path": relpath(scna_path) if scna_path else None,
            "methylation_path": relpath(methyl_path),
            "clinical_path": relpath(clinical_path),
            "rppa_gene_path": relpath(rppa_gene_path) if rppa_gene_path else None,
            "rppa_analyte_path": relpath(rppa_analyte_path) if rppa_analyte_path else None,
            "expression_gene_count": len(expr_means),
            "methylation_gene_count": len(methyl_means),
            "rppa_gene_count": len(rppa_gene_medians.get(cohort, {})),
            "rppa_phospho_count": len(rppa_phospho_medians.get(cohort, {})),
            "expression_sample_count": expr_n,
            "methylation_sample_count": methyl_n,
            "clinical_summary": clinical_summary_for_output(clinical_summary),
        }
    return cohort_data, source_manifest, expression_means, methylation_means, rppa_gene_medians, rppa_phospho_medians


def comparison_label(comparison_id: str) -> str:
    mapping = {
        "pan_cancer_background": "target TCGA cohort vs selected multi-cancer TCGA pan-cancer background",
        "stage_late_vs_early": "late stage (III/IV) vs early stage (I/II)",
        "purity_high_vs_low": "high tumor purity (Q4) vs low tumor purity (Q1)",
        "cohort_recurrent": "recurrent cohort signal without subgroup split",
        "overall_survival_association": "molecular subgroup contrast against overall survival",
    }
    return mapping[comparison_id]


def score_bundle(effect_strength: float, n_effective: int, source_reliability: float = 0.9) -> tuple[dict[str, Any], dict[str, Any]]:
    study_type_score = 0.78
    sample_size_score = min(n_effective / 500.0, 1.0)
    stat_score = min(max(effect_strength, 0.15), 1.0)
    composite = round(mean([study_type_score, sample_size_score, stat_score, source_reliability]), 4)
    if composite >= 0.85:
        grade = "A"
    elif composite >= 0.70:
        grade = "B"
    elif composite >= 0.55:
        grade = "C"
    elif composite >= 0.40:
        grade = "D"
    else:
        grade = "E"
    belief = round(min(0.95, composite * 0.84), 4)
    plausibility = round(min(1.0, belief + 0.14), 4)
    score = {
        "study_type_score": study_type_score,
        "impact_score": source_reliability,
        "stat_significance_score": stat_score,
        "sample_size_score": round(sample_size_score, 4),
        "llm_confidence": None,
        "composite_score": composite,
        "grade_level": grade,
        "ds_belief": belief,
        "ds_plausibility": plausibility,
        "ds_ignorance": round(max(0.0, plausibility - belief), 4),
    }
    uncertainty = {
        "evidence_grade": grade,
        "grade_framework": "custom_omics",
        "confidence_score": composite,
        "study_quality_score": study_type_score,
        "sample_size_score": round(sample_size_score, 4),
        "statistical_support_score": stat_score,
        "source_reliability_score": source_reliability,
        "llm_confidence": None,
        "ds_mass_support": belief,
        "ds_mass_against": None,
        "ds_mass_uncertainty": round(max(0.0, plausibility - belief), 4),
        "ds_belief": belief,
        "ds_plausibility": plausibility,
        "contradiction_risk": None,
        "missingness_notes": None,
        "review_status": "pending",
    }
    return score, uncertainty


def compact_entity_ref(entity_id: str | None, label: str, entity_type: str) -> dict[str, Any]:
    return {
        "id": entity_id,
        "label": label,
        "entity_type": entity_type,
    }


def build_compact_confidence(
    *,
    score: dict[str, Any],
    uncertainty: dict[str, Any],
    extraction_confidence: float | None,
    data_completeness: float,
) -> dict[str, Any]:
    return {
        "grade": uncertainty["evidence_grade"],
        "score": uncertainty["confidence_score"],
        "study_quality_score": uncertainty["study_quality_score"],
        "statistical_support_score": uncertainty["statistical_support_score"],
        "source_reliability_score": uncertainty["source_reliability_score"],
        "sample_size_score": uncertainty["sample_size_score"],
        "extraction_confidence": extraction_confidence,
        "ds_belief": score["ds_belief"],
        "ds_plausibility": score["ds_plausibility"],
        "ds_ignorance": score["ds_ignorance"],
        "contradiction_risk": uncertainty["contradiction_risk"],
        "data_completeness": data_completeness,
        "review_status": uncertainty["review_status"],
    }


def make_evidence(
    *,
    cohort: str,
    family: str,
    label: str,
    predicate: str,
    direction: str,
    evidence_category: str,
    feature_layer: str,
    metric: str,
    gene: str,
    source_file: Path,
    value: float,
    effect_size: float,
    sample_count: int,
    comparison_id: str,
    comparison_type: str | None,
    sample_grouping: str,
    group_a_name: str | None,
    group_b_name: str | None,
    n_group_a: int | None,
    n_group_b: int | None,
    clinical_summary: dict[str, Any],
    summary_text: str,
    platform_specific_fields: dict[str, Any],
    subject_id: str | None = None,
    subject_entity_type: str = "Gene",
    subject_aliases: list[str] | None = None,
    subject_db_source: str | None = None,
    subject_normalized_from: list[str] | None = None,
    source_record_key: str | None = None,
    data_completeness: float = 0.95,
    statistical_method: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    created_at = now_iso()
    record_key = source_record_key or gene
    evidence_id = f"TCGAEVID2-{stable_hash(cohort, feature_layer, predicate, metric, record_key, comparison_id)}"
    cohort_id = f"COHORT-{stable_hash(cohort)}"
    subject_entity_id = subject_id or (f"HGNC:{gene}" if subject_entity_type == "Gene" else f"ENTITY:{stable_hash(gene)}")
    disease_entity_id = f"DISEASE:{family}"

    score, uncertainty = score_bundle(min(abs(effect_size), 1.0), sample_count)
    group_size_total = 0
    if n_group_a:
        group_size_total += n_group_a
    if n_group_b:
        group_size_total += n_group_b

    stats = {
        "p_value": None,
        "q_value": None,
        "fold_change": round(value, 4) if evidence_category in {"expression_regulation", "epigenetic_regulation"} else None,
        "confidence_interval": None,
        "sample_size": sample_count,
        "effect_size": round(abs(effect_size), 4),
        "hazard_ratio": None,
        "mutation_frequency": round(value, 4) if predicate == "mutated_in" and comparison_id == "cohort_recurrent" else None,
        "ceres_score": None,
        "codependency_coefficient": None,
        "synergy_score": None,
        "statistical_method": statistical_method or "thresholded cohort/subgroup comparison on LinkedOmics TCGA matrices",
    }
    if evidence_category == "clinical_prognosis":
        stats["fold_change"] = None
        stats["statistical_method"] = statistical_method or "overall survival contrast using molecular subgrouping on LinkedOmics TCGA clinical annotations"

    proposition_polarity = "uncertain" if evidence_category == "epigenetic_regulation" else ("oncogenic" if direction in {"increase", "positive"} else "tumor_suppressive")
    claim_scope = "gene_disease"
    hypothesis_level = "molecular"
    if evidence_category == "clinical_prognosis":
        claim_scope = "clinical_outcome"
        hypothesis_level = "clinical"
        if "poor_prognosis" in predicate:
            proposition_polarity = "pro_tumor"
        elif "favorable_prognosis" in predicate:
            proposition_polarity = "anti_tumor"

    support = {
        "supports_target_discovery": True,
        "supports_mechanism_inference": True,
        "supports_drug_repositioning": False,
    }
    comparison = {
        "sample_grouping": sample_grouping,
        "comparison_type": comparison_type,
        "comparison_label": comparison_label(comparison_id),
        "group_a": group_a_name,
        "group_b": group_b_name,
        "n_total": sample_count,
        "n_effective": group_size_total or sample_count,
        "n_group_a": n_group_a,
        "n_group_b": n_group_b,
    }
    confidence = build_compact_confidence(
        score=score,
        uncertainty=uncertainty,
        extraction_confidence=score["composite_score"],
        data_completeness=data_completeness,
    )
    source_context = {
        "study_object": f"{cohort} tumor cohort",
        "group_axis": comparison_id,
        **platform_specific_fields,
    }

    evidence = {
        "node_type": "evidence",
        "id": evidence_id,
        "record_id": evidence_id,
        "source_family": "tcga_linkedomics",
        "disease": family,
        "cohort_id": cohort_id,
        "subject": compact_entity_ref(subject_entity_id, gene, subject_entity_type),
        "object": compact_entity_ref(disease_entity_id, family, "Disease"),
        "outcome": None,
        "predicate": predicate,
        "direction": direction,
        "polarity": proposition_polarity,
        "claim_scope": claim_scope,
        "hypothesis_level": hypothesis_level,
        "support": support,
        "evidence_category": evidence_category,
        "feature_layer": feature_layer,
        "aggregation_level": "gene_by_group" if comparison_id != "cohort_recurrent" else "gene_by_cohort",
        "summary": summary_text,
        "metric": metric,
        "value": round(value, 4),
        "unit": None,
        "comparison": comparison,
        "statistics": stats,
        "confidence": confidence,
        "citation": None,
        "source_context": source_context,
        "reviewer_comments": None,
        "created_at": created_at,
        "updated_at": None,
        "version": "2.0.0",
        "provenance": {
            "source_family": "tcga_linkedomics",
            "source_dataset": "LinkedOmics-TCGA",
            "source_release": "01_28_2016",
            "source_record_id": record_key,
            "source_file": relpath(source_file),
            "source_url": None,
            "citation_doi": None,
            "citation_pmid": None,
            "platform": source_file.name,
            "assay": metric,
            "sequencing_or_screen_id": cohort,
            "batch_id": "01_28_2016",
            "analysis_pipeline": "tcga_atomic_extraction_v2",
            "extractor_name": "scripts/extract_tcga_atomic_evidence_v2.py",
            "extractor_version": "2.0.0",
            "extracted_at": created_at,
            "study_design": "observational_omics",
            "clinical_stage": "omics_cohort",
            "licence": None,
            "is_primary_observation": True,
            "source_priority": 2 if family in {"HCC", "CRC"} else 3,
        },
        "aggregation_parent_ids": [],
    }

    cohort_node = {
        "node_type": "cohort",
        "id": cohort_id,
        "cohort_label": f"{cohort} ({label})",
        "disease_name": family,
        "lineage": None,
        "cohort_type": "tumor_cohort",
        "n_effective": sample_count,
    }
    entity_nodes = [
        {
            "node_type": "entity",
            "id": subject_entity_id,
            "name": gene,
            "entity_type": subject_entity_type,
            "db_source": subject_db_source or ("HGNC" if subject_entity_type == "Gene" else "custom"),
            "aliases": list(dict.fromkeys(subject_aliases or [gene])),
            "normalized_from": list(dict.fromkeys(subject_normalized_from or [gene])),
        },
        {
            "node_type": "entity",
            "id": disease_entity_id,
            "name": family,
            "entity_type": "Disease",
            "db_source": "custom_disease_family",
            "aliases": [family],
            "normalized_from": [family],
        },
    ]
    return evidence, cohort_node, entity_nodes


def subgroup_indices(sample_ids: list[str], subgroup_defs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = []
    for subgroup in subgroup_defs:
        a = [idx for idx, sid in enumerate(sample_ids) if sid in subgroup["group_a_ids"]]
        b = [idx for idx, sid in enumerate(sample_ids) if sid in subgroup["group_b_ids"]]
        if len(a) >= MIN_GROUP_SIZE and len(b) >= MIN_GROUP_SIZE:
            indexed.append({**subgroup, "group_a_indices": a, "group_b_indices": b})
    return indexed


def prognosis_direction(median_ratio: float | None, event_rate_delta: float | None) -> str | None:
    signal = 0
    if median_ratio is not None:
        if median_ratio <= 1.0 / PROGNOSIS_MIN_MEDIAN_OS_RATIO:
            signal -= 1
        elif median_ratio >= PROGNOSIS_MIN_MEDIAN_OS_RATIO:
            signal += 1
    if event_rate_delta is not None:
        if event_rate_delta >= PROGNOSIS_MIN_EVENT_RATE_DELTA:
            signal -= 1
        elif event_rate_delta <= -PROGNOSIS_MIN_EVENT_RATE_DELTA:
            signal += 1
    if signal <= -1:
        return "poor"
    if signal >= 1:
        return "favorable"
    return None


def prognosis_effect_size(median_ratio: float | None, event_rate_delta: float | None) -> float:
    ratio_component = 0.0
    if median_ratio and median_ratio > 0:
        ratio_component = abs(median_ratio - 1.0)
    event_component = abs(event_rate_delta or 0.0)
    return max(ratio_component, event_component)


def process_prognosis_continuous(
    *,
    path: Path,
    cohort: str,
    family: str,
    label: str,
    clinical_summary: dict[str, Any],
    feature_layer: str,
    high_label: str,
    poor_predicate: str,
    favorable_predicate: str,
    metric: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    evidence_nodes = []
    cohort_nodes = []
    entity_nodes = []
    counts = Counter()
    survival_records = clinical_summary["survival_records"]
    if len(survival_records) < PROGNOSIS_MIN_GROUP_SIZE * 2:
        return evidence_nodes, cohort_nodes, entity_nodes, counts

    with open_text(path) as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        sample_ids = header[1:]
        for row in reader:
            if not row:
                continue
            gene = row[0].strip()
            values = [safe_float(v) for v in row[1:]]
            pairs = []
            for sample_id, value in zip(sample_ids, values, strict=False):
                if value is None or sample_id not in survival_records:
                    continue
                record = survival_records[sample_id]
                pairs.append((sample_id, value, float(record["overall_survival_days"]), int(record["status"])))
            if len(pairs) < PROGNOSIS_MIN_GROUP_SIZE * 2:
                continue
            pairs.sort(key=lambda item: item[1])
            half = len(pairs) // 2
            low_group = pairs[:half]
            high_group = pairs[-half:]
            if len(low_group) < PROGNOSIS_MIN_GROUP_SIZE or len(high_group) < PROGNOSIS_MIN_GROUP_SIZE:
                continue
            high_events = sum(item[3] for item in high_group)
            low_events = sum(item[3] for item in low_group)
            if high_events < PROGNOSIS_MIN_EVENT_COUNT and low_events < PROGNOSIS_MIN_EVENT_COUNT:
                continue
            high_median_os = median_or_none([item[2] for item in high_group])
            low_median_os = median_or_none([item[2] for item in low_group])
            if high_median_os is None or low_median_os is None or low_median_os == 0:
                continue
            median_ratio = high_median_os / low_median_os
            event_rate_delta = (high_events / len(high_group)) - (low_events / len(low_group))
            prognosis = prognosis_direction(median_ratio, event_rate_delta)
            if prognosis is None:
                continue
            predicate = poor_predicate if prognosis == "poor" else favorable_predicate
            direction = "negative" if prognosis == "poor" else "positive"
            effect_size = prognosis_effect_size(median_ratio, event_rate_delta)
            summary = (
                f"{gene} {high_label} is associated with {'poorer' if prognosis == 'poor' else 'more favorable'} overall survival "
                f"in {family} ({cohort}); median OS ratio={round(median_ratio, 4)}, event-rate delta={round(event_rate_delta, 4)}"
            )
            evidence, cohort_node, ents = make_evidence(
                cohort=cohort,
                family=family,
                label=label,
                predicate=predicate,
                direction=direction,
                evidence_category="clinical_prognosis",
                feature_layer=feature_layer,
                metric=metric,
                gene=gene,
                source_file=path,
                value=median_ratio,
                effect_size=effect_size,
                sample_count=len(high_group) + len(low_group),
                comparison_id="overall_survival_association",
                comparison_type="high_vs_low",
                sample_grouping="overall_survival_association",
                group_a_name=f"high_{feature_layer}",
                group_b_name=f"low_{feature_layer}",
                n_group_a=len(high_group),
                n_group_b=len(low_group),
                clinical_summary=clinical_summary,
                summary_text=summary,
                platform_specific_fields={
                    "prognosis_axis": "overall_survival",
                    "median_os_high_days": round(high_median_os, 4),
                    "median_os_low_days": round(low_median_os, 4),
                    "event_rate_high": round(high_events / len(high_group), 4),
                    "event_rate_low": round(low_events / len(low_group), 4),
                    "event_rate_delta": round(event_rate_delta, 4),
                },
            )
            evidence_nodes.append(evidence)
            cohort_nodes.append(cohort_node)
            entity_nodes.extend(ents)
            counts["overall_survival_association"] += 1

    return evidence_nodes, cohort_nodes, entity_nodes, counts


def process_prognosis_binary(
    *,
    path: Path,
    cohort: str,
    family: str,
    label: str,
    clinical_summary: dict[str, Any],
    feature_layer: str,
    altered_label: str,
    poor_predicate: str,
    favorable_predicate: str,
    metric: str,
    altered_fn,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    evidence_nodes = []
    cohort_nodes = []
    entity_nodes = []
    counts = Counter()
    survival_records = clinical_summary["survival_records"]
    if len(survival_records) < PROGNOSIS_MIN_GROUP_SIZE * 2:
        return evidence_nodes, cohort_nodes, entity_nodes, counts

    with open_text(path) as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        sample_ids = header[1:]
        for row in reader:
            if not row:
                continue
            gene = row[0].strip()
            values = [safe_float(v) for v in row[1:]]
            altered_group = []
            baseline_group = []
            for sample_id, value in zip(sample_ids, values, strict=False):
                if value is None or sample_id not in survival_records:
                    continue
                record = survival_records[sample_id]
                packed = (sample_id, value, float(record["overall_survival_days"]), int(record["status"]))
                if altered_fn(value):
                    altered_group.append(packed)
                else:
                    baseline_group.append(packed)
            if len(altered_group) < PROGNOSIS_MIN_GROUP_SIZE or len(baseline_group) < PROGNOSIS_MIN_GROUP_SIZE:
                continue
            altered_events = sum(item[3] for item in altered_group)
            baseline_events = sum(item[3] for item in baseline_group)
            if altered_events < PROGNOSIS_MIN_EVENT_COUNT and baseline_events < PROGNOSIS_MIN_EVENT_COUNT:
                continue
            altered_median_os = median_or_none([item[2] for item in altered_group])
            baseline_median_os = median_or_none([item[2] for item in baseline_group])
            if altered_median_os is None or baseline_median_os is None or baseline_median_os == 0:
                continue
            median_ratio = altered_median_os / baseline_median_os
            event_rate_delta = (altered_events / len(altered_group)) - (baseline_events / len(baseline_group))
            prognosis = prognosis_direction(median_ratio, event_rate_delta)
            if prognosis is None:
                continue
            predicate = poor_predicate if prognosis == "poor" else favorable_predicate
            direction = "negative" if prognosis == "poor" else "positive"
            effect_size = prognosis_effect_size(median_ratio, event_rate_delta)
            summary = (
                f"{gene} {altered_label} is associated with {'poorer' if prognosis == 'poor' else 'more favorable'} overall survival "
                f"in {family} ({cohort}); median OS ratio={round(median_ratio, 4)}, event-rate delta={round(event_rate_delta, 4)}"
            )
            evidence, cohort_node, ents = make_evidence(
                cohort=cohort,
                family=family,
                label=label,
                predicate=predicate,
                direction=direction,
                evidence_category="clinical_prognosis",
                feature_layer=feature_layer,
                metric=metric,
                gene=gene,
                source_file=path,
                value=median_ratio,
                effect_size=effect_size,
                sample_count=len(altered_group) + len(baseline_group),
                comparison_id="overall_survival_association",
                comparison_type="altered_vs_unaltered",
                sample_grouping="overall_survival_association",
                group_a_name=f"{altered_label}_present",
                group_b_name=f"{altered_label}_absent",
                n_group_a=len(altered_group),
                n_group_b=len(baseline_group),
                clinical_summary=clinical_summary,
                summary_text=summary,
                platform_specific_fields={
                    "prognosis_axis": "overall_survival",
                    "median_os_altered_days": round(altered_median_os, 4),
                    "median_os_unaltered_days": round(baseline_median_os, 4),
                    "event_rate_altered": round(altered_events / len(altered_group), 4),
                    "event_rate_unaltered": round(baseline_events / len(baseline_group), 4),
                    "event_rate_delta": round(event_rate_delta, 4),
                },
            )
            evidence_nodes.append(evidence)
            cohort_nodes.append(cohort_node)
            entity_nodes.extend(ents)
            counts["overall_survival_association"] += 1

    return evidence_nodes, cohort_nodes, entity_nodes, counts


def process_expression_like(
    *,
    path: Path,
    cohort: str,
    family: str,
    label: str,
    clinical_summary: dict[str, Any],
    overall_maps: dict[str, dict[str, float]],
    overall_threshold: float,
    subgroup_threshold: float,
    evidence_category: str,
    positive_predicate: str,
    negative_predicate: str,
    metric: str,
    feature_layer: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    evidence_nodes = []
    cohort_nodes = []
    entity_nodes = []
    counts = Counter()

    with open_text(path) as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        sample_ids = header[1:]
        subgroup_defs = subgroup_indices(sample_ids, clinical_summary["subgroup_definitions"])
        sample_count = len(sample_ids)

        for row in reader:
            if not row:
                continue
            gene = row[0].strip()
            values = [safe_float(v) for v in row[1:]]
            clean = [v for v in values if v is not None]
            if not gene or not clean:
                continue

            bg_values = [mapping[gene] for other, mapping in overall_maps.items() if other != cohort and gene in mapping]
            if bg_values:
                delta = mean(clean) - mean(bg_values)
                if abs(delta) >= overall_threshold:
                    predicate = positive_predicate if delta > 0 else negative_predicate
                    direction = "increase" if delta > 0 else "decrease"
                    summary = f"{gene} {predicate} {family} in {cohort} relative to selected TCGA pan-cancer tumor background"
                    evidence, cohort_node, ents = make_evidence(
                        cohort=cohort,
                        family=family,
                        label=label,
                        predicate=predicate,
                        direction=direction,
                        evidence_category=evidence_category,
                        feature_layer=feature_layer,
                        metric=metric,
                        gene=gene,
                        source_file=path,
                        value=delta,
                        effect_size=abs(delta),
                        sample_count=sample_count,
                        comparison_id="pan_cancer_background",
                        comparison_type="lineage_enriched_vs_background",
                        sample_grouping="pan_cancer_background",
                        group_a_name=f"{cohort}_tumor_cohort",
                        group_b_name="selected_tcga_pan_cancer_background",
                        n_group_a=sample_count,
                        n_group_b=None,
                        clinical_summary=clinical_summary,
                        summary_text=summary,
                        platform_specific_fields={"background_cohorts": [c for c in overall_maps if c != cohort], "group_axis": "overall_pan_cancer"},
                    )
                    evidence_nodes.append(evidence)
                    cohort_nodes.append(cohort_node)
                    entity_nodes.extend(ents)
                    counts["overall_pan_cancer"] += 1

            for subgroup in subgroup_defs:
                group_a = [values[idx] for idx in subgroup["group_a_indices"] if values[idx] is not None]
                group_b = [values[idx] for idx in subgroup["group_b_indices"] if values[idx] is not None]
                if len(group_a) < MIN_GROUP_SIZE or len(group_b) < MIN_GROUP_SIZE:
                    continue
                delta = mean(group_a) - mean(group_b)
                if abs(delta) < subgroup_threshold:
                    continue
                predicate = positive_predicate if delta > 0 else negative_predicate
                direction = "increase" if delta > 0 else "decrease"
                summary = f"{gene} {predicate} {family} in {cohort} for {comparison_label(subgroup['comparison_id'])}"
                evidence, cohort_node, ents = make_evidence(
                    cohort=cohort,
                    family=family,
                    label=label,
                    predicate=predicate,
                    direction=direction,
                    evidence_category=evidence_category,
                    feature_layer=feature_layer,
                    metric=metric,
                    gene=gene,
                    source_file=path,
                    value=delta,
                    effect_size=abs(delta),
                    sample_count=sample_count,
                    comparison_id=subgroup["comparison_id"],
                    comparison_type=subgroup["comparison_type"],
                    sample_grouping=subgroup["sample_grouping"],
                    group_a_name=subgroup["group_a_name"],
                    group_b_name=subgroup["group_b_name"],
                    n_group_a=len(group_a),
                    n_group_b=len(group_b),
                    clinical_summary=clinical_summary,
                    summary_text=summary,
                    platform_specific_fields={"group_axis": subgroup["comparison_id"], "group_a_mean": round(mean(group_a), 4), "group_b_mean": round(mean(group_b), 4)},
                )
                evidence_nodes.append(evidence)
                cohort_nodes.append(cohort_node)
                entity_nodes.extend(ents)
                counts[subgroup["comparison_id"]] += 1

    return evidence_nodes, cohort_nodes, entity_nodes, counts


def process_rppa_signal(
    *,
    path: Path,
    cohort: str,
    family: str,
    label: str,
    clinical_summary: dict[str, Any],
    overall_maps: dict[str, dict[str, float]],
    overall_min_median_delta: float,
    overall_min_effect_size: float,
    subgroup_min_median_delta: float,
    subgroup_min_effect_size: float,
    evidence_category: str,
    positive_predicate: str,
    negative_predicate: str,
    metric: str,
    feature_layer: str,
    subject_builder,
    assay_kind: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    evidence_nodes = []
    cohort_nodes = []
    entity_nodes = []
    counts = Counter()

    with open_text(path) as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        sample_ids = header[1:]
        subgroup_defs = subgroup_indices(sample_ids, clinical_summary["subgroup_definitions"])
        sample_count = len(sample_ids)

        for row in reader:
            if not row:
                continue
            feature = row[0].strip()
            subject_meta = subject_builder(feature)
            if not feature or subject_meta is None:
                continue
            values = [safe_float(v) for v in row[1:]]
            clean = [v for v in values if v is not None]
            if not clean:
                continue
            completeness = completeness_ratio(len(clean), sample_count)
            if completeness < 1.0 - RPPA_MAX_MISSING_RATE:
                continue

            bg_values = [mapping[feature] for other, mapping in overall_maps.items() if other != cohort and feature in mapping]
            if bg_values:
                cohort_median = median_or_none(clean)
                bg_median = median_or_none(bg_values)
                if cohort_median is not None and bg_median is not None:
                    delta = cohort_median - bg_median
                    effect = cohen_d(clean, bg_values)
                    if abs(delta) >= overall_min_median_delta and effect >= overall_min_effect_size:
                        predicate = positive_predicate if delta > 0 else negative_predicate
                        direction = "increase" if delta > 0 else "decrease"
                        summary = f"{feature} {predicate} {family} in {cohort} relative to selected TCGA pan-cancer protein background"
                        evidence, cohort_node, ents = make_evidence(
                            cohort=cohort,
                            family=family,
                            label=label,
                            predicate=predicate,
                            direction=direction,
                            evidence_category=evidence_category,
                            feature_layer=feature_layer,
                            metric=metric,
                            gene=subject_meta["subject_label"],
                            source_file=path,
                            value=delta,
                            effect_size=effect,
                            sample_count=sample_count,
                            comparison_id="pan_cancer_background",
                            comparison_type="lineage_enriched_vs_background",
                            sample_grouping="pan_cancer_background",
                            group_a_name=f"{cohort}_tumor_cohort",
                            group_b_name="selected_tcga_pan_cancer_background",
                            n_group_a=len(clean),
                            n_group_b=len(bg_values),
                            clinical_summary=clinical_summary,
                            summary_text=summary,
                            platform_specific_fields={
                                "background_cohorts": [c for c in overall_maps if c != cohort],
                                "group_axis": "overall_pan_cancer",
                                "assay_kind": assay_kind,
                                "cohort_median": round(cohort_median, 4),
                                "background_median": round(bg_median, 4),
                                "cohen_d": round(effect, 4),
                                "missing_rate": round(1.0 - completeness, 4),
                                **({"gene_anchor": subject_meta["gene_anchor"], "analyte_label": subject_meta["analyte_label"]} if "gene_anchor" in subject_meta else {}),
                            },
                            subject_id=subject_meta["subject_id"],
                            subject_entity_type=subject_meta["subject_entity_type"],
                            subject_aliases=subject_meta["subject_aliases"],
                            subject_db_source=subject_meta["subject_db_source"],
                            subject_normalized_from=subject_meta["subject_normalized_from"],
                            source_record_key=feature,
                            data_completeness=completeness,
                            statistical_method="thresholded RPPA cohort/background comparison using median shift, Cohen's d, and missingness gating on LinkedOmics TCGA matrices",
                        )
                        evidence_nodes.append(evidence)
                        cohort_nodes.append(cohort_node)
                        entity_nodes.extend(ents)
                        counts["overall_pan_cancer"] += 1

            for subgroup in subgroup_defs:
                group_a = [values[idx] for idx in subgroup["group_a_indices"] if values[idx] is not None]
                group_b = [values[idx] for idx in subgroup["group_b_indices"] if values[idx] is not None]
                if len(group_a) < MIN_GROUP_SIZE or len(group_b) < MIN_GROUP_SIZE:
                    continue
                coverage_a = completeness_ratio(len(group_a), len(subgroup["group_a_indices"]))
                coverage_b = completeness_ratio(len(group_b), len(subgroup["group_b_indices"]))
                if coverage_a < 1.0 - RPPA_MAX_MISSING_RATE or coverage_b < 1.0 - RPPA_MAX_MISSING_RATE:
                    continue
                median_a = median_or_none(group_a)
                median_b = median_or_none(group_b)
                if median_a is None or median_b is None:
                    continue
                delta = median_a - median_b
                effect = cohen_d(group_a, group_b)
                if abs(delta) < subgroup_min_median_delta or effect < subgroup_min_effect_size:
                    continue
                predicate = positive_predicate if delta > 0 else negative_predicate
                direction = "increase" if delta > 0 else "decrease"
                summary = f"{feature} {predicate} {family} in {cohort} for {comparison_label(subgroup['comparison_id'])}"
                evidence, cohort_node, ents = make_evidence(
                    cohort=cohort,
                    family=family,
                    label=label,
                    predicate=predicate,
                    direction=direction,
                    evidence_category=evidence_category,
                    feature_layer=feature_layer,
                    metric=metric,
                    gene=subject_meta["subject_label"],
                    source_file=path,
                    value=delta,
                    effect_size=effect,
                    sample_count=sample_count,
                    comparison_id=subgroup["comparison_id"],
                    comparison_type=subgroup["comparison_type"],
                    sample_grouping=subgroup["sample_grouping"],
                    group_a_name=subgroup["group_a_name"],
                    group_b_name=subgroup["group_b_name"],
                    n_group_a=len(group_a),
                    n_group_b=len(group_b),
                    clinical_summary=clinical_summary,
                    summary_text=summary,
                    platform_specific_fields={
                        "group_axis": subgroup["comparison_id"],
                        "assay_kind": assay_kind,
                        "group_a_median": round(median_a, 4),
                        "group_b_median": round(median_b, 4),
                        "cohen_d": round(effect, 4),
                        "group_a_missing_rate": round(1.0 - coverage_a, 4),
                        "group_b_missing_rate": round(1.0 - coverage_b, 4),
                        **({"gene_anchor": subject_meta["gene_anchor"], "analyte_label": subject_meta["analyte_label"]} if "gene_anchor" in subject_meta else {}),
                    },
                    subject_id=subject_meta["subject_id"],
                    subject_entity_type=subject_meta["subject_entity_type"],
                    subject_aliases=subject_meta["subject_aliases"],
                    subject_db_source=subject_meta["subject_db_source"],
                    subject_normalized_from=subject_meta["subject_normalized_from"],
                    source_record_key=feature,
                    data_completeness=min(coverage_a, coverage_b),
                    statistical_method="thresholded RPPA subgroup comparison using median shift, Cohen's d, and missingness gating on LinkedOmics TCGA matrices",
                )
                evidence_nodes.append(evidence)
                cohort_nodes.append(cohort_node)
                entity_nodes.extend(ents)
                counts[subgroup["comparison_id"]] += 1

    return evidence_nodes, cohort_nodes, entity_nodes, counts


def process_rppa_prognosis(
    *,
    path: Path,
    cohort: str,
    family: str,
    label: str,
    clinical_summary: dict[str, Any],
    feature_layer: str,
    high_label: str,
    poor_predicate: str,
    favorable_predicate: str,
    metric: str,
    subject_builder,
    assay_kind: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    evidence_nodes = []
    cohort_nodes = []
    entity_nodes = []
    counts = Counter()
    survival_records = clinical_summary["survival_records"]
    if len(survival_records) < PROGNOSIS_MIN_GROUP_SIZE * 2:
        return evidence_nodes, cohort_nodes, entity_nodes, counts

    with open_text(path) as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        sample_ids = header[1:]
        for row in reader:
            if not row:
                continue
            feature = row[0].strip()
            subject_meta = subject_builder(feature)
            if not feature or subject_meta is None:
                continue
            values = [safe_float(v) for v in row[1:]]
            pairs = []
            for sample_id, value in zip(sample_ids, values, strict=False):
                if value is None or sample_id not in survival_records:
                    continue
                record = survival_records[sample_id]
                pairs.append((sample_id, value, float(record["overall_survival_days"]), int(record["status"])))
            completeness = completeness_ratio(len(pairs), len(survival_records))
            if completeness < 1.0 - RPPA_MAX_MISSING_RATE or len(pairs) < PROGNOSIS_MIN_GROUP_SIZE * 2:
                continue
            pairs.sort(key=lambda item: item[1])
            half = len(pairs) // 2
            low_group = pairs[:half]
            high_group = pairs[-half:]
            if len(low_group) < PROGNOSIS_MIN_GROUP_SIZE or len(high_group) < PROGNOSIS_MIN_GROUP_SIZE:
                continue
            high_events = sum(item[3] for item in high_group)
            low_events = sum(item[3] for item in low_group)
            if high_events < PROGNOSIS_MIN_EVENT_COUNT and low_events < PROGNOSIS_MIN_EVENT_COUNT:
                continue
            high_median_os = median_or_none([item[2] for item in high_group])
            low_median_os = median_or_none([item[2] for item in low_group])
            if high_median_os is None or low_median_os is None or low_median_os == 0:
                continue
            median_ratio = high_median_os / low_median_os
            event_rate_delta = (high_events / len(high_group)) - (low_events / len(low_group))
            prognosis = prognosis_direction(median_ratio, event_rate_delta)
            if prognosis is None:
                continue
            predicate = poor_predicate if prognosis == "poor" else favorable_predicate
            direction = "negative" if prognosis == "poor" else "positive"
            effect_size = prognosis_effect_size(median_ratio, event_rate_delta)
            summary = (
                f"{subject_meta['subject_label']} {high_label} is associated with {'poorer' if prognosis == 'poor' else 'more favorable'} overall survival "
                f"in {family} ({cohort}); median OS ratio={round(median_ratio, 4)}, event-rate delta={round(event_rate_delta, 4)}"
            )
            evidence, cohort_node, ents = make_evidence(
                cohort=cohort,
                family=family,
                label=label,
                predicate=predicate,
                direction=direction,
                evidence_category="clinical_prognosis",
                feature_layer=feature_layer,
                metric=metric,
                gene=subject_meta["subject_label"],
                source_file=path,
                value=median_ratio,
                effect_size=effect_size,
                sample_count=len(high_group) + len(low_group),
                comparison_id="overall_survival_association",
                comparison_type="high_vs_low",
                sample_grouping="overall_survival_association",
                group_a_name=f"high_{feature_layer}",
                group_b_name=f"low_{feature_layer}",
                n_group_a=len(high_group),
                n_group_b=len(low_group),
                clinical_summary=clinical_summary,
                summary_text=summary,
                platform_specific_fields={
                    "prognosis_axis": "overall_survival",
                    "assay_kind": assay_kind,
                    "median_os_high_days": round(high_median_os, 4),
                    "median_os_low_days": round(low_median_os, 4),
                    "event_rate_high": round(high_events / len(high_group), 4),
                    "event_rate_low": round(low_events / len(low_group), 4),
                    "event_rate_delta": round(event_rate_delta, 4),
                    "missing_rate": round(1.0 - completeness, 4),
                    **({"gene_anchor": subject_meta["gene_anchor"], "analyte_label": subject_meta["analyte_label"]} if "gene_anchor" in subject_meta else {}),
                },
                subject_id=subject_meta["subject_id"],
                subject_entity_type=subject_meta["subject_entity_type"],
                subject_aliases=subject_meta["subject_aliases"],
                subject_db_source=subject_meta["subject_db_source"],
                subject_normalized_from=subject_meta["subject_normalized_from"],
                source_record_key=feature,
                data_completeness=completeness,
                statistical_method="overall survival contrast using RPPA-based molecular subgrouping with missingness gating on LinkedOmics TCGA clinical annotations",
            )
            evidence_nodes.append(evidence)
            cohort_nodes.append(cohort_node)
            entity_nodes.extend(ents)
            counts["overall_survival_association"] += 1

    return evidence_nodes, cohort_nodes, entity_nodes, counts


def process_mutation(
    *,
    path: Path,
    cohort: str,
    family: str,
    label: str,
    clinical_summary: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    evidence_nodes = []
    cohort_nodes = []
    entity_nodes = []
    counts = Counter()
    with open_text(path) as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        sample_ids = header[1:]
        subgroup_defs = subgroup_indices(sample_ids, clinical_summary["subgroup_definitions"])
        sample_count = len(sample_ids)
        for row in reader:
            if not row:
                continue
            gene = row[0].strip()
            values = [safe_float(v) for v in row[1:]]
            clean = [v for v in values if v is not None]
            if not gene or not clean:
                continue
            freq = sum(1 for v in clean if v > 0) / len(clean)
            if freq >= MUTATION_OVERALL_MIN_FREQ:
                summary = f"{gene} is recurrently mutated in {family} ({cohort})"
                evidence, cohort_node, ents = make_evidence(
                    cohort=cohort,
                    family=family,
                    label=label,
                    predicate="mutated_in",
                    direction="positive",
                    evidence_category="gene_variation",
                    feature_layer="mutation",
                    metric="mutation_frequency",
                    gene=gene,
                    source_file=path,
                    value=freq,
                    effect_size=freq,
                    sample_count=sample_count,
                    comparison_id="cohort_recurrent",
                    comparison_type=None,
                    sample_grouping="cohort_recurrence",
                    group_a_name=f"{cohort}_tumor_cohort",
                    group_b_name=None,
                    n_group_a=sample_count,
                    n_group_b=None,
                    clinical_summary=clinical_summary,
                    summary_text=summary,
                    platform_specific_fields={"group_axis": "cohort_recurrent"},
                )
                evidence_nodes.append(evidence)
                cohort_nodes.append(cohort_node)
                entity_nodes.extend(ents)
                counts["cohort_recurrent"] += 1

            for subgroup in subgroup_defs:
                group_a = [values[idx] for idx in subgroup["group_a_indices"] if values[idx] is not None]
                group_b = [values[idx] for idx in subgroup["group_b_indices"] if values[idx] is not None]
                if len(group_a) < MIN_GROUP_SIZE or len(group_b) < MIN_GROUP_SIZE:
                    continue
                freq_a = sum(1 for v in group_a if v > 0) / len(group_a)
                freq_b = sum(1 for v in group_b if v > 0) / len(group_b)
                delta = freq_a - freq_b
                if abs(delta) < MUTATION_SUBGROUP_MIN_DELTA or max(freq_a, freq_b) < MUTATION_OVERALL_MIN_FREQ:
                    continue
                direction = "positive" if delta > 0 else "negative"
                summary = f"{gene} mutation frequency differs in {family} ({cohort}) for {comparison_label(subgroup['comparison_id'])}"
                evidence, cohort_node, ents = make_evidence(
                    cohort=cohort,
                    family=family,
                    label=label,
                    predicate="mutated_in",
                    direction=direction,
                    evidence_category="gene_variation",
                    feature_layer="mutation",
                    metric="mutation_frequency_delta",
                    gene=gene,
                    source_file=path,
                    value=delta,
                    effect_size=abs(delta),
                    sample_count=sample_count,
                    comparison_id=subgroup["comparison_id"],
                    comparison_type=subgroup["comparison_type"],
                    sample_grouping=subgroup["sample_grouping"],
                    group_a_name=subgroup["group_a_name"],
                    group_b_name=subgroup["group_b_name"],
                    n_group_a=len(group_a),
                    n_group_b=len(group_b),
                    clinical_summary=clinical_summary,
                    summary_text=summary,
                    platform_specific_fields={"group_axis": subgroup["comparison_id"], "group_a_frequency": round(freq_a, 4), "group_b_frequency": round(freq_b, 4)},
                )
                evidence_nodes.append(evidence)
                cohort_nodes.append(cohort_node)
                entity_nodes.extend(ents)
                counts[subgroup["comparison_id"]] += 1

    return evidence_nodes, cohort_nodes, entity_nodes, counts


def process_scna(
    *,
    path: Path,
    cohort: str,
    family: str,
    label: str,
    clinical_summary: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    evidence_nodes = []
    cohort_nodes = []
    entity_nodes = []
    counts = Counter()
    with open_text(path) as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        sample_ids = header[1:]
        subgroup_defs = subgroup_indices(sample_ids, clinical_summary["subgroup_definitions"])
        sample_count = len(sample_ids)
        for row in reader:
            if not row:
                continue
            gene = row[0].strip()
            values = [safe_float(v) for v in row[1:]]
            clean = [v for v in values if v is not None]
            if not gene or not clean:
                continue
            amp_freq = sum(1 for v in clean if v >= CNV_AMP_THRESHOLD) / len(clean)
            del_freq = sum(1 for v in clean if v <= CNV_DEL_THRESHOLD) / len(clean)
            for predicate, freq, direction, tag in [
                ("amplified_in", amp_freq, "increase", "amp"),
                ("deleted_in", del_freq, "decrease", "del"),
            ]:
                if freq >= CNV_OVERALL_MIN_FREQ:
                    summary = f"{gene} is recurrently {predicate.replace('_in', '')} in {family} ({cohort})"
                    evidence, cohort_node, ents = make_evidence(
                        cohort=cohort,
                        family=family,
                        label=label,
                        predicate=predicate,
                        direction=direction,
                        evidence_category="gene_variation",
                        feature_layer="copy_number",
                        metric=f"copy_number_{tag}_frequency",
                        gene=gene,
                        source_file=path,
                        value=freq,
                        effect_size=freq,
                        sample_count=sample_count,
                        comparison_id="cohort_recurrent",
                        comparison_type=None,
                        sample_grouping="cohort_recurrence",
                        group_a_name=f"{cohort}_tumor_cohort",
                        group_b_name=None,
                        n_group_a=sample_count,
                        n_group_b=None,
                        clinical_summary=clinical_summary,
                        summary_text=summary,
                        platform_specific_fields={"group_axis": "cohort_recurrent"},
                    )
                    evidence_nodes.append(evidence)
                    cohort_nodes.append(cohort_node)
                    entity_nodes.extend(ents)
                    counts[f"cohort_recurrent_{tag}"] += 1

            for subgroup in subgroup_defs:
                group_a = [values[idx] for idx in subgroup["group_a_indices"] if values[idx] is not None]
                group_b = [values[idx] for idx in subgroup["group_b_indices"] if values[idx] is not None]
                if len(group_a) < MIN_GROUP_SIZE or len(group_b) < MIN_GROUP_SIZE:
                    continue
                for predicate, threshold, direction, tag in [
                    ("amplified_in", CNV_AMP_THRESHOLD, "increase", "amp"),
                    ("deleted_in", CNV_DEL_THRESHOLD, "decrease", "del"),
                ]:
                    if tag == "amp":
                        freq_a = sum(1 for v in group_a if v >= threshold) / len(group_a)
                        freq_b = sum(1 for v in group_b if v >= threshold) / len(group_b)
                    else:
                        freq_a = sum(1 for v in group_a if v <= threshold) / len(group_a)
                        freq_b = sum(1 for v in group_b if v <= threshold) / len(group_b)
                    delta = freq_a - freq_b
                    if abs(delta) < CNV_SUBGROUP_MIN_DELTA or max(freq_a, freq_b) < CNV_OVERALL_MIN_FREQ:
                        continue
                    summary = f"{gene} {predicate.replace('_in', '')} frequency differs in {family} ({cohort}) for {comparison_label(subgroup['comparison_id'])}"
                    evidence, cohort_node, ents = make_evidence(
                        cohort=cohort,
                        family=family,
                        label=label,
                        predicate=predicate,
                        direction=direction if delta > 0 else ("decrease" if direction == "increase" else "increase"),
                        evidence_category="gene_variation",
                        feature_layer="copy_number",
                        metric=f"copy_number_{tag}_frequency_delta",
                        gene=gene,
                        source_file=path,
                        value=delta,
                        effect_size=abs(delta),
                        sample_count=sample_count,
                        comparison_id=subgroup["comparison_id"],
                        comparison_type=subgroup["comparison_type"],
                        sample_grouping=subgroup["sample_grouping"],
                        group_a_name=subgroup["group_a_name"],
                        group_b_name=subgroup["group_b_name"],
                        n_group_a=len(group_a),
                        n_group_b=len(group_b),
                        clinical_summary=clinical_summary,
                        summary_text=summary,
                        platform_specific_fields={"group_axis": subgroup["comparison_id"], "group_a_frequency": round(freq_a, 4), "group_b_frequency": round(freq_b, 4)},
                    )
                    evidence_nodes.append(evidence)
                    cohort_nodes.append(cohort_node)
                    entity_nodes.extend(ents)
                    counts[f"{subgroup['comparison_id']}_{tag}"] += 1
    return evidence_nodes, cohort_nodes, entity_nodes, counts


def collect_evidence(
    cohort_data: dict[str, Any],
    expression_means: dict[str, dict[str, float]],
    methylation_means: dict[str, dict[str, float]],
    rppa_gene_medians: dict[str, dict[str, float]],
    rppa_phospho_medians: dict[str, dict[str, float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    evidence_nodes = []
    cohort_nodes = []
    entity_nodes = []
    cohort_stats: dict[str, Any] = {}

    for cohort, payload in cohort_data.items():
        config = payload["config"]
        clinical_summary = payload["clinical_summary"]
        local_counts = Counter()
        print(f"[extract] {cohort}: starting evidence extraction", flush=True)

        processors = [
            process_expression_like(
                path=payload["paths"]["expression"],
                cohort=cohort,
                family=config["family"],
                label=config["label"],
                clinical_summary=clinical_summary,
                overall_maps=expression_means,
                overall_threshold=EXPR_OVERALL_MIN_DELTA,
                subgroup_threshold=EXPR_SUBGROUP_MIN_DELTA,
                evidence_category="expression_regulation",
                positive_predicate="upregulated_in",
                negative_predicate="downregulated_in",
                metric="log2_expression_delta",
                feature_layer="expression",
            ),
            process_expression_like(
                path=payload["paths"]["methylation"],
                cohort=cohort,
                family=config["family"],
                label=config["label"],
                clinical_summary=clinical_summary,
                overall_maps=methylation_means,
                overall_threshold=METHYL_OVERALL_MIN_DELTA,
                subgroup_threshold=METHYL_SUBGROUP_MIN_DELTA,
                evidence_category="epigenetic_regulation",
                positive_predicate="hypermethylated_in",
                negative_predicate="hypomethylated_in",
                metric="gene_methylation_delta",
                feature_layer="methylation",
            ),
            process_mutation(
                path=payload["paths"]["mutation"],
                cohort=cohort,
                family=config["family"],
                label=config["label"],
                clinical_summary=clinical_summary,
            ),
            process_prognosis_continuous(
                path=payload["paths"]["expression"],
                cohort=cohort,
                family=config["family"],
                label=config["label"],
                clinical_summary=clinical_summary,
                feature_layer="expression",
                high_label="high expression",
                poor_predicate="high_expression_associated_with_poor_prognosis_in",
                favorable_predicate="high_expression_associated_with_favorable_prognosis_in",
                metric="overall_survival_high_vs_low_ratio",
            ),
            process_prognosis_continuous(
                path=payload["paths"]["methylation"],
                cohort=cohort,
                family=config["family"],
                label=config["label"],
                clinical_summary=clinical_summary,
                feature_layer="methylation",
                high_label="hypermethylation",
                poor_predicate="hypermethylation_associated_with_poor_prognosis_in",
                favorable_predicate="hypermethylation_associated_with_favorable_prognosis_in",
                metric="overall_survival_high_vs_low_ratio",
            ),
            process_prognosis_binary(
                path=payload["paths"]["mutation"],
                cohort=cohort,
                family=config["family"],
                label=config["label"],
                clinical_summary=clinical_summary,
                feature_layer="mutation",
                altered_label="mutation",
                poor_predicate="mutation_associated_with_poor_prognosis_in",
                favorable_predicate="mutation_associated_with_favorable_prognosis_in",
                metric="overall_survival_altered_vs_unaltered_ratio",
                altered_fn=lambda value: value > 0,
            ),
        ]
        if payload["paths"]["scna"] is not None:
            processors.append(
                process_scna(
                    path=payload["paths"]["scna"],
                    cohort=cohort,
                    family=config["family"],
                    label=config["label"],
                    clinical_summary=clinical_summary,
                )
            )
            processors.append(
                process_prognosis_binary(
                    path=payload["paths"]["scna"],
                    cohort=cohort,
                    family=config["family"],
                    label=config["label"],
                    clinical_summary=clinical_summary,
                    feature_layer="copy_number_amplification",
                    altered_label="amplification",
                    poor_predicate="amplification_associated_with_poor_prognosis_in",
                    favorable_predicate="amplification_associated_with_favorable_prognosis_in",
                    metric="overall_survival_altered_vs_unaltered_ratio",
                    altered_fn=lambda value: value >= CNV_AMP_THRESHOLD,
                )
            )
            processors.append(
                process_prognosis_binary(
                    path=payload["paths"]["scna"],
                    cohort=cohort,
                    family=config["family"],
                    label=config["label"],
                    clinical_summary=clinical_summary,
                    feature_layer="copy_number_deletion",
                    altered_label="deletion",
                    poor_predicate="deletion_associated_with_poor_prognosis_in",
                    favorable_predicate="deletion_associated_with_favorable_prognosis_in",
                    metric="overall_survival_altered_vs_unaltered_ratio",
                    altered_fn=lambda value: value <= CNV_DEL_THRESHOLD,
                )
            )
        if payload["paths"]["rppa_gene"] is not None and cohort in rppa_gene_medians:
            processors.append(
                process_rppa_signal(
                    path=payload["paths"]["rppa_gene"],
                    cohort=cohort,
                    family=config["family"],
                    label=config["label"],
                    clinical_summary=clinical_summary,
                    overall_maps=rppa_gene_medians,
                    overall_min_median_delta=RPPA_GENE_OVERALL_MIN_MEDIAN_DELTA,
                    overall_min_effect_size=RPPA_GENE_OVERALL_MIN_COHEN_D,
                    subgroup_min_median_delta=RPPA_GENE_SUBGROUP_MIN_MEDIAN_DELTA,
                    subgroup_min_effect_size=RPPA_GENE_SUBGROUP_MIN_COHEN_D,
                    evidence_category="protein_abundance",
                    positive_predicate="upregulated_in",
                    negative_predicate="downregulated_in",
                    metric="rppa_protein_median_delta",
                    feature_layer="protein_abundance",
                    subject_builder=build_rppa_gene_subject,
                    assay_kind="rppa_gene",
                )
            )
            processors.append(
                process_rppa_prognosis(
                    path=payload["paths"]["rppa_gene"],
                    cohort=cohort,
                    family=config["family"],
                    label=config["label"],
                    clinical_summary=clinical_summary,
                    feature_layer="protein_abundance",
                    high_label="high protein abundance",
                    poor_predicate="high_protein_abundance_associated_with_poor_prognosis_in",
                    favorable_predicate="high_protein_abundance_associated_with_favorable_prognosis_in",
                    metric="overall_survival_high_vs_low_ratio",
                    subject_builder=build_rppa_gene_subject,
                    assay_kind="rppa_gene",
                )
            )
        if payload["paths"]["rppa_analyte"] is not None and cohort in rppa_phospho_medians:
            processors.append(
                process_rppa_signal(
                    path=payload["paths"]["rppa_analyte"],
                    cohort=cohort,
                    family=config["family"],
                    label=config["label"],
                    clinical_summary=clinical_summary,
                    overall_maps=rppa_phospho_medians,
                    overall_min_median_delta=RPPA_PHOSPHO_OVERALL_MIN_MEDIAN_DELTA,
                    overall_min_effect_size=RPPA_PHOSPHO_OVERALL_MIN_COHEN_D,
                    subgroup_min_median_delta=RPPA_PHOSPHO_SUBGROUP_MIN_MEDIAN_DELTA,
                    subgroup_min_effect_size=RPPA_PHOSPHO_SUBGROUP_MIN_COHEN_D,
                    evidence_category="protein_activation",
                    positive_predicate="hyperphosphorylated_in",
                    negative_predicate="hypophosphorylated_in",
                    metric="rppa_phospho_median_delta",
                    feature_layer="protein_activation",
                    subject_builder=build_rppa_phospho_subject,
                    assay_kind="rppa_phospho",
                )
            )
            processors.append(
                process_rppa_prognosis(
                    path=payload["paths"]["rppa_analyte"],
                    cohort=cohort,
                    family=config["family"],
                    label=config["label"],
                    clinical_summary=clinical_summary,
                    feature_layer="protein_activation",
                    high_label="high phosphoprotein signal",
                    poor_predicate="high_phosphoprotein_signal_associated_with_poor_prognosis_in",
                    favorable_predicate="high_phosphoprotein_signal_associated_with_favorable_prognosis_in",
                    metric="overall_survival_high_vs_low_ratio",
                    subject_builder=build_rppa_phospho_subject,
                    assay_kind="rppa_phospho",
                )
            )

        for nodes, cohorts_, entities, counts in processors:
            evidence_nodes.extend(nodes)
            cohort_nodes.extend(cohorts_)
            entity_nodes.extend(entities)
            local_counts.update(counts)
            print(f"[extract] {cohort}: processor added {sum(counts.values())} evidence -> {dict(counts)}", flush=True)

        cohort_stats[cohort] = {
            "family": config["family"],
            "clinical_summary": clinical_summary_for_output(clinical_summary),
            "selected_atomic_evidence_counts": dict(sorted(local_counts.items())),
        }
        print(f"[extract] {cohort}: total selected {sum(local_counts.values())}", flush=True)

    uniq_cohorts = sorted({node["id"]: node for node in cohort_nodes}.values(), key=lambda item: item["id"])
    uniq_entities = sorted({node["id"]: node for node in entity_nodes}.values(), key=lambda item: item["id"])
    return evidence_nodes, uniq_cohorts, uniq_entities, cohort_stats


def select_final_evidence(evidence_nodes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = sorted(evidence_nodes, key=lambda item: item["id"])
    summary = {
        "strategy": "threshold_only",
        "rationale": "Final reasoning graph keeps all evidence passing biologically motivated per-type thresholds and does not apply additional top-k truncation.",
        "pre_count": len(evidence_nodes),
        "post_count": len(selected),
    }
    return selected, summary


def filter_support_nodes(
    evidence_nodes: list[dict[str, Any]],
    cohort_nodes: list[dict[str, Any]],
    entity_nodes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cohort_ids = {evidence["cohort_id"] for evidence in evidence_nodes}
    entity_ids = set()
    for evidence in evidence_nodes:
        entity_ids.add(evidence["subject"]["id"])
        if evidence.get("object") and evidence["object"].get("id"):
            entity_ids.add(evidence["object"]["id"])
        if evidence.get("outcome") and evidence["outcome"].get("id"):
            entity_ids.add(evidence["outcome"]["id"])
    selected_cohorts = sorted((node for node in cohort_nodes if node["id"] in cohort_ids), key=lambda item: item["id"])
    selected_entities = sorted((node for node in entity_nodes if node["id"] in entity_ids), key=lambda item: item["id"])
    return selected_cohorts, selected_entities


def build_edges(evidence_nodes: list[dict[str, Any]], cohort_nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cohort_ids = {node["id"] for node in cohort_nodes}
    edges: list[dict[str, Any]] = []
    for evidence in evidence_nodes:
        cohort_id = evidence["cohort_id"]
        gene_id = evidence["subject"]["id"]
        subject_type = evidence["subject"]["entity_type"]
        disease_id = evidence["object"]["id"] if evidence.get("object") else None
        if cohort_id in cohort_ids:
            edges.append(
                {
                    "source": evidence["id"],
                    "target": cohort_id,
                    "relation": "OBSERVED_IN",
                    "relation_family": "semantic_anchor",
                    "entity_type": "Cohort",
                    "linking_score": None,
                    "similarity_score": None,
                    "rationale": "TCGA atomic evidence observed in TCGA cohort context.",
                    "support_strength": None,
                    "evidence_count": 1,
                    "created_by_rule": "tcga_atomic:observed_in",
                }
            )
        edges.extend(
            [
                {
                    "source": evidence["id"],
                    "target": gene_id,
                    "relation": "LINKED_TO",
                    "relation_family": "semantic_anchor",
                    "entity_type": subject_type,
                    "linking_score": 1.0,
                    "similarity_score": None,
                    "rationale": "Evidence subject entity link.",
                    "support_strength": None,
                    "evidence_count": 1,
                    "created_by_rule": "tcga_atomic:link_subject_entity",
                },
                {
                    "source": evidence["id"],
                    "target": disease_id,
                    "relation": "LINKED_TO",
                    "relation_family": "semantic_anchor",
                    "entity_type": "Disease",
                    "linking_score": 1.0,
                    "similarity_score": None,
                    "rationale": "Evidence target disease link.",
                    "support_strength": None,
                    "evidence_count": 1,
                    "created_by_rule": "tcga_atomic:link_disease",
                },
            ]
        )
    return edges


def summarize_selected_cohorts(evidence_nodes: list[dict[str, Any]], cohort_stats: dict[str, Any]) -> dict[str, Any]:
    selected_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for evidence in evidence_nodes:
        cohort_name = evidence["provenance"]["sequencing_or_screen_id"]
        grouping = evidence["comparison"]["sample_grouping"]
        selected_counts[cohort_name][grouping] += 1

    summary = {}
    for cohort_name, stats in cohort_stats.items():
        summary[cohort_name] = {
            **stats,
            "selected_compact_counts_by_grouping": dict(sorted(selected_counts.get(cohort_name, Counter()).items())),
        }
    return summary


def build_graph(evidence_nodes: list[dict[str, Any]], cohort_nodes: list[dict[str, Any]], entity_nodes: list[dict[str, Any]], source_manifest: dict[str, Any], selection_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "2.0.0",
        "graph_metadata": {
            "graph_id": "tcga_atomic_compact_latest",
            "graph_name": "Compact TCGA atomic evidence aligned to compact unified multisource KG schema",
            "created_by": "scripts/extract_tcga_atomic_evidence_v2.py",
            "created_at": now_iso(),
            "source_families": ["tcga_linkedomics"],
            "disease_scope": sorted({node["disease"] for node in evidence_nodes}),
            "build_parameters": {
                "priority_cohorts": PRIORITY_COHORTS,
                "thresholds": {
                    "expr_overall_min_delta": EXPR_OVERALL_MIN_DELTA,
                    "expr_subgroup_min_delta": EXPR_SUBGROUP_MIN_DELTA,
                    "methyl_overall_min_delta": METHYL_OVERALL_MIN_DELTA,
                    "methyl_subgroup_min_delta": METHYL_SUBGROUP_MIN_DELTA,
                    "mutation_overall_min_freq": MUTATION_OVERALL_MIN_FREQ,
                    "mutation_subgroup_min_delta": MUTATION_SUBGROUP_MIN_DELTA,
                    "cnv_overall_min_freq": CNV_OVERALL_MIN_FREQ,
                    "cnv_subgroup_min_delta": CNV_SUBGROUP_MIN_DELTA,
                    "cnv_amp_threshold": CNV_AMP_THRESHOLD,
                    "cnv_del_threshold": CNV_DEL_THRESHOLD,
                    "rppa_max_missing_rate": RPPA_MAX_MISSING_RATE,
                    "rppa_gene_overall_min_median_delta": RPPA_GENE_OVERALL_MIN_MEDIAN_DELTA,
                    "rppa_gene_overall_min_cohen_d": RPPA_GENE_OVERALL_MIN_COHEN_D,
                    "rppa_gene_subgroup_min_median_delta": RPPA_GENE_SUBGROUP_MIN_MEDIAN_DELTA,
                    "rppa_gene_subgroup_min_cohen_d": RPPA_GENE_SUBGROUP_MIN_COHEN_D,
                    "rppa_phospho_overall_min_median_delta": RPPA_PHOSPHO_OVERALL_MIN_MEDIAN_DELTA,
                    "rppa_phospho_overall_min_cohen_d": RPPA_PHOSPHO_OVERALL_MIN_COHEN_D,
                    "rppa_phospho_subgroup_min_median_delta": RPPA_PHOSPHO_SUBGROUP_MIN_MEDIAN_DELTA,
                    "rppa_phospho_subgroup_min_cohen_d": RPPA_PHOSPHO_SUBGROUP_MIN_COHEN_D,
                    "min_group_size": MIN_GROUP_SIZE,
                },
                "rationale": {
                    "expression": "Absolute log2 mean difference >= 2.5 for pan-cancer contrast and >= 1.5 for within-cohort subgroup contrasts, retaining only strong transcriptomic shifts.",
                    "methylation": "Absolute gene-level methylation shift >= 0.25 for pan-cancer contrast and >= 0.15 for subgroup contrasts, retaining only stronger methylation changes.",
                    "mutation": "Recurrent mutation evidence requires cohort frequency >= 10%; subgroup enrichment requires absolute frequency difference >= 8%, allowing a modest recovery of clearly recurrent mutational events.",
                    "copy_number": "Cohort recurrent CNV evidence requires amplification/deletion frequency >= 50%; subgroup enrichment requires absolute frequency difference >= 25%, restricting the final graph to only the strongest and broadest copy-number events.",
                    "protein_abundance": "RPPA total-protein evidence requires missingness <= 20% plus median-shift and Cohen's d thresholds (overall >= 0.50 and 0.80; subgroup >= 0.35 and 0.60) so only robust protein-level signals enter the final graph.",
                    "protein_activation": "RPPA phospho-analyte evidence requires explicit phospho markers, missingness <= 20%, and stricter median-shift/Cohen's d thresholds (overall >= 0.50 and 0.90; subgroup >= 0.40 and 0.70) to emphasize higher-specificity activation signals.",
                    "prognosis": "Overall-survival association requires subgroup size >= 40, event count >= 15 in at least one arm, event-rate delta >= 0.20, and median-OS ratio >= 1.40, retaining only strong directional prognosis signals.",
                    "selection": "The final reasoning graph is threshold-defined rather than top-k-truncated so graph membership follows explicit biological entry criteria.",
                },
                "selection": selection_summary,
                "source_manifest": source_manifest,
            },
            "notes": "Final TCGA reasoning graph derived from explicit per-type inclusion thresholds, now including RPPA total-protein and phospho-analyte evidence under the same compact evidence-node schema and without additional top-k truncation.",
        },
        "nodes": sorted(evidence_nodes + cohort_nodes + entity_nodes, key=lambda item: (item["node_type"], item["id"])),
        "edges": build_edges(evidence_nodes, cohort_nodes),
    }


def fast_validate(graph: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    node_ids = [node["id"] for node in graph["nodes"]]
    if len(node_ids) != len(set(node_ids)):
        errors.append("duplicate node ids")
    known = set(node_ids)
    for edge in graph["edges"]:
        if edge["relation"] not in ALLOWED_EDGE_RELATIONS:
            errors.append(f"invalid relation: {edge['relation']}")
        if edge["source"] not in known or edge["target"] not in known:
            errors.append(f"orphan edge: {edge['source']}->{edge['target']}")
    return errors


def build_stats(graph: dict[str, Any], cohort_stats: dict[str, Any], alignment_summary: dict[str, Any]) -> dict[str, Any]:
    evidence_nodes = [node for node in graph["nodes"] if node["node_type"] == "evidence"]
    return {
        "graph_id": graph["graph_metadata"]["graph_id"],
        "disease_scope": graph["graph_metadata"]["disease_scope"],
        "node_counts": {
            "total": len(graph["nodes"]),
            "evidence": len(evidence_nodes),
            "entity": sum(1 for node in graph["nodes"] if node["node_type"] == "entity"),
            "cohort": sum(1 for node in graph["nodes"] if node["node_type"] == "cohort"),
        },
        "edge_counts": {
            "total": len(graph["edges"]),
            "by_relation": dict(sorted(Counter(edge["relation"] for edge in graph["edges"]).items())),
        },
        "evidence_distributions": {
            "by_disease": dict(sorted(Counter(node["disease"] for node in evidence_nodes).items())),
            "by_evidence_category": dict(sorted(Counter(node["evidence_category"] for node in evidence_nodes).items())),
            "by_feature_layer": dict(sorted(Counter(node["feature_layer"] for node in evidence_nodes).items())),
            "by_predicate": dict(sorted(Counter(node["predicate"] for node in evidence_nodes).items())),
            "by_sample_grouping": dict(sorted(Counter((node["comparison"]["sample_grouping"] or "none") for node in evidence_nodes).items())),
            "by_comparison_label": dict(sorted(Counter((node["comparison"]["comparison_label"] or "none") for node in evidence_nodes).items())),
            "by_grade": dict(sorted(Counter(node["confidence"]["grade"] for node in evidence_nodes).items())),
        },
        "cohort_stats": cohort_stats,
        "quality": {
            "confidence_score": summarize_numeric([node["confidence"]["score"] for node in evidence_nodes]),
            "effect_size": summarize_numeric([node["statistics"]["effect_size"] for node in evidence_nodes]),
            "data_completeness": summarize_numeric([node["confidence"]["data_completeness"] for node in evidence_nodes]),
            "by_feature_layer": {
                layer: {
                    "count": len(nodes),
                    "confidence_score": summarize_numeric([node["confidence"]["score"] for node in nodes]),
                    "effect_size": summarize_numeric([node["statistics"]["effect_size"] for node in nodes]),
                    "data_completeness": summarize_numeric([node["confidence"]["data_completeness"] for node in nodes]),
                }
                for layer, nodes in sorted(
                    (
                        layer,
                        [node for node in evidence_nodes if node["feature_layer"] == layer],
                    )
                    for layer in {node["feature_layer"] for node in evidence_nodes}
                )
            },
        },
        "pubmed_alignment": alignment_summary,
    }


def build_audit(graph: dict[str, Any], validation_errors: list[str], source_manifest: dict[str, Any], alignment_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    node_ids = {node["id"] for node in graph["nodes"]}
    orphan_edges = [edge for edge in graph["edges"] if edge["source"] not in node_ids or edge["target"] not in node_ids]
    evidence_nodes = [node for node in graph["nodes"] if node["node_type"] == "evidence"]
    return {
        "graph_id": graph["graph_metadata"]["graph_id"],
        "timestamp": graph["graph_metadata"]["created_at"],
        "input_sources": source_manifest,
        "checks": {
            "schema_validation": {
                "validator": "basic_fast_validation",
                "passed": not validation_errors,
                "error_count": len(validation_errors),
                "errors": validation_errors,
            },
            "orphan_edges": {
                "count": len(orphan_edges),
                "examples": orphan_edges[:10],
            },
            "required_field_completeness": {
                "source_file": round(sum(1 for node in evidence_nodes if node["provenance"]["source_file"]) / len(evidence_nodes), 4),
                "sample_size": round(sum(1 for node in evidence_nodes if node["statistics"]["sample_size"] is not None) / len(evidence_nodes), 4),
                "effect_size": round(sum(1 for node in evidence_nodes if node["statistics"]["effect_size"] is not None) / len(evidence_nodes), 4),
                "subgroup_metadata": round(sum(1 for node in evidence_nodes if node["comparison"]["sample_grouping"]) / len(evidence_nodes), 4),
            },
            "pubmed_alignment_candidate_count": len(alignment_candidates),
        },
    }


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    print("[main] loading TCGA sources", flush=True)
    cohort_data, source_manifest, expression_means, methylation_means, rppa_gene_medians, rppa_phospho_medians = load_sources()
    print("[main] collecting evidence", flush=True)
    evidence_nodes, cohort_nodes, entity_nodes, cohort_stats = collect_evidence(
        cohort_data,
        expression_means,
        methylation_means,
        rppa_gene_medians,
        rppa_phospho_medians,
    )
    print(
        f"[main] evidence collected: evidence={len(evidence_nodes)} cohorts={len(cohort_nodes)} entities={len(entity_nodes)}",
        flush=True,
    )
    selected_evidence_nodes, selection_summary = select_final_evidence(evidence_nodes)
    selected_cohort_nodes, selected_entity_nodes = filter_support_nodes(selected_evidence_nodes, cohort_nodes, entity_nodes)
    compact_cohort_stats = summarize_selected_cohorts(selected_evidence_nodes, cohort_stats)
    print(
        f"[main] threshold selection: selected={len(selected_evidence_nodes)} cohorts={len(selected_cohort_nodes)} entities={len(selected_entity_nodes)}",
        flush=True,
    )
    graph = build_graph(selected_evidence_nodes, selected_cohort_nodes, selected_entity_nodes, source_manifest, selection_summary)
    print("[main] graph assembled", flush=True)
    validation_errors = fast_validate(graph)
    print(f"[main] validation done: errors={len(validation_errors)}", flush=True)
    alignment_candidates, alignment_summary = build_pubmed_alignment_candidates(selected_evidence_nodes)
    print(f"[main] pubmed alignment candidates={len(alignment_candidates)}", flush=True)
    stats = build_stats(graph, compact_cohort_stats, alignment_summary)
    audit = build_audit(graph, validation_errors, source_manifest, alignment_candidates)

    (OUTPUT_DIR / "tcga_atomic_compact_latest.json").write_text(json.dumps(graph, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("[main] wrote tcga_atomic_compact_latest.json", flush=True)
    (OUTPUT_DIR / "tcga_atomic_compact_latest_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "tcga_atomic_compact_latest_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "tcga_pubmed_alignment_candidates_compact_latest.json").write_text(json.dumps(alignment_candidates, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("[main] finished all outputs", flush=True)


if __name__ == "__main__":
    main()

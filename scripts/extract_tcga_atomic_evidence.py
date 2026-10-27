from __future__ import annotations

import csv
import gzip
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
except ImportError:  # pragma: no cover
    jsonschema = None


ROOT = Path(__file__).resolve().parents[1]
from knowledge_build_paths import LINKEDOMICS_ROOT, source_relpath
SCHEMA_PATH = ROOT / "kg" / "schema" / "multisource_evidence_kg_schema_v1.json"
PUBMED_ALIGNED_PATH = ROOT / "kg" / "aligned" / "pubmed_aligned_latest.json"
OUTPUT_DIR = ROOT / "kg" / "aligned"

PRIORITY_COHORTS = [
    {"cohort": "LIHC-TCGA", "family": "HCC", "label": "Hepatocellular carcinoma"},
    {"cohort": "COADREAD-TCGA", "family": "CRC", "label": "Colorectal adenocarcinoma"},
    {"cohort": "BRCA-TCGA", "family": "BRCA", "label": "Breast carcinoma"},
    {"cohort": "LUAD-TCGA", "family": "NSCLC", "label": "Lung adenocarcinoma"},
    {"cohort": "LUSC-TCGA", "family": "NSCLC", "label": "Lung squamous cell carcinoma"},
    {"cohort": "GBM-TCGA", "family": "GLIOMA", "label": "Glioblastoma"},
    {"cohort": "OV-TCGA", "family": "OVARIAN", "label": "Ovarian carcinoma"},
    {"cohort": "KIRC-TCGA", "family": "RCC", "label": "Renal clear cell carcinoma"},
    {"cohort": "LAML-TCGA", "family": "AML", "label": "Acute myeloid leukemia"},
    {"cohort": "PAAD-TCGA", "family": "PAAD", "label": "Pancreatic adenocarcinoma"},
    {"cohort": "HNSC-TCGA", "family": "HNSC", "label": "Head and neck squamous cell carcinoma"},
    {"cohort": "STAD-TCGA", "family": "GASTROESOPH", "label": "Stomach adenocarcinoma"},
]

TOP_N = {
    "expression_up": 80,
    "expression_down": 80,
    "mutation": 80,
    "amplification": 60,
    "deletion": 60,
}

EXPRESSION_MIN_DELTA = 1.0
MUTATION_MIN_FREQ = 0.05
MUTATION_MIN_DELTA = 0.03
CNV_MIN_FREQ = 0.08
CNV_MIN_DELTA = 0.03
CNV_AMP_THRESHOLD = 0.3
CNV_DEL_THRESHOLD = -0.3

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
    "ASSERTS",
    "OBSERVED_IN",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_hash(*parts: str) -> str:
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()[:16].upper()


def relpath(path: Path) -> str:
    return source_relpath(path)


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


def mean_clean(values: list[float]) -> float | None:
    return mean(values) if values else None


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return path.open("r", encoding="utf-8", errors="ignore")


def find_file(cohort_dir: Path, predicate) -> Path | None:
    for path in sorted(cohort_dir.iterdir()):
        if path.is_file() and predicate(path.name):
            return path
    return None


def parse_matrix_header(path: Path) -> tuple[list[str], int]:
    with open_text(path) as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
    sample_ids = header[1:]
    return sample_ids, len(sample_ids)


def parse_expression_gene_means(path: Path) -> tuple[dict[str, float], int]:
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
            values = [safe_float(value) for value in row[1:]]
            clean = [value for value in values if value is not None]
            if not gene or not clean:
                continue
            gene_means[gene] = mean(clean)
    return gene_means, sample_count


def parse_mutation_frequencies(path: Path) -> tuple[dict[str, float], int]:
    gene_freqs: dict[str, float] = {}
    sample_count = 0
    with open_text(path) as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        sample_count = len(header) - 1
        for row in reader:
            if not row:
                continue
            gene = row[0].strip()
            values = [safe_float(value) for value in row[1:]]
            clean = [value for value in values if value is not None]
            if not gene or not clean:
                continue
            gene_freqs[gene] = sum(1 for value in clean if value > 0) / len(clean)
    return gene_freqs, sample_count


def parse_scna_profiles(path: Path) -> tuple[dict[str, dict[str, float]], int]:
    gene_profiles: dict[str, dict[str, float]] = {}
    sample_count = 0
    with open_text(path) as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        sample_count = len(header) - 1
        for row in reader:
            if not row:
                continue
            gene = row[0].strip()
            values = [safe_float(value) for value in row[1:]]
            clean = [value for value in values if value is not None]
            if not gene or not clean:
                continue
            gene_profiles[gene] = {
                "mean_value": mean(clean),
                "amp_freq": sum(1 for value in clean if value >= CNV_AMP_THRESHOLD) / len(clean),
                "del_freq": sum(1 for value in clean if value <= CNV_DEL_THRESHOLD) / len(clean),
            }
    return gene_profiles, sample_count


def parse_clinical_summary(path: Path) -> dict[str, Any]:
    row_map: dict[str, list[str]] = {}
    with open_text(path) as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        sample_ids = header[1:]
        for row in reader:
            if not row:
                continue
            row_map[row[0]] = row[1:]

    stage_values = [value for value in row_map.get("pathologic_stage", []) if value and value != "NA"]
    purity_values = [safe_float(value) for value in row_map.get("Tumor_purity", [])]
    purity_clean = [value for value in purity_values if value is not None]
    return {
        "clinical_sample_count": len(sample_ids),
        "pathologic_stage_distribution": dict(sorted(Counter(stage_values).items())),
        "mean_tumor_purity": round(mean(purity_clean), 4) if purity_clean else None,
    }


def load_tcga_sources() -> tuple[dict[str, Any], dict[str, Any]]:
    cohort_data: dict[str, Any] = {}
    source_manifest: dict[str, Any] = {}
    for config in PRIORITY_COHORTS:
        cohort = config["cohort"]
        cohort_dir = LINKEDOMICS_ROOT / cohort
        expr_path = find_file(cohort_dir, lambda name: "RNAseq" in name and "RSEM_log2.cct.gz" in name)
        mut_path = find_file(cohort_dir, lambda name: "Mutation" in name and "Gene__Firehose_MutSig2CV.cbt" in name)
        scna_path = find_file(cohort_dir, lambda name: "SCNA" in name and "Gene__Firehose_GISTIC2.cct.gz" in name)
        clinical_path = find_file(cohort_dir, lambda name: "Clinical" in name and name.endswith(".tsi"))
        if not expr_path or not mut_path or not clinical_path:
            raise FileNotFoundError(f"Missing required files for {cohort}")

        expr_means, expr_n = parse_expression_gene_means(expr_path)
        mut_freqs, mut_n = parse_mutation_frequencies(mut_path)
        scna_profiles, scna_n = parse_scna_profiles(scna_path) if scna_path else ({}, 0)
        clinical_summary = parse_clinical_summary(clinical_path)

        cohort_data[cohort] = {
            "config": config,
            "paths": {
                "expression": expr_path,
                "mutation": mut_path,
                "scna": scna_path,
                "clinical": clinical_path,
            },
            "expression_means": expr_means,
            "expression_n": expr_n,
            "mutation_freqs": mut_freqs,
            "mutation_n": mut_n,
            "scna_profiles": scna_profiles,
            "scna_n": scna_n,
            "clinical_summary": clinical_summary,
        }
        source_manifest[cohort] = {
            "family": config["family"],
            "label": config["label"],
            "expression_path": relpath(expr_path),
            "mutation_path": relpath(mut_path),
            "scna_path": relpath(scna_path) if scna_path else None,
            "clinical_path": relpath(clinical_path),
            "expression_gene_count": len(expr_means),
            "mutation_gene_count": len(mut_freqs),
            "scna_gene_count": len(scna_profiles),
            "expression_sample_count": expr_n,
            "mutation_sample_count": mut_n,
            "scna_sample_count": scna_n,
            "clinical_summary": clinical_summary,
        }
    return cohort_data, source_manifest


def background_mean(feature_maps: dict[str, dict[str, float]], target_cohort: str, gene: str) -> float | None:
    values = [mapping[gene] for cohort, mapping in feature_maps.items() if cohort != target_cohort and gene in mapping]
    return mean(values) if values else None


def grade_from_score(value: float) -> str:
    if value >= 0.85:
        return "A"
    if value >= 0.7:
        return "B"
    if value >= 0.55:
        return "C"
    if value >= 0.4:
        return "D"
    return "E"


def build_score_bundle(effect_strength: float, sample_count: int, source_reliability: float = 0.88) -> tuple[dict[str, Any], dict[str, Any]]:
    study_type_score = 0.76
    sample_size_score = min(sample_count / 500.0, 1.0)
    stat_support_score = max(0.2, min(effect_strength, 1.0))
    composite = round(mean([study_type_score, sample_size_score, stat_support_score, source_reliability]), 4)
    grade = grade_from_score(composite)
    ds_belief = round(min(0.95, composite * 0.82), 4)
    ds_plausibility = round(min(1.0, ds_belief + 0.15), 4)
    score = {
        "study_type_score": study_type_score,
        "impact_score": source_reliability,
        "stat_significance_score": stat_support_score,
        "sample_size_score": round(sample_size_score, 4),
        "llm_confidence": None,
        "composite_score": composite,
        "grade_level": grade,
        "ds_belief": ds_belief,
        "ds_plausibility": ds_plausibility,
        "ds_ignorance": round(max(0.0, ds_plausibility - ds_belief), 4),
    }
    uncertainty = {
        "evidence_grade": grade,
        "grade_framework": "custom_omics",
        "confidence_score": composite,
        "study_quality_score": study_type_score,
        "sample_size_score": round(sample_size_score, 4),
        "statistical_support_score": stat_support_score,
        "source_reliability_score": source_reliability,
        "llm_confidence": None,
        "ds_mass_support": ds_belief,
        "ds_mass_against": None,
        "ds_mass_uncertainty": round(max(0.0, ds_plausibility - ds_belief), 4),
        "ds_belief": ds_belief,
        "ds_plausibility": ds_plausibility,
        "contradiction_risk": None,
        "missingness_notes": None,
        "review_status": "pending",
    }
    return score, uncertainty


def disease_identifier(family: str) -> list[dict[str, Any]]:
    return [{"db_source": "custom_disease_family", "dbid": family, "label": family, "score": 1.0}]


def build_entity_mention(name: str, entity_type: str, role: str) -> dict[str, Any]:
    return {
        "name": name,
        "entity_type": entity_type,
        "normalized_id": name,
        "normalized_name": name,
        "db_source": "HGNC" if entity_type in {"Gene", "Protein"} else "custom",
        "dbid": name,
        "linking_score": 1.0,
        "role": role,
    }


def make_evidence_node(
    *,
    cohort: str,
    family: str,
    label: str,
    feature_type: str,
    gene: str,
    value: float,
    delta: float | None,
    sample_count: int,
    source_file: Path,
    clinical_summary: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    created_at = now_iso()
    predicate: str
    direction: str
    evidence_category: str
    metric: str
    phenotype: str
    bio_mechanism: str
    feature_layer: str
    effect_strength: float
    value_text: float | str

    if feature_type == "expression_up":
        predicate = "upregulated_in"
        direction = "increase"
        evidence_category = "expression_regulation"
        metric = "mean_log2_expression_delta_vs_priority_background"
        phenotype = f"{gene} shows higher expression in {family} ({cohort}) than the selected TCGA priority-cohort background"
        bio_mechanism = f"{gene} exhibits cohort-enriched expression in {family}"
        feature_layer = "expression"
        effect_strength = min(abs(delta or 0.0) / 3.0, 1.0)
        value_text = round(delta or 0.0, 4)
    elif feature_type == "expression_down":
        predicate = "downregulated_in"
        direction = "decrease"
        evidence_category = "expression_regulation"
        metric = "mean_log2_expression_delta_vs_priority_background"
        phenotype = f"{gene} shows lower expression in {family} ({cohort}) than the selected TCGA priority-cohort background"
        bio_mechanism = f"{gene} exhibits cohort-depleted expression in {family}"
        feature_layer = "expression"
        effect_strength = min(abs(delta or 0.0) / 3.0, 1.0)
        value_text = round(delta or 0.0, 4)
    elif feature_type == "mutation":
        predicate = "mutated_in"
        direction = "positive"
        evidence_category = "gene_variation"
        metric = "mutation_frequency"
        phenotype = f"{gene} is recurrently mutated in {family} ({cohort})"
        bio_mechanism = f"{gene} carries elevated mutation burden in {family}"
        feature_layer = "mutation"
        effect_strength = min((value + max(delta or 0.0, 0.0)) / 0.5, 1.0)
        value_text = round(value, 4)
    elif feature_type == "amplification":
        predicate = "amplified_in"
        direction = "increase"
        evidence_category = "gene_variation"
        metric = "copy_number_amplification_frequency"
        phenotype = f"{gene} shows recurrent copy-number amplification in {family} ({cohort})"
        bio_mechanism = f"{gene} has elevated amplification frequency in {family}"
        feature_layer = "copy_number"
        effect_strength = min((value + max(delta or 0.0, 0.0)) / 0.6, 1.0)
        value_text = round(value, 4)
    elif feature_type == "deletion":
        predicate = "deleted_in"
        direction = "decrease"
        evidence_category = "gene_variation"
        metric = "copy_number_deletion_frequency"
        phenotype = f"{gene} shows recurrent copy-number deletion in {family} ({cohort})"
        bio_mechanism = f"{gene} has elevated deletion frequency in {family}"
        feature_layer = "copy_number"
        effect_strength = min((value + max(delta or 0.0, 0.0)) / 0.6, 1.0)
        value_text = round(value, 4)
    else:  # pragma: no cover
        raise ValueError(feature_type)

    score, uncertainty = build_score_bundle(effect_strength, sample_count)
    evidence_id = f"TCGAEVID-{stable_hash(cohort, feature_type, gene)}"
    claim_id = f"CLAIM-{evidence_id}"
    cohort_id = f"COHORT-{stable_hash(cohort)}"
    gene_entity_id = f"HGNC:{gene}"
    disease_entity_id = f"DISEASE:{family}"

    cohort_context = {
        "disease_name": family,
        "disease_family": family,
        "disease_identifiers": disease_identifier(family),
        "cohort_id": cohort_id,
        "cohort_label": f"{cohort} ({label})",
        "cohort_type": "tumor_cohort",
        "lineage": None,
        "tissue": label,
        "sample_origin": f"{cohort} tumor cohort",
        "sample_grouping": "cohort summary",
        "comparison_type": None,
        "comparison_label": "vs selected TCGA priority-cohort background",
        "case_definition": f"{cohort} tumor samples",
        "control_definition": "selected TCGA priority-cohort background",
        "n_total": sample_count,
        "n_effective": sample_count,
        "n_cases": sample_count,
        "n_controls": None,
        "n_samples": sample_count,
        "n_patients": sample_count,
        "n_cell_lines": None,
        "effective_unit": "sample",
        "platform_summary": source_file.name,
        "batch_summary": None,
        "context_summary": f"pathologic_stage={clinical_summary.get('pathologic_stage_distribution')}, mean_tumor_purity={clinical_summary.get('mean_tumor_purity')}",
    }

    proposition = {
        "claim_id": claim_id,
        "subject_entities": [build_entity_mention(gene, "Gene", "subject")],
        "predicate": predicate,
        "object_entities": [build_entity_mention(family, "Disease", "object")],
        "outcome_entities": [],
        "direction": direction,
        "polarity": "oncogenic" if feature_type in {"expression_up", "mutation", "amplification"} else "tumor_suppressive",
        "claim_scope": "gene_disease",
        "summary": f"{gene} {predicate} {family} in {cohort}",
        "normalized_claim_text": f"{gene}|{predicate}|{family}|{cohort}",
        "hypothesis_level": "molecular",
        "supports_target_discovery": True,
        "supports_mechanism_inference": True,
        "supports_drug_repositioning": False,
    }

    statistics = {
        "p_value": None,
        "q_value": None,
        "fold_change": round(delta, 4) if delta is not None else None,
        "confidence_interval": None,
        "sample_size": sample_count,
        "effect_size": round(abs(delta), 4) if delta is not None else round(value, 4),
        "hazard_ratio": None,
        "mutation_frequency": round(value, 4) if feature_type == "mutation" else None,
        "ceres_score": None,
        "codependency_coefficient": None,
        "synergy_score": None,
        "statistical_method": "cohort mean comparison against selected-priority background",
    }

    evidence_payload = {
        "evidence_category": evidence_category,
        "aggregation_level": "gene_by_cohort",
        "atomic_observations": [
            {
                "observation_id": f"OBS-{evidence_id}",
                "feature_layer": feature_layer,
                "subject_label": gene,
                "subject_id": gene,
                "object_label": family,
                "object_id": family,
                "metric": metric,
                "value": value_text,
                "unit": None,
                "direction": direction if direction in {"increase", "decrease", "positive", "negative", "mixed", "present", "absent", "unknown"} else "unknown",
                "comparator": "selected TCGA priority-cohort background",
                "group_a": cohort,
                "group_b": "priority_background",
                "supporting_n": sample_count,
                "p_value": None,
                "q_value": None,
                "effect_size": statistics["effect_size"],
                "hazard_ratio": None,
                "ci_lower": None,
                "ci_upper": None,
                "mutation_frequency": round(value, 4) if feature_type == "mutation" else None,
                "dependency_score": None,
                "viability_effect": None,
                "codependency_score": None,
                "synergy_score": None,
                "text_evidence": phenotype,
                "is_primary": True,
            }
        ],
        "summary_statistics": statistics,
        "platform_specific_fields": {
            "tcga_cohort": cohort,
            "priority_family": family,
            "source_metric_value": round(value, 4),
            "background_delta": round(delta, 4) if delta is not None else None,
            "source_file_name": source_file.name,
        },
        "data_completeness": 0.92,
    }

    evidence = {
        "node_type": "evidence",
        "id": evidence_id,
        "evidence_id": evidence_id,
        "source_family": "tcga_linkedomics",
        "source": {
            "doi": None,
            "pmid": None,
            "title": f"LinkedOmics {cohort} {feature_type} atomic evidence",
            "authors": [],
            "journal": "LinkedOmics TCGA",
            "year": 2016,
            "impact_factor": None,
            "jcr_quartile": None,
            "citation_count": None,
            "pdf_path": None,
            "dataset_name": "LinkedOmics-TCGA",
            "file_path": relpath(source_file),
            "release": "01_28_2016",
        },
        "pico": {
            "study_object": f"{cohort} tumor cohort",
            "intervention": f"{feature_type} cohort summary",
            "comparison": "selected TCGA priority-cohort background",
            "outcome_metrics": [metric],
        },
        "core_entities": [
            build_entity_mention(gene, "Gene", "subject"),
            build_entity_mention(family, "Disease", "object"),
        ],
        "target_disease": family,
        "bio_mechanism": bio_mechanism,
        "phenotype": phenotype,
        "experimental_context": cohort_context["context_summary"],
        "study_design": "observational_omics",
        "clinical_stage": "omics_cohort",
        "statistics": statistics,
        "score": score,
        "source_text": phenotype,
        "extraction_confidence": round(score["composite_score"], 4),
        "linked_tarkg_entities": [gene, family],
        "evidence_relations": [],
        "merged_from": [],
        "review_status": "pending",
        "reviewer_comments": None,
        "created_at": created_at,
        "updated_at": None,
        "version": "1.0.0",
        "provenance": {
            "source_family": "tcga_linkedomics",
            "source_dataset": "LinkedOmics-TCGA",
            "source_release": "01_28_2016",
            "source_record_id": evidence_id,
            "source_file": relpath(source_file),
            "source_url": None,
            "citation_doi": None,
            "citation_pmid": None,
            "platform": "RNAseq" if "expression" in feature_type else ("SNP_6.0" if feature_type in {"amplification", "deletion"} else "mutation_call"),
            "assay": metric,
            "sequencing_or_screen_id": cohort,
            "batch_id": "01_28_2016",
            "analysis_pipeline": "tcga_atomic_extraction_v1",
            "extractor_name": "scripts/extract_tcga_atomic_evidence.py",
            "extractor_version": "1.0.0",
            "extracted_at": created_at,
            "study_design": "observational_omics",
            "clinical_stage": "omics_cohort",
            "licence": None,
            "is_primary_observation": True,
            "source_priority": 2 if family in {"HCC", "CRC"} else 3,
        },
        "cohort_context": cohort_context,
        "proposition": proposition,
        "evidence_payload": evidence_payload,
        "uncertainty": uncertainty,
        "graph_relation_slots": {
            "supports": [],
            "contradicts": [],
            "complements": [],
            "redundant_with": [],
            "same_entity_context": [],
            "same_cohort_context": [],
            "refines": [],
            "extends": [],
            "replicates": [],
            "linked_entities": [gene_entity_id, disease_entity_id],
        },
        "aggregation_parent_ids": [],
    }

    proposition_node = {
        "node_type": "proposition",
        "id": claim_id,
        "claim_id": claim_id,
        "summary": proposition["summary"],
        "predicate": proposition["predicate"],
        "direction": proposition["direction"],
        "supports_target_discovery": proposition["supports_target_discovery"],
        "supports_mechanism_inference": proposition["supports_mechanism_inference"],
        "supports_drug_repositioning": proposition["supports_drug_repositioning"],
    }

    cohort_node = {
        "node_type": "cohort",
        "id": cohort_id,
        "cohort_label": cohort_context["cohort_label"],
        "disease_name": family,
        "lineage": None,
        "cohort_type": "tumor_cohort",
        "n_effective": sample_count,
    }

    entity_nodes = [
        {
            "node_type": "entity",
            "id": gene_entity_id,
            "name": gene,
            "entity_type": "Gene",
            "db_source": "HGNC",
            "aliases": [gene],
            "normalized_from": [gene],
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

    return evidence, proposition_node, cohort_node, {"gene_entity_id": gene_entity_id, "disease_entity_id": disease_entity_id}, entity_nodes


def extract_evidence(cohort_data: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    expression_maps = {cohort: payload["expression_means"] for cohort, payload in cohort_data.items()}
    mutation_maps = {cohort: payload["mutation_freqs"] for cohort, payload in cohort_data.items()}
    amp_maps = {cohort: {gene: values["amp_freq"] for gene, values in payload["scna_profiles"].items()} for cohort, payload in cohort_data.items()}
    del_maps = {cohort: {gene: values["del_freq"] for gene, values in payload["scna_profiles"].items()} for cohort, payload in cohort_data.items()}

    evidence_nodes: list[dict[str, Any]] = []
    proposition_nodes: list[dict[str, Any]] = []
    cohort_nodes: list[dict[str, Any]] = []
    entity_nodes: list[dict[str, Any]] = []
    cohort_stats: dict[str, Any] = {}

    for cohort, payload in cohort_data.items():
        config = payload["config"]
        selected_records: list[tuple[str, str, float, float | None, int, Path]] = []

        expression_candidates_up: list[tuple[float, str, float, float]] = []
        expression_candidates_down: list[tuple[float, str, float, float]] = []
        for gene, cohort_mean in payload["expression_means"].items():
            bg = background_mean(expression_maps, cohort, gene)
            if bg is None:
                continue
            delta = cohort_mean - bg
            if delta >= EXPRESSION_MIN_DELTA:
                expression_candidates_up.append((delta, gene, cohort_mean, delta))
            elif delta <= -EXPRESSION_MIN_DELTA:
                expression_candidates_down.append((abs(delta), gene, cohort_mean, delta))

        mutation_candidates: list[tuple[float, str, float, float]] = []
        for gene, freq in payload["mutation_freqs"].items():
            bg = background_mean(mutation_maps, cohort, gene)
            if bg is None:
                continue
            delta = freq - bg
            if freq >= MUTATION_MIN_FREQ and delta >= MUTATION_MIN_DELTA:
                mutation_candidates.append((freq + delta, gene, freq, delta))

        amp_candidates: list[tuple[float, str, float, float]] = []
        del_candidates: list[tuple[float, str, float, float]] = []
        for gene, profile in payload["scna_profiles"].items():
            amp_bg = background_mean(amp_maps, cohort, gene)
            del_bg = background_mean(del_maps, cohort, gene)
            amp_freq = profile["amp_freq"]
            del_freq = profile["del_freq"]
            amp_delta = amp_freq - amp_bg if amp_bg is not None else None
            del_delta = del_freq - del_bg if del_bg is not None else None
            if amp_delta is not None and amp_freq >= CNV_MIN_FREQ and amp_delta >= CNV_MIN_DELTA:
                amp_candidates.append((amp_freq + amp_delta, gene, amp_freq, amp_delta))
            if del_delta is not None and del_freq >= CNV_MIN_FREQ and del_delta >= CNV_MIN_DELTA:
                del_candidates.append((del_freq + del_delta, gene, del_freq, del_delta))

        for _, gene, value, delta in sorted(expression_candidates_up, reverse=True)[: TOP_N["expression_up"]]:
            selected_records.append(("expression_up", gene, value, delta, payload["expression_n"], payload["paths"]["expression"]))
        for _, gene, value, delta in sorted(expression_candidates_down, reverse=True)[: TOP_N["expression_down"]]:
            selected_records.append(("expression_down", gene, value, delta, payload["expression_n"], payload["paths"]["expression"]))
        for _, gene, value, delta in sorted(mutation_candidates, reverse=True)[: TOP_N["mutation"]]:
            selected_records.append(("mutation", gene, value, delta, payload["mutation_n"], payload["paths"]["mutation"]))
        for _, gene, value, delta in sorted(amp_candidates, reverse=True)[: TOP_N["amplification"]]:
            selected_records.append(("amplification", gene, value, delta, payload["scna_n"], payload["paths"]["scna"]))
        for _, gene, value, delta in sorted(del_candidates, reverse=True)[: TOP_N["deletion"]]:
            selected_records.append(("deletion", gene, value, delta, payload["scna_n"], payload["paths"]["scna"]))

        per_type_counter = Counter()
        for feature_type, gene, value, delta, sample_count, source_file in selected_records:
            evidence, proposition_node, cohort_node, _, entity_node_list = make_evidence_node(
                cohort=cohort,
                family=config["family"],
                label=config["label"],
                feature_type=feature_type,
                gene=gene,
                value=value,
                delta=delta,
                sample_count=sample_count,
                source_file=source_file,
                clinical_summary=payload["clinical_summary"],
            )
            evidence_nodes.append(evidence)
            proposition_nodes.append(proposition_node)
            cohort_nodes.append(cohort_node)
            entity_nodes.extend(entity_node_list)
            per_type_counter[feature_type] += 1

        cohort_stats[cohort] = {
            "family": config["family"],
            "clinical_summary": payload["clinical_summary"],
            "selected_atomic_evidence_counts": dict(sorted(per_type_counter.items())),
            "candidate_counts_before_topn": {
                "expression_up": len(expression_candidates_up),
                "expression_down": len(expression_candidates_down),
                "mutation": len(mutation_candidates),
                "amplification": len(amp_candidates),
                "deletion": len(del_candidates),
            },
        }

    unique_props = sorted({node["id"]: node for node in proposition_nodes}.values(), key=lambda item: item["id"])
    unique_cohorts = sorted({node["id"]: node for node in cohort_nodes}.values(), key=lambda item: item["id"])
    unique_entities = sorted({node["id"]: node for node in entity_nodes}.values(), key=lambda item: item["id"])
    return evidence_nodes, unique_props, unique_cohorts, unique_entities, cohort_stats


def build_edges(evidence_nodes: list[dict[str, Any]], proposition_nodes: list[dict[str, Any]], cohort_nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    proposition_ids = {node["id"] for node in proposition_nodes}
    cohort_ids = {node["id"] for node in cohort_nodes}
    edges: list[dict[str, Any]] = []
    for evidence in evidence_nodes:
        proposition_id = evidence["proposition"]["claim_id"]
        cohort_id = evidence["cohort_context"]["cohort_id"]
        gene_id = f"HGNC:{evidence['proposition']['subject_entities'][0]['name']}"
        disease_id = f"DISEASE:{evidence['target_disease']}"
        if proposition_id in proposition_ids:
            edges.append(
                {
                    "source": evidence["id"],
                    "target": proposition_id,
                    "relation": "ASSERTS",
                    "relation_family": "aggregation_trace",
                    "entity_type": None,
                    "linking_score": None,
                    "similarity_score": None,
                    "rationale": "TCGA atomic evidence asserts its proposition node.",
                    "support_strength": evidence["uncertainty"]["confidence_score"],
                    "evidence_count": 1,
                    "created_by_rule": "tcga_atomic:asserts",
                }
            )
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
            edges.append(
                {
                    "source": proposition_id,
                    "target": cohort_id,
                    "relation": "OBSERVED_IN",
                    "relation_family": "semantic_anchor",
                    "entity_type": "Cohort",
                    "linking_score": None,
                    "similarity_score": None,
                    "rationale": "Aligned proposition grounded in the same TCGA cohort.",
                    "support_strength": None,
                    "evidence_count": 1,
                    "created_by_rule": "tcga_atomic:proposition_observed_in",
                }
            )
        edges.extend(
            [
                {
                    "source": evidence["id"],
                    "target": gene_id,
                    "relation": "LINKED_TO",
                    "relation_family": "semantic_anchor",
                    "entity_type": "Gene",
                    "linking_score": 1.0,
                    "similarity_score": None,
                    "rationale": "Evidence subject gene link.",
                    "support_strength": None,
                    "evidence_count": 1,
                    "created_by_rule": "tcga_atomic:link_subject_gene",
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


def load_pubmed_gene_index() -> dict[str, dict[str, list[dict[str, Any]]]]:
    data = json.loads(PUBMED_ALIGNED_PATH.read_text(encoding="utf-8"))
    entity_by_id = {
        node["id"]: node
        for node in data["nodes"]
        if node.get("node_type") == "entity"
    }
    gene_links: dict[str, set[str]] = defaultdict(set)
    for edge in data["edges"]:
        if edge.get("relation") != "LINKED_TO":
            continue
        evidence_id = edge.get("source")
        entity = entity_by_id.get(edge.get("target"))
        if not entity or entity.get("entity_type") not in {"Gene", "Protein"}:
            continue
        gene_links[evidence_id].add((entity.get("name") or "").upper())
        for alias in entity.get("aliases") or []:
            alias_value = (alias or "").upper()
            if alias_value:
                gene_links[evidence_id].add(alias_value)

    index: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for node in data["nodes"]:
        if node.get("node_type") != "evidence":
            continue
        disease = node.get("disease")
        gene_names = set(gene_links.get(node["id"], set()))
        subject = node.get("subject") or {}
        if subject.get("entity_type") in {"Gene", "Protein"} and subject.get("label"):
            gene_names.add(str(subject["label"]).upper())
        for gene_name in gene_names:
            if gene_name:
                index[disease][gene_name].append(
                    {
                        "pubmed_evidence_id": node["id"],
                        "predicate": node["predicate"],
                        "polarity": node["polarity"],
                        "confidence_score": node["confidence"]["score"],
                        "summary": node["summary"],
                    }
                )
    return index


def classify_alignment_hint(tcga_predicate: str, pubmed_predicate: str, pubmed_polarity: str | None) -> str:
    promoting_tcga = tcga_predicate in {"upregulated_in", "mutated_in", "amplified_in"}
    suppressing_tcga = tcga_predicate in {"downregulated_in", "deleted_in"}
    if promoting_tcga and pubmed_predicate == "promotes_mechanism":
        return "support_like"
    if suppressing_tcga and pubmed_predicate == "suppresses_mechanism":
        return "support_like"
    if promoting_tcga and pubmed_predicate == "suppresses_mechanism":
        return "contradiction_like"
    if suppressing_tcga and pubmed_predicate == "promotes_mechanism":
        return "contradiction_like"
    if tcga_predicate == "mutated_in":
        return "complement_like"
    if pubmed_polarity in {"oncogenic", "pro_tumor", "tumor_suppressive", "anti_tumor"}:
        return "entity_context_only"
    return "undetermined"


def build_pubmed_alignment_candidates(evidence_nodes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pubmed_index = load_pubmed_gene_index()
    candidates: list[dict[str, Any]] = []
    counts = Counter()
    for evidence in evidence_nodes:
        disease = evidence["disease"]
        if disease not in {"HCC", "CRC"}:
            continue
        gene = evidence["subject"]["label"].upper()
        matches = sorted(pubmed_index.get(disease, {}).get(gene, []), key=lambda item: item["confidence_score"], reverse=True)[:20]
        for match in matches:
            hint = classify_alignment_hint(evidence["predicate"], match["predicate"], match["polarity"])
            candidate = {
                "tcga_evidence_id": evidence["id"],
                "pubmed_evidence_id": match["pubmed_evidence_id"],
                "shared_disease_family": disease,
                "shared_gene": gene,
                "candidate_relation": "shared_normalized_entity_context",
                "semantic_hint": hint,
                "tcga_predicate": evidence["predicate"],
                "pubmed_predicate": match["predicate"],
                "pubmed_polarity": match["polarity"],
                "tcga_confidence": evidence["confidence"]["score"],
                "pubmed_confidence": match["confidence_score"],
                "rationale": f"Shared canonical disease family {disease} and shared gene/protein anchor {gene}; candidate kept for later cross-source relation construction.",
                "pubmed_summary": match["summary"],
            }
            candidates.append(candidate)
            counts[hint] += 1
    return candidates, {"total_candidates": len(candidates), "by_hint": dict(sorted(counts.items()))}


def build_graph(
    evidence_nodes: list[dict[str, Any]],
    proposition_nodes: list[dict[str, Any]],
    cohort_nodes: list[dict[str, Any]],
    entity_nodes: list[dict[str, Any]],
    source_manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "graph_metadata": {
            "graph_id": "tcga_atomic_aligned_v1",
            "graph_name": "TCGA atomic evidence aligned to unified multisource KG schema",
            "created_by": "scripts/extract_tcga_atomic_evidence.py",
            "created_at": now_iso(),
            "source_families": ["tcga_linkedomics"],
            "disease_scope": sorted({node["target_disease"] for node in evidence_nodes}),
            "build_parameters": {
                "priority_cohorts": PRIORITY_COHORTS,
                "top_n": TOP_N,
                "thresholds": {
                    "expression_min_delta": EXPRESSION_MIN_DELTA,
                    "mutation_min_freq": MUTATION_MIN_FREQ,
                    "mutation_min_delta": MUTATION_MIN_DELTA,
                    "cnv_min_freq": CNV_MIN_FREQ,
                    "cnv_min_delta": CNV_MIN_DELTA,
                    "cnv_amp_threshold": CNV_AMP_THRESHOLD,
                    "cnv_del_threshold": CNV_DEL_THRESHOLD,
                },
                "source_manifest": source_manifest,
            },
            "notes": "High-priority TCGA cohort atomic evidence standardized into the unified evidence schema and harmonized to PubMed disease families.",
        },
        "nodes": sorted(evidence_nodes + proposition_nodes + cohort_nodes + entity_nodes, key=lambda item: (item["node_type"], item["id"])),
        "edges": build_edges(evidence_nodes, proposition_nodes, cohort_nodes),
    }


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


def build_stats(graph: dict[str, Any], cohort_stats: dict[str, Any], alignment_summary: dict[str, Any]) -> dict[str, Any]:
    evidence_nodes = [node for node in graph["nodes"] if node["node_type"] == "evidence"]
    return {
        "graph_id": graph["graph_metadata"]["graph_id"],
        "disease_scope": graph["graph_metadata"]["disease_scope"],
        "node_counts": {
            "total": len(graph["nodes"]),
            "evidence": len(evidence_nodes),
            "entity": sum(1 for node in graph["nodes"] if node["node_type"] == "entity"),
            "proposition": sum(1 for node in graph["nodes"] if node["node_type"] == "proposition"),
            "cohort": sum(1 for node in graph["nodes"] if node["node_type"] == "cohort"),
        },
        "edge_counts": {
            "total": len(graph["edges"]),
            "by_relation": dict(sorted(Counter(edge["relation"] for edge in graph["edges"]).items())),
        },
        "evidence_distributions": {
            "by_disease": dict(sorted(Counter(node["target_disease"] for node in evidence_nodes).items())),
            "by_cohort": dict(sorted(Counter(node["cohort_context"]["cohort_label"] for node in evidence_nodes).items())),
            "by_evidence_category": dict(sorted(Counter(node["evidence_payload"]["evidence_category"] for node in evidence_nodes).items())),
            "by_predicate": dict(sorted(Counter(node["proposition"]["predicate"] for node in evidence_nodes).items())),
            "by_grade": dict(sorted(Counter(node["uncertainty"]["evidence_grade"] for node in evidence_nodes).items())),
        },
        "cohort_stats": cohort_stats,
        "quality": {
            "confidence_score": summarize_numeric([node["uncertainty"]["confidence_score"] for node in evidence_nodes]),
            "composite_score": summarize_numeric([node["score"]["composite_score"] for node in evidence_nodes]),
            "effect_size": summarize_numeric([node["statistics"]["effect_size"] for node in evidence_nodes]),
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
                "validator": "jsonschema" if jsonschema is not None else "basic_validation",
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
                "gene_subject": round(sum(1 for node in evidence_nodes if node["proposition"]["subject_entities"]) / len(evidence_nodes), 4),
            },
            "pubmed_alignment_candidate_count": len(alignment_candidates),
        },
    }


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    cohort_data, source_manifest = load_tcga_sources()
    evidence_nodes, proposition_nodes, cohort_nodes, entity_nodes, cohort_stats = extract_evidence(cohort_data)
    graph = build_graph(evidence_nodes, proposition_nodes, cohort_nodes, entity_nodes, source_manifest)
    validation_errors = schema_validate(graph)
    alignment_candidates, alignment_summary = build_pubmed_alignment_candidates(evidence_nodes)
    stats = build_stats(graph, cohort_stats, alignment_summary)
    audit = build_audit(graph, validation_errors, source_manifest, alignment_candidates)

    (OUTPUT_DIR / "tcga_atomic_aligned_v1.json").write_text(json.dumps(graph, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "tcga_atomic_aligned_v1_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "tcga_atomic_aligned_v1_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "tcga_pubmed_alignment_candidates_v1.json").write_text(json.dumps(alignment_candidates, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

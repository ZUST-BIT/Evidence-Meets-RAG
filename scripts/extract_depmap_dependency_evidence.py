from __future__ import annotations

import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import pandas as pd

from build_source_overlap_selection import DEPMAP_ONCOTREE_TO_FAMILY, FAMILY_DESCRIPTIONS
from extract_tcga_atomic_evidence import (
    PRIORITY_COHORTS,
    build_pubmed_alignment_candidates,
    ensure_dir,
    now_iso,
    relpath,
    stable_hash,
    summarize_numeric,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "kg" / "aligned"
from knowledge_build_paths import DEPMAP_ROOT

MODEL_PATH = DEPMAP_ROOT / "model-condition-mapping" / "Model.csv"
GENE_EFFECT_PATH = DEPMAP_ROOT / "crispr-screen" / "CRISPRGeneEffect.csv"
GENE_DEPENDENCY_PATH = DEPMAP_ROOT / "crispr-screen" / "CRISPRGeneDependency.csv"
EXPRESSION_PATH = DEPMAP_ROOT / "expression" / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
COPY_NUMBER_PATH = DEPMAP_ROOT / "copy-number" / "OmicsCNGeneWGS.csv"
HOTSPOT_MUTATION_PATH = DEPMAP_ROOT / "mutations" / "OmicsSomaticMutationsMatrixHotspot.csv"
DAMAGING_MUTATION_PATH = DEPMAP_ROOT / "mutations" / "OmicsSomaticMutationsMatrixDamaging.csv"
GLOBAL_SIGNATURE_PATH = DEPMAP_ROOT / "global-genomic-features" / "OmicsGlobalSignatures.csv"
MOLECULAR_SUBTYPE_PATH = DEPMAP_ROOT / "global-genomic-features" / "OmicsInferredMolecularSubtypes.csv"
PORTAL_COMPOUNDS_PATH = DEPMAP_ROOT / "model-condition-mapping" / "PortalCompounds.csv"
GDSC1_RESPONSE_PATH = DEPMAP_ROOT / "drug-response" / "GDSC1_fitted_dose_response_24Jul22.csv"
GDSC2_RESPONSE_PATH = DEPMAP_ROOT / "drug-response" / "GDSC2_fitted_dose_response_24Jul22.csv"
GDSC_COMPOUND_METADATA_PATH = DEPMAP_ROOT / "drug-response" / "screened_compounds_rel_8.4.csv"

TARGET_FAMILIES = sorted({config["family"] for config in PRIORITY_COHORTS})
DEPENDENCY_POSITIVE_PROBABILITY = 0.50
MIN_FAMILY_MODELS = 8

RECURRENT_EFFECT_MAX = -1.00
RECURRENT_MEAN_PROB_MIN = 0.65
RECURRENT_PREVALENCE_MIN = 0.65
RECURRENT_BACKGROUND_PREVALENCE_MAX = 0.80

SELECTIVE_EFFECT_MAX = -0.65
SELECTIVE_DELTA_MAX = -0.20
SELECTIVE_MEAN_PROB_MIN = 0.50
SELECTIVE_PREVALENCE_DELTA_MIN = 0.15
SELECTIVE_COHEN_D_MIN = 0.65
SELECTIVE_BACKGROUND_PREVALENCE_MAX = 0.75

CONTEXT_ANCHOR_EFFECT_MAX = -0.60
CONTEXT_ANCHOR_MEAN_PROB_MIN = 0.50
CONTEXT_ANCHOR_PREVALENCE_MIN = 0.50
CONTEXT_ANCHOR_BACKGROUND_PREVALENCE_MAX = 0.90

EXPR_CONTEXT_MIN_GROUP = 6
EXPR_CONTEXT_MIN_MEDIAN = -0.55
EXPR_CONTEXT_MIN_DELTA = -0.20
EXPR_CONTEXT_MIN_PREVALENCE_DELTA = 0.15
EXPR_CONTEXT_MIN_COHEN_D = 0.60
EXPR_CONTEXT_MIN_SPLIT = 1.00

CROSS_EXPR_CONTEXT_MIN_GROUP = 6
CROSS_EXPR_CONTEXT_MIN_MEDIAN = -0.65
CROSS_EXPR_CONTEXT_MIN_DELTA = -0.25
CROSS_EXPR_CONTEXT_MIN_PREVALENCE_DELTA = 0.18
CROSS_EXPR_CONTEXT_MIN_COHEN_D = 0.60
CROSS_EXPR_CONTEXT_MIN_SPLIT = 1.00

COPY_CONTEXT_MIN_GROUP = 5
COPY_CONTEXT_MIN_MEDIAN = -0.55
COPY_CONTEXT_MIN_DELTA = -0.20
COPY_CONTEXT_MIN_PREVALENCE_DELTA = 0.15
COPY_CONTEXT_MIN_COHEN_D = 0.60
COPY_GAIN_THRESHOLD = 1.20
COPY_BASELINE_MAX = 0.90

CROSS_COPY_CONTEXT_MIN_GROUP = 5
CROSS_COPY_CONTEXT_MIN_MEDIAN = -0.65
CROSS_COPY_CONTEXT_MIN_DELTA = -0.22
CROSS_COPY_CONTEXT_MIN_PREVALENCE_DELTA = 0.18
CROSS_COPY_CONTEXT_MIN_COHEN_D = 0.60

MUTATION_CONTEXT_MIN_POSITIVE = 4
MUTATION_CONTEXT_MIN_NEGATIVE = 8
MUTATION_CONTEXT_MIN_MEDIAN = -0.60
MUTATION_CONTEXT_MIN_DELTA = -0.25
MUTATION_CONTEXT_MIN_PREVALENCE_DELTA = 0.15
MUTATION_CONTEXT_MIN_COHEN_D = 0.55

CROSS_MUTATION_CONTEXT_MIN_POSITIVE = 4
CROSS_MUTATION_CONTEXT_MIN_NEGATIVE = 8
CROSS_MUTATION_CONTEXT_MIN_MEDIAN = -0.65
CROSS_MUTATION_CONTEXT_MIN_DELTA = -0.25
CROSS_MUTATION_CONTEXT_MIN_PREVALENCE_DELTA = 0.15
CROSS_MUTATION_CONTEXT_MIN_COHEN_D = 0.55

PHENOTYPE_CONTEXT_MIN_POSITIVE = 8
PHENOTYPE_CONTEXT_MIN_NEGATIVE = 8
PHENOTYPE_CONTEXT_MIN_MEDIAN = -0.60
PHENOTYPE_CONTEXT_MIN_DELTA = -0.25
PHENOTYPE_CONTEXT_MIN_PREVALENCE_DELTA = 0.20
PHENOTYPE_CONTEXT_MIN_COHEN_D = 0.65

SUBTYPE_CONTEXT_MIN_POSITIVE = 6
SUBTYPE_CONTEXT_MIN_NEGATIVE = 10
SUBTYPE_CONTEXT_MIN_MEDIAN = -0.60
SUBTYPE_CONTEXT_MIN_DELTA = -0.25
SUBTYPE_CONTEXT_MIN_PREVALENCE_DELTA = 0.20
SUBTYPE_CONTEXT_MIN_COHEN_D = 0.65

METASTATIC_CONTEXT_MIN_POSITIVE = 5
METASTATIC_CONTEXT_MIN_NEGATIVE = 5
METASTATIC_CONTEXT_MIN_MEDIAN = -0.60
METASTATIC_CONTEXT_MIN_DELTA = -0.20
METASTATIC_CONTEXT_MIN_PREVALENCE_DELTA = 0.15
METASTATIC_CONTEXT_MIN_COHEN_D = 0.55

CODEPENDENCY_MIN_SHARED_MODELS = 15
CODEPENDENCY_MIN_CORRELATION = 0.70
CODEPENDENCY_MIN_MEAN_PROBABILITY = 0.55

DRUG_RESPONSE_MIN_MODELS = 8
DRUG_RESPONSE_BACKGROUND_MIN_MODELS = 40
DRUG_RESPONSE_FAMILY_Z_MAX = -0.90
DRUG_RESPONSE_FAMILY_AUC_MAX = 0.82
DRUG_RESPONSE_Z_DELTA_MAX = -0.20
DRUG_RESPONSE_AUC_DELTA_MAX = -0.03

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

PRIMARY_METASTASIS_LABELS = {
    "Primary": "primary_model_state",
    "Metastatic": "metastatic_model_state",
}

SIGNATURE_CONTEXTS = {
    "MSIScore": ("high_msi", "low_msi"),
    "CIN": ("high_cin", "low_cin"),
    "Aneuploidy": ("high_aneuploidy", "low_aneuploidy"),
    "Ploidy": ("high_ploidy", "low_ploidy"),
    "LoHFraction": ("high_loh_fraction", "low_loh_fraction"),
}

MATRIX_META_COLUMNS = {
    "Unnamed: 0",
    "SequencingID",
    "ModelConditionID",
    "ModelID",
    "IsDefaultEntryForMC",
    "IsDefaultEntryForModel",
}


def score_bundle(effect_strength: float, n_effective: int, source_reliability: float = 0.95) -> tuple[dict[str, Any], dict[str, Any]]:
    study_type_score = 0.9
    sample_size_score = min(n_effective / 150.0, 1.0)
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
    belief = round(min(0.97, composite * 0.86), 4)
    plausibility = round(min(1.0, belief + 0.12), 4)
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
        "grade_framework": "custom_functional_dependency",
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


def compact_entity_ref(entity_id: str | None, label: str, entity_type: str) -> dict[str, Any]:
    return {
        "id": entity_id,
        "label": label,
        "entity_type": entity_type,
    }


def family_cohort_id(family: str) -> str:
    return f"COHORT-{stable_hash('DEPMAP', family)}"


def standardize_gene_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.copy()
    renamed.columns = [str(column).split(" (")[0].strip() for column in renamed.columns]
    if renamed.columns.has_duplicates:
        renamed = renamed.T.groupby(level=0).mean().T
    return renamed


def build_gene_entity(gene: str) -> dict[str, Any]:
    return {
        "id": f"HGNC:{gene}",
        "label": gene,
        "entity_type": "Gene",
        "db_source": "HGNC",
        "aliases": [gene],
        "normalized_from": [gene],
    }


def build_disease_entity(family: str, family_label: str) -> dict[str, Any]:
    return {
        "id": f"DISEASE:{family}",
        "label": family,
        "entity_type": "Disease",
        "db_source": "custom_disease_family",
        "aliases": [family, family_label],
        "normalized_from": [family],
    }


def build_context_entity(label: str, entity_type: str = "Phenotype", aliases: list[str] | None = None) -> dict[str, Any]:
    return {
        "id": f"{entity_type.upper()}:{stable_hash(label)}",
        "label": label,
        "entity_type": entity_type,
        "db_source": "custom_depmap_context",
        "aliases": aliases or [label],
        "normalized_from": [label],
    }


def build_drug_entity(
    compound_id: str,
    label: str,
    *,
    synonyms: list[str] | None = None,
    normalized_from: list[str] | None = None,
) -> dict[str, Any]:
    aliases = [label]
    for synonym in synonyms or []:
        synonym = str(synonym).strip()
        if synonym and synonym not in aliases:
            aliases.append(synonym)
    return {
        "id": f"DRUG:{compound_id}",
        "label": label,
        "entity_type": "Drug",
        "db_source": "depmap_portal_compounds",
        "aliases": aliases,
        "normalized_from": normalized_from or [label, compound_id],
    }


def completeness_ratio(observed: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(observed / total, 4)


def pooled_cohen_d(group_a: pd.Series | np.ndarray, group_b: pd.Series | np.ndarray) -> float:
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return 0.0
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled = (((len(a) - 1) * var_a) + ((len(b) - 1) * var_b)) / (len(a) + len(b) - 2)
    if pooled <= 1e-6:
        return 0.0 if np.mean(a) == np.mean(b) else min(abs(np.mean(a) - np.mean(b)) * 2.0, 3.0)
    return abs((np.mean(a) - np.mean(b)) / np.sqrt(pooled))


def align_binary_union(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    all_index = left.index.union(right.index)
    all_columns = left.columns.union(right.columns)
    merged = left.reindex(index=all_index, columns=all_columns, fill_value=0).astype(bool) | right.reindex(index=all_index, columns=all_columns, fill_value=0).astype(bool)
    return merged.astype("float32")


def load_model_metadata() -> pd.DataFrame:
    model = pd.read_csv(
        MODEL_PATH,
        usecols=[
            "ModelID",
            "CellLineName",
            "SangerModelID",
            "COSMICID",
            "OncotreeLineage",
            "OncotreePrimaryDisease",
            "OncotreeSubtype",
            "OncotreeCode",
            "PrimaryOrMetastasis",
        ],
    )
    model["family"] = model["OncotreeCode"].map(DEPMAP_ONCOTREE_TO_FAMILY)
    model = model[model["family"].isin(TARGET_FAMILIES)].copy()
    model = model[model["OncotreePrimaryDisease"] != "Non-Cancerous"].copy()
    model = model.drop_duplicates(subset=["ModelID"]).reset_index(drop=True)
    return model


def load_dependency_matrices(model_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    gene_effect = pd.read_csv(GENE_EFFECT_PATH, index_col=0)
    gene_dependency = pd.read_csv(GENE_DEPENDENCY_PATH, index_col=0)
    gene_effect = standardize_gene_columns(gene_effect)
    gene_dependency = standardize_gene_columns(gene_dependency)
    index = gene_effect.index.intersection(gene_dependency.index).intersection(pd.Index(model_ids))
    columns = gene_effect.columns.intersection(gene_dependency.columns)
    gene_effect = gene_effect.loc[index, columns].astype("float32")
    gene_dependency = gene_dependency.loc[index, columns].astype("float32")
    return gene_effect, gene_dependency


def load_dense_omics_matrix(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "IsDefaultEntryForModel" in frame.columns:
        frame = frame[frame["IsDefaultEntryForModel"].astype(str).str.lower().eq("yes")]
    value_columns = [column for column in frame.columns if column not in MATRIX_META_COLUMNS]
    matrix = frame.set_index("ModelID")[value_columns]
    matrix = standardize_gene_columns(matrix)
    matrix = matrix.groupby(level=0).first()
    return matrix.apply(pd.to_numeric, errors="coerce").astype("float32")


def load_binary_omics_matrix(path: Path) -> pd.DataFrame:
    matrix = load_dense_omics_matrix(path)
    return (matrix.fillna(0) > 0).astype("float32")


def load_global_signatures() -> pd.DataFrame:
    frame = pd.read_csv(GLOBAL_SIGNATURE_PATH)
    frame = frame[frame["IsDefaultEntryForModel"].astype(str).str.lower().eq("yes")]
    matrix = frame.set_index("ModelID")[list(SIGNATURE_CONTEXTS.keys()) + ["WGD"]]
    return matrix.apply(pd.to_numeric, errors="coerce").astype("float32")


def load_molecular_subtypes() -> pd.DataFrame:
    frame = pd.read_csv(MOLECULAR_SUBTYPE_PATH).set_index("ModelID")
    frame = frame.replace({"True": 1, "False": 0, True: 1, False: 0})
    return frame.apply(pd.to_numeric, errors="coerce").fillna(0).astype("float32")


def split_aliases(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [part.strip() for part in str(value).split(";") if part and part.strip()]


def load_gdsc_drug_response(model: pd.DataFrame) -> pd.DataFrame:
    if "ModelID" not in model.columns:
        model = model.reset_index()
        if "ModelID" not in model.columns and "index" in model.columns:
            model = model.rename(columns={"index": "ModelID"})

    portal_compounds = pd.read_csv(
        PORTAL_COMPOUNDS_PATH,
        usecols=[
            "CompoundID",
            "CompoundName",
            "GeneSymbolOfTargets",
            "TargetOrMechanism",
            "ChEMBLID",
            "SampleIDs",
            "Synonyms",
        ],
    )
    compound_token_map: dict[str, dict[str, Any]] = {}
    for _, row in portal_compounds.iterrows():
        metadata = {
            "compound_id": row["CompoundID"],
            "compound_name": row["CompoundName"],
            "gene_targets": row["GeneSymbolOfTargets"],
            "target_or_mechanism": row["TargetOrMechanism"],
            "chembl_id": row["ChEMBLID"],
            "synonyms": split_aliases(row.get("Synonyms")),
        }
        for token in split_aliases(row.get("SampleIDs")):
            if token.startswith("GDSC1:") or token.startswith("GDSC2:"):
                compound_token_map[token] = metadata

    gdsc_compounds = pd.read_csv(GDSC_COMPOUND_METADATA_PATH)
    gdsc_compounds["compound_key"] = gdsc_compounds["DRUG_ID"].astype(str)
    gdsc_compound_map = gdsc_compounds.set_index("compound_key").to_dict("index")

    model_meta = model.drop_duplicates(subset=["ModelID"]).copy()
    model_meta["COSMICID"] = pd.to_numeric(model_meta["COSMICID"], errors="coerce")
    sanger_map = model_meta.dropna(subset=["SangerModelID"]).drop_duplicates(subset=["SangerModelID"]).set_index("SangerModelID")["ModelID"]
    cosmic_map = model_meta.dropna(subset=["COSMICID"]).drop_duplicates(subset=["COSMICID"]).set_index("COSMICID")["ModelID"]

    frames: list[pd.DataFrame] = []
    for dataset, path in [("GDSC1", GDSC1_RESPONSE_PATH), ("GDSC2", GDSC2_RESPONSE_PATH)]:
        frame = pd.read_csv(
            path,
            usecols=[
                "SANGER_MODEL_ID",
                "COSMIC_ID",
                "DRUG_ID",
                "DRUG_NAME",
                "PUTATIVE_TARGET",
                "PATHWAY_NAME",
                "LN_IC50",
                "AUC",
                "Z_SCORE",
            ],
        )
        frame["dataset"] = dataset
        frame["COSMIC_ID"] = pd.to_numeric(frame["COSMIC_ID"], errors="coerce")
        frame["ModelID"] = frame["SANGER_MODEL_ID"].map(sanger_map)
        unmatched = frame["ModelID"].isna()
        frame.loc[unmatched, "ModelID"] = frame.loc[unmatched, "COSMIC_ID"].map(cosmic_map)
        frame = frame[frame["ModelID"].notna()].copy()
        frame["token"] = frame.apply(lambda row: f"{dataset}:{int(row['DRUG_ID'])}", axis=1)
        frame["compound_meta"] = frame["token"].map(compound_token_map)
        frame["compound_key"] = frame["DRUG_ID"].astype(int).astype(str)
        frame["gdsc_meta"] = frame["compound_key"].map(gdsc_compound_map)
        frame["CompoundID"] = frame["compound_meta"].apply(lambda value: value.get("compound_id") if isinstance(value, dict) else None)
        frame["PortalCompoundName"] = frame["compound_meta"].apply(lambda value: value.get("compound_name") if isinstance(value, dict) else None)
        frame["GeneTargets"] = frame["compound_meta"].apply(lambda value: value.get("gene_targets") if isinstance(value, dict) else None)
        frame["TargetOrMechanism"] = frame["compound_meta"].apply(lambda value: value.get("target_or_mechanism") if isinstance(value, dict) else None)
        frame["ChEMBLID"] = frame["compound_meta"].apply(lambda value: value.get("chembl_id") if isinstance(value, dict) else None)
        frame["Synonyms"] = frame["compound_meta"].apply(lambda value: value.get("synonyms") if isinstance(value, dict) else [])
        frame["GDSCDrugName"] = frame["gdsc_meta"].apply(lambda value: value.get("DRUG_NAME") if isinstance(value, dict) else None)
        frame["GDSCTarget"] = frame["gdsc_meta"].apply(lambda value: value.get("TARGET") if isinstance(value, dict) else None)
        frame["GDSCPathway"] = frame["gdsc_meta"].apply(lambda value: value.get("TARGET_PATHWAY") if isinstance(value, dict) else None)
        frames.append(frame)

    response = pd.concat(frames, ignore_index=True)
    response = response.merge(
        model_meta[
            [
                "ModelID",
                "family",
                "OncotreeCode",
                "OncotreePrimaryDisease",
                "CellLineName",
            ]
        ],
        on="ModelID",
        how="inner",
    )
    response["CompoundID"] = response["CompoundID"].fillna(response.apply(lambda row: f"{row['dataset']}-{int(row['DRUG_ID'])}", axis=1))
    response["DrugLabel"] = response["PortalCompoundName"].fillna(response["DRUG_NAME"]).fillna(response["GDSCDrugName"])
    response["DrugTargets"] = response["GeneTargets"].fillna(response["PUTATIVE_TARGET"]).fillna(response["GDSCTarget"])
    response["DrugMechanism"] = response["TargetOrMechanism"].fillna(response["GDSCPathway"])
    return response


def init_family_stats(family: str, family_label: str, family_models: pd.DataFrame, pool_size: int) -> dict[str, Any]:
    return {
        "family": family,
        "description": family_label,
        "model_count": int(len(family_models.index)),
        "model_fraction_of_selected_pool": round(len(family_models.index) / pool_size, 4),
        "oncotree_codes": dict(sorted(Counter(family_models["OncotreeCode"]).items())),
        "primary_diseases": dict(sorted(Counter(family_models["OncotreePrimaryDisease"]).items())),
        "selected_evidence_counts": {},
        "context_anchor_gene_count": 0,
        "selected_dependency_gene_count": 0,
    }


def make_evidence(
    *,
    family: str,
    family_label: str,
    predicate: str,
    evidence_category: str,
    feature_layer: str,
    metric: str,
    subject_entity: dict[str, Any],
    object_entity: dict[str, Any] | None,
    outcome_entity: dict[str, Any] | None,
    comparison_id: str,
    comparison_label: str,
    comparison_type: str | None,
    group_a_label: str,
    group_b_label: str | None,
    sample_count: int,
    n_group_a: int,
    n_group_b: int | None,
    value: float,
    effect_size: float,
    summary_text: str,
    platform_specific_fields: dict[str, Any],
    source_file: str,
    source_record_key: str,
    data_completeness: float,
    statistical_method: str,
    claim_scope: str = "gene_disease",
    direction: str = "negative_selection",
    polarity: str = "pro_tumor",
    supports_target_discovery: bool = True,
    supports_mechanism_inference: bool = True,
    supports_drug_repositioning: bool = False,
    aggregation_level: str = "gene_by_lineage_panel",
    provenance_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    created_at = now_iso()
    cohort_id = family_cohort_id(family)
    evidence_id = f"DEEPMAP-{stable_hash(family, predicate, feature_layer, metric, source_record_key, comparison_id)}"
    disease_entity = build_disease_entity(family, family_label)
    score, uncertainty = score_bundle(min(abs(effect_size), 1.0), max(n_group_a, 1))

    evidence = {
        "node_type": "evidence",
        "id": evidence_id,
        "record_id": evidence_id,
        "source_family": "depmap",
        "disease": family,
        "cohort_id": cohort_id,
        "subject": compact_entity_ref(subject_entity["id"], subject_entity["label"], subject_entity["entity_type"]),
        "object": compact_entity_ref(object_entity["id"], object_entity["label"], object_entity["entity_type"]) if object_entity else compact_entity_ref(disease_entity["id"], disease_entity["label"], disease_entity["entity_type"]),
        "outcome": compact_entity_ref(outcome_entity["id"], outcome_entity["label"], outcome_entity["entity_type"]) if outcome_entity else None,
        "predicate": predicate,
        "direction": direction,
        "polarity": polarity,
        "claim_scope": claim_scope,
        "hypothesis_level": "functional",
        "support": {
            "supports_target_discovery": supports_target_discovery,
            "supports_mechanism_inference": supports_mechanism_inference,
            "supports_drug_repositioning": supports_drug_repositioning,
        },
        "evidence_category": evidence_category,
        "feature_layer": feature_layer,
        "aggregation_level": aggregation_level,
        "summary": summary_text,
        "metric": metric,
        "value": round(float(value), 4),
        "unit": None,
        "comparison": {
            "sample_grouping": comparison_id,
            "comparison_type": comparison_type,
            "comparison_label": comparison_label,
            "group_a": group_a_label,
            "group_b": group_b_label,
            "n_total": sample_count,
            "n_effective": sample_count,
            "n_group_a": n_group_a,
            "n_group_b": n_group_b,
        },
        "statistics": {
            "p_value": None,
            "q_value": None,
            "fold_change": None,
            "confidence_interval": None,
            "sample_size": sample_count,
            "effect_size": round(abs(effect_size), 4),
            "hazard_ratio": None,
            "mutation_frequency": None,
            "ceres_score": round(float(value), 4) if metric.startswith("chronos_gene_effect") else None,
            "codependency_coefficient": round(float(value), 4) if metric == "dependency_profile_correlation" else None,
            "synergy_score": None,
            "statistical_method": statistical_method,
        },
        "confidence": build_compact_confidence(
            score=score,
            uncertainty=uncertainty,
            extraction_confidence=score["composite_score"],
            data_completeness=data_completeness,
        ),
        "citation": None,
        "source_context": platform_specific_fields,
        "reviewer_comments": None,
        "created_at": created_at,
        "updated_at": None,
        "version": "2.0.0",
        "provenance": {
            "source_family": "depmap",
            "source_dataset": "DepMap-26Q1",
            "source_release": "26Q1",
            "source_record_id": source_record_key,
            "source_file": source_file,
            "source_url": None,
            "citation_doi": None,
            "citation_pmid": None,
            "platform": "DepMap integrated model matrices",
            "assay": metric,
            "sequencing_or_screen_id": family,
            "batch_id": "26Q1",
            "analysis_pipeline": "depmap_atomic_extraction_v2",
            "extractor_name": "scripts/extract_depmap_dependency_evidence.py",
            "extractor_version": "2.0.0",
            "extracted_at": created_at,
            "study_design": "functional_genomics_cell_line_panel",
            "clinical_stage": "in_vitro_model_panel",
            "licence": None,
            "is_primary_observation": True,
            "source_priority": 2 if family in {"HCC", "CRC"} else 3,
        },
        "aggregation_parent_ids": [],
    }
    if provenance_overrides:
        evidence["provenance"].update(provenance_overrides)
    cohort_node = {
        "node_type": "cohort",
        "id": cohort_id,
        "cohort_label": f"DepMap {family} lineage panel ({family_label})",
        "disease_name": family,
        "lineage": family,
        "cohort_type": "cell_line_panel",
        "n_effective": n_group_a,
    }
    entity_nodes: list[dict[str, Any]] = []
    for entity in [subject_entity, object_entity, outcome_entity, disease_entity]:
        if entity is None:
            continue
        entity_nodes.append(
            {
                "node_type": "entity",
                "id": entity["id"],
                "name": entity["label"],
                "entity_type": entity["entity_type"],
                "db_source": entity["db_source"],
                "aliases": entity["aliases"],
                "normalized_from": entity["normalized_from"],
            }
        )
    return evidence, cohort_node, entity_nodes


def append_bundle(
    *,
    evidence_nodes: list[dict[str, Any]],
    cohort_nodes: list[dict[str, Any]],
    entity_nodes: list[dict[str, Any]],
    local_counts: Counter[str],
    bundle: tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]] | None,
    count_key: str,
) -> None:
    if bundle is None:
        return
    evidence, cohort_node, entities = bundle
    evidence_nodes.append(evidence)
    cohort_nodes.append(cohort_node)
    entity_nodes.extend(entities)
    local_counts[count_key] += 1


def build_base_evidence(
    *,
    family: str,
    family_label: str,
    family_models: pd.DataFrame,
    background_models: pd.DataFrame,
    fam_ge: pd.DataFrame,
    bg_ge: pd.DataFrame,
    fam_dp: pd.DataFrame,
    bg_dp: pd.DataFrame,
    local_counts: Counter[str],
    evidence_nodes: list[dict[str, Any]],
    cohort_nodes: list[dict[str, Any]],
    entity_nodes: list[dict[str, Any]],
) -> tuple[set[str], set[str]]:
    fam_median = fam_ge.median(axis=0)
    bg_median = bg_ge.median(axis=0)
    fam_mean_prob = fam_dp.mean(axis=0)
    bg_mean_prob = bg_dp.mean(axis=0)
    fam_prevalence = (fam_dp >= DEPENDENCY_POSITIVE_PROBABILITY).mean(axis=0)
    bg_prevalence = (bg_dp >= DEPENDENCY_POSITIVE_PROBABILITY).mean(axis=0)

    selected_genes: set[str] = set()
    anchor_genes: set[str] = set(
        fam_median.index[
            (fam_median <= CONTEXT_ANCHOR_EFFECT_MAX)
            & (fam_mean_prob >= CONTEXT_ANCHOR_MEAN_PROB_MIN)
            & (fam_prevalence >= CONTEXT_ANCHOR_PREVALENCE_MIN)
            & (bg_prevalence <= CONTEXT_ANCHOR_BACKGROUND_PREVALENCE_MAX)
        ]
    )

    recurrent_mask = (
        (fam_median <= RECURRENT_EFFECT_MAX)
        & (fam_mean_prob >= RECURRENT_MEAN_PROB_MIN)
        & (fam_prevalence >= RECURRENT_PREVALENCE_MIN)
        & (bg_prevalence <= RECURRENT_BACKGROUND_PREVALENCE_MAX)
    )
    for gene in fam_median.index[recurrent_mask]:
        subject_entity = build_gene_entity(gene)
        summary = (
            f"{family} DepMap models show recurrent dependency on {gene}; "
            f"median Chronos effect={round(float(fam_median[gene]), 4)}, "
            f"mean dependency probability={round(float(fam_mean_prob[gene]), 4)}, "
            f"dependency prevalence={round(float(fam_prevalence[gene]), 4)}"
        )
        bundle = make_evidence(
            family=family,
            family_label=family_label,
            predicate="dependency_in",
            evidence_category="functional_dependency",
            feature_layer="crispr_dependency",
            metric="chronos_gene_effect_median",
            subject_entity=subject_entity,
            object_entity=None,
            outcome_entity=None,
            comparison_id="cohort_recurrence",
            comparison_label="recurrent lineage dependency signal without subgroup split",
            comparison_type=None,
            group_a_label=f"{family}_depmap_models",
            group_b_label=None,
            sample_count=len(family_models.index),
            n_group_a=len(family_models.index),
            n_group_b=None,
            value=float(fam_median[gene]),
            effect_size=abs(float(fam_median[gene])),
            summary_text=summary,
            platform_specific_fields={
                "model_panel_kind": "depmap_family_panel",
                "family_model_count": int(len(family_models.index)),
                "background_model_count": int(len(background_models.index)),
                "family_median_gene_effect": round(float(fam_median[gene]), 4),
                "family_mean_dependency_probability": round(float(fam_mean_prob[gene]), 4),
                "family_dependency_prevalence": round(float(fam_prevalence[gene]), 4),
                "background_dependency_prevalence": round(float(bg_prevalence[gene]), 4),
                "background_mean_dependency_probability": round(float(bg_mean_prob[gene]), 4),
                "oncotree_codes": sorted(set(family_models["OncotreeCode"])),
                "primary_diseases": sorted(set(family_models["OncotreePrimaryDisease"])),
            },
            source_file=relpath(GENE_EFFECT_PATH),
            source_record_key=f"{family}:{gene}:dependency",
            data_completeness=1.0,
            statistical_method="thresholded DepMap Chronos family-level recurrent dependency aggregation",
        )
        append_bundle(
            evidence_nodes=evidence_nodes,
            cohort_nodes=cohort_nodes,
            entity_nodes=entity_nodes,
            local_counts=local_counts,
            bundle=bundle,
            count_key="dependency_in",
        )
        selected_genes.add(gene)

    for gene in fam_median.index[
        (fam_median <= SELECTIVE_EFFECT_MAX)
        & ((fam_median - bg_median) <= SELECTIVE_DELTA_MAX)
        & (fam_mean_prob >= SELECTIVE_MEAN_PROB_MIN)
        & ((fam_prevalence - bg_prevalence) >= SELECTIVE_PREVALENCE_DELTA_MIN)
        & (bg_prevalence <= SELECTIVE_BACKGROUND_PREVALENCE_MAX)
    ]:
        subject_entity = build_gene_entity(gene)
        delta = float(fam_median[gene] - bg_median[gene])
        effect_size = max(abs(delta), pooled_cohen_d(fam_ge[gene], bg_ge[gene]))
        if effect_size < SELECTIVE_COHEN_D_MIN:
            continue
        summary = (
            f"{family} DepMap models show selective dependency on {gene} relative to the selected multi-cancer background; "
            f"family median Chronos effect={round(float(fam_median[gene]), 4)}, "
            f"background median={round(float(bg_median[gene]), 4)}, "
            f"Cohen's d={round(effect_size, 4)}"
        )
        bundle = make_evidence(
            family=family,
            family_label=family_label,
            predicate="selective_dependency_in",
            evidence_category="functional_dependency",
            feature_layer="crispr_dependency",
            metric="chronos_gene_effect_delta_vs_background",
            subject_entity=subject_entity,
            object_entity=None,
            outcome_entity=None,
            comparison_id="pan_cancer_background",
            comparison_label="target DepMap lineage panel vs selected multi-cancer background panel",
            comparison_type="lineage_enriched_vs_background",
            group_a_label=f"{family}_depmap_models",
            group_b_label="selected_depmap_background_models",
            sample_count=int(len(family_models.index) + len(background_models.index)),
            n_group_a=len(family_models.index),
            n_group_b=len(background_models.index),
            value=delta,
            effect_size=effect_size,
            summary_text=summary,
            platform_specific_fields={
                "model_panel_kind": "depmap_family_vs_background",
                "family_model_count": int(len(family_models.index)),
                "background_model_count": int(len(background_models.index)),
                "family_median_gene_effect": round(float(fam_median[gene]), 4),
                "background_median_gene_effect": round(float(bg_median[gene]), 4),
                "family_mean_dependency_probability": round(float(fam_mean_prob[gene]), 4),
                "background_mean_dependency_probability": round(float(bg_mean_prob[gene]), 4),
                "family_dependency_prevalence": round(float(fam_prevalence[gene]), 4),
                "background_dependency_prevalence": round(float(bg_prevalence[gene]), 4),
                "dependency_prevalence_delta": round(float(fam_prevalence[gene] - bg_prevalence[gene]), 4),
                "cohen_d": round(effect_size, 4),
                "oncotree_codes": sorted(set(family_models["OncotreeCode"])),
                "primary_diseases": sorted(set(family_models["OncotreePrimaryDisease"])),
            },
            source_file=relpath(GENE_EFFECT_PATH),
            source_record_key=f"{family}:{gene}:selective_dependency",
            data_completeness=1.0,
            statistical_method="thresholded DepMap Chronos lineage-selective dependency contrast against multi-cancer background",
        )
        append_bundle(
            evidence_nodes=evidence_nodes,
            cohort_nodes=cohort_nodes,
            entity_nodes=entity_nodes,
            local_counts=local_counts,
            bundle=bundle,
            count_key="selective_dependency_in",
        )
        selected_genes.add(gene)

    return selected_genes, anchor_genes


def add_same_gene_context_evidence(
    *,
    family: str,
    family_label: str,
    family_models: pd.DataFrame,
    fam_ge: pd.DataFrame,
    fam_dp: pd.DataFrame,
    context_matrix: pd.DataFrame,
    genes: set[str],
    local_counts: Counter[str],
    evidence_nodes: list[dict[str, Any]],
    cohort_nodes: list[dict[str, Any]],
    entity_nodes: list[dict[str, Any]],
    context_kind: str,
) -> set[str]:
    family_index = family_models.index.intersection(context_matrix.index)
    if len(family_index) < MIN_FAMILY_MODELS:
        return set()
    ge = fam_ge.loc[family_index]
    dp = fam_dp.loc[family_index]
    retained_context_genes: set[str] = set()

    for gene in sorted(genes.intersection(context_matrix.columns).intersection(ge.columns)):
        context_values = context_matrix.loc[family_index, gene].dropna()
        if context_kind == "expression":
            if len(context_values) < EXPR_CONTEXT_MIN_GROUP * 2:
                continue
            q1, q3 = context_values.quantile([0.25, 0.75])
            positive_index = context_values.index[context_values >= q3]
            baseline_index = context_values.index[context_values <= q1]
            if len(positive_index) < EXPR_CONTEXT_MIN_GROUP or len(baseline_index) < EXPR_CONTEXT_MIN_GROUP:
                continue
            if float(context_values.loc[positive_index].median() - context_values.loc[baseline_index].median()) < EXPR_CONTEXT_MIN_SPLIT:
                continue
            predicate = "high_expression_associated_with_dependency_in"
            evidence_category = "expression_dependency_context"
            feature_layer = "expression_dependency_context"
            metric = "chronos_gene_effect_high_expression_vs_low_expression"
            comparison_id = "gene_expression_context"
            comparison_label = "high-expression gene context vs low-expression gene context"
            object_label = f"{gene} high expression context"
            source_file = f"{relpath(GENE_EFFECT_PATH)} | {relpath(EXPRESSION_PATH)}"
            min_median = EXPR_CONTEXT_MIN_MEDIAN
            min_delta = EXPR_CONTEXT_MIN_DELTA
            min_prev_delta = EXPR_CONTEXT_MIN_PREVALENCE_DELTA
            min_d = EXPR_CONTEXT_MIN_COHEN_D
        elif context_kind == "copy_number":
            positive_index = context_values.index[context_values >= COPY_GAIN_THRESHOLD]
            baseline_index = context_values.index[context_values <= COPY_BASELINE_MAX]
            if len(positive_index) < COPY_CONTEXT_MIN_GROUP or len(baseline_index) < COPY_CONTEXT_MIN_GROUP:
                continue
            predicate = "copy_number_gain_associated_with_dependency_in"
            evidence_category = "copy_number_dependency_context"
            feature_layer = "copy_number_dependency_context"
            metric = "chronos_gene_effect_copy_gain_vs_baseline"
            comparison_id = "copy_number_context"
            comparison_label = "copy-number gain context vs baseline copy-number context"
            object_label = f"{gene} copy-number gain context"
            source_file = f"{relpath(GENE_EFFECT_PATH)} | {relpath(COPY_NUMBER_PATH)}"
            min_median = COPY_CONTEXT_MIN_MEDIAN
            min_delta = COPY_CONTEXT_MIN_DELTA
            min_prev_delta = COPY_CONTEXT_MIN_PREVALENCE_DELTA
            min_d = COPY_CONTEXT_MIN_COHEN_D
        else:
            positive_index = context_values.index[context_values > 0]
            baseline_index = context_values.index[context_values == 0]
            if len(positive_index) < MUTATION_CONTEXT_MIN_POSITIVE or len(baseline_index) < MUTATION_CONTEXT_MIN_NEGATIVE:
                continue
            predicate = "mutation_associated_with_dependency_in"
            evidence_category = "mutation_dependency_context"
            feature_layer = "mutation_dependency_context"
            metric = "chronos_gene_effect_mutated_vs_unmutated"
            comparison_id = "mutation_context"
            comparison_label = "mutation-positive context vs mutation-negative context"
            object_label = f"{gene} mutated context"
            source_file = f"{relpath(GENE_EFFECT_PATH)} | {relpath(HOTSPOT_MUTATION_PATH)} | {relpath(DAMAGING_MUTATION_PATH)}"
            min_median = MUTATION_CONTEXT_MIN_MEDIAN
            min_delta = MUTATION_CONTEXT_MIN_DELTA
            min_prev_delta = MUTATION_CONTEXT_MIN_PREVALENCE_DELTA
            min_d = MUTATION_CONTEXT_MIN_COHEN_D

        group_a_effect = ge.loc[positive_index, gene].dropna()
        group_b_effect = ge.loc[baseline_index, gene].dropna()
        group_a_prob = dp.loc[positive_index, gene].dropna()
        group_b_prob = dp.loc[baseline_index, gene].dropna()
        if len(group_a_effect) < 4 or len(group_b_effect) < 4:
            continue
        group_a_median = float(group_a_effect.median())
        group_b_median = float(group_b_effect.median())
        delta = group_a_median - group_b_median
        prevalence_delta = float((group_a_prob >= DEPENDENCY_POSITIVE_PROBABILITY).mean() - (group_b_prob >= DEPENDENCY_POSITIVE_PROBABILITY).mean())
        effect_size = pooled_cohen_d(group_a_effect, group_b_effect)
        if group_a_median > min_median or delta > min_delta or prevalence_delta < min_prev_delta or effect_size < min_d:
            continue

        summary = (
            f"{family} DepMap models show stronger dependency on {gene} in {object_label}; "
            f"context median effect={round(group_a_median, 4)}, baseline median={round(group_b_median, 4)}, "
            f"prevalence delta={round(prevalence_delta, 4)}"
        )
        bundle = make_evidence(
            family=family,
            family_label=family_label,
            predicate=predicate,
            evidence_category=evidence_category,
            feature_layer=feature_layer,
            metric=metric,
            subject_entity=build_gene_entity(gene),
            object_entity=build_context_entity(object_label, aliases=[object_label, gene]),
            outcome_entity=build_disease_entity(family, family_label),
            comparison_id=comparison_id,
            comparison_label=comparison_label,
            comparison_type="context_positive_vs_context_negative",
            group_a_label="context_positive_models",
            group_b_label="context_negative_models",
            sample_count=len(group_a_effect) + len(group_b_effect),
            n_group_a=len(group_a_effect),
            n_group_b=len(group_b_effect),
            value=delta,
            effect_size=max(abs(delta), effect_size),
            summary_text=summary,
            platform_specific_fields={
                "context_gene": gene,
                "family_model_count": int(len(family_models.index)),
                "context_positive_model_count": int(len(group_a_effect)),
                "context_negative_model_count": int(len(group_b_effect)),
                "context_positive_median_gene_effect": round(group_a_median, 4),
                "context_negative_median_gene_effect": round(group_b_median, 4),
                "context_positive_mean_dependency_probability": round(float(group_a_prob.mean()), 4),
                "context_negative_mean_dependency_probability": round(float(group_b_prob.mean()), 4),
                "context_prevalence_delta": round(prevalence_delta, 4),
                "cohen_d": round(effect_size, 4),
            },
            source_file=source_file,
            source_record_key=f"{family}:{gene}:{context_kind}_context",
            data_completeness=completeness_ratio(len(group_a_effect) + len(group_b_effect), len(family_index)),
            statistical_method=f"thresholded within-family DepMap dependency contrast in {context_kind} context",
        )
        append_bundle(
            evidence_nodes=evidence_nodes,
            cohort_nodes=cohort_nodes,
            entity_nodes=entity_nodes,
            local_counts=local_counts,
            bundle=bundle,
            count_key=predicate,
        )
        retained_context_genes.add(gene)

    return retained_context_genes


def add_cross_gene_context_evidence(
    *,
    family: str,
    family_label: str,
    family_models: pd.DataFrame,
    fam_ge: pd.DataFrame,
    fam_dp: pd.DataFrame,
    context_matrix: pd.DataFrame,
    subject_genes: set[str],
    context_genes: set[str],
    local_counts: Counter[str],
    evidence_nodes: list[dict[str, Any]],
    cohort_nodes: list[dict[str, Any]],
    entity_nodes: list[dict[str, Any]],
    context_kind: str,
) -> None:
    family_index = family_models.index.intersection(context_matrix.index)
    if len(family_index) < MIN_FAMILY_MODELS or not subject_genes or not context_genes:
        return
    ge = fam_ge.loc[family_index]
    dp = fam_dp.loc[family_index]

    for context_gene in sorted(context_genes.intersection(context_matrix.columns)):
        context_values = context_matrix.loc[family_index, context_gene].dropna()
        if context_kind == "expression":
            if len(context_values) < CROSS_EXPR_CONTEXT_MIN_GROUP * 2:
                continue
            q1, q3 = context_values.quantile([0.25, 0.75])
            positive_index = context_values.index[context_values >= q3]
            baseline_index = context_values.index[context_values <= q1]
            if len(positive_index) < CROSS_EXPR_CONTEXT_MIN_GROUP or len(baseline_index) < CROSS_EXPR_CONTEXT_MIN_GROUP:
                continue
            if float(context_values.loc[positive_index].median() - context_values.loc[baseline_index].median()) < CROSS_EXPR_CONTEXT_MIN_SPLIT:
                continue
            predicate = "high_expression_associated_with_dependency_in"
            evidence_category = "expression_dependency_context"
            feature_layer = "expression_dependency_context"
            metric = "chronos_gene_effect_cross_gene_high_expression_vs_low_expression"
            comparison_id = "gene_expression_context"
            comparison_label = "high-expression gene context vs low-expression gene context"
            source_file = f"{relpath(GENE_EFFECT_PATH)} | {relpath(EXPRESSION_PATH)}"
            min_median = CROSS_EXPR_CONTEXT_MIN_MEDIAN
            min_delta = CROSS_EXPR_CONTEXT_MIN_DELTA
            min_prev_delta = CROSS_EXPR_CONTEXT_MIN_PREVALENCE_DELTA
            min_d = CROSS_EXPR_CONTEXT_MIN_COHEN_D
            object_label = f"{context_gene} high expression context"
        elif context_kind == "copy_number":
            positive_index = context_values.index[context_values >= COPY_GAIN_THRESHOLD]
            baseline_index = context_values.index[context_values <= COPY_BASELINE_MAX]
            if len(positive_index) < CROSS_COPY_CONTEXT_MIN_GROUP or len(baseline_index) < CROSS_COPY_CONTEXT_MIN_GROUP:
                continue
            predicate = "copy_number_gain_associated_with_dependency_in"
            evidence_category = "copy_number_dependency_context"
            feature_layer = "copy_number_dependency_context"
            metric = "chronos_gene_effect_cross_gene_copy_gain_vs_baseline"
            comparison_id = "copy_number_context"
            comparison_label = "copy-number gain context vs baseline copy-number context"
            source_file = f"{relpath(GENE_EFFECT_PATH)} | {relpath(COPY_NUMBER_PATH)}"
            min_median = CROSS_COPY_CONTEXT_MIN_MEDIAN
            min_delta = CROSS_COPY_CONTEXT_MIN_DELTA
            min_prev_delta = CROSS_COPY_CONTEXT_MIN_PREVALENCE_DELTA
            min_d = CROSS_COPY_CONTEXT_MIN_COHEN_D
            object_label = f"{context_gene} copy-number gain context"
        else:
            positive_index = context_values.index[context_values > 0]
            baseline_index = context_values.index[context_values == 0]
            if len(positive_index) < CROSS_MUTATION_CONTEXT_MIN_POSITIVE or len(baseline_index) < CROSS_MUTATION_CONTEXT_MIN_NEGATIVE:
                continue
            predicate = "mutation_associated_with_dependency_in"
            evidence_category = "mutation_dependency_context"
            feature_layer = "mutation_dependency_context"
            metric = "chronos_gene_effect_cross_gene_mutated_vs_unmutated"
            comparison_id = "mutation_context"
            comparison_label = "mutation-positive context vs mutation-negative context"
            source_file = f"{relpath(GENE_EFFECT_PATH)} | {relpath(HOTSPOT_MUTATION_PATH)} | {relpath(DAMAGING_MUTATION_PATH)}"
            min_median = CROSS_MUTATION_CONTEXT_MIN_MEDIAN
            min_delta = CROSS_MUTATION_CONTEXT_MIN_DELTA
            min_prev_delta = CROSS_MUTATION_CONTEXT_MIN_PREVALENCE_DELTA
            min_d = CROSS_MUTATION_CONTEXT_MIN_COHEN_D
            object_label = f"{context_gene} mutated context"

        for subject_gene in sorted(subject_genes.intersection(ge.columns)):
            if subject_gene == context_gene:
                continue
            group_a_effect = ge.loc[positive_index, subject_gene].dropna()
            group_b_effect = ge.loc[baseline_index, subject_gene].dropna()
            group_a_prob = dp.loc[positive_index, subject_gene].dropna()
            group_b_prob = dp.loc[baseline_index, subject_gene].dropna()
            if len(group_a_effect) < 4 or len(group_b_effect) < 4:
                continue
            group_a_median = float(group_a_effect.median())
            group_b_median = float(group_b_effect.median())
            delta = group_a_median - group_b_median
            prevalence_delta = float((group_a_prob >= DEPENDENCY_POSITIVE_PROBABILITY).mean() - (group_b_prob >= DEPENDENCY_POSITIVE_PROBABILITY).mean())
            effect_size = pooled_cohen_d(group_a_effect, group_b_effect)
            if group_a_median > min_median or delta > min_delta or prevalence_delta < min_prev_delta or effect_size < min_d:
                continue

            summary = (
                f"{family} DepMap models show stronger dependency on {subject_gene} in {object_label}; "
                f"context median effect={round(group_a_median, 4)}, baseline median={round(group_b_median, 4)}, "
                f"prevalence delta={round(prevalence_delta, 4)}"
            )
            bundle = make_evidence(
                family=family,
                family_label=family_label,
                predicate=predicate,
                evidence_category=evidence_category,
                feature_layer=feature_layer,
                metric=metric,
                subject_entity=build_gene_entity(subject_gene),
                object_entity=build_context_entity(object_label, aliases=[object_label, context_gene]),
                outcome_entity=build_disease_entity(family, family_label),
                comparison_id=comparison_id,
                comparison_label=comparison_label,
                comparison_type="context_positive_vs_context_negative",
                group_a_label="context_positive_models",
                group_b_label="context_negative_models",
                sample_count=len(group_a_effect) + len(group_b_effect),
                n_group_a=len(group_a_effect),
                n_group_b=len(group_b_effect),
                value=delta,
                effect_size=max(abs(delta), effect_size),
                summary_text=summary,
                platform_specific_fields={
                    "context_gene": context_gene,
                    "family_model_count": int(len(family_models.index)),
                    "context_positive_model_count": int(len(group_a_effect)),
                    "context_negative_model_count": int(len(group_b_effect)),
                    "context_positive_median_gene_effect": round(group_a_median, 4),
                    "context_negative_median_gene_effect": round(group_b_median, 4),
                    "context_positive_mean_dependency_probability": round(float(group_a_prob.mean()), 4),
                    "context_negative_mean_dependency_probability": round(float(group_b_prob.mean()), 4),
                    "context_prevalence_delta": round(prevalence_delta, 4),
                    "cohen_d": round(effect_size, 4),
                },
                source_file=source_file,
                source_record_key=f"{family}:{subject_gene}:{context_gene}:{context_kind}_cross_context",
                data_completeness=completeness_ratio(len(group_a_effect) + len(group_b_effect), len(family_index)),
                statistical_method=f"thresholded within-family DepMap dependency contrast under cross-gene {context_kind} context",
            )
            append_bundle(
                evidence_nodes=evidence_nodes,
                cohort_nodes=cohort_nodes,
                entity_nodes=entity_nodes,
                local_counts=local_counts,
                bundle=bundle,
                count_key=predicate,
            )


def add_boolean_context_dependency(
    *,
    family: str,
    family_label: str,
    family_models: pd.DataFrame,
    fam_ge: pd.DataFrame,
    fam_dp: pd.DataFrame,
    context_matrix: pd.DataFrame,
    genes: set[str],
    local_counts: Counter[str],
    evidence_nodes: list[dict[str, Any]],
    cohort_nodes: list[dict[str, Any]],
    entity_nodes: list[dict[str, Any]],
    predicate: str,
    evidence_category: str,
    feature_layer: str,
    metric: str,
    comparison_id: str,
    comparison_label: str,
    source_file: str,
    min_positive: int,
    min_negative: int,
    min_median: float,
    min_delta: float,
    min_prev_delta: float,
    min_d: float,
) -> None:
    family_index = family_models.index.intersection(context_matrix.index)
    if len(family_index) < MIN_FAMILY_MODELS:
        return
    ge = fam_ge.loc[family_index]
    dp = fam_dp.loc[family_index]

    for context_name in sorted(context_matrix.columns):
        values = context_matrix.loc[family_index, context_name].fillna(0)
        positive_index = values.index[values > 0]
        negative_index = values.index[values == 0]
        if len(positive_index) < min_positive or len(negative_index) < min_negative:
            continue

        for gene in sorted(genes.intersection(ge.columns)):
            group_a_effect = ge.loc[positive_index, gene].dropna()
            group_b_effect = ge.loc[negative_index, gene].dropna()
            group_a_prob = dp.loc[positive_index, gene].dropna()
            group_b_prob = dp.loc[negative_index, gene].dropna()
            if len(group_a_effect) < min_positive or len(group_b_effect) < min_negative:
                continue
            group_a_median = float(group_a_effect.median())
            group_b_median = float(group_b_effect.median())
            delta = group_a_median - group_b_median
            prevalence_delta = float((group_a_prob >= DEPENDENCY_POSITIVE_PROBABILITY).mean() - (group_b_prob >= DEPENDENCY_POSITIVE_PROBABILITY).mean())
            effect_size = pooled_cohen_d(group_a_effect, group_b_effect)
            if group_a_median > min_median or delta > min_delta or prevalence_delta < min_prev_delta or effect_size < min_d:
                continue

            summary = (
                f"{family} DepMap models show stronger dependency on {gene} in {context_name} context; "
                f"context median effect={round(group_a_median, 4)}, baseline median={round(group_b_median, 4)}, "
                f"prevalence delta={round(prevalence_delta, 4)}"
            )
            bundle = make_evidence(
                family=family,
                family_label=family_label,
                predicate=predicate,
                evidence_category=evidence_category,
                feature_layer=feature_layer,
                metric=metric,
                subject_entity=build_gene_entity(gene),
                object_entity=build_context_entity(context_name),
                outcome_entity=build_disease_entity(family, family_label),
                comparison_id=comparison_id,
                comparison_label=comparison_label,
                comparison_type="context_positive_vs_context_negative",
                group_a_label="context_positive_models",
                group_b_label="context_negative_models",
                sample_count=len(group_a_effect) + len(group_b_effect),
                n_group_a=len(group_a_effect),
                n_group_b=len(group_b_effect),
                value=delta,
                effect_size=max(abs(delta), effect_size),
                summary_text=summary,
                platform_specific_fields={
                    "context_name": context_name,
                    "family_model_count": int(len(family_models.index)),
                    "context_positive_model_count": int(len(group_a_effect)),
                    "context_negative_model_count": int(len(group_b_effect)),
                    "context_positive_median_gene_effect": round(group_a_median, 4),
                    "context_negative_median_gene_effect": round(group_b_median, 4),
                    "context_positive_mean_dependency_probability": round(float(group_a_prob.mean()), 4),
                    "context_negative_mean_dependency_probability": round(float(group_b_prob.mean()), 4),
                    "context_prevalence_delta": round(prevalence_delta, 4),
                    "cohen_d": round(effect_size, 4),
                },
                source_file=source_file,
                source_record_key=f"{family}:{gene}:{context_name}:{predicate}",
                data_completeness=completeness_ratio(len(group_a_effect) + len(group_b_effect), len(family_index)),
                statistical_method="thresholded within-family DepMap dependency contrast under boolean molecular context",
            )
            append_bundle(
                evidence_nodes=evidence_nodes,
                cohort_nodes=cohort_nodes,
                entity_nodes=entity_nodes,
                local_counts=local_counts,
                bundle=bundle,
                count_key=predicate,
            )


def add_signature_context_dependency(
    *,
    family: str,
    family_label: str,
    family_models: pd.DataFrame,
    fam_ge: pd.DataFrame,
    fam_dp: pd.DataFrame,
    signatures: pd.DataFrame,
    genes: set[str],
    local_counts: Counter[str],
    evidence_nodes: list[dict[str, Any]],
    cohort_nodes: list[dict[str, Any]],
    entity_nodes: list[dict[str, Any]],
) -> None:
    family_index = family_models.index.intersection(signatures.index)
    if len(family_index) < MIN_FAMILY_MODELS:
        return
    ge = fam_ge.loc[family_index]
    dp = fam_dp.loc[family_index]

    for signature, (high_label, low_label) in SIGNATURE_CONTEXTS.items():
        values = signatures.loc[family_index, signature].dropna()
        if len(values) < PHENOTYPE_CONTEXT_MIN_POSITIVE + PHENOTYPE_CONTEXT_MIN_NEGATIVE:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        positive_index = values.index[values >= q3]
        negative_index = values.index[values <= q1]
        if len(positive_index) < PHENOTYPE_CONTEXT_MIN_POSITIVE or len(negative_index) < PHENOTYPE_CONTEXT_MIN_NEGATIVE:
            continue

        for gene in sorted(genes.intersection(ge.columns)):
            group_a_effect = ge.loc[positive_index, gene].dropna()
            group_b_effect = ge.loc[negative_index, gene].dropna()
            group_a_prob = dp.loc[positive_index, gene].dropna()
            group_b_prob = dp.loc[negative_index, gene].dropna()
            if len(group_a_effect) < PHENOTYPE_CONTEXT_MIN_POSITIVE or len(group_b_effect) < PHENOTYPE_CONTEXT_MIN_NEGATIVE:
                continue
            group_a_median = float(group_a_effect.median())
            group_b_median = float(group_b_effect.median())
            delta = group_a_median - group_b_median
            prevalence_delta = float((group_a_prob >= DEPENDENCY_POSITIVE_PROBABILITY).mean() - (group_b_prob >= DEPENDENCY_POSITIVE_PROBABILITY).mean())
            effect_size = pooled_cohen_d(group_a_effect, group_b_effect)
            if group_a_median > PHENOTYPE_CONTEXT_MIN_MEDIAN or delta > PHENOTYPE_CONTEXT_MIN_DELTA or prevalence_delta < PHENOTYPE_CONTEXT_MIN_PREVALENCE_DELTA or effect_size < PHENOTYPE_CONTEXT_MIN_COHEN_D:
                continue

            context_name = f"{signature}:{high_label}"
            summary = (
                f"{family} DepMap models show stronger dependency on {gene} in {high_label} context; "
                f"context median effect={round(group_a_median, 4)}, baseline median={round(group_b_median, 4)}, "
                f"prevalence delta={round(prevalence_delta, 4)}"
            )
            bundle = make_evidence(
                family=family,
                family_label=family_label,
                predicate="phenotype_associated_with_dependency_in",
                evidence_category="phenotype_dependency_context",
                feature_layer="phenotype_dependency_context",
                metric=f"chronos_gene_effect_{signature.lower()}_high_vs_low",
                subject_entity=build_gene_entity(gene),
                object_entity=build_context_entity(context_name, aliases=[context_name, high_label, low_label]),
                outcome_entity=build_disease_entity(family, family_label),
                comparison_id="phenotype_context",
                comparison_label=f"high {signature} phenotype context vs low {signature} phenotype context",
                comparison_type="high_vs_low",
                group_a_label=high_label,
                group_b_label=low_label,
                sample_count=len(group_a_effect) + len(group_b_effect),
                n_group_a=len(group_a_effect),
                n_group_b=len(group_b_effect),
                value=delta,
                effect_size=max(abs(delta), effect_size),
                summary_text=summary,
                platform_specific_fields={
                    "signature_name": signature,
                    "context_positive_model_count": int(len(group_a_effect)),
                    "context_negative_model_count": int(len(group_b_effect)),
                    "context_positive_median_gene_effect": round(group_a_median, 4),
                    "context_negative_median_gene_effect": round(group_b_median, 4),
                    "context_positive_mean_dependency_probability": round(float(group_a_prob.mean()), 4),
                    "context_negative_mean_dependency_probability": round(float(group_b_prob.mean()), 4),
                    "context_prevalence_delta": round(prevalence_delta, 4),
                    "cohen_d": round(effect_size, 4),
                },
                source_file=f"{relpath(GENE_EFFECT_PATH)} | {relpath(GLOBAL_SIGNATURE_PATH)}",
                source_record_key=f"{family}:{gene}:{signature}:phenotype_dependency",
                data_completeness=completeness_ratio(len(group_a_effect) + len(group_b_effect), len(family_index)),
                statistical_method="thresholded within-family DepMap dependency contrast under quantitative phenotype signatures",
            )
            append_bundle(
                evidence_nodes=evidence_nodes,
                cohort_nodes=cohort_nodes,
                entity_nodes=entity_nodes,
                local_counts=local_counts,
                bundle=bundle,
                count_key="phenotype_associated_with_dependency_in",
            )


def add_metastatic_context_dependency(
    *,
    family: str,
    family_label: str,
    family_models: pd.DataFrame,
    fam_ge: pd.DataFrame,
    fam_dp: pd.DataFrame,
    genes: set[str],
    local_counts: Counter[str],
    evidence_nodes: list[dict[str, Any]],
    cohort_nodes: list[dict[str, Any]],
    entity_nodes: list[dict[str, Any]],
) -> None:
    values = family_models["PrimaryOrMetastasis"].fillna("")
    positive_index = values.index[values == "Metastatic"]
    negative_index = values.index[values == "Primary"]
    if len(positive_index) < METASTATIC_CONTEXT_MIN_POSITIVE or len(negative_index) < METASTATIC_CONTEXT_MIN_NEGATIVE:
        return

    for gene in sorted(genes.intersection(fam_ge.columns)):
        group_a_effect = fam_ge.loc[positive_index, gene].dropna()
        group_b_effect = fam_ge.loc[negative_index, gene].dropna()
        group_a_prob = fam_dp.loc[positive_index, gene].dropna()
        group_b_prob = fam_dp.loc[negative_index, gene].dropna()
        if len(group_a_effect) < METASTATIC_CONTEXT_MIN_POSITIVE or len(group_b_effect) < METASTATIC_CONTEXT_MIN_NEGATIVE:
            continue
        group_a_median = float(group_a_effect.median())
        group_b_median = float(group_b_effect.median())
        delta = group_a_median - group_b_median
        prevalence_delta = float((group_a_prob >= DEPENDENCY_POSITIVE_PROBABILITY).mean() - (group_b_prob >= DEPENDENCY_POSITIVE_PROBABILITY).mean())
        effect_size = pooled_cohen_d(group_a_effect, group_b_effect)
        if group_a_median > METASTATIC_CONTEXT_MIN_MEDIAN or delta > METASTATIC_CONTEXT_MIN_DELTA or prevalence_delta < METASTATIC_CONTEXT_MIN_PREVALENCE_DELTA or effect_size < METASTATIC_CONTEXT_MIN_COHEN_D:
            continue

        summary = (
            f"{family} metastatic DepMap models show stronger dependency on {gene} than primary models; "
            f"metastatic median effect={round(group_a_median, 4)}, primary median={round(group_b_median, 4)}, "
            f"prevalence delta={round(prevalence_delta, 4)}"
        )
        bundle = make_evidence(
            family=family,
            family_label=family_label,
            predicate="metastatic_state_associated_with_dependency_in",
            evidence_category="cell_state_dependency_context",
            feature_layer="cell_state_dependency_context",
            metric="chronos_gene_effect_metastatic_vs_primary",
            subject_entity=build_gene_entity(gene),
            object_entity=build_context_entity(PRIMARY_METASTASIS_LABELS["Metastatic"]),
            outcome_entity=build_disease_entity(family, family_label),
            comparison_id="metastatic_state_context",
            comparison_label="metastatic cell-state context vs primary cell-state context",
            comparison_type="metastatic_vs_primary",
            group_a_label="metastatic_models",
            group_b_label="primary_models",
            sample_count=len(group_a_effect) + len(group_b_effect),
            n_group_a=len(group_a_effect),
            n_group_b=len(group_b_effect),
            value=delta,
            effect_size=max(abs(delta), effect_size),
            summary_text=summary,
            platform_specific_fields={
                "family_model_count": int(len(family_models.index)),
                "metastatic_model_count": int(len(group_a_effect)),
                "primary_model_count": int(len(group_b_effect)),
                "metastatic_median_gene_effect": round(group_a_median, 4),
                "primary_median_gene_effect": round(group_b_median, 4),
                "prevalence_delta": round(prevalence_delta, 4),
                "cohen_d": round(effect_size, 4),
            },
            source_file=relpath(MODEL_PATH),
            source_record_key=f"{family}:{gene}:metastatic_dependency",
            data_completeness=completeness_ratio(len(group_a_effect) + len(group_b_effect), len(family_models.index)),
            statistical_method="thresholded within-family DepMap dependency contrast between metastatic and primary model states",
        )
        append_bundle(
            evidence_nodes=evidence_nodes,
            cohort_nodes=cohort_nodes,
            entity_nodes=entity_nodes,
            local_counts=local_counts,
            bundle=bundle,
            count_key="metastatic_state_associated_with_dependency_in",
        )


def add_codependency_evidence(
    *,
    family: str,
    family_label: str,
    family_models: pd.DataFrame,
    fam_ge: pd.DataFrame,
    fam_dp: pd.DataFrame,
    selected_genes: set[str],
    anchor_genes: set[str],
    local_counts: Counter[str],
    evidence_nodes: list[dict[str, Any]],
    cohort_nodes: list[dict[str, Any]],
    entity_nodes: list[dict[str, Any]],
) -> None:
    candidate_genes = sorted((selected_genes or anchor_genes).intersection(fam_ge.columns))
    if len(candidate_genes) < 2 or len(family_models.index) < CODEPENDENCY_MIN_SHARED_MODELS:
        return

    fam_ge_candidate = fam_ge[candidate_genes]
    fam_dp_candidate = fam_dp[candidate_genes]
    corr = fam_ge_candidate.corr(method="pearson")
    mean_prob = fam_dp_candidate.mean(axis=0)

    for left_gene, right_gene in combinations(candidate_genes, 2):
        corr_value = float(corr.loc[left_gene, right_gene])
        if np.isnan(corr_value) or corr_value < CODEPENDENCY_MIN_CORRELATION:
            continue
        if float(mean_prob[left_gene]) < CODEPENDENCY_MIN_MEAN_PROBABILITY or float(mean_prob[right_gene]) < CODEPENDENCY_MIN_MEAN_PROBABILITY:
            continue
        summary = (
            f"{family} DepMap models show co-dependency between {left_gene} and {right_gene}; "
            f"dependency profile correlation={round(corr_value, 4)}"
        )
        bundle = make_evidence(
            family=family,
            family_label=family_label,
            predicate="codependent_with",
            evidence_category="gene_codependency",
            feature_layer="codependency",
            metric="dependency_profile_correlation",
            subject_entity=build_gene_entity(left_gene),
            object_entity=build_gene_entity(right_gene),
            outcome_entity=build_disease_entity(family, family_label),
            comparison_id="codependency_context",
            comparison_label="within-lineage dependency profile correlation",
            comparison_type="positive_correlation",
            group_a_label=f"{family}_depmap_models",
            group_b_label=None,
            sample_count=len(family_models.index),
            n_group_a=len(family_models.index),
            n_group_b=None,
            value=corr_value,
            effect_size=corr_value,
            summary_text=summary,
            platform_specific_fields={
                "family_model_count": int(len(family_models.index)),
                "subject_mean_dependency_probability": round(float(mean_prob[left_gene]), 4),
                "object_mean_dependency_probability": round(float(mean_prob[right_gene]), 4),
                "correlation_method": "pearson",
            },
            source_file=f"{relpath(GENE_EFFECT_PATH)} | {relpath(GENE_DEPENDENCY_PATH)}",
            source_record_key=f"{family}:{left_gene}:{right_gene}:codependency",
            data_completeness=1.0,
            statistical_method="thresholded within-family DepMap dependency profile correlation",
        )
        append_bundle(
            evidence_nodes=evidence_nodes,
            cohort_nodes=cohort_nodes,
            entity_nodes=entity_nodes,
            local_counts=local_counts,
            bundle=bundle,
            count_key="codependent_with",
        )


def add_drug_response_evidence(
    *,
    family: str,
    family_label: str,
    family_models: pd.DataFrame,
    gdsc_response: pd.DataFrame,
    local_counts: Counter[str],
    evidence_nodes: list[dict[str, Any]],
    cohort_nodes: list[dict[str, Any]],
    entity_nodes: list[dict[str, Any]],
) -> None:
    family_response = gdsc_response[gdsc_response["family"] == family].copy()
    if family_response.empty:
        return

    grouped = family_response.groupby(
        [
            "dataset",
            "DRUG_ID",
            "CompoundID",
            "DrugLabel",
            "DrugTargets",
            "DrugMechanism",
            "ChEMBLID",
        ],
        dropna=False,
    )

    for (dataset, drug_id, compound_id, drug_label, drug_targets, drug_mechanism, chembl_id), frame in grouped:
        family_model_ids = sorted(set(frame["ModelID"]))
        if len(family_model_ids) < DRUG_RESPONSE_MIN_MODELS:
            continue
        background = gdsc_response[
            (gdsc_response["dataset"] == dataset)
            & (gdsc_response["DRUG_ID"] == drug_id)
            & (gdsc_response["family"] != family)
        ].copy()
        background_model_ids = sorted(set(background["ModelID"]))
        if len(background_model_ids) < DRUG_RESPONSE_BACKGROUND_MIN_MODELS:
            continue

        family_z_median = float(frame["Z_SCORE"].median())
        family_auc_median = float(frame["AUC"].median())
        family_ln_ic50_median = float(frame["LN_IC50"].median())
        background_z_median = float(background["Z_SCORE"].median())
        background_auc_median = float(background["AUC"].median())
        background_ln_ic50_median = float(background["LN_IC50"].median())
        z_delta = family_z_median - background_z_median
        auc_delta = family_auc_median - background_auc_median
        if (
            family_z_median > DRUG_RESPONSE_FAMILY_Z_MAX
            or family_auc_median > DRUG_RESPONSE_FAMILY_AUC_MAX
            or z_delta > DRUG_RESPONSE_Z_DELTA_MAX
            or auc_delta > DRUG_RESPONSE_AUC_DELTA_MAX
        ):
            continue

        synonyms = sorted({alias for aliases in frame["Synonyms"] for alias in (aliases if isinstance(aliases, list) else [])})
        drug_entity = build_drug_entity(
            compound_id=str(compound_id),
            label=str(drug_label),
            synonyms=synonyms,
            normalized_from=[str(drug_label), str(compound_id), f"{dataset}:{int(drug_id)}"],
        )
        disease_entity = build_disease_entity(family, family_label)
        effect_size = max(abs(z_delta), abs(family_z_median))
        summary = (
            f"{family} DepMap-aligned GDSC models are sensitive to {drug_label}; "
            f"family median Z score={round(family_z_median, 4)}, background median Z score={round(background_z_median, 4)}, "
            f"family median AUC={round(family_auc_median, 4)}"
        )
        bundle = make_evidence(
            family=family,
            family_label=family_label,
            predicate="sensitive_to",
            evidence_category="drug_sensitivity",
            feature_layer="gdsc_drug_response",
            metric="gdsc_median_z_score_delta_vs_background",
            subject_entity=disease_entity,
            object_entity=drug_entity,
            outcome_entity=None,
            comparison_id="gdsc_pan_cancer_background",
            comparison_label="target DepMap lineage panel vs selected multi-cancer GDSC background drug response",
            comparison_type="drug_sensitive_vs_background",
            group_a_label=f"{family}_depmap_models",
            group_b_label="selected_depmap_background_models",
            sample_count=len(family_model_ids) + len(background_model_ids),
            n_group_a=len(family_model_ids),
            n_group_b=len(background_model_ids),
            value=z_delta,
            effect_size=effect_size,
            summary_text=summary,
            platform_specific_fields={
                "response_dataset": dataset,
                "family_model_count": int(len(family_models.index)),
                "drug_response_model_count": int(len(family_model_ids)),
                "background_response_model_count": int(len(background_model_ids)),
                "family_median_z_score": round(family_z_median, 4),
                "background_median_z_score": round(background_z_median, 4),
                "z_score_delta": round(z_delta, 4),
                "family_median_auc": round(family_auc_median, 4),
                "background_median_auc": round(background_auc_median, 4),
                "auc_delta": round(auc_delta, 4),
                "family_median_ln_ic50": round(family_ln_ic50_median, 4),
                "background_median_ln_ic50": round(background_ln_ic50_median, 4),
                "gdsc_drug_id": int(drug_id),
                "compound_id": str(compound_id),
                "compound_targets": drug_targets,
                "compound_mechanism": drug_mechanism,
                "chembl_id": chembl_id,
                "responsive_model_ids": family_model_ids,
            },
            source_file=f"{relpath(GDSC1_RESPONSE_PATH) if dataset == 'GDSC1' else relpath(GDSC2_RESPONSE_PATH)} | {relpath(PORTAL_COMPOUNDS_PATH)} | {relpath(GDSC_COMPOUND_METADATA_PATH)}",
            source_record_key=f"{family}:{dataset}:{int(drug_id)}:drug_sensitivity",
            data_completeness=completeness_ratio(len(family_model_ids), len(family_models.index)),
            statistical_method="thresholded family-level GDSC fitted dose-response aggregation aligned to DepMap models",
            claim_scope="disease_drug",
            direction="drug_sensitivity",
            polarity="anti_tumor",
            supports_target_discovery=True,
            supports_mechanism_inference=False,
            supports_drug_repositioning=True,
            aggregation_level="drug_by_lineage_panel",
            provenance_overrides={
                "source_dataset": "DepMap-26Q1|GDSC-8.4",
                "source_release": "26Q1|8.4",
                "platform": "DepMap-aligned GDSC fitted dose response",
                "assay": "gdsc_fitted_dose_response",
                "sequencing_or_screen_id": dataset,
                "analysis_pipeline": "depmap_atomic_drug_response_v1",
                "study_design": "pharmacogenomic_cell_line_panel",
            },
        )
        append_bundle(
            evidence_nodes=evidence_nodes,
            cohort_nodes=cohort_nodes,
            entity_nodes=entity_nodes,
            local_counts=local_counts,
            bundle=bundle,
            count_key="sensitive_to",
        )


def collect_depmap_evidence() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    model = load_model_metadata().set_index("ModelID")
    gene_effect, gene_dependency = load_dependency_matrices(model.index.tolist())
    model = model.loc[gene_effect.index].copy()
    gdsc_response = load_gdsc_drug_response(model.reset_index())

    expression = load_dense_omics_matrix(EXPRESSION_PATH)
    expression = expression.loc[expression.index.intersection(gene_effect.index), expression.columns.intersection(gene_effect.columns)]
    copy_number = load_dense_omics_matrix(COPY_NUMBER_PATH)
    copy_number = copy_number.loc[copy_number.index.intersection(gene_effect.index), copy_number.columns.intersection(gene_effect.columns)]
    hotspot_mutation = load_binary_omics_matrix(HOTSPOT_MUTATION_PATH)
    damaging_mutation = load_binary_omics_matrix(DAMAGING_MUTATION_PATH)
    mutation = align_binary_union(hotspot_mutation, damaging_mutation)
    mutation = mutation.loc[mutation.index.intersection(gene_effect.index), mutation.columns.intersection(gene_effect.columns)]
    molecular_subtypes = load_molecular_subtypes().loc[lambda frame: frame.index.intersection(gene_effect.index)]
    global_signatures = load_global_signatures().loc[lambda frame: frame.index.intersection(gene_effect.index)]

    evidence_nodes: list[dict[str, Any]] = []
    cohort_nodes: list[dict[str, Any]] = []
    entity_nodes: list[dict[str, Any]] = []
    cohort_stats: dict[str, Any] = {}

    for family in TARGET_FAMILIES:
        family_models = model[model["family"] == family]
        if len(family_models.index) < MIN_FAMILY_MODELS:
            continue
        background_models = model[model["family"] != family]
        family_label = FAMILY_DESCRIPTIONS.get(family, family)
        fam_ge = gene_effect.loc[family_models.index]
        bg_ge = gene_effect.loc[background_models.index]
        fam_dp = gene_dependency.loc[family_models.index]
        bg_dp = gene_dependency.loc[background_models.index]
        local_counts: Counter[str] = Counter()

        selected_genes, anchor_genes = build_base_evidence(
            family=family,
            family_label=family_label,
            family_models=family_models,
            background_models=background_models,
            fam_ge=fam_ge,
            bg_ge=bg_ge,
            fam_dp=fam_dp,
            bg_dp=bg_dp,
            local_counts=local_counts,
            evidence_nodes=evidence_nodes,
            cohort_nodes=cohort_nodes,
            entity_nodes=entity_nodes,
        )

        expression_context_genes = add_same_gene_context_evidence(
            family=family,
            family_label=family_label,
            family_models=family_models,
            fam_ge=fam_ge,
            fam_dp=fam_dp,
            context_matrix=expression,
            genes=anchor_genes,
            local_counts=local_counts,
            evidence_nodes=evidence_nodes,
            cohort_nodes=cohort_nodes,
            entity_nodes=entity_nodes,
            context_kind="expression",
        )
        copy_number_context_genes = add_same_gene_context_evidence(
            family=family,
            family_label=family_label,
            family_models=family_models,
            fam_ge=fam_ge,
            fam_dp=fam_dp,
            context_matrix=copy_number,
            genes=anchor_genes,
            local_counts=local_counts,
            evidence_nodes=evidence_nodes,
            cohort_nodes=cohort_nodes,
            entity_nodes=entity_nodes,
            context_kind="copy_number",
        )
        mutation_context_genes = add_same_gene_context_evidence(
            family=family,
            family_label=family_label,
            family_models=family_models,
            fam_ge=fam_ge,
            fam_dp=fam_dp,
            context_matrix=mutation,
            genes=anchor_genes,
            local_counts=local_counts,
            evidence_nodes=evidence_nodes,
            cohort_nodes=cohort_nodes,
            entity_nodes=entity_nodes,
            context_kind="mutation",
        )
        context_refinement_genes = selected_genes or anchor_genes

        add_cross_gene_context_evidence(
            family=family,
            family_label=family_label,
            family_models=family_models,
            fam_ge=fam_ge,
            fam_dp=fam_dp,
            context_matrix=expression,
            subject_genes=selected_genes,
            context_genes=expression_context_genes,
            local_counts=local_counts,
            evidence_nodes=evidence_nodes,
            cohort_nodes=cohort_nodes,
            entity_nodes=entity_nodes,
            context_kind="expression",
        )
        add_cross_gene_context_evidence(
            family=family,
            family_label=family_label,
            family_models=family_models,
            fam_ge=fam_ge,
            fam_dp=fam_dp,
            context_matrix=copy_number,
            subject_genes=selected_genes,
            context_genes=copy_number_context_genes,
            local_counts=local_counts,
            evidence_nodes=evidence_nodes,
            cohort_nodes=cohort_nodes,
            entity_nodes=entity_nodes,
            context_kind="copy_number",
        )
        add_cross_gene_context_evidence(
            family=family,
            family_label=family_label,
            family_models=family_models,
            fam_ge=fam_ge,
            fam_dp=fam_dp,
            context_matrix=mutation,
            subject_genes=selected_genes,
            context_genes=mutation_context_genes,
            local_counts=local_counts,
            evidence_nodes=evidence_nodes,
            cohort_nodes=cohort_nodes,
            entity_nodes=entity_nodes,
            context_kind="mutation",
        )

        add_boolean_context_dependency(
            family=family,
            family_label=family_label,
            family_models=family_models,
            fam_ge=fam_ge,
            fam_dp=fam_dp,
            context_matrix=molecular_subtypes,
            genes=context_refinement_genes,
            local_counts=local_counts,
            evidence_nodes=evidence_nodes,
            cohort_nodes=cohort_nodes,
            entity_nodes=entity_nodes,
            predicate="subtype_associated_with_dependency_in",
            evidence_category="subtype_dependency_context",
            feature_layer="subtype_dependency_context",
            metric="chronos_gene_effect_subtype_positive_vs_negative",
            comparison_id="subtype_context",
            comparison_label="molecular subtype-positive context vs subtype-negative context",
            source_file=f"{relpath(GENE_EFFECT_PATH)} | {relpath(MOLECULAR_SUBTYPE_PATH)}",
            min_positive=SUBTYPE_CONTEXT_MIN_POSITIVE,
            min_negative=SUBTYPE_CONTEXT_MIN_NEGATIVE,
            min_median=SUBTYPE_CONTEXT_MIN_MEDIAN,
            min_delta=SUBTYPE_CONTEXT_MIN_DELTA,
            min_prev_delta=SUBTYPE_CONTEXT_MIN_PREVALENCE_DELTA,
            min_d=SUBTYPE_CONTEXT_MIN_COHEN_D,
        )
        add_signature_context_dependency(
            family=family,
            family_label=family_label,
            family_models=family_models,
            fam_ge=fam_ge,
            fam_dp=fam_dp,
            signatures=global_signatures,
            genes=context_refinement_genes,
            local_counts=local_counts,
            evidence_nodes=evidence_nodes,
            cohort_nodes=cohort_nodes,
            entity_nodes=entity_nodes,
        )
        add_metastatic_context_dependency(
            family=family,
            family_label=family_label,
            family_models=family_models,
            fam_ge=fam_ge,
            fam_dp=fam_dp,
            genes=context_refinement_genes,
            local_counts=local_counts,
            evidence_nodes=evidence_nodes,
            cohort_nodes=cohort_nodes,
            entity_nodes=entity_nodes,
        )
        add_codependency_evidence(
            family=family,
            family_label=family_label,
            family_models=family_models,
            fam_ge=fam_ge,
            fam_dp=fam_dp,
            selected_genes=context_refinement_genes,
            anchor_genes=set(),
            local_counts=local_counts,
            evidence_nodes=evidence_nodes,
            cohort_nodes=cohort_nodes,
            entity_nodes=entity_nodes,
        )
        add_drug_response_evidence(
            family=family,
            family_label=family_label,
            family_models=family_models,
            gdsc_response=gdsc_response,
            local_counts=local_counts,
            evidence_nodes=evidence_nodes,
            cohort_nodes=cohort_nodes,
            entity_nodes=entity_nodes,
        )

        family_stats = init_family_stats(family, family_label, family_models, len(model.index))
        family_stats["selected_evidence_counts"] = dict(sorted(local_counts.items()))
        family_stats["context_anchor_gene_count"] = len(anchor_genes)
        family_stats["selected_dependency_gene_count"] = len(selected_genes)
        cohort_stats[family] = family_stats

    uniq_cohorts = sorted({node["id"]: node for node in cohort_nodes}.values(), key=lambda item: item["id"])
    uniq_entities = sorted({node["id"]: node for node in entity_nodes}.values(), key=lambda item: item["id"])
    return evidence_nodes, uniq_cohorts, uniq_entities, cohort_stats


def build_edges(evidence_nodes: list[dict[str, Any]], cohort_nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cohort_ids = {node["id"] for node in cohort_nodes}
    edges: list[dict[str, Any]] = []
    for evidence in evidence_nodes:
        if evidence["cohort_id"] in cohort_ids:
            edges.append(
                {
                    "source": evidence["id"],
                    "target": evidence["cohort_id"],
                    "relation": "OBSERVED_IN",
                    "relation_family": "semantic_anchor",
                    "entity_type": "Cohort",
                    "linking_score": None,
                    "similarity_score": None,
                    "rationale": "DepMap atomic evidence observed in lineage-specific cell-line panel context.",
                    "support_strength": None,
                    "evidence_count": 1,
                    "created_by_rule": "depmap_atomic:observed_in",
                }
            )
        for entity_ref, rule in [
            (evidence["subject"], "depmap_atomic:link_subject_entity"),
            (evidence.get("object"), "depmap_atomic:link_object_entity"),
            (evidence.get("outcome"), "depmap_atomic:link_outcome_entity"),
        ]:
            if not entity_ref or not entity_ref.get("id"):
                continue
            edges.append(
                {
                    "source": evidence["id"],
                    "target": entity_ref["id"],
                    "relation": "LINKED_TO",
                    "relation_family": "semantic_anchor",
                    "entity_type": entity_ref["entity_type"],
                    "linking_score": 1.0,
                    "similarity_score": None,
                    "rationale": "Evidence-to-entity semantic link.",
                    "support_strength": None,
                    "evidence_count": 1,
                    "created_by_rule": rule,
                }
            )
    return edges


def build_graph(evidence_nodes: list[dict[str, Any]], cohort_nodes: list[dict[str, Any]], entity_nodes: list[dict[str, Any]], cohort_stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "2.0.0",
        "graph_metadata": {
            "graph_id": "depmap_atomic_compact_latest",
            "graph_name": "Compact DepMap atomic evidence aligned to compact unified multisource KG schema",
            "created_by": "scripts/extract_depmap_dependency_evidence.py",
            "created_at": now_iso(),
            "source_families": ["depmap"],
            "disease_scope": sorted({node["disease"] for node in evidence_nodes}),
            "build_parameters": {
                "target_families": TARGET_FAMILIES,
                "thresholds": {
                    "dependency_positive_probability": DEPENDENCY_POSITIVE_PROBABILITY,
                    "min_family_models": MIN_FAMILY_MODELS,
                    "recurrent_effect_max": RECURRENT_EFFECT_MAX,
                    "recurrent_mean_prob_min": RECURRENT_MEAN_PROB_MIN,
                    "recurrent_prevalence_min": RECURRENT_PREVALENCE_MIN,
                    "recurrent_background_prevalence_max": RECURRENT_BACKGROUND_PREVALENCE_MAX,
                    "selective_effect_max": SELECTIVE_EFFECT_MAX,
                    "selective_delta_max": SELECTIVE_DELTA_MAX,
                    "selective_mean_prob_min": SELECTIVE_MEAN_PROB_MIN,
                    "selective_prevalence_delta_min": SELECTIVE_PREVALENCE_DELTA_MIN,
                    "selective_cohen_d_min": SELECTIVE_COHEN_D_MIN,
                    "selective_background_prevalence_max": SELECTIVE_BACKGROUND_PREVALENCE_MAX,
                    "context_anchor_effect_max": CONTEXT_ANCHOR_EFFECT_MAX,
                    "context_anchor_mean_prob_min": CONTEXT_ANCHOR_MEAN_PROB_MIN,
                    "context_anchor_prevalence_min": CONTEXT_ANCHOR_PREVALENCE_MIN,
                    "context_anchor_background_prevalence_max": CONTEXT_ANCHOR_BACKGROUND_PREVALENCE_MAX,
                    "expression_context_min_group": EXPR_CONTEXT_MIN_GROUP,
                    "cross_expression_context_min_group": CROSS_EXPR_CONTEXT_MIN_GROUP,
                    "copy_context_min_group": COPY_CONTEXT_MIN_GROUP,
                    "cross_copy_context_min_group": CROSS_COPY_CONTEXT_MIN_GROUP,
                    "mutation_context_min_positive": MUTATION_CONTEXT_MIN_POSITIVE,
                    "cross_mutation_context_min_positive": CROSS_MUTATION_CONTEXT_MIN_POSITIVE,
                    "drug_response_min_models": DRUG_RESPONSE_MIN_MODELS,
                    "drug_response_background_min_models": DRUG_RESPONSE_BACKGROUND_MIN_MODELS,
                    "drug_response_family_z_max": DRUG_RESPONSE_FAMILY_Z_MAX,
                    "drug_response_family_auc_max": DRUG_RESPONSE_FAMILY_AUC_MAX,
                    "drug_response_z_delta_max": DRUG_RESPONSE_Z_DELTA_MAX,
                    "drug_response_auc_delta_max": DRUG_RESPONSE_AUC_DELTA_MAX,
                    "subtype_context_min_positive": SUBTYPE_CONTEXT_MIN_POSITIVE,
                    "phenotype_context_min_positive": PHENOTYPE_CONTEXT_MIN_POSITIVE,
                    "metastatic_context_min_positive": METASTATIC_CONTEXT_MIN_POSITIVE,
                    "codependency_min_shared_models": CODEPENDENCY_MIN_SHARED_MODELS,
                    "codependency_min_correlation": CODEPENDENCY_MIN_CORRELATION,
                },
                "rationale": {
                    "family_dependency": "Retain only strong recurrent and lineage-selective DepMap Chronos dependencies so common essentials do not dominate the final graph.",
                    "expression_context": "Retain high-expression dependency context only when the context-positive group remains strongly dependent, expression split is substantial, prevalence improves materially, and Cohen's d is large enough to support expression addiction-like evidence.",
                    "copy_number_context": "Retain copy-number gain dependency context only for stronger gain-addiction patterns against low-copy or baseline models within the same lineage panel.",
                    "mutation_context": "Retain mutation-associated dependency context only for sufficiently frequent altered subsets with materially stronger dependency than the unaltered background.",
                    "drug_response": "Retain GDSC drug sensitivity evidence only when a DepMap-aligned lineage panel shows low median drug-response Z score, low AUC, and materially stronger sensitivity than the multi-cancer background with enough screened models.",
                    "cross_gene_context": "Expand cross-gene expression, copy-number, and mutation contexts only from already validated same-gene context anchors to raise DepMap coverage without opening the graph to unconstrained context-gene combinatorics.",
                    "subtype_and_phenotype_context": "Retain molecular subtype, phenotype-signature, and metastatic-state contexts only when they create strong within-family dependency contrasts rather than weak annotations.",
                    "codependency": "Retain gene-gene codependency only when selected lineage-relevant dependency genes show strong positive correlation across family models.",
                    "selection": "The final DepMap graph is threshold-defined rather than top-k-truncated so graph membership follows explicit functional and context-dependent entry criteria.",
                },
                "cohort_stats": cohort_stats,
            },
            "notes": "Final DepMap atomic reasoning graph with recurrent dependency, lineage-selective dependency, same-gene and cross-gene context-conditioned dependency, codependency, and GDSC-aligned drug sensitivity evidence under a shared compact evidence schema.",
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
    by_feature_layer = {
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
    }
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
            "by_feature_layer": by_feature_layer,
        },
        "pubmed_alignment": alignment_summary,
    }


def build_audit(graph: dict[str, Any], validation_errors: list[str], alignment_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    node_ids = {node["id"] for node in graph["nodes"]}
    orphan_edges = [edge for edge in graph["edges"] if edge["source"] not in node_ids or edge["target"] not in node_ids]
    evidence_nodes = [node for node in graph["nodes"] if node["node_type"] == "evidence"]
    return {
        "graph_id": graph["graph_metadata"]["graph_id"],
        "timestamp": graph["graph_metadata"]["created_at"],
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
                "source_file": round(sum(1 for node in evidence_nodes if node["provenance"]["source_file"]) / len(evidence_nodes), 4) if evidence_nodes else 0.0,
                "sample_size": round(sum(1 for node in evidence_nodes if node["statistics"]["sample_size"] is not None) / len(evidence_nodes), 4) if evidence_nodes else 0.0,
                "effect_size": round(sum(1 for node in evidence_nodes if node["statistics"]["effect_size"] is not None) / len(evidence_nodes), 4) if evidence_nodes else 0.0,
                "comparison_metadata": round(sum(1 for node in evidence_nodes if node["comparison"]["sample_grouping"]) / len(evidence_nodes), 4) if evidence_nodes else 0.0,
            },
            "pubmed_alignment_candidate_count": len(alignment_candidates),
        },
    }


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    print("[main] collecting DepMap atomic evidence", flush=True)
    evidence_nodes, cohort_nodes, entity_nodes, cohort_stats = collect_depmap_evidence()
    print(
        f"[main] evidence collected: evidence={len(evidence_nodes)} cohorts={len(cohort_nodes)} entities={len(entity_nodes)}",
        flush=True,
    )
    graph = build_graph(evidence_nodes, cohort_nodes, entity_nodes, cohort_stats)
    validation_errors = fast_validate(graph)
    print(f"[main] validation done: errors={len(validation_errors)}", flush=True)
    alignment_candidates, alignment_summary = build_pubmed_alignment_candidates(evidence_nodes)
    print(f"[main] pubmed alignment candidates={len(alignment_candidates)}", flush=True)
    stats = build_stats(graph, cohort_stats, alignment_summary)
    audit = build_audit(graph, validation_errors, alignment_candidates)

    (OUTPUT_DIR / "depmap_atomic_compact_latest.json").write_text(json.dumps(graph, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "depmap_atomic_compact_latest_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "depmap_atomic_compact_latest_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "depmap_pubmed_alignment_candidates_compact_latest.json").write_text(json.dumps(alignment_candidates, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("[main] finished all outputs", flush=True)


if __name__ == "__main__":
    main()

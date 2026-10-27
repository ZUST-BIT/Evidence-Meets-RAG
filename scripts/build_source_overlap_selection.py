from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
KG_DIR = ROOT / "kg"
OUTPUT_DIR = KG_DIR / "coverage"

from knowledge_build_paths import LINKEDOMICS_ROOT, DEPMAP_ROOT, PUBMED_ROOT, source_relpath
DEPMAP_MODEL_PATH = DEPMAP_ROOT / "model-condition-mapping" / "Model.csv"
DEPMAP_MODELCONDITION_PATH = DEPMAP_ROOT / "model-condition-mapping" / "ModelCondition.csv"
PUBMED_KG_ROOT = PUBMED_ROOT


TCGA_CODE_TO_FAMILY: dict[str, dict[str, str]] = {
    "ACC": {"family": "ACC", "label": "Adrenocortical carcinoma"},
    "BLCA": {"family": "BLCA", "label": "Bladder urothelial carcinoma"},
    "BRCA": {"family": "BRCA", "label": "Breast carcinoma"},
    "CESC": {"family": "CERVICAL", "label": "Cervical carcinoma"},
    "CHOL": {"family": "BILIARY_TRACT", "label": "Cholangiocarcinoma"},
    "COADREAD": {"family": "CRC", "label": "Colorectal adenocarcinoma"},
    "DLBC": {"family": "LYMPHOMA", "label": "Diffuse large B-cell lymphoma"},
    "ESCA": {"family": "GASTROESOPH", "label": "Esophageal carcinoma"},
    "GBM": {"family": "GLIOMA", "label": "Glioblastoma"},
    "GBMLGG": {"family": "GLIOMA", "label": "Glioma"},
    "HNSC": {"family": "HNSC", "label": "Head and neck squamous cell carcinoma"},
    "KICH": {"family": "RCC", "label": "Kidney chromophobe / renal carcinoma"},
    "KIPAN": {"family": "RCC", "label": "Pan-kidney renal carcinoma"},
    "KIRC": {"family": "RCC", "label": "Kidney renal clear cell carcinoma"},
    "KIRP": {"family": "RCC", "label": "Kidney renal papillary cell carcinoma"},
    "LAML": {"family": "AML", "label": "Acute myeloid leukemia"},
    "LGG": {"family": "GLIOMA", "label": "Lower grade glioma"},
    "LIHC": {"family": "HCC", "label": "Hepatocellular carcinoma"},
    "LUAD": {"family": "NSCLC", "label": "Lung adenocarcinoma"},
    "LUSC": {"family": "NSCLC", "label": "Lung squamous cell carcinoma"},
    "MESO": {"family": "MESO", "label": "Mesothelioma"},
    "OV": {"family": "OVARIAN", "label": "Ovarian serous carcinoma"},
    "PAAD": {"family": "PAAD", "label": "Pancreatic adenocarcinoma"},
    "PCPG": {"family": "PCPG", "label": "Pheochromocytoma and paraganglioma"},
    "PRAD": {"family": "PRAD", "label": "Prostate adenocarcinoma"},
    "SARC": {"family": "SARCOMA", "label": "Sarcoma"},
    "SKCM": {"family": "MELANOMA", "label": "Cutaneous melanoma"},
    "STAD": {"family": "GASTROESOPH", "label": "Stomach adenocarcinoma"},
    "STES": {"family": "GASTROESOPH", "label": "Stomach and esophageal carcinoma"},
    "TGCT": {"family": "TGCT", "label": "Testicular germ cell tumor"},
    "THCA": {"family": "THYROID", "label": "Thyroid carcinoma"},
    "THYM": {"family": "THYMOMA", "label": "Thymoma"},
    "UCEC": {"family": "UCEC", "label": "Endometrial carcinoma"},
    "UCS": {"family": "UCEC", "label": "Uterine carcinosarcoma"},
    "UVM": {"family": "MELANOMA", "label": "Uveal melanoma"},
}

FAMILY_DESCRIPTIONS: dict[str, str] = {
    "HCC": "Hepatocellular carcinoma",
    "CRC": "Colorectal cancer",
    "BRCA": "Breast carcinoma",
    "NSCLC": "Non-small cell lung cancer",
    "GLIOMA": "Glioma",
    "OVARIAN": "Ovarian carcinoma",
    "RCC": "Renal cell carcinoma",
    "AML": "Acute myeloid leukemia",
    "PAAD": "Pancreatic adenocarcinoma",
    "HNSC": "Head and neck squamous cell carcinoma",
    "GASTROESOPH": "Esophagogastric carcinoma",
    "PRAD": "Prostate adenocarcinoma",
    "MELANOMA": "Melanoma",
    "MESO": "Mesothelioma",
    "BLCA": "Bladder carcinoma",
    "UCEC": "Endometrial/uterine carcinoma",
    "BILIARY_TRACT": "Biliary tract carcinoma",
    "SARCOMA": "Sarcoma",
    "THYROID": "Thyroid carcinoma",
    "LYMPHOMA": "Lymphoma",
    "CERVICAL": "Cervical carcinoma",
    "ACC": "Adrenocortical carcinoma",
    "PCPG": "Pheochromocytoma/paraganglioma",
    "TGCT": "Testicular germ cell tumor",
    "THYMOMA": "Thymoma",
    "OTHER": "Other or unmapped cancers",
}

EVAL_ALIGNED_TIER2_ORDER = [
    "BRCA",
    "NSCLC",
    "GLIOMA",
    "OVARIAN",
    "RCC",
    "AML",
    "PAAD",
    "HNSC",
    "GASTROESOPH",
    "PRAD",
    "MELANOMA",
    "MESO",
    "BLCA",
    "UCEC",
    "BILIARY_TRACT",
    "SARCOMA",
    "THYROID",
    "LYMPHOMA",
    "CERVICAL",
    "ACC",
]

PUBMED_DIR_TO_FAMILY = {
    "hcc": "HCC",
    "crc": "CRC",
}

DEPMAP_ONCOTREE_TO_FAMILY = {
    "HCC": "HCC",
    "LIHC": "HCC",
    "COAD": "CRC",
    "READ": "CRC",
    "BRCA": "BRCA",
    "IDC": "BRCA",
    "ILC": "BRCA",
    "LUAD": "NSCLC",
    "LUSC": "NSCLC",
    "NSCLC": "NSCLC",
    "LCLC": "NSCLC",
    "GB": "GLIOMA",
    "ASTR": "GLIOMA",
    "ODG": "GLIOMA",
    "HGSOC": "OVARIAN",
    "SOC": "OVARIAN",
    "OVT": "OVARIAN",
    "CCRCC": "RCC",
    "RCC": "RCC",
    "KICH": "RCC",
    "KIRC": "RCC",
    "KIRP": "RCC",
    "AML": "AML",
    "AMLNOS": "AML",
    "AMLMRC": "AML",
    "PAAD": "PAAD",
    "PDAC": "PAAD",
    "HNSC": "HNSC",
    "OCSC": "HNSC",
    "LXSC": "HNSC",
    "STAD": "GASTROESOPH",
    "ESCA": "GASTROESOPH",
    "ESCC": "GASTROESOPH",
    "PRAD": "PRAD",
    "MEL": "MELANOMA",
    "SKCM": "MELANOMA",
    "UVM": "MELANOMA",
    "PLEMESO": "MESO",
    "PLBMESO": "MESO",
    "BLCA": "BLCA",
    "UCEC": "UCEC",
    "UCS": "UCEC",
    "IHCH": "BILIARY_TRACT",
    "CHOL": "BILIARY_TRACT",
    "SARC": "SARCOMA",
    "ES": "SARCOMA",
    "RMS": "SARCOMA",
    "OS": "SARCOMA",
    "THCA": "THYROID",
    "DLBCLNOS": "LYMPHOMA",
    "BL": "LYMPHOMA",
    "NHL": "LYMPHOMA",
    "CESC": "CERVICAL",
    "ACC": "ACC",
    "PCPG": "PCPG",
    "TGCT": "TGCT",
    "THYM": "THYMOMA",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "family",
        "description",
        "priority_tier",
        "in_tcga",
        "tcga_cohort_count",
        "tcga_cohorts",
        "in_depmap",
        "depmap_model_count",
        "depmap_primary_diseases",
        "depmap_oncotree_codes",
        "in_pubmed",
        "pubmed_datasets",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_normalization_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "family",
        "description",
        "tcga_aliases",
        "depmap_oncotree_aliases",
        "depmap_primary_disease_aliases",
        "depmap_subtype_aliases",
        "pubmed_aliases",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def load_tcga_summary() -> dict[str, dict[str, Any]]:
    family_to_cohorts: dict[str, list[str]] = defaultdict(list)
    for cohort_dir in sorted(p for p in LINKEDOMICS_ROOT.iterdir() if p.is_dir() and p.name.endswith("-TCGA")):
        code = cohort_dir.name.removesuffix("-TCGA")
        meta = TCGA_CODE_TO_FAMILY.get(code, {"family": "OTHER", "label": code})
        family_to_cohorts[meta["family"]].append(cohort_dir.name)

    summary: dict[str, dict[str, Any]] = {}
    for family, cohorts in sorted(family_to_cohorts.items()):
        summary[family] = {
            "family": family,
            "description": FAMILY_DESCRIPTIONS.get(family, family),
            "cohort_count": len(cohorts),
            "cohorts": sorted(cohorts),
        }
    return summary


def depmap_family_from_row(row: dict[str, str]) -> str:
    oncotree_code = (row.get("OncotreeCode") or "").strip().upper()
    if oncotree_code in DEPMAP_ONCOTREE_TO_FAMILY:
        return DEPMAP_ONCOTREE_TO_FAMILY[oncotree_code]

    primary = (row.get("OncotreePrimaryDisease") or "").strip().lower()
    subtype = (row.get("OncotreeSubtype") or "").strip().lower()
    lineage = (row.get("OncotreeLineage") or "").strip().lower()
    tissue = (row.get("TissueOrigin") or "").strip().lower()
    hay = " ".join([primary, subtype, lineage, tissue])

    if primary == "hepatocellular carcinoma" or subtype == "hepatocellular carcinoma":
        return "HCC"
    if primary == "colorectal adenocarcinoma" or subtype in {
        "colon adenocarcinoma",
        "rectal adenocarcinoma",
        "colorectal adenocarcinoma",
    }:
        return "CRC"
    if "breast" in primary or "breast" in subtype:
        return "BRCA"
    if primary == "non-small cell lung cancer" or subtype in {
        "lung adenocarcinoma",
        "lung squamous cell carcinoma",
    }:
        return "NSCLC"
    if primary == "adult-type diffuse glioma" or "glioma" in subtype or "glioblastoma" in subtype or "astrocytoma" in subtype:
        return "GLIOMA"
    if primary == "ovarian epithelial tumor" or "ovarian" in subtype or "fallopian" in subtype:
        return "OVARIAN"
    if primary == "renal cell carcinoma" or "renal" in subtype or "kidney" in subtype:
        return "RCC"
    if primary == "acute myeloid leukemia" or subtype == "acute myeloid leukemia":
        return "AML"
    if primary == "pancreatic adenocarcinoma" or subtype == "pancreatic adenocarcinoma":
        return "PAAD"
    if primary == "head and neck squamous cell carcinoma" or "squamous cell carcinoma" in subtype and any(
        token in subtype for token in ["oral cavity", "larynx", "head and neck"]
    ):
        return "HNSC"
    if primary == "esophagogastric adenocarcinoma" or any(
        token in subtype for token in ["esophageal", "stomach adenocarcinoma", "gastric"]
    ):
        return "GASTROESOPH"
    if primary == "prostate adenocarcinoma" or subtype == "prostate adenocarcinoma":
        return "PRAD"
    if primary == "melanoma" or "melanoma" in subtype:
        return "MELANOMA"
    if primary == "pleural mesothelioma" or "mesothelioma" in subtype:
        return "MESO"
    if primary == "bladder urothelial carcinoma" or "urothelial" in subtype:
        return "BLCA"
    if primary == "endometrial carcinoma" or "endometrial" in subtype or "uterine" in subtype:
        return "UCEC"
    if primary == "intraductal papillary neoplasm of the bile duct" or any(
        token in subtype for token in ["cholangio", "bile duct", "intrahepatic cholangiocarcinoma"]
    ):
        return "BILIARY_TRACT"
    if "sarcoma" in primary or "sarcoma" in subtype or primary in {"ewing sarcoma", "osteosarcoma"}:
        return "SARCOMA"
    if "thyroid" in primary or "thyroid" in subtype:
        return "THYROID"
    if any(token in primary for token in ["lymphoma", "b-cell", "t-cell", "hodgkin"]) or any(
        token in subtype for token in ["lymphoma", "b-cell", "t-cell"]
    ):
        return "LYMPHOMA"
    if "cervical" in primary or "cervical" in subtype:
        return "CERVICAL"
    if "adrenocortical" in primary or "adrenocortical" in subtype:
        return "ACC"
    if "pheochromocytoma" in primary or "paraganglioma" in primary or "pheochromocytoma" in subtype or "paraganglioma" in subtype:
        return "PCPG"
    if "germ cell" in primary or "testicular" in primary or "germ cell" in subtype or "testicular" in subtype:
        return "TGCT"
    if "thymoma" in primary or "thymoma" in subtype:
        return "THYMOMA"
    return "OTHER"


def load_depmap_summary() -> dict[str, Any]:
    family_counters: dict[str, Counter[str]] = defaultdict(Counter)
    family_subtypes: dict[str, Counter[str]] = defaultdict(Counter)
    family_codes: dict[str, Counter[str]] = defaultdict(Counter)
    family_model_ids: dict[str, list[str]] = defaultdict(list)

    with DEPMAP_MODEL_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            family = depmap_family_from_row(row)
            family_model_ids[family].append(row["ModelID"])
            if row.get("OncotreePrimaryDisease"):
                family_counters[family][row["OncotreePrimaryDisease"]] += 1
            if row.get("OncotreeSubtype"):
                family_subtypes[family][row["OncotreeSubtype"]] += 1
            if row.get("OncotreeCode"):
                family_codes[family][row["OncotreeCode"]] += 1

    summary: dict[str, Any] = {}
    for family, model_ids in sorted(family_model_ids.items()):
        summary[family] = {
            "family": family,
            "description": FAMILY_DESCRIPTIONS.get(family, family),
            "model_count": len(model_ids),
            "model_ids_preview": sorted(model_ids)[:10],
            "primary_diseases": family_counters[family].most_common(10),
            "subtypes": family_subtypes[family].most_common(10),
            "oncotree_codes": family_codes[family].most_common(10),
        }
    return summary


def load_modelcondition_metadata() -> dict[str, Any]:
    with DEPMAP_MODELCONDITION_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
    return {
        "path": str(DEPMAP_MODELCONDITION_PATH),
        "columns": header,
        "contains_disease_annotations": False,
        "note": "ModelCondition.csv stores growth and treatment context; disease labels are not present there and should be joined from Model.csv via ModelID.",
    }


def load_pubmed_summary() -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for disease_dir in sorted(p for p in PUBMED_KG_ROOT.iterdir() if p.is_dir()):
        family = PUBMED_DIR_TO_FAMILY.get(disease_dir.name.lower(), disease_dir.name.upper())
        summary[family] = {
            "family": family,
            "description": FAMILY_DESCRIPTIONS.get(family, family),
            "datasets": [
                source_relpath(disease_dir / "evidence_nodes.json"),
                source_relpath(disease_dir / "evidence_graph.json"),
            ],
        }
    return summary


def build_overlap_payload() -> dict[str, Any]:
    tcga_summary = load_tcga_summary()
    depmap_summary = load_depmap_summary()
    pubmed_summary = load_pubmed_summary()
    modelcondition_meta = load_modelcondition_metadata()

    all_families = sorted(set(tcga_summary) | set(depmap_summary) | set(pubmed_summary))
    family_rows: list[dict[str, Any]] = []
    overlap_details: dict[str, Any] = {}
    normalization_aliases: dict[str, dict[str, Any]] = {}

    for family in all_families:
        tcga = tcga_summary.get(family)
        depmap = depmap_summary.get(family)
        pubmed = pubmed_summary.get(family)
        in_tcga = tcga is not None
        in_depmap = depmap is not None
        in_pubmed = pubmed is not None

        if family in {"HCC", "CRC"} and in_tcga and in_depmap and in_pubmed:
            priority_tier = "tier_1_triple_overlap"
        elif in_tcga and in_depmap:
            priority_tier = "tier_2_tcga_depmap_overlap"
        else:
            priority_tier = "source_specific_only"

        row = {
            "family": family,
            "description": FAMILY_DESCRIPTIONS.get(family, family),
            "priority_tier": priority_tier,
            "in_tcga": in_tcga,
            "tcga_cohort_count": tcga["cohort_count"] if tcga else 0,
            "tcga_cohorts": ",".join(tcga["cohorts"]) if tcga else "",
            "in_depmap": in_depmap,
            "depmap_model_count": depmap["model_count"] if depmap else 0,
            "depmap_primary_diseases": ",".join(name for name, _ in (depmap["primary_diseases"][:5] if depmap else [])),
            "depmap_oncotree_codes": ",".join(code for code, _ in (depmap["oncotree_codes"][:5] if depmap else [])),
            "in_pubmed": in_pubmed,
            "pubmed_datasets": ",".join(pubmed["datasets"]) if pubmed else "",
        }
        family_rows.append(row)
        normalization_aliases[family] = {
            "family": family,
            "description": FAMILY_DESCRIPTIONS.get(family, family),
            "tcga_aliases": sorted(
                {
                    cohort.removesuffix("-TCGA")
                    for cohort in (tcga.get("cohorts", []) if tcga else [])
                }
            ),
            "depmap_oncotree_aliases": [code for code, _ in (depmap.get("oncotree_codes", []) if depmap else [])],
            "depmap_primary_disease_aliases": [name for name, _ in (depmap.get("primary_diseases", []) if depmap else [])],
            "depmap_subtype_aliases": [name for name, _ in (depmap.get("subtypes", []) if depmap else [])],
            "pubmed_aliases": sorted({Path(dataset).parts[-2] for dataset in (pubmed.get("datasets", []) if pubmed else [])}),
        }
        overlap_details[family] = {
            "description": FAMILY_DESCRIPTIONS.get(family, family),
            "priority_tier": priority_tier,
            "tcga": tcga,
            "depmap": depmap,
            "pubmed": pubmed,
            "semantic_normalization": normalization_aliases[family],
        }

    tcga_depmap_shared = [family for family in all_families if family in tcga_summary and family in depmap_summary]
    triple_shared = [family for family in all_families if family in tcga_summary and family in depmap_summary and family in pubmed_summary]

    tier_2_ranked = sorted(
        [family for family in tcga_depmap_shared if family not in {"HCC", "CRC"}],
        key=lambda family: (
            0 if family in EVAL_ALIGNED_TIER2_ORDER else 1,
            EVAL_ALIGNED_TIER2_ORDER.index(family) if family in EVAL_ALIGNED_TIER2_ORDER else 999,
            -(depmap_summary.get(family, {}).get("model_count", 0)),
            family,
        ),
    )

    return {
        "schema_version": "1.0.0",
        "selection_goal": "Select disease overlaps for unified evidence KG versions spanning TCGA, DepMap, and PubMed.",
        "selection_logic": {
            "tier_1": "Diseases present in TCGA, DepMap, and current PubMed evidence KG. Because PubMed currently covers only HCC and CRC, these are the immediate triple-overlap priorities.",
            "tier_2": "Additional diseases present in both TCGA and DepMap, used to expand the graph into a multi-cancer version after the HCC/CRC core.",
        },
        "semantic_normalization_policy": {
            "purpose": "Unify disease names when TCGA cohort codes, DepMap Oncotree labels, and PubMed directory names refer to the same cancer semantically.",
            "canonical_family_unit": "broad disease family used for graph coverage selection",
            "normalization_priority": [
                "DepMap OncotreeCode exact mapping",
                "TCGA cohort-code to family mapping",
                "DepMap disease/subtype/lineage semantic fallback",
                "PubMed disease-directory mapping",
            ],
            "examples": [
                {
                    "family": "HCC",
                    "tcga_aliases": ["LIHC"],
                    "depmap_aliases": ["HCC", "Hepatocellular Carcinoma"],
                    "pubmed_aliases": ["hcc"],
                },
                {
                    "family": "CRC",
                    "tcga_aliases": ["COADREAD"],
                    "depmap_aliases": ["COAD", "READ", "Colorectal Adenocarcinoma", "Colon Adenocarcinoma"],
                    "pubmed_aliases": ["crc"],
                },
            ],
        },
        "data_sources": {
            "tcga": {
                "root": str(LINKEDOMICS_ROOT),
                "disease_label_origin": "Directory names ending with -TCGA, mapped by TCGA cohort code.",
            },
            "depmap": {
                "root": str(DEPMAP_MODEL_PATH.parent.parent),
                "disease_label_origin": {
                    "primary_file": str(DEPMAP_MODEL_PATH),
                    "fields": [
                        "OncotreeLineage",
                        "OncotreePrimaryDisease",
                        "OncotreeSubtype",
                        "OncotreeCode",
                        "TissueOrigin",
                    ],
                },
                "context_file": modelcondition_meta,
            },
            "pubmed": {
                "root": str(PUBMED_KG_ROOT),
                "disease_label_origin": "Disease-specific directories under kg/pubmed-evidence-kg/",
            },
        },
        "selected_overlap_versions": {
            "tcga_depmap_pubmed_overlap_v1": {
                "description": "Triple-overlap graph version for direct multi-source fusion.",
                "selected_families": ["HCC", "CRC"],
            },
            "tcga_depmap_overlap_v1": {
                "description": "Expanded multi-cancer graph version using diseases shared by TCGA and DepMap.",
                "selected_families": tcga_depmap_shared,
            },
            "tcga_depmap_overlap_priority_v1": {
                "description": "Tiered ordering for building the expanded graph after HCC/CRC.",
                "tier_1_triple_overlap": ["HCC", "CRC"],
                "tier_2_tcga_depmap_overlap": tier_2_ranked,
            },
        },
        "summary_counts": {
            "tcga_family_count": len(tcga_summary),
            "depmap_family_count": len(depmap_summary),
            "pubmed_family_count": len(pubmed_summary),
            "tcga_depmap_shared_family_count": len(tcga_depmap_shared),
            "triple_shared_family_count": len(triple_shared),
        },
        "family_normalization_aliases": normalization_aliases,
        "overlap_details": overlap_details,
    }


def main() -> None:
    payload = build_overlap_payload()
    write_json(OUTPUT_DIR / "source_overlap_selection_v1.json", payload)

    rows = []
    for family, detail in payload["overlap_details"].items():
        tcga = detail.get("tcga") or {}
        depmap = detail.get("depmap") or {}
        pubmed = detail.get("pubmed") or {}
        rows.append(
            {
                "family": family,
                "description": detail["description"],
                "priority_tier": detail["priority_tier"],
                "in_tcga": bool(tcga),
                "tcga_cohort_count": tcga.get("cohort_count", 0),
                "tcga_cohorts": ",".join(tcga.get("cohorts", [])),
                "in_depmap": bool(depmap),
                "depmap_model_count": depmap.get("model_count", 0),
                "depmap_primary_diseases": ",".join(name for name, _ in depmap.get("primary_diseases", [])[:5]),
                "depmap_oncotree_codes": ",".join(code for code, _ in depmap.get("oncotree_codes", [])[:5]),
                "in_pubmed": bool(pubmed),
                "pubmed_datasets": ",".join(pubmed.get("datasets", [])),
            }
        )

    write_tsv(OUTPUT_DIR / "source_overlap_selection_v1.tsv", rows)

    normalization_rows = []
    for family, aliases in payload["family_normalization_aliases"].items():
        normalization_rows.append(
            {
                "family": family,
                "description": aliases["description"],
                "tcga_aliases": ",".join(aliases["tcga_aliases"]),
                "depmap_oncotree_aliases": ",".join(aliases["depmap_oncotree_aliases"]),
                "depmap_primary_disease_aliases": ",".join(aliases["depmap_primary_disease_aliases"]),
                "depmap_subtype_aliases": ",".join(aliases["depmap_subtype_aliases"]),
                "pubmed_aliases": ",".join(aliases["pubmed_aliases"]),
            }
        )
    write_normalization_tsv(OUTPUT_DIR / "source_overlap_family_normalization_v1.tsv", normalization_rows)


if __name__ == "__main__":
    main()

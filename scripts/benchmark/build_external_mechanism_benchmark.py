from __future__ import annotations

import gzip
import io
import json
import re
import subprocess
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from build_benchmark_dataset import attach_gold_standard, build_gold_standard, trim_text


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "dataset" / "raw"
OUTPUT_DIR = ROOT / "dataset" / "evaluated"

INT0GEN_DRIVERS_URL = "https://www.intogen.org/download?file=IntOGen-Drivers-20200201.zip"
INT0GEN_COHORTS_URL = "https://www.intogen.org/download?file=IntOGen-Cohorts-20200201.zip"
OPENTARGETS_BASE = "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/25.03/output"
SIGNOR_LATEST_URL = "https://signor.uniroma2.it/releases/getLatestRelease.php"
PATHWAYCOMMONS_GMT_URL = "https://download.baderlab.org/PathwayCommons/PC2/v14/pc-hgnc.gmt.gz"
USER_AGENT = "EvidenceMeetsRAG/1.0"

EXTERNAL_BENCHMARK_PATH = OUTPUT_DIR / "external_mechanism_benchmark.jsonl"
EXTERNAL_MACHINE_PATH = OUTPUT_DIR / "external_mechanism_benchmark_machine_evaluable.jsonl"
EXTERNAL_PILOT_PATH = OUTPUT_DIR / "external_mechanism_benchmark_pilot_subset_machine_evaluable.jsonl"
EXTERNAL_STATS_PATH = OUTPUT_DIR / "external_mechanism_benchmark_stats.json"
EVIDENCE_GRAPH_DOCS_PATH = ROOT / "dataset" / "rag" / "knowledge_sources" / "multi_cancer" / "evidence_graph_docs.jsonl"

SUPPORTED_FAMILIES = {
    "aml",
    "brca",
    "crc",
    "gastroesoph",
    "glioma",
    "hcc",
    "hnsc",
    "nsclc",
    "ovarian",
    "paad",
    "rcc",
}

FAMILY_LABELS = {
    "aml": "AML",
    "brca": "BRCA",
    "crc": "CRC",
    "gastroesoph": "GASTROESOPH",
    "glioma": "GLIOMA",
    "hcc": "HCC",
    "hnsc": "HNSC",
    "nsclc": "NSCLC",
    "ovarian": "OVARIAN",
    "paad": "PAAD",
    "rcc": "RCC",
}

FAMILY_PATTERNS = [
    ("hcc", ["hepatocellular carcinoma", "liver hepatocellular carcinoma", "hcc", "lihc", "liver cancer"]),
    (
        "crc",
        [
            "colorectal cancer",
            "colorectal carcinoma",
            "colorectal adenocarcinoma",
            "colon cancer",
            "rectal cancer",
            "colon adenocarcinoma",
            "rectal adenocarcinoma",
            "coad",
            "read",
            "coadread",
            "colorectal tumour",
            "colorectal tumor",
            "metastatic colorectal",
            "lynch syndrome",
        ],
    ),
    ("nsclc", ["non small cell lung cancer", "non-small cell lung cancer", "nsclc", "luad", "lusc", "lung adenocarcinoma", "lung squamous cell carcinoma"]),
    ("brca", ["breast cancer", "breast carcinoma", "invasive breast carcinoma", "brca"]),
    ("glioma", ["glioma", "glioblastoma", "gbm", "oligodendroglioma", "astrocytoma"]),
    ("ovarian", ["ovarian cancer", "ovarian carcinoma", "ovarian epithelial tumor", "ovarian clear cell cancer"]),
    ("rcc", ["renal cell carcinoma", "kidney renal clear cell carcinoma", "kidney cancer", "kirc", "rcc"]),
    ("aml", ["acute myeloid leukemia", "acute myelogenous leukemia", "aml", "laml"]),
    ("paad", ["pancreatic adenocarcinoma", "pancreatic cancer", "paad"]),
    ("hnsc", ["head and neck squamous cell carcinoma", "head and neck cancer", "hnsc"]),
    ("gastroesoph", ["gastric cancer", "stomach adenocarcinoma", "gastroesophageal adenocarcinoma", "esophageal adenocarcinoma", "esophageal cancer", "esca", "esad", "stad"]),
]

INT0GEN_CANCER_CODE_TO_FAMILY = {
    "LIHC": "hcc",
    "COAD": "crc",
    "READ": "crc",
    "COADREAD": "crc",
    "CRC": "crc",
    "BRCA": "brca",
    "LUAD": "nsclc",
    "LUSC": "nsclc",
    "NSCLC": "nsclc",
    "GBM": "glioma",
    "LGG": "glioma",
    "OV": "ovarian",
    "KIRC": "rcc",
    "KIRP": "rcc",
    "KIDNEY": "rcc",
    "LAML": "aml",
    "AML": "aml",
    "PAAD": "paad",
    "HNSC": "hnsc",
    "STAD": "gastroesoph",
    "ESCA": "gastroesoph",
    "ESAD": "gastroesoph",
}

CANCER_TERMS = [
    "cancer",
    "carcinoma",
    "tumor",
    "tumour",
    "neoplasm",
    "oncolog",
    "adenocarcinoma",
    "glioma",
    "glioblastoma",
    "leukemia",
    "melanoma",
    "lymphoma",
    "sarcoma",
    "hepatocellular",
    "colorectal",
    "metastatic",
]

MALIGNANCY_TERMS = [
    "cancer",
    "carcinoma",
    "adenocarcinoma",
    "glioma",
    "glioblastoma",
    "leukemia",
    "lymphoma",
    "sarcoma",
    "tumor",
    "tumour",
    "neoplasm",
    "malignant",
    "metastatic",
]

NON_MALIGNANT_EXCLUDE_TERMS = [
    "benign",
    "cyst",
    "cystic",
    "adenoma",
    "polyp",
    "hypoplasia",
    "insufficiency",
    "diverticula",
    "diverticulitis",
    "mass",
    "abscess",
    "aplasi",
    "leiomyoma",
    "lipoma",
    "lymphangioma",
    "gonadoblastoma",
]

MECHANISM_TERMS = [
    "gene",
    "protein",
    "mutation",
    "mutations",
    "variant",
    "variants",
    "fusion",
    "amplification",
    "deletion",
    "methylation",
    "expression",
    "pathway",
    "signaling",
    "mechanism",
    "oncogenic",
    "tumor suppress",
    "driver",
    "biomarker",
    "dependency",
    "vulnerability",
    "synthetic lethal",
    "resistance",
    "sensitivity",
    "target",
    "targeted",
    "receptor",
    "kinase",
    "cell line",
    "xenograft",
    "organoid",
    "screen",
    "crispr",
    "knockout",
    "knockdown",
]

THERAPY_OVERRIDE_TERMS = [
    "resistance",
    "acquired resistance",
    "biomarker",
    "mutation",
    "mutations",
    "variant",
    "variants",
    "gene",
    "protein",
    "pathway",
    "signaling",
    "target",
    "targeted",
    "dependency",
    "vulnerability",
    "sensitivity",
    "mechanism",
    "driver",
    "receptor",
]

PATHWAY_CLASS_KEYWORDS = {
    "rtk_mapk": ["egfr", "erk", "mapk", "ras", "raf", "mek"],
    "pi3k_akt_mtor": ["pi3k", "akt", "mtor", "pten"],
    "wnt_beta_catenin": ["wnt", "beta-catenin", "ctnnb1", "apc"],
    "tgfb_emt": ["tgf", "emt", "smad"],
    "p53_cell_cycle": ["p53", "cell cycle", "rb1", "checkpoint", "cdk", "cyclin"],
    "dna_damage_repair": ["dna repair", "homologous recombination", "atm", "atr", "brca", "mismatch repair"],
    "apoptosis": ["apoptosis", "bcl2", "caspase"],
    "jak_stat": ["jak", "stat", "interferon"],
    "notch": ["notch"],
    "hypoxia_angiogenesis": ["hypoxia", "hif", "angiogenesis", "vegf"],
}

STRONG_SIGNOR_MECHANISMS = {
    "phosphorylation",
    "transcriptional_regulation",
    "dephosphorylation",
    "ubiquitination",
    "deubiquitination",
    "methylation",
    "demethylation",
    "acetylation",
    "deacetylation",
    "sumoylation",
    "desumoylation",
    "glycosylation",
    "degradation",
    "cleavage",
    "stabilization",
    "destabilization",
}

WEAK_SIGNOR_MECHANISMS = {
    "binding",
    "complex_formation",
}

HARD_QUESTION_EXCLUDES = [
    "first-line treatment",
    "second-line treatment",
    "third-line treatment",
    "appropriate treatment",
    "recommended treatment",
    "standard of care",
    "clinical practice",
    "guideline",
    "screening",
    "surveillance",
    "cost-effective",
    "cost effective",
    "contraindication",
    "diagnosis",
    "diagnostic",
    "surgery",
    "surgical",
    "resection",
    "radiotherapy",
    "chemoradiotherapy",
    "dose",
    "dosage",
    "adverse event",
    "toxicity",
    "quality of life",
    "smoke",
    "smoking",
    "diet",
    "exercise",
    "letter of invitation",
    "effective for",
    "laparotomy",
    "frozen-section",
    "frozen section",
    "sentinel lymph node",
    "intraoperative",
    "second-look",
    "tnm classification",
    "tnm version",
    "preferred therapy",
    "routinely performed",
    "routinely determined",
    "routine determination",
    "initial chemotherapy regimen",
    "most appropriate",
    "chemotherapy regimen",
]

SURVIVAL_HEAVY_TERMS = [
    "survival",
    "overall survival",
    "progression free survival",
    "progression-free survival",
    "relapse free survival",
    "relapse-free survival",
    "disease free survival",
    "disease-free survival",
    "prognosis",
    "prognostic",
]

GENE_SYMBOL_PATTERN = re.compile(r"\b[A-Z0-9][A-Z0-9\-]{1,10}\b")
GENE_STOPWORDS = {
    "DNA",
    "RNA",
    "MRI",
    "CT",
    "CTL",
    "CRC",
    "HCC",
    "AML",
    "NSCLC",
    "BRCA",
    "PAAD",
    "HNSC",
    "RCC",
    "GBM",
}

GRAPH_DRIVERLIKE_STRONG_TERMS = (
    "integrated supports oncogenic mechanism",
    "amplified",
    "deleted",
    "mutated",
    "hotspot",
    "damaging mutation",
    "driver",
)
GRAPH_DRIVERLIKE_WEAK_TERMS = (
    "upregulated",
    "downregulated",
    "hypermethylated",
    "hypomethylated",
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def unique_preserve(values: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def norm_text(value: str | None) -> str:
    if not value:
        return ""
    value = re.sub(r"[^a-z0-9]+", " ", str(value).lower())
    return re.sub(r"\s+", " ", value).strip()


def safe_trim_text(value: Any, limit: int = 1600) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return trim_text(str(value), limit=limit)


def cancer_relevant(*values: str | None) -> bool:
    text = " ".join(str(value or "") for value in values).lower()
    return any(term in text for term in CANCER_TERMS)


def infer_priority_family(*values: str | None) -> str:
    text = " ".join(str(value or "") for value in values).lower()
    for family, aliases in FAMILY_PATTERNS:
        if any(alias in text for alias in aliases):
            return family
    return "mixed_oncology"


def family_label(family: str) -> str:
    return FAMILY_LABELS.get(family, family.upper())


def parse_html_listing(url: str) -> list[str]:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=180)
    response.raise_for_status()
    names = re.findall(r'href="([^"]+\.parquet)"', response.text)
    return sorted({name for name in names if name.endswith(".parquet")})


def download_file(url: str, destination: Path) -> Path:
    ensure_dir(destination.parent)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    if temp_path.exists():
        temp_path.unlink()
    last_error: Exception | None = None
    for _attempt in range(3):
        try:
            subprocess.run(
                [
                    "curl",
                    "-L",
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--retry",
                    "5",
                    "--retry-all-errors",
                    "--user-agent",
                    USER_AGENT,
                    "-o",
                    str(temp_path),
                    url,
                ],
                check=True,
            )
            temp_path.replace(destination)
            return destination
        except Exception as exc:
            last_error = exc
            if temp_path.exists():
                temp_path.unlink()
    if last_error is not None:
        raise last_error
    return destination


def extract_zip_member(zip_path: Path, member_suffix: str, destination: Path) -> Path:
    ensure_dir(destination.parent)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    with zipfile.ZipFile(zip_path) as archive:
        member_name = next(name for name in archive.namelist() if name.endswith(member_suffix))
        with archive.open(member_name) as source, destination.open("wb") as target:
            target.write(source.read())
    return destination


def extract_gene_symbols(*values: str | None, limit: int = 8) -> list[str]:
    text = " ".join(str(value or "") for value in values)
    symbols = []
    seen = set()
    for token in GENE_SYMBOL_PATTERN.findall(text):
        if len(token) < 3:
            continue
        if token in GENE_STOPWORDS:
            continue
        if token not in seen:
            seen.add(token)
            symbols.append(token)
        if len(symbols) >= limit:
            break
    return symbols


def has_mechanism_anchor(*values: str | None) -> bool:
    raw_text = " ".join(str(value or "") for value in values)
    text = raw_text.lower()
    if any(term in text for term in MECHANISM_TERMS):
        return True
    return bool(extract_gene_symbols(raw_text))


def has_therapy_override(*values: str | None) -> bool:
    text = " ".join(str(value or "") for value in values).lower()
    return any(term in text for term in THERAPY_OVERRIDE_TERMS)


def strict_literature_candidate(
    question: str,
    answer: str,
    evidence_text: str,
    *,
    source: str,
    q_type: str = "",
) -> bool:
    if not cancer_relevant(question, answer, evidence_text):
        return False
    if not has_mechanism_anchor(question, answer, evidence_text):
        return False

    question_lc = question.lower()
    if any(term in question_lc for term in HARD_QUESTION_EXCLUDES):
        return False
    if any(term in question_lc for term in SURVIVAL_HEAVY_TERMS) and not has_therapy_override(question, answer, evidence_text):
        return False

    if source == "pubmedqa":
        if not cancer_relevant(question):
            return False
        if not (has_mechanism_anchor(question) or has_therapy_override(question)):
            return False

    if source == "evidence_inference_qa":
        if not cancer_relevant(question):
            return False
        explicit_molecular = any(
            term in question_lc
            for term in [
                "gene",
                "protein",
                "mutation",
                "variant",
                "biomarker",
                "expression",
                "methylation",
                "copy number",
                "receptor",
                "kinase",
                "pathway",
                "signaling",
                "hla",
                "ctl",
            ]
        ) or bool(re.search(r"\b[A-Z0-9]+[*-][A-Z0-9]+\b", question))
        if not explicit_molecular:
            return False
        if any(term in question_lc for term in ["smoke", "smoking", "placebo", "internet group", "invitation", "quality of life", "survival", "rfs", "pfs", "effective", "response rate", "baseline", "mg/kg", "folfiri", "xeliri", "irinotecan", "capecitabine", "pain and discomfort", "treatment problem"]):
            return False
        if any(term in question_lc for term in SURVIVAL_HEAVY_TERMS):
            return False
        if not any(
            term in " ".join([question_lc, evidence_text.lower()])
            for term in [
                "gene",
                "protein",
                "mutation",
                "variant",
                "expression",
                "methylation",
                "copy number",
                "biomarker",
                "target",
                "receptor",
                "kinase",
                "cell line",
                "pathway",
                "signaling",
            ]
        ) and not extract_gene_symbols(question, evidence_text):
            return False

    if source == "bioasq":
        if q_type not in {"yesno", "factoid", "list"}:
            return False
        if not cancer_relevant(question) and not ("target" in question_lc and cancer_relevant(evidence_text, answer)):
            return False
        if not has_mechanism_anchor(question) and not ("target" in question_lc and cancer_relevant(evidence_text, answer)):
            return False
        if "treatment" in question_lc and not has_therapy_override(question, answer, evidence_text):
            return False

    if source == "pubmedqa":
        if ("therapeutic decisions" in question_lc or "role in prediction of" in question_lc and "response" in question_lc) and not has_therapy_override(question, answer, evidence_text):
            return False

    return True


def infer_scenario_tags(*values: str | None) -> list[str]:
    text = " ".join(str(value or "") for value in values).lower()
    tags = []
    if any(term in text for term in ["mechanism", "pathway", "signaling", "oncogenic", "tumor suppress", "driver", "resistance"]):
        tags.append("mechanism")
    if any(term in text for term in ["gene", "protein", "mutation", "variant", "expression", "methylation", "copy number", "biomarker", "receptor", "kinase"]):
        tags.append("molecular")
    if any(term in text for term in ["cell line", "xenograft", "organoid", "screen", "assay", "cohort", "crispr"]):
        tags.append("experimental")
    if any(term in text for term in ["target", "dependency", "vulnerability", "sensitivity", "drug target", "inhibitor"]):
        tags.append("target")
    return sorted(set(tags or ["molecular"]))


def infer_qa_function(source: str, task_type: str, metadata: dict[str, Any]) -> str:
    if source == "pubmedqa":
        return "reasoning"
    if source == "evidence_inference_qa":
        return "comparison"
    if source == "bioasq":
        q_type = str(metadata.get("bioasq_type") or "").lower()
        if q_type == "factoid":
            return "description"
        if q_type == "list":
            return "summary"
        return "reasoning"
    if source == "intogen":
        return "reasoning" if "method" in task_type else "description"
    if source == "open_targets":
        return "target" if "association" in task_type else "description"
    if source == "signor":
        return "reasoning"
    if source == "pathway_commons":
        return "description"
    if source == "external_multisource":
        return "summary"
    return "description"


def finalize_record(
    record: dict[str, Any],
    *,
    source_dataset: str,
    source_family: str,
) -> dict[str, Any]:
    gold = record.get("gold_standard") or {}
    metadata = record.setdefault("metadata", {})
    priority_family = metadata.get("priority_cancer_family") or infer_priority_family(
        record.get("oncology_focus"),
        record.get("question"),
        record.get("answer"),
        record.get("evidence_text"),
    )
    metadata["priority_cancer_family"] = priority_family
    metadata["benchmark_track"] = "external_mechanism"
    metadata["source_dataset"] = source_dataset
    metadata["source_family"] = source_family
    metadata["scenario_tags"] = metadata.get("scenario_tags") or infer_scenario_tags(
        record.get("question"),
        record.get("answer"),
        record.get("evidence_text"),
    )
    metadata["qa_evidence_function"] = metadata.get("qa_evidence_function") or infer_qa_function(
        record.get("source", ""),
        record.get("task_type", ""),
        metadata,
    )
    metadata["entity_matchable"] = bool(priority_family in SUPPORTED_FAMILIES and ((record.get("entities") or {}).get("gene_symbol") or (record.get("entities") or {}).get("gene_symbols")))
    metadata["coverage_group"] = "hcc_crc_pubmed_extension" if priority_family in {"hcc", "crc"} and source_family == "literature_qa" else "multi_cancer_core"
    metadata["normalized_entity_anchors"] = infer_normalized_entity_anchors(record)
    metadata["eval_answer_options"] = metadata.get("eval_answer_options") or infer_eval_answer_options(record)
    if gold and not gold.get("normalized_answer"):
        gold["normalized_answer"] = norm_text(gold.get("class_label") or gold.get("canonical_answer"))
    return record


def parse_literal_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        flattened = []
        for item in value:
            if isinstance(item, list):
                flattened.extend(str(part).strip() for part in item if str(part).strip())
            elif str(item).strip():
                flattened.append(str(item).strip())
        return flattened
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            raw = json.loads(text.replace("'", '"'))
            return [str(item).strip() for item in raw if str(item).strip()]
        except Exception:
            pass
    return [text]


def looks_like_gene_symbol(value: Any) -> bool:
    token = str(value or "").strip().upper()
    return bool(token) and 2 < len(token) <= 12 and re.fullmatch(r"[A-Z0-9][A-Z0-9\-]*", token) is not None and token not in GENE_STOPWORDS


def normalize_symbol_token(value: Any) -> str:
    return re.sub(r"[^A-Z0-9\-]+", "", str(value or "").upper())


def classify_pathway_name(name: Any) -> str | None:
    text = str(name or "").strip().lower()
    for label, keywords in PATHWAY_CLASS_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return label
    return None


def normalize_signor_effect(value: Any) -> str | None:
    text = norm_text(str(value or ""))
    if not text:
        return None
    if any(term in text for term in ["up regulates", "up regulates quantity", "up regulates activity", "activates", "stabilizes", "forms complex"]):
        return "activating_or_upregulating"
    if any(term in text for term in ["down regulates", "down regulates quantity", "down regulates activity", "inhibits", "destabilizes"]):
        return "inhibitory_or_downregulating"
    return text.replace(" ", "_")


def signor_direct_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"t", "true", "1", "yes", "y"}


def effect_sign(effect: str | None) -> int | None:
    if effect == "activating_or_upregulating":
        return 1
    if effect == "inhibitory_or_downregulating":
        return -1
    return None


def compose_causal_effect(first: str | None, second: str | None) -> str | None:
    first_sign = effect_sign(first)
    second_sign = effect_sign(second)
    if first_sign is None or second_sign is None:
        return None
    return "activating_or_upregulating" if first_sign * second_sign > 0 else "inhibitory_or_downregulating"


def build_pathway_gene_indexes(pathway_rows: list[dict[str, Any]]) -> tuple[dict[str, set[str]], dict[str, list[str]]]:
    gene_to_classes: dict[str, set[str]] = defaultdict(set)
    gene_to_pathways: dict[str, list[str]] = defaultdict(list)
    for row in pathway_rows:
        pathway_class = str(row.get("pathway_class") or "").strip()
        pathway_name = str(row.get("pathway_name") or "").strip()
        for gene in row.get("genes") or []:
            token = normalize_symbol_token(gene)
            if not looks_like_gene_symbol(token):
                continue
            if pathway_class:
                gene_to_classes[token].add(pathway_class)
            if pathway_name and pathway_name not in gene_to_pathways[token]:
                gene_to_pathways[token].append(pathway_name)
    return gene_to_classes, gene_to_pathways


def family_gene_supports(support_index: dict[tuple[str, str], set[str]], family: str, *genes: str) -> dict[str, set[str]]:
    return {
        gene: set(support_index.get((family, gene), set()))
        for gene in genes
        if looks_like_gene_symbol(gene)
    }


def candidate_causal_families(support_index: dict[tuple[str, str], set[str]], *genes: str) -> list[str]:
    families = set()
    for family, gene in support_index:
        if gene in genes and support_index[(family, gene)]:
            families.add(family)
    return sorted(families)


def score_direct_causal_fit(
    *,
    family: str,
    entity_a: str,
    entity_b: str,
    mechanism: str,
    direct: Any,
    sentence: Any,
    pmid: Any,
    support_index: dict[tuple[str, str], set[str]],
    gene_to_classes: dict[str, set[str]],
) -> tuple[int, dict[str, Any]]:
    source_layers = set(support_index.get((family, entity_a), set()))
    target_layers = set(support_index.get((family, entity_b), set()))
    shared_pathway_classes = sorted(gene_to_classes.get(entity_a, set()).intersection(gene_to_classes.get(entity_b, set())))
    score = 0
    if source_layers:
        score += 3
    if target_layers:
        score += 2
    if signor_direct_bool(direct):
        score += 2
    if mechanism in STRONG_SIGNOR_MECHANISMS:
        score += 2
    elif mechanism and mechanism not in WEAK_SIGNOR_MECHANISMS:
        score += 1
    if shared_pathway_classes:
        score += 2
    if sentence and str(sentence).strip():
        score += 1
    if pmid and str(pmid).strip():
        score += 1
    if entity_a == entity_b:
        score -= 4
    return score, {
        "source_layers": sorted(source_layers),
        "target_layers": sorted(target_layers),
        "shared_pathway_classes": shared_pathway_classes,
    }


def score_indirect_causal_fit(
    *,
    family: str,
    entity_a: str,
    intermediate_gene: str,
    entity_b: str,
    edge1: dict[str, Any],
    edge2: dict[str, Any],
    support_index: dict[tuple[str, str], set[str]],
    gene_to_classes: dict[str, set[str]],
) -> tuple[int, dict[str, Any]]:
    family_support = family_gene_supports(support_index, family, entity_a, intermediate_gene, entity_b)
    triple_shared_classes = sorted(
        gene_to_classes.get(entity_a, set())
        .intersection(gene_to_classes.get(intermediate_gene, set()))
        .intersection(gene_to_classes.get(entity_b, set()))
    )
    pair_shared_classes = sorted(
        gene_to_classes.get(entity_a, set()).intersection(gene_to_classes.get(entity_b, set()))
    )
    score = 0
    support_gene_count = sum(1 for layers in family_support.values() if layers)
    score += support_gene_count * 2
    if signor_direct_bool(edge1.get("DIRECT")):
        score += 2
    if signor_direct_bool(edge2.get("DIRECT")):
        score += 2
    if str(edge1.get("normalized_mechanism") or "") in STRONG_SIGNOR_MECHANISMS:
        score += 1
    if str(edge2.get("normalized_mechanism") or "") in STRONG_SIGNOR_MECHANISMS:
        score += 1
    if triple_shared_classes:
        score += 3
    elif pair_shared_classes:
        score += 1
    if edge1.get("PMID") and edge2.get("PMID") and str(edge1.get("PMID")) != str(edge2.get("PMID")):
        score += 1
    if entity_a == entity_b or entity_a == intermediate_gene or intermediate_gene == entity_b:
        score -= 4
    return score, {
        "family_support": {gene: sorted(layers) for gene, layers in family_support.items() if layers},
        "shared_pathway_classes": triple_shared_classes or pair_shared_classes,
    }


def extend_aliases_with_gene_symbols(aliases: list[str], *texts: Any) -> list[str]:
    expanded = []
    seen = set()
    for alias in aliases:
        alias_text = str(alias or "").strip()
        if alias_text and alias_text not in seen:
            seen.add(alias_text)
            expanded.append(alias_text)
    for symbol in extract_gene_symbols(*(str(text or "") for text in texts)):
        token = normalize_symbol_token(symbol)
        if looks_like_gene_symbol(token) and token not in seen:
            seen.add(token)
            expanded.append(token)
    return expanded


def choose_entity_normalized_answer(aliases: list[str], *texts: Any) -> str:
    for alias in aliases:
        token = normalize_symbol_token(alias)
        if looks_like_gene_symbol(token):
            return token
    for symbol in extract_gene_symbols(*(str(text or "") for text in texts)):
        token = normalize_symbol_token(symbol)
        if looks_like_gene_symbol(token):
            return token
    return norm_text(aliases[0]) if aliases else ""


def infer_eval_answer_options(record: dict[str, Any]) -> list[str]:
    task_type = str(record.get("task_type") or "")
    if task_type == "driver_role_assignment":
        return ["activating", "loss_of_function", "ambiguous"]
    if task_type in {"target_disease_association_strength", "intogen_support_strength", "cgc_support_strength"}:
        return ["high", "moderate", "low"]
    if task_type == "causal_relation_effect":
        return ["activating_or_upregulating", "inhibitory_or_downregulating", "unknown"]
    if task_type == "pathway_class_membership":
        return ["rtk_mapk", "pi3k_akt_mtor", "wnt_beta_catenin", "tgfb_emt", "p53_cell_cycle", "dna_damage_repair", "apoptosis", "jak_stat", "notch", "hypoxia_angiogenesis"]
    if task_type == "cross_source_support_bundle":
        return [
            "causal_relation+driver+pathway_context+target_association",
            "causal_relation+driver+target_association",
            "causal_relation+pathway_context+target_association",
        ]
    if task_type == "cross_source_corroboration_status":
        return ["yes", "no"]
    if task_type in {"mechanism_bioasq_yesno", "mechanism_literature_reasoning"}:
        return ["yes", "no"]
    metadata = record.get("metadata") or {}
    choices = [norm_text(choice) for choice in metadata.get("choices") or [] if norm_text(choice)]
    return unique_preserve(choices)


def infer_normalized_entity_anchors(record: dict[str, Any]) -> dict[str, Any]:
    entities = record.get("entities") or {}
    anchors: dict[str, Any] = {}
    family = infer_priority_family(record.get("oncology_focus"), entities.get("disease"), entities.get("disease_name"))
    if family:
        anchors["disease_family"] = family
    gene_symbols = []
    for value in [entities.get("gene_symbol"), entities.get("target_gene_symbol")] + list(entities.get("gene_symbols") or []):
        token = normalize_symbol_token(value)
        if looks_like_gene_symbol(token):
            gene_symbols.append(token)
    gene_symbols = unique_preserve(gene_symbols)
    if gene_symbols:
        anchors["gene_symbols"] = gene_symbols
    if entities.get("pathway_name"):
        anchors["pathway_name"] = entities.get("pathway_name")
    if entities.get("cohort"):
        anchors["cohort"] = entities.get("cohort")
    return anchors


def build_external_support_index(intogen_df: pd.DataFrame, open_targets_payload: dict[str, pd.DataFrame]) -> dict[tuple[str, str], set[str]]:
    support = defaultdict(set)
    for row in intogen_df.to_dict(orient="records"):
        family = row.get("priority_cancer_family")
        gene = str(row.get("SYMBOL") or "").strip().upper()
        if family in SUPPORTED_FAMILIES and looks_like_gene_symbol(gene):
            support[(family, gene)].add("intogen_driver")
    target_map = {
        row["id"]: str(row.get("approvedSymbol") or "").strip().upper()
        for row in open_targets_payload["targets"].to_dict(orient="records")
        if row.get("approvedSymbol")
    }
    for label, frame in [("open_targets_overall", open_targets_payload["associations"]), ("open_targets_intogen", open_targets_payload["intogen"]), ("open_targets_cgc", open_targets_payload["cgc"])]:
        for row in frame.to_dict(orient="records"):
            family = row.get("priority_cancer_family")
            gene = target_map.get(row.get("targetId"))
            if family in SUPPORTED_FAMILIES and looks_like_gene_symbol(gene):
                support[(family, gene)].add(label)
    return support


def build_graph_driverlike_index(path: Path = EVIDENCE_GRAPH_DOCS_PATH) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    index: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "strong_count": 0,
            "weak_count": 0,
            "matched_terms": [],
            "titles": [],
        }
    )
    for row in read_jsonl(path):
        disease_tags = row.get("disease_tags") or []
        family = infer_priority_family(*disease_tags) if disease_tags else "mixed_oncology"
        if family not in SUPPORTED_FAMILIES:
            family = infer_priority_family(row.get("title"), row.get("text"))
        if family not in SUPPORTED_FAMILIES:
            continue
        text = norm_text(" ".join(str(row.get(key) or "") for key in ["title", "text"]))
        strong_hits = [term for term in GRAPH_DRIVERLIKE_STRONG_TERMS if term in text]
        weak_hits = [term for term in GRAPH_DRIVERLIKE_WEAK_TERMS if term in text]
        if not strong_hits and not weak_hits:
            continue
        genes = unique_preserve(
            [
                normalize_symbol_token(value)
                for value in (row.get("entity_names") or [])
                if looks_like_gene_symbol(value)
            ]
        )
        if not genes:
            continue
        title = str(row.get("title") or "").strip()
        for gene in genes:
            summary = index[(family, gene)]
            if strong_hits:
                summary["strong_count"] += 1
            if weak_hits:
                summary["weak_count"] += 1
            summary["matched_terms"] = unique_preserve(summary["matched_terms"] + strong_hits + weak_hits)
            if title and title not in summary["titles"] and len(summary["titles"]) < 4:
                summary["titles"].append(title)
    return dict(index)


def has_graph_driverlike_signal(summary: dict[str, Any] | None) -> bool:
    if not summary:
        return False
    return int(summary.get("strong_count") or 0) >= 1 or int(summary.get("weak_count") or 0) >= 2


def materialize_signor_raw() -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_dir = RAW_DIR / "signor"
    ensure_dir(raw_dir)
    latest_path = download_file(SIGNOR_LATEST_URL, raw_dir / "signor_latest.tsv")
    df = pd.read_csv(latest_path, sep="\t")
    for column in ["ENTITYA", "ENTITYB", "TYPEA", "TYPEB", "DATABASEA", "DATABASEB", "EFFECT", "MECHANISM", "SENTENCE", "PMID", "DIRECT"]:
        if column not in df.columns:
            df[column] = None
    supported = df[
        df["TYPEA"].astype(str).str.lower().eq("protein")
        & df["TYPEB"].astype(str).str.lower().eq("protein")
        & df["DATABASEA"].astype(str).str.upper().eq("UNIPROT")
        & df["DATABASEB"].astype(str).str.upper().eq("UNIPROT")
        & df["ENTITYA"].astype(str).map(looks_like_gene_symbol)
        & df["ENTITYB"].astype(str).map(looks_like_gene_symbol)
        & df["EFFECT"].notna()
    ].copy()
    supported["normalized_effect"] = supported["EFFECT"].map(normalize_signor_effect)
    supported["normalized_mechanism"] = supported["MECHANISM"].astype(str).str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
    supported = supported[supported["normalized_effect"].notna()].copy()
    supported.to_csv(raw_dir / "signor_supported.tsv", sep="\t", index=False)
    stats = {
        "total_rows": int(len(df)),
        "supported_rows": int(len(supported)),
        "by_effect": {str(k): int(v) for k, v in supported["normalized_effect"].value_counts().to_dict().items()},
    }
    write_json(raw_dir / "signor_supported_stats.json", stats)
    return supported, stats


def materialize_pathwaycommons_raw() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_dir = RAW_DIR / "pathwaycommons"
    ensure_dir(raw_dir)
    gmt_path = download_file(PATHWAYCOMMONS_GMT_URL, raw_dir / "pc-hgnc.gmt.gz")
    pathway_rows: list[dict[str, Any]] = []
    with gzip.open(gmt_path, "rt", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            pathway_name = parts[0].strip()
            pathway_source = parts[1].strip()
            pathway_class = classify_pathway_name(pathway_name)
            if not pathway_class:
                continue
            genes = []
            seen = set()
            for gene in parts[2:]:
                token = str(gene or "").strip().upper()
                if looks_like_gene_symbol(token) and token not in seen:
                    seen.add(token)
                    genes.append(token)
            if not genes:
                continue
            pathway_rows.append(
                {
                    "pathway_name": pathway_name,
                    "pathway_source": pathway_source,
                    "pathway_class": pathway_class,
                    "genes": genes,
                }
            )
    write_jsonl(raw_dir / "pathwaycommons_cancer_pathways.jsonl", pathway_rows)
    stats = {
        "selected_pathways": len(pathway_rows),
        "by_pathway_class": dict(Counter(row["pathway_class"] for row in pathway_rows)),
    }
    write_json(raw_dir / "pathwaycommons_supported_stats.json", stats)
    return pathway_rows, stats


def normalize_pubmedqa() -> list[dict[str, Any]]:
    records = []
    for split_dir in sorted((RAW_DIR / "pubmedqa").glob("pqal_fold*")):
        for file_path in sorted(split_dir.glob("*.json")):
            payload = json.loads(file_path.read_text())
            split_name = f"{split_dir.name}_{file_path.stem}"
            for pmid, item in payload.items():
                question = safe_trim_text(item.get("QUESTION"))
                contexts = [safe_trim_text(text) for text in item.get("CONTEXTS") or []]
                long_answer = safe_trim_text(item.get("LONG_ANSWER"))
                decision = str(item.get("final_decision") or "").strip().lower()
                priority_family = infer_priority_family(question, long_answer, " ".join(contexts[:3]))
                if priority_family not in SUPPORTED_FAMILIES:
                    continue
                if not strict_literature_candidate(question, long_answer, " ".join(contexts[:3]), source="pubmedqa"):
                    continue
                gene_symbols = extract_gene_symbols(question, long_answer, " ".join(contexts[:2]))
                answer_text = "Yes." if decision == "yes" else ("No." if decision == "no" else "Maybe.")
                record = {
                    "id": f"pubmedqa-{split_name}-{pmid}",
                    "source": "pubmedqa",
                    "task_type": "mechanism_literature_reasoning",
                    "question": question,
                    "answer": answer_text,
                    "answer_type": "categorical",
                    "oncology_focus": family_label(priority_family),
                    "evidence_text": trim_text(" ".join(contexts[:2] + [long_answer])),
                    "entities": {
                        "disease": family_label(priority_family),
                        "pmid": pmid,
                        "gene_symbols": gene_symbols,
                    },
                    "provenance": {
                        "source_record_id": pmid,
                        "source_dataset": "PubMedQA",
                        "split": split_name,
                    },
                    "metadata": {
                        "pubmedqa_decision": decision,
                        "mesh_terms": item.get("MESHES") or [],
                        "year": item.get("YEAR"),
                    },
                }
                records.append(
                    finalize_record(
                        attach_gold_standard(
                            record,
                            build_gold_standard(
                                "categorical",
                                canonical_answer=decision,
                                acceptable_answers=[decision],
                                class_label=decision,
                                evaluable=decision in {"yes", "no", "maybe"},
                            ),
                        ),
                        source_dataset="PubMedQA",
                        source_family="literature_qa",
                    )
                )
    return records


def normalize_evidence_inference() -> list[dict[str, Any]]:
    records = []
    for csv_path in sorted((RAW_DIR / "evidence-inference-qa").glob("*.csv")):
        df = pd.read_csv(csv_path)
        split_name = csv_path.stem.replace("evidence_inference_qa_", "")
        for row in df.to_dict(orient="records"):
            question = safe_trim_text(row.get("question"))
            context = safe_trim_text(row.get("context"))
            priority_family = infer_priority_family(question, context)
            if priority_family not in SUPPORTED_FAMILIES:
                continue
            if not strict_literature_candidate(question, "", context, source="evidence_inference_qa"):
                continue
            answers = parse_literal_list(row.get("answer"))
            label = norm_text(answers[0]) if answers else ""
            if not label:
                continue
            gene_symbols = extract_gene_symbols(question, context)
            record = {
                "id": f"evidence-inference-{split_name}-{row.get('id')}",
                "source": "evidence_inference_qa",
                "task_type": "mechanism_evidence_comparison",
                "question": question,
                "answer": answers[0],
                "answer_type": "categorical",
                "oncology_focus": family_label(priority_family),
                "evidence_text": context,
                "entities": {
                    "disease": family_label(priority_family),
                    "document_id": row.get("document_id"),
                    "question_id": row.get("question_id"),
                    "gene_symbols": gene_symbols,
                },
                "provenance": {
                    "source_record_id": row.get("id"),
                    "source_dataset": "Evidence Inference QA",
                    "split": split_name,
                },
                "metadata": {
                    "question_type": row.get("type"),
                    "choices": parse_literal_list(row.get("choices")),
                },
            }
            records.append(
                finalize_record(
                    attach_gold_standard(
                        record,
                        build_gold_standard(
                            "categorical",
                            canonical_answer=label,
                            acceptable_answers=answers,
                            class_label=label,
                            evaluable=True,
                        ),
                    ),
                    source_dataset="Evidence Inference QA",
                    source_family="literature_qa",
                )
            )
    return records


def normalize_bioasq() -> list[dict[str, Any]]:
    records = []
    for json_path in sorted((RAW_DIR / "bioasq" / "test").glob("*.json")):
        payload = json.loads(json_path.read_text())
        split_name = json_path.stem
        for question in payload.get("questions") or []:
            body = safe_trim_text(question.get("body"))
            snippets = question.get("snippets") or []
            snippet_text = safe_trim_text(" ".join((snippet.get("text") or "") for snippet in snippets[:3]))
            ideal_answer = safe_trim_text(" ".join(question.get("ideal_answer") or []))
            q_type = str(question.get("type") or "").lower()
            priority_family = infer_priority_family(body, snippet_text, ideal_answer)
            if priority_family not in SUPPORTED_FAMILIES:
                continue
            if not strict_literature_candidate(body, ideal_answer, snippet_text, source="bioasq", q_type=q_type):
                continue

            exact_answer = question.get("exact_answer")
            gold = None
            answer_text = ideal_answer or body
            answer_type = "free_text"
            if q_type == "yesno":
                label = norm_text(str(exact_answer))
                if label not in {"yes", "no"}:
                    continue
                answer_text = "Yes." if label == "yes" else "No."
                answer_type = "categorical"
                gold = build_gold_standard("categorical", canonical_answer=label, acceptable_answers=[label], class_label=label, evaluable=True)
            elif q_type == "factoid":
                aliases = extend_aliases_with_gene_symbols(parse_literal_list(exact_answer), exact_answer, body, ideal_answer, snippet_text)
                if not aliases:
                    continue
                answer_text = aliases[0]
                answer_type = "entity"
                gold = build_gold_standard(
                    "entity",
                    canonical_answer=aliases[0],
                    normalized_answer=choose_entity_normalized_answer(aliases, exact_answer, body, ideal_answer, snippet_text),
                    acceptable_answers=aliases,
                    class_label=norm_text(aliases[0]),
                    evaluable=True,
                )
            elif q_type == "list":
                aliases = extend_aliases_with_gene_symbols(parse_literal_list(exact_answer), exact_answer, body, ideal_answer, snippet_text)
                if not aliases:
                    continue
                answer_text = "; ".join(aliases[:5])
                answer_type = "entity"
                gold = build_gold_standard(
                    "entity",
                    canonical_answer=aliases[0],
                    normalized_answer=choose_entity_normalized_answer(aliases, exact_answer, body, ideal_answer, snippet_text),
                    acceptable_answers=aliases,
                    class_label=norm_text(aliases[0]),
                    evaluable=True,
                )
            else:
                continue

            gene_symbols = extract_gene_symbols(body, snippet_text, ideal_answer, answer_text)
            record = {
                "id": f"bioasq-{split_name}-{question.get('id')}",
                "source": "bioasq",
                "task_type": f"mechanism_bioasq_{q_type}",
                "question": body,
                "answer": answer_text,
                "answer_type": answer_type,
                "oncology_focus": family_label(priority_family),
                "evidence_text": trim_text(" ".join([snippet_text, ideal_answer])),
                "entities": {
                    "disease": family_label(priority_family),
                    "bioasq_type": q_type,
                    "gene_symbols": gene_symbols,
                },
                "provenance": {
                    "source_record_id": question.get("id"),
                    "source_dataset": "BioASQ",
                    "split": split_name,
                },
                "metadata": {
                    "bioasq_type": q_type,
                    "document_count": len(question.get("documents") or []),
                },
            }
            records.append(
                finalize_record(
                    attach_gold_standard(record, gold),
                    source_dataset="BioASQ",
                    source_family="literature_qa",
                )
            )
    return records


def map_intogen_family(cancer_type: Any, cancer_name: Any, cohort: Any) -> str | None:
    code = str(cancer_type or "").strip().upper()
    if code in INT0GEN_CANCER_CODE_TO_FAMILY:
        family = INT0GEN_CANCER_CODE_TO_FAMILY[code]
        return family if family in SUPPORTED_FAMILIES else None
    inferred = infer_priority_family(str(cancer_name or ""), str(cohort or ""))
    return inferred if inferred in SUPPORTED_FAMILIES else None


def materialize_intogen_raw() -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_dir = RAW_DIR / "intogen"
    ensure_dir(raw_dir)
    drivers_zip = download_file(INT0GEN_DRIVERS_URL, raw_dir / "IntOGen-Drivers-20200201.zip")
    cohorts_zip = download_file(INT0GEN_COHORTS_URL, raw_dir / "IntOGen-Cohorts-20200201.zip")
    drivers_tsv = extract_zip_member(drivers_zip, "Compendium_Cancer_Genes.tsv", raw_dir / "Compendium_Cancer_Genes.tsv")
    cohorts_tsv = extract_zip_member(cohorts_zip, "cohorts.tsv", raw_dir / "cohorts.tsv")

    drivers = pd.read_csv(drivers_tsv, sep="\t")
    cohorts = pd.read_csv(cohorts_tsv, sep="\t")
    drivers["priority_cancer_family"] = drivers.apply(
        lambda row: map_intogen_family(row.get("CANCER_TYPE"), row.get("CANCER_TYPE"), row.get("COHORT")),
        axis=1,
    )
    merged = drivers.merge(
        cohorts[["COHORT", "CANCER_TYPE_NAME", "REFERENCE", "SAMPLES", "WEB_SHORT_COHORT_NAME", "WEB_LONG_COHORT_NAME"]],
        on="COHORT",
        how="left",
        suffixes=("", "_cohort"),
    )
    merged["priority_cancer_family"] = merged.apply(
        lambda row: row.get("priority_cancer_family")
        or map_intogen_family(row.get("CANCER_TYPE"), row.get("CANCER_TYPE_NAME"), row.get("COHORT")),
        axis=1,
    )
    supported = merged[merged["priority_cancer_family"].isin(SUPPORTED_FAMILIES)].copy()
    supported.to_csv(raw_dir / "Compendium_Cancer_Genes_supported.tsv", sep="\t", index=False)
    stats = {
        "total_compendium_rows": int(len(drivers)),
        "supported_rows": int(len(supported)),
        "by_family": {str(k): int(v) for k, v in supported["priority_cancer_family"].value_counts().to_dict().items()},
    }
    write_json(raw_dir / "intogen_supported_stats.json", stats)
    return supported, stats


def normalize_intogen_role(role: Any) -> str:
    value = str(role or "").strip().lower()
    if value == "act":
        return "activating"
    if value == "lof":
        return "loss_of_function"
    if value:
        return value.replace(" ", "_")
    return "unknown"


def normalize_intogen(df: pd.DataFrame | None = None) -> list[dict[str, Any]]:
    if df is None:
        df, _stats = materialize_intogen_raw()
    records = []
    for row in df.to_dict(orient="records"):
        family = row.get("priority_cancer_family")
        if family not in SUPPORTED_FAMILIES:
            continue
        gene_symbol = str(row.get("SYMBOL") or "").strip()
        if not gene_symbol:
            continue
        methods = [part.strip() for part in str(row.get("METHODS") or "").split(",") if part.strip()]
        role = normalize_intogen_role(row.get("ROLE"))
        cohort_label = row.get("WEB_SHORT_COHORT_NAME") or row.get("COHORT")
        cancer_name = row.get("CANCER_TYPE_NAME") or row.get("CANCER_TYPE")
        evidence_text = trim_text(
            f"IntOGen compendium driver record for {gene_symbol} in {cancer_name}. "
            f"Role={role}; methods={', '.join(methods) or 'NA'}; qvalue={row.get('QVALUE_COMBINATION')}; "
            f"mutations={row.get('MUTATIONS')}; samples={row.get('SAMPLES')}; percent_samples={row.get('%_SAMPLES_COHORT')}."
        )
        role_record = {
            "id": f"intogen-role-{gene_symbol}-{row.get('COHORT')}",
            "source": "intogen",
            "task_type": "driver_role_assignment",
            "question": f"In IntOGen, what driver role is assigned to {gene_symbol} in {cancer_name} cohort {cohort_label}?",
            "answer": role,
            "answer_type": "categorical",
            "oncology_focus": family_label(family),
            "evidence_text": evidence_text,
            "entities": {
                "disease": family_label(family),
                "gene_symbol": gene_symbol,
                "transcript": row.get("TRANSCRIPT"),
                "cohort": row.get("COHORT"),
                "cancer_type_code": row.get("CANCER_TYPE"),
            },
            "provenance": {
                "source_record_id": f"{gene_symbol}:{row.get('COHORT')}",
                "source_dataset": "IntOGen Compendium Cancer Genes",
                "reference": row.get("REFERENCE"),
            },
            "metadata": {
                "priority_cancer_family": family,
                "method_count": len(methods),
                "methods": methods,
                "qvalue_combination": row.get("QVALUE_COMBINATION"),
                "sample_fraction": row.get("%_SAMPLES_COHORT"),
                "scenario_tags": ["mechanism", "molecular", "target"],
            },
        }
        records.append(
            finalize_record(
                attach_gold_standard(
                    role_record,
                    build_gold_standard(
                        "categorical",
                        canonical_answer=role,
                        acceptable_answers=[role],
                        class_label=role,
                        evaluable=True,
                    ),
                ),
                source_dataset="IntOGen",
                source_family="driver_catalog",
            )
        )

        method_answer = "yes" if len(methods) > 1 else "no"
        method_record = {
            "id": f"intogen-method-{gene_symbol}-{row.get('COHORT')}",
            "source": "intogen",
            "task_type": "driver_multimethod_support",
            "question": f"In IntOGen, is {gene_symbol} supported by multiple driver-calling methods in {cancer_name} cohort {cohort_label}?",
            "answer": "Yes." if method_answer == "yes" else "No.",
            "answer_type": "boolean",
            "oncology_focus": family_label(family),
            "evidence_text": evidence_text,
            "entities": {
                "disease": family_label(family),
                "gene_symbol": gene_symbol,
                "cohort": row.get("COHORT"),
            },
            "provenance": {
                "source_record_id": f"{gene_symbol}:{row.get('COHORT')}",
                "source_dataset": "IntOGen Compendium Cancer Genes",
                "reference": row.get("REFERENCE"),
            },
            "metadata": {
                "priority_cancer_family": family,
                "method_count": len(methods),
                "methods": methods,
                "qvalue_combination": row.get("QVALUE_COMBINATION"),
                "scenario_tags": ["mechanism", "molecular"],
            },
        }
        records.append(
            finalize_record(
                attach_gold_standard(
                    method_record,
                    build_gold_standard(
                        "boolean",
                        canonical_answer=method_answer,
                        acceptable_answers=[method_answer, "true" if method_answer == "yes" else "false"],
                        binary_answer=method_answer == "yes",
                        class_label="multimethod" if method_answer == "yes" else "single_method",
                        evaluable=True,
                    ),
                ),
                source_dataset="IntOGen",
                source_family="driver_catalog",
            )
        )
    return records


def map_open_targets_family(name: Any) -> str | None:
    text = str(name or "").strip().lower()
    if not text:
        return None
    if any(term in text for term in NON_MALIGNANT_EXCLUDE_TERMS):
        return None
    if any(term in text for term in ["measurement", "response measurement", "trait", "assay", "screen", "level", "secretion", "count", "activity"]):
        return None
    if not any(term in text for term in MALIGNANCY_TERMS):
        return None
    family = infer_priority_family(text)
    return family if family in SUPPORTED_FAMILIES else None


def materialize_open_targets_raw() -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    raw_dir = RAW_DIR / "opentargets"
    ensure_dir(raw_dir)

    disease_remote = f"{OPENTARGETS_BASE}/disease/disease.parquet"
    disease_local = raw_dir / "disease.parquet"
    if not disease_local.exists():
        pd.read_parquet(disease_remote).to_parquet(disease_local, index=False)

    disease_df = pd.read_parquet(disease_local, columns=["id", "name", "description"])
    disease_df["priority_cancer_family"] = disease_df["name"].map(map_open_targets_family)
    disease_supported = disease_df[disease_df["priority_cancer_family"].isin(SUPPORTED_FAMILIES)].copy()
    disease_supported.to_parquet(raw_dir / "disease_supported.parquet", index=False)
    supported_disease_ids = set(disease_supported["id"].tolist())
    disease_family_map = dict(zip(disease_supported["id"], disease_supported["priority_cancer_family"]))

    def remote_dir_to_local_parquet(remote_dir: str, columns: list[str], local_name: str, filters) -> pd.DataFrame:
        local_path = raw_dir / local_name
        if local_path.exists():
            return pd.read_parquet(local_path)
        part_names = parse_html_listing(remote_dir)
        frames = []
        cache_dir = raw_dir / "_part_cache" / local_name.replace(".parquet", "")
        for name in part_names:
            local_part = download_file(f"{remote_dir}/{name}", cache_dir / name)
            chunk = pd.read_parquet(local_part, columns=columns)
            chunk = filters(chunk)
            if len(chunk):
                frames.append(chunk)
        result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
        result.to_parquet(local_path, index=False)
        return result

    assoc_df = remote_dir_to_local_parquet(
        f"{OPENTARGETS_BASE}/association_overall_direct",
        ["diseaseId", "targetId", "score", "evidenceCount"],
        "association_overall_direct_supported.parquet",
        lambda frame: frame[
            frame["diseaseId"].isin(supported_disease_ids)
            & frame["targetId"].notna()
            & (frame["score"] >= 0.15)
            & (frame["evidenceCount"] >= 1)
        ].copy(),
    )

    datasource_assoc_df = remote_dir_to_local_parquet(
        f"{OPENTARGETS_BASE}/association_by_datasource_direct",
        ["datatypeId", "datasourceId", "diseaseId", "targetId", "score", "evidenceCount"],
        "association_by_datasource_direct_supported.parquet",
        lambda frame: frame[
            frame["diseaseId"].isin(supported_disease_ids)
            & frame["targetId"].notna()
            & frame["datasourceId"].isin(["intogen", "cancer_gene_census"])
            & (frame["score"] >= 0.10)
            & (frame["evidenceCount"] >= 1)
        ].copy(),
    )
    intogen_df = datasource_assoc_df[datasource_assoc_df["datasourceId"] == "intogen"].copy()
    cgc_df = datasource_assoc_df[datasource_assoc_df["datasourceId"] == "cancer_gene_census"].copy()

    target_ids = set(assoc_df["targetId"].dropna().tolist()) | set(intogen_df["targetId"].dropna().tolist()) | set(cgc_df["targetId"].dropna().tolist())
    target_df = remote_dir_to_local_parquet(
        f"{OPENTARGETS_BASE}/target",
        ["id", "approvedSymbol", "approvedName", "biotype", "functionDescriptions", "targetClass", "proteinIds", "dbXrefs"],
        "target_supported.parquet",
        lambda frame: frame[
            frame["id"].isin(target_ids)
            & frame["approvedSymbol"].notna()
            & (frame["biotype"] == "protein_coding")
        ].copy(),
    )

    stats = {
        "disease_supported_rows": int(len(disease_supported)),
        "association_supported_rows": int(len(assoc_df)),
        "datasource_intogen_supported_rows": int(len(intogen_df)),
        "datasource_cancer_gene_census_supported_rows": int(len(cgc_df)),
        "target_supported_rows": int(len(target_df)),
        "by_family": {str(k): int(v) for k, v in disease_supported["priority_cancer_family"].value_counts().to_dict().items()},
    }
    write_json(raw_dir / "opentargets_supported_stats.json", stats)

    assoc_df["priority_cancer_family"] = assoc_df["diseaseId"].map(disease_family_map)
    intogen_df["priority_cancer_family"] = intogen_df["diseaseId"].map(disease_family_map)
    cgc_df["priority_cancer_family"] = cgc_df["diseaseId"].map(disease_family_map)
    return {
        "diseases": disease_supported,
        "associations": assoc_df,
        "intogen": intogen_df,
        "cgc": cgc_df,
        "targets": target_df,
    }, stats


def association_strength_label(score: float) -> str:
    if score >= 0.5:
        return "high"
    if score >= 0.3:
        return "moderate"
    return "low"


def normalize_open_targets(payload: dict[str, pd.DataFrame] | None = None) -> list[dict[str, Any]]:
    if payload is None:
        payload, _stats = materialize_open_targets_raw()
    diseases = payload["diseases"]
    associations = payload["associations"]
    intogen = payload["intogen"]
    cgc = payload["cgc"]
    targets = payload["targets"]

    disease_name_map = dict(zip(diseases["id"], diseases["name"]))
    disease_family_map = dict(zip(diseases["id"], diseases["priority_cancer_family"]))
    target_map = {
        row["id"]: row
        for row in targets.to_dict(orient="records")
        if row.get("approvedSymbol")
    }

    records = []

    assoc_best = (
        associations.sort_values(["score", "evidenceCount"], ascending=[False, False])
        .drop_duplicates(subset=["diseaseId", "targetId"])
        .to_dict(orient="records")
    )
    for row in assoc_best:
        target = target_map.get(row["targetId"])
        family = row.get("priority_cancer_family")
        if target is None or family not in SUPPORTED_FAMILIES:
            continue
        disease_name = disease_name_map.get(row["diseaseId"]) or family_label(family)
        gene_symbol = target["approvedSymbol"]
        score = float(row["score"])
        strength = association_strength_label(score)
        record = {
            "id": f"opentargets-assoc-{row['targetId']}-{row['diseaseId']}",
            "source": "open_targets",
            "task_type": "target_disease_association_strength",
            "question": f"In Open Targets, how strong is the direct target-disease association between {gene_symbol} and {disease_name}?",
            "answer": strength,
            "answer_type": "categorical",
            "oncology_focus": family_label(family),
            "evidence_text": trim_text(
                f"Open Targets direct association for {gene_symbol} and {disease_name}: score={score:.3f}, evidence_count={int(row['evidenceCount'])}."
            ),
            "entities": {
                "disease": family_label(family),
                "disease_id": row["diseaseId"],
                "gene_symbol": gene_symbol,
                "target_ensembl_id": row["targetId"],
            },
            "provenance": {
                "source_record_id": f"{row['targetId']}:{row['diseaseId']}",
                "source_dataset": "Open Targets association_overall_direct",
            },
            "metadata": {
                "priority_cancer_family": family,
                "association_score": score,
                "evidence_count": int(row["evidenceCount"]),
                "scenario_tags": ["molecular", "target"],
            },
        }
        records.append(
            finalize_record(
                attach_gold_standard(
                    record,
                    build_gold_standard(
                        "categorical",
                        canonical_answer=strength,
                        acceptable_answers=[strength],
                        class_label=strength,
                        evaluable=True,
                    ),
                ),
                source_dataset="Open Targets",
                source_family="target_platform",
            )
        )

    def build_datasource_records(frame: pd.DataFrame, source_label: str, task_suffix: str) -> list[dict[str, Any]]:
        built = []
        ranked = frame.sort_values(["score", "evidenceCount"], ascending=[False, False]).drop_duplicates(subset=["diseaseId", "targetId"])
        for row in ranked.to_dict(orient="records"):
            target = target_map.get(row["targetId"])
            family = row.get("priority_cancer_family")
            if target is None or family not in SUPPORTED_FAMILIES:
                continue
            disease_name = disease_name_map.get(row["diseaseId"]) or family_label(family)
            gene_symbol = target["approvedSymbol"]
            score = float(row["score"])
            strength = association_strength_label(score)
            evidence_text = trim_text(
                f"Open Targets {source_label} direct datasource association for {gene_symbol} and {disease_name}. "
                f"score={score:.3f}; evidence_count={int(row['evidenceCount'])}; datatype={row.get('datatypeId')}."
            )
            record = {
                "id": f"opentargets-{task_suffix}-{row['targetId']}-{row['diseaseId']}",
                "source": "open_targets",
                "task_type": f"{task_suffix}_support_strength",
                "question": f"According to Open Targets {source_label} support, how strong is the gene-disease association between {gene_symbol} and {disease_name}?",
                "answer": strength,
                "answer_type": "categorical",
                "oncology_focus": family_label(family),
                "evidence_text": evidence_text,
                "entities": {
                    "disease": family_label(family),
                    "disease_id": row["diseaseId"],
                    "gene_symbol": gene_symbol,
                    "target_ensembl_id": row["targetId"],
                },
                "provenance": {
                    "source_record_id": f"{row['targetId']}:{row['diseaseId']}",
                    "source_dataset": f"Open Targets {source_label} direct datasource association",
                },
                "metadata": {
                    "priority_cancer_family": family,
                    "association_score": score,
                    "evidence_count": int(row["evidenceCount"]),
                    "datatype_id": row.get("datatypeId"),
                    "scenario_tags": ["mechanism", "molecular", "target"],
                },
            }
            built.append(
                finalize_record(
                    attach_gold_standard(
                        record,
                        build_gold_standard(
                            "categorical",
                            canonical_answer=strength,
                            acceptable_answers=[strength],
                            class_label=strength,
                            evaluable=True,
                        ),
                    ),
                    source_dataset="Open Targets",
                    source_family="target_platform",
                )
            )
        return built

    records.extend(build_datasource_records(intogen, "IntOGen", "intogen"))
    records.extend(build_datasource_records(cgc, "Cancer Gene Census", "cgc"))
    return records


def normalize_signor(
    support_index: dict[tuple[str, str], set[str]],
    pathway_rows: list[dict[str, Any]] | None = None,
    signor_df: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    if signor_df is None:
        signor_df, _stats = materialize_signor_raw()
    if pathway_rows is None:
        pathway_rows, _pathway_stats = materialize_pathwaycommons_raw()
    gene_to_classes, _gene_to_pathways = build_pathway_gene_indexes(pathway_rows)
    records = []
    ranked_rows = signor_df.sort_values(["DIRECT", "PMID"], ascending=[False, True]).to_dict(orient="records")
    for row in ranked_rows:
        entity_a = str(row.get("ENTITYA") or "").strip().upper()
        entity_b = str(row.get("ENTITYB") or "").strip().upper()
        effect = str(row.get("normalized_effect") or "").strip()
        mechanism = str(row.get("normalized_mechanism") or "").strip()
        if not (
            looks_like_gene_symbol(entity_a)
            and looks_like_gene_symbol(entity_b)
            and effect in {"activating_or_upregulating", "inhibitory_or_downregulating"}
        ):
            continue
        families = candidate_causal_families(support_index, entity_a, entity_b)
        if not families:
            continue
        for family in families:
            fit_score, fit_meta = score_direct_causal_fit(
                family=family,
                entity_a=entity_a,
                entity_b=entity_b,
                mechanism=mechanism,
                direct=row.get("DIRECT"),
                sentence=row.get("SENTENCE"),
                pmid=row.get("PMID"),
                support_index=support_index,
                gene_to_classes=gene_to_classes,
            )
            if fit_score < 6:
                continue
            shared_pathway_classes = fit_meta["shared_pathway_classes"]
            if not (
                shared_pathway_classes
                or mechanism in STRONG_SIGNOR_MECHANISMS
                or (fit_meta["source_layers"] and fit_meta["target_layers"])
            ):
                continue
            if mechanism in WEAK_SIGNOR_MECHANISMS and not shared_pathway_classes:
                continue
            record = {
                "id": f"signor-effect-{family}-{entity_a}-{entity_b}-{row.get('SIGNOR_ID') or row.get('PMID')}",
                "source": "signor",
                "task_type": "causal_relation_effect",
                "question": f"In external cancer mechanism resources, what signed causal effect is supported from {entity_a} to {entity_b} for {family_label(family)}-relevant oncology context?",
                "answer": effect,
                "answer_type": "categorical",
                "oncology_focus": family_label(family),
                "evidence_text": trim_text(
                    f"SIGNOR relation {entity_a} -> {entity_b}. effect={row.get('EFFECT')}; mechanism={row.get('MECHANISM')}; direct={row.get('DIRECT')}; "
                    f"shared_pathway_classes={', '.join(shared_pathway_classes) if shared_pathway_classes else 'none'}; sentence={row.get('SENTENCE')}; pmid={row.get('PMID')}."
                ),
                "entities": {
                    "disease": family_label(family),
                    "gene_symbol": entity_a,
                    "target_gene_symbol": entity_b,
                    "gene_symbols": [entity_a, entity_b],
                },
                "provenance": {
                    "source_record_id": row.get("SIGNOR_ID") or f"{entity_a}:{entity_b}:{row.get('PMID')}",
                    "source_dataset": "SIGNOR",
                    "pmid": row.get("PMID"),
                },
                "metadata": {
                    "priority_cancer_family": family,
                    "signor_mechanism": row.get("normalized_mechanism"),
                    "signor_direct": str(row.get("DIRECT") or ""),
                    "causal_support_mode": "direct_supported",
                    "causal_suitability_score": fit_score,
                    "shared_pathway_classes": shared_pathway_classes,
                    "scenario_tags": ["mechanism", "molecular"],
                    "external_support_layers": sorted(set(fit_meta["source_layers"]).union(fit_meta["target_layers"])),
                },
            }
            records.append(
                finalize_record(
                    attach_gold_standard(
                        record,
                        build_gold_standard(
                            "categorical",
                            canonical_answer=effect,
                            acceptable_answers=[effect],
                            class_label=effect,
                            evaluable=True,
                        ),
                    ),
                    source_dataset="SIGNOR",
                    source_family="causal_signaling",
                )
            )
    return records


def normalize_external_indirect_causal(
    support_index: dict[tuple[str, str], set[str]],
    pathway_rows: list[dict[str, Any]],
    signor_df: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    if signor_df is None:
        signor_df, _stats = materialize_signor_raw()
    gene_to_classes, _gene_to_pathways = build_pathway_gene_indexes(pathway_rows)
    edges_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    edges_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in signor_df.to_dict(orient="records"):
        entity_a = str(row.get("ENTITYA") or "").strip().upper()
        entity_b = str(row.get("ENTITYB") or "").strip().upper()
        effect = str(row.get("normalized_effect") or "").strip()
        if not (
            looks_like_gene_symbol(entity_a)
            and looks_like_gene_symbol(entity_b)
            and effect in {"activating_or_upregulating", "inhibitory_or_downregulating"}
            and entity_a != entity_b
        ):
            continue
        edge = dict(row)
        edge["ENTITYA"] = entity_a
        edge["ENTITYB"] = entity_b
        edges_by_source[entity_a].append(edge)
        edges_by_target[entity_b].append(edge)

    best_records: dict[tuple[str, str, str], dict[str, Any]] = {}
    for intermediate_gene, incoming_edges in edges_by_target.items():
        outgoing_edges = edges_by_source.get(intermediate_gene, [])
        if not outgoing_edges:
            continue
        for edge1 in incoming_edges:
            entity_a = edge1["ENTITYA"]
            if entity_a == intermediate_gene:
                continue
            for edge2 in outgoing_edges:
                entity_b = edge2["ENTITYB"]
                if entity_b in {entity_a, intermediate_gene}:
                    continue
                composed = compose_causal_effect(
                    str(edge1.get("normalized_effect") or ""),
                    str(edge2.get("normalized_effect") or ""),
                )
                if not composed:
                    continue
                families = candidate_causal_families(support_index, entity_a, intermediate_gene, entity_b)
                for family in families:
                    fit_score, fit_meta = score_indirect_causal_fit(
                        family=family,
                        entity_a=entity_a,
                        intermediate_gene=intermediate_gene,
                        entity_b=entity_b,
                        edge1=edge1,
                        edge2=edge2,
                        support_index=support_index,
                        gene_to_classes=gene_to_classes,
                    )
                    if fit_score < 8:
                        continue
                    shared_pathway_classes = fit_meta["shared_pathway_classes"]
                    if not shared_pathway_classes:
                        continue
                    union_support_layers = sorted(
                        {
                            layer
                            for layers in fit_meta["family_support"].values()
                            for layer in layers
                        }
                    )
                    record = {
                        "id": f"external-causal-chain-{family}-{entity_a}-{intermediate_gene}-{entity_b}",
                        "source": "external_multisource",
                        "task_type": "causal_relation_effect",
                        "question": (
                            f"In external cancer mechanism resources, what signed causal effect is supported from {entity_a} to {entity_b} "
                            f"for {family_label(family)}-relevant pathway context?"
                        ),
                        "answer": composed,
                        "answer_type": "categorical",
                        "oncology_focus": family_label(family),
                        "evidence_text": trim_text(
                            f"Indirect external mechanism chain for {family_label(family)}: "
                            f"{entity_a} --{edge1.get('EFFECT')}--> {intermediate_gene} (mechanism={edge1.get('MECHANISM')}; direct={edge1.get('DIRECT')}; pmid={edge1.get('PMID')}) ; "
                            f"{intermediate_gene} --{edge2.get('EFFECT')}--> {entity_b} (mechanism={edge2.get('MECHANISM')}; direct={edge2.get('DIRECT')}; pmid={edge2.get('PMID')}). "
                            f"Shared pathway classes={', '.join(shared_pathway_classes)}; external support layers={', '.join(union_support_layers) if union_support_layers else 'none'}."
                        ),
                        "entities": {
                            "disease": family_label(family),
                            "gene_symbol": entity_a,
                            "target_gene_symbol": entity_b,
                            "gene_symbols": [entity_a, intermediate_gene, entity_b],
                        },
                        "provenance": {
                            "source_record_id": f"{edge1.get('SIGNOR_ID') or edge1.get('PMID')}::{edge2.get('SIGNOR_ID') or edge2.get('PMID')}",
                            "source_dataset": "SIGNOR/Pathway Commons/IntOGen/Open Targets",
                            "pmid": [edge1.get("PMID"), edge2.get("PMID")],
                        },
                        "metadata": {
                            "priority_cancer_family": family,
                            "causal_support_mode": "indirect_path_supported",
                            "causal_intermediate_gene": intermediate_gene,
                            "causal_path_length": 2,
                            "causal_suitability_score": fit_score,
                            "shared_pathway_classes": shared_pathway_classes,
                            "external_support_layers": union_support_layers,
                            "scenario_tags": ["mechanism", "molecular", "target"],
                        },
                    }
                    finalized = finalize_record(
                        attach_gold_standard(
                            record,
                            build_gold_standard(
                                "categorical",
                                canonical_answer=composed,
                                acceptable_answers=[composed],
                                class_label=composed,
                                evaluable=True,
                            ),
                        ),
                        source_dataset="External Multisource",
                        source_family="causal_mechanism_bundle",
                    )
                    key = (family, entity_a, entity_b)
                    if key not in best_records or fit_score > int((best_records[key].get("metadata") or {}).get("causal_suitability_score") or 0):
                        best_records[key] = finalized
    return sorted(best_records.values(), key=lambda row: row["id"])


def normalize_pathwaycommons(
    support_index: dict[tuple[str, str], set[str]],
    pathway_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if pathway_rows is None:
        pathway_rows, _stats = materialize_pathwaycommons_raw()
    gene_to_memberships = defaultdict(list)
    for row in pathway_rows:
        for gene in row["genes"]:
            gene_to_memberships[gene].append(row)
    records = []
    for (family, gene), layers in sorted(support_index.items()):
        memberships = sorted(
            gene_to_memberships.get(gene, []),
            key=lambda row: (row["pathway_class"], row["pathway_name"]),
        )
        if family not in SUPPORTED_FAMILIES or not memberships:
            continue
        membership = memberships[0]
        pathway_class = membership["pathway_class"]
        record = {
            "id": f"pathwaycommons-class-{family}-{gene}-{pathway_class}",
            "source": "pathway_commons",
            "task_type": "pathway_class_membership",
            "question": f"In external pathway resources, which cancer mechanism pathway class best matches {gene} in {family_label(family)} context?",
            "answer": pathway_class,
            "answer_type": "categorical",
            "oncology_focus": family_label(family),
            "evidence_text": trim_text(
                f"Pathway Commons membership for {gene}: pathway={membership['pathway_name']}; pathway_source={membership['pathway_source']}; pathway_class={pathway_class}."
            ),
            "entities": {
                "disease": family_label(family),
                "gene_symbol": gene,
                "pathway_name": membership["pathway_name"],
            },
            "provenance": {
                "source_record_id": f"{gene}:{membership['pathway_name']}",
                "source_dataset": "Pathway Commons",
            },
            "metadata": {
                "priority_cancer_family": family,
                "pathway_name": membership["pathway_name"],
                "pathway_source": membership["pathway_source"],
                "scenario_tags": ["mechanism", "molecular"],
                "external_support_layers": sorted(layers),
            },
        }
        records.append(
            finalize_record(
                attach_gold_standard(
                    record,
                    build_gold_standard(
                        "categorical",
                        canonical_answer=pathway_class,
                        acceptable_answers=[pathway_class],
                        class_label=pathway_class,
                        evaluable=True,
                    ),
                ),
                source_dataset="Pathway Commons",
                source_family="pathway_membership",
            )
        )
    return records


def normalize_cross_source_support_bundle(
    support_index: dict[tuple[str, str], set[str]],
    signor_df: pd.DataFrame,
    pathway_rows: list[dict[str, Any]],
    graph_driverlike_index: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    signor_genes = set()
    for row in signor_df.to_dict(orient="records"):
        signor_genes.add(str(row.get("ENTITYA") or "").strip().upper())
        signor_genes.add(str(row.get("ENTITYB") or "").strip().upper())
    pathway_genes = {gene for row in pathway_rows for gene in row["genes"]}
    graph_driverlike_index = graph_driverlike_index or {}
    records = []
    for (family, gene), layers in sorted(support_index.items()):
        bundle_layers = set()
        if "intogen_driver" in layers:
            bundle_layers.add("driver")
        if any(layer.startswith("open_targets_") for layer in layers):
            bundle_layers.add("target_association")
        if gene in signor_genes:
            bundle_layers.add("causal_relation")
        if gene in pathway_genes:
            bundle_layers.add("pathway_context")
        if len(bundle_layers) < 3 or "target_association" not in bundle_layers:
            continue
        graph_driver_summary = graph_driverlike_index.get((family, gene))
        graph_driverlike = has_graph_driverlike_signal(graph_driver_summary)
        if "driver" not in bundle_layers and graph_driverlike:
            continue
        if "driver" in bundle_layers and ("pathway_context" not in bundle_layers or not graph_driverlike):
            continue
        answer = "+".join(sorted(bundle_layers))
        record = {
            "id": f"external-support-bundle-{family}-{gene}",
            "source": "external_multisource",
            "task_type": "cross_source_support_bundle",
            "question": f"Which external evidence bundle supports {gene} in {family_label(family)} mechanism-focused oncology resources?",
            "answer": answer,
            "answer_type": "categorical",
            "oncology_focus": family_label(family),
            "evidence_text": trim_text(
                f"External support layers for {gene} in {family_label(family)} include {', '.join(sorted(bundle_layers))}."
            ),
            "entities": {
                "disease": family_label(family),
                "gene_symbol": gene,
            },
            "provenance": {
                "source_record_id": f"{family}:{gene}",
                "source_dataset": "IntOGen/Open Targets/SIGNOR/Pathway Commons",
            },
            "metadata": {
                "priority_cancer_family": family,
                "external_support_layers": sorted(bundle_layers),
                "scenario_tags": ["mechanism", "molecular", "target"],
                "qa_evidence_function": "summary",
                "graph_alignment_policy": "bundle_driver_strict_v1",
                "graph_driverlike_signal": graph_driverlike,
                "graph_driverlike_strong_count": int((graph_driver_summary or {}).get("strong_count") or 0),
                "graph_driverlike_weak_count": int((graph_driver_summary or {}).get("weak_count") or 0),
                "graph_driverlike_terms": list((graph_driver_summary or {}).get("matched_terms") or []),
                "graph_driverlike_examples": list((graph_driver_summary or {}).get("titles") or []),
            },
        }
        records.append(
            finalize_record(
                attach_gold_standard(
                    record,
                    build_gold_standard(
                        "categorical",
                        canonical_answer=answer,
                        acceptable_answers=[answer],
                        class_label=answer,
                        evaluable=True,
                    ),
                ),
                source_dataset="External Multisource",
                source_family="cross_source_support",
            )
        )
    return records


def normalize_mechanism_trace_intermediate(indirect_causal_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for row in indirect_causal_records:
        metadata = row.get("metadata") or {}
        entities = row.get("entities") or {}
        intermediate_gene = str(metadata.get("causal_intermediate_gene") or "").strip().upper()
        source_gene = str(entities.get("gene_symbol") or "").strip().upper()
        target_gene = str(entities.get("target_gene_symbol") or "").strip().upper()
        if not (looks_like_gene_symbol(source_gene) and looks_like_gene_symbol(target_gene) and looks_like_gene_symbol(intermediate_gene)):
            continue
        record = {
            "id": f"mechanism-trace-intermediate-{row['id']}",
            "source": "external_multisource",
            "task_type": "mechanism_trace_intermediate_gene",
            "question": (
                f"In external mechanism-trace resources, which intermediate gene links {source_gene} to {target_gene} "
                f"in {row.get('oncology_focus')} pathway context?"
            ),
            "answer": intermediate_gene,
            "answer_type": "entity",
            "oncology_focus": row.get("oncology_focus"),
            "evidence_text": row.get("evidence_text"),
            "entities": {
                "disease": row.get("oncology_focus"),
                "gene_symbol": source_gene,
                "target_gene_symbol": target_gene,
                "gene_symbols": [source_gene, intermediate_gene, target_gene],
            },
            "provenance": dict(row.get("provenance") or {}),
            "metadata": {
                **metadata,
                "scenario_tags": sorted(set((metadata.get("scenario_tags") or []) + ["mechanism", "trace"])),
                "qa_evidence_function": "reasoning",
            },
        }
        records.append(
            finalize_record(
                attach_gold_standard(
                    record,
                    build_gold_standard(
                        "entity",
                        canonical_answer=intermediate_gene,
                        acceptable_answers=[intermediate_gene],
                        entity_ids={},
                        evaluable=True,
                    ),
                ),
                source_dataset="External Multisource",
                source_family="mechanism_trace",
            )
        )
    return records


def normalize_cross_source_corroboration(
    support_index: dict[tuple[str, str], set[str]],
    signor_df: pd.DataFrame,
    pathway_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    signor_genes = set()
    for row in signor_df.to_dict(orient="records"):
        signor_genes.add(str(row.get("ENTITYA") or "").strip().upper())
        signor_genes.add(str(row.get("ENTITYB") or "").strip().upper())
    pathway_genes = {gene for row in pathway_rows for gene in row["genes"]}
    records = []
    for (family, gene), layers in sorted(support_index.items()):
        bundle_layers = set()
        if "intogen_driver" in layers:
            bundle_layers.add("driver")
        if any(layer.startswith("open_targets_") for layer in layers):
            bundle_layers.add("target_association")
        if gene in signor_genes:
            bundle_layers.add("causal_relation")
        if gene in pathway_genes:
            bundle_layers.add("pathway_context")
        has_target = "target_association" in bundle_layers
        has_mechanism_context = bool(bundle_layers.intersection({"causal_relation", "pathway_context"}))
        if not has_target:
            continue
        answer = "yes" if has_mechanism_context else "no"
        record = {
            "id": f"cross-source-corroboration-{family}-{gene}",
            "source": "external_multisource",
            "task_type": "cross_source_corroboration_status",
            "question": (
                f"Across external oncology resources, is there corroborated support for {gene} in {family_label(family)} "
                f"that spans both target-association evidence and mechanism or pathway context?"
            ),
            "answer": answer,
            "answer_type": "boolean",
            "oncology_focus": family_label(family),
            "evidence_text": trim_text(
                f"External support layers for {gene} in {family_label(family)} include {', '.join(sorted(bundle_layers)) if bundle_layers else 'none'}."
            ),
            "entities": {
                "disease": family_label(family),
                "gene_symbol": gene,
            },
            "provenance": {
                "source_record_id": f"{family}:{gene}",
                "source_dataset": "IntOGen/Open Targets/SIGNOR/Pathway Commons",
            },
            "metadata": {
                "priority_cancer_family": family,
                "external_support_layers": sorted(bundle_layers),
                "scenario_tags": ["mechanism", "molecular", "source_aware"],
                "qa_evidence_function": "reasoning",
            },
        }
        records.append(
            finalize_record(
                attach_gold_standard(
                    record,
                    build_gold_standard(
                        "boolean",
                        canonical_answer=answer,
                        acceptable_answers=[answer],
                        binary_answer=(answer == "yes"),
                        evaluable=True,
                    ),
                ),
                source_dataset="External Multisource",
                source_family="cross_source_corroboration",
            )
        )
    return records


def deterministic_take(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (row.get("metadata", {}).get("priority_cancer_family"), row["id"]))
    if len(ordered) <= limit:
        return ordered
    selected = []
    buckets = defaultdict(list)
    for row in ordered:
        buckets[(row.get("metadata") or {}).get("priority_cancer_family")].append(row)
    while len(selected) < limit:
        progressed = False
        for family in sorted(buckets):
            if buckets[family] and len(selected) < limit:
                selected.append(buckets[family].pop(0))
                progressed = True
        if not progressed:
            break
    return selected[:limit]


def deterministic_take_balanced_binary(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    yes_rows = [row for row in rows if norm_text(str(row.get("answer") or "")) == "yes"]
    no_rows = [row for row in rows if norm_text(str(row.get("answer") or "")) == "no"]
    selected = deterministic_take(yes_rows, limit // 2) + deterministic_take(no_rows, limit // 2)
    if len(selected) < limit:
        selected_ids = {row["id"] for row in selected}
        remaining = [row for row in rows if row["id"] not in selected_ids]
        selected.extend(deterministic_take(remaining, limit - len(selected)))
    deduped = {row["id"]: row for row in selected}
    return sorted(deduped.values(), key=lambda row: row["id"])[:limit]


def causal_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    entities = row.get("entities") or {}
    metadata = row.get("metadata") or {}
    return (
        row.get("source"),
        metadata.get("causal_support_mode"),
        entities.get("gene_symbol"),
        metadata.get("causal_intermediate_gene"),
        entities.get("target_gene_symbol"),
        row.get("answer"),
    )


def deterministic_take_diverse_causal(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            -int((row.get("metadata") or {}).get("causal_suitability_score") or 0),
            (row.get("metadata") or {}).get("priority_cancer_family") or "",
            row["id"],
        ),
    )
    unique_rows = []
    duplicate_rows = []
    seen_signatures = set()
    for row in ordered:
        signature = causal_signature(row)
        if signature not in seen_signatures:
            seen_signatures.add(signature)
            unique_rows.append(row)
        else:
            duplicate_rows.append(row)
    selected = deterministic_take(unique_rows, limit)
    if len(selected) >= limit:
        return selected[:limit]
    remaining_ids = {row["id"] for row in selected}
    fill_pool = [row for row in duplicate_rows if row["id"] not in remaining_ids]
    selected.extend(deterministic_take(fill_pool, limit - len(selected)))
    return selected[:limit]


def apply_source_caps(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_bucket = defaultdict(list)
    for row in records:
        bucket = row["source"]
        if row["source"] == "intogen":
            bucket = f"intogen::{row['task_type']}"
        elif row["source"] == "open_targets":
            bucket = f"open_targets::{row['task_type']}"
        elif row["source"] == "bioasq":
            bucket = f"bioasq::{(row.get('metadata') or {}).get('bioasq_type')}"
        elif row["source"] in {"signor", "pathway_commons", "external_multisource"}:
            bucket = f"{row['source']}::{row['task_type']}"
        by_bucket[bucket].append(row)

    caps = {
        "pubmedqa": 80,
        "evidence_inference_qa": 20,
        "bioasq::yesno": 24,
        "bioasq::factoid": 24,
        "bioasq::list": 24,
        "intogen::driver_role_assignment": 90,
        "intogen::driver_multimethod_support": 60,
        "open_targets::target_disease_association_strength": 70,
        "open_targets::intogen_support_strength": 40,
        "open_targets::cgc_support_strength": 46,
        "signor::causal_relation_effect": 12,
        "external_multisource::causal_relation_effect": 12,
        "external_multisource::mechanism_trace_intermediate_gene": 12,
        "external_multisource::cross_source_corroboration_status": 12,
        "pathway_commons::pathway_class_membership": 70,
        "external_multisource::cross_source_support_bundle": 70,
    }

    selected = []
    for bucket, rows in by_bucket.items():
        if bucket in {"signor::causal_relation_effect", "external_multisource::causal_relation_effect"}:
            selected.extend(deterministic_take_diverse_causal(rows, caps.get(bucket, len(rows))))
        elif bucket == "external_multisource::cross_source_corroboration_status":
            selected.extend(deterministic_take_balanced_binary(rows, caps.get(bucket, len(rows))))
        else:
            selected.extend(deterministic_take(rows, caps.get(bucket, len(rows))))
    deduped = {row["id"]: row for row in selected}
    return sorted(deduped.values(), key=lambda row: row["id"])


def build_pilot(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_source = defaultdict(list)
    for row in records:
        by_source[row["source"]].append(row)
    pilot = []
    source_limits = {
        "pubmedqa": 16,
        "bioasq": 16,
        "evidence_inference_qa": 8,
        "intogen": 30,
        "open_targets": 30,
        "signor": 20,
        "pathway_commons": 20,
        "external_multisource": 20,
    }
    for source, limit in source_limits.items():
        pilot.extend(deterministic_take(by_source.get(source, []), limit))
    deduped = {row["id"]: row for row in pilot}
    return sorted(deduped.values(), key=lambda row: row["id"])


def build_stats(records: list[dict[str, Any]], pilot: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total_records": len(records),
        "machine_evaluable_records": sum(1 for row in records if row.get("is_machine_evaluable")),
        "by_source": dict(Counter(row["source"] for row in records)),
        "by_task_type": dict(Counter(row["task_type"] for row in records)),
        "by_answer_format": dict(Counter((row.get("gold_standard") or {}).get("answer_format") for row in records)),
        "by_priority_cancer_family": dict(Counter((row.get("metadata") or {}).get("priority_cancer_family") for row in records)),
        "by_scenario_tag": dict(Counter(tag for row in records for tag in (row.get("metadata") or {}).get("scenario_tags") or [])),
        "by_qa_evidence_function": dict(Counter((row.get("metadata") or {}).get("qa_evidence_function") for row in records)),
        "by_source_dataset": dict(Counter((row.get("metadata") or {}).get("source_dataset") for row in records)),
        "pilot_subset": {
            "count": len(pilot),
            "by_source": dict(Counter(row["source"] for row in pilot)),
            "by_answer_format": dict(Counter((row.get("gold_standard") or {}).get("answer_format") for row in pilot)),
            "by_priority_cancer_family": dict(Counter((row.get("metadata") or {}).get("priority_cancer_family") for row in pilot)),
        },
    }


def main() -> None:
    intogen_df, _intogen_stats = materialize_intogen_raw()
    open_targets_payload, _open_targets_stats = materialize_open_targets_raw()
    signor_df, _signor_stats = materialize_signor_raw()
    pathway_rows, _pathway_stats = materialize_pathwaycommons_raw()
    support_index = build_external_support_index(intogen_df, open_targets_payload)
    graph_driverlike_index = build_graph_driverlike_index()

    records = []
    records.extend(normalize_pubmedqa())
    records.extend(normalize_evidence_inference())
    records.extend(normalize_bioasq())
    records.extend(normalize_intogen(intogen_df))
    records.extend(normalize_open_targets(open_targets_payload))
    records.extend(normalize_signor(support_index, pathway_rows, signor_df))
    indirect_causal_records = normalize_external_indirect_causal(support_index, pathway_rows, signor_df)
    records.extend(indirect_causal_records)
    records.extend(normalize_mechanism_trace_intermediate(indirect_causal_records))
    records.extend(normalize_pathwaycommons(support_index, pathway_rows))
    records.extend(normalize_cross_source_support_bundle(support_index, signor_df, pathway_rows, graph_driverlike_index))
    records.extend(normalize_cross_source_corroboration(support_index, signor_df, pathway_rows))
    deduped = {}
    for row in records:
        gold = row.get("gold_standard") or {}
        canonical = norm_text(str(gold.get("canonical_answer") or row.get("answer") or ""))
        key = (row.get("source"), norm_text(row.get("question")), canonical)
        deduped.setdefault(key, row)
    capped = apply_source_caps(list(deduped.values()))
    machine_evaluable = [row for row in capped if row.get("is_machine_evaluable")]
    pilot = build_pilot(machine_evaluable)
    write_jsonl(EXTERNAL_BENCHMARK_PATH, capped)
    write_jsonl(EXTERNAL_MACHINE_PATH, machine_evaluable)
    write_jsonl(EXTERNAL_PILOT_PATH, pilot)
    write_json(EXTERNAL_STATS_PATH, build_stats(capped, pilot))


if __name__ == "__main__":
    print("Benchmark-construction helper. Run scripts/benchmark/build_reasoning_benchmark_formal_v2.py --help for the released 571-question benchmark.")

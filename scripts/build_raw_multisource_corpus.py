from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import pandas as pd

from build_source_overlap_selection import DEPMAP_ONCOTREE_TO_FAMILY, FAMILY_DESCRIPTIONS
from extract_depmap_dependency_evidence import (
    load_binary_omics_matrix,
    load_dependency_matrices,
    load_dense_omics_matrix,
    load_gdsc_drug_response,
    load_global_signatures,
    load_model_metadata,
    load_molecular_subtypes,
)
from extract_tcga_atomic_evidence import PRIORITY_COHORTS, find_file, open_text
from prepare_rag_knowledge_sources import (
    chunk_text,
    doc_record,
    edge_record,
    extract_pdf_text,
    stable_hash,
    unique_preserve,
)


ROOT = Path(__file__).resolve().parents[1]
from knowledge_build_paths import LINKEDOMICS_ROOT, DEPMAP_ROOT, PAPERS_ROOT, source_relpath
PUBMED_GRAPH_PATH = ROOT / "kg" / "aligned" / "pubmed_aligned_normalized_latest.json"

TCGA_OUTPUT_DIR = ROOT / "kg" / "tcga-multiomics"
DEPMAP_OUTPUT_DIR = ROOT / "kg" / "depmap"
RAW_OUTPUT_DIR = ROOT / "kg" / "raw-multisource"

TCGA_SOURCE = "raw_tcga_multiomics"
DEPMAP_SOURCE = "raw_depmap"
PAPER_SOURCE = "raw_paper_fulltext"

TARGET_FAMILIES = {
    "AML",
    "BRCA",
    "CRC",
    "GASTROESOPH",
    "GLIOMA",
    "HCC",
    "HNSC",
    "NSCLC",
    "OVARIAN",
    "PAAD",
    "RCC",
}
PAPER_FAMILIES = {"HCC", "CRC"}

TCGA_BLOCK_SIZE = 80
DEP_BLOCK_SIZE = 120
MODEL_BLOCK_SIZE = 40
PAPER_CHUNK_SIZE = 1400
PAPER_CHUNK_OVERLAP = 250
TOP_EXAMPLE_COUNT = 3

TCGA_FILE_PATTERNS = {
    "expression": lambda name: "RNAseq" in name and "RSEM_log2.cct.gz" in name,
    "mutation": lambda name: "Mutation" in name and "Gene__Firehose_MutSig2CV.cbt" in name,
    "scna": lambda name: "SCNA" in name and "Gene__Firehose_GISTIC2.cct.gz" in name,
    "methylation": lambda name: "Methylation" in name and "__Gene__" in name and "Prepocessor.cct.gz" in name,
    "clinical": lambda name: "Clinical" in name and name.endswith(".tsi"),
    "rppa_gene": lambda name: "RPPA" in name and "__Gene__" in name,
    "rppa_analyte": lambda name: "RPPA" in name and "__Analyte__" in name,
}

PAPER_DIRS = {
    "HCC": PAPERS_ROOT / "hcc_papers",
    "CRC": PAPERS_ROOT / "crc_papers",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def relpath(path: Path) -> str:
    return source_relpath(path)


def disease_tags(family: str) -> list[str]:
    description = FAMILY_DESCRIPTIONS.get(family, family)
    aliases = {
        "HCC": ["HCC", "hepatocellular carcinoma", "liver hepatocellular carcinoma", "LIHC"],
        "CRC": ["CRC", "colorectal cancer", "colorectal adenocarcinoma", "COADREAD", "COAD", "READ"],
        "NSCLC": ["NSCLC", "non-small cell lung cancer", "LUAD", "LUSC"],
        "BRCA": ["BRCA", "breast carcinoma", "breast cancer"],
        "GLIOMA": ["GLIOMA", "glioma", "glioblastoma", "GBM"],
        "OVARIAN": ["OVARIAN", "ovarian carcinoma", "ovarian cancer", "OV"],
        "RCC": ["RCC", "renal cell carcinoma", "KIRC"],
        "AML": ["AML", "acute myeloid leukemia", "LAML"],
        "PAAD": ["PAAD", "pancreatic adenocarcinoma", "pancreatic cancer"],
        "HNSC": ["HNSC", "head and neck squamous cell carcinoma"],
        "GASTROESOPH": ["GASTROESOPH", "esophagogastric carcinoma", "stomach adenocarcinoma", "STAD"],
    }
    return unique_preserve([family, description, *aliases.get(family, [])])


def coverage_groups_for_family(family: str) -> list[str]:
    groups = []
    if family in TARGET_FAMILIES:
        groups.append("tcga_depmap_multi_cancer_core")
    if family in PAPER_FAMILIES:
        groups.append("tcga_depmap_pubmed_extension")
    return groups


def safe_float(value: Any) -> float | None:
    try:
        if value in ("", None, "NA"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_numeric_pairs(pairs: list[tuple[str, float]]) -> dict[str, Any]:
    if not pairs:
        return {"count": 0}
    values = [value for _, value in pairs]
    ordered = sorted(pairs, key=lambda item: item[1])
    low_examples = [f"{sid}:{round(val, 4)}" for sid, val in ordered[:TOP_EXAMPLE_COUNT]]
    high_examples = [f"{sid}:{round(val, 4)}" for sid, val in ordered[-TOP_EXAMPLE_COUNT:]]
    return {
        "count": len(values),
        "mean": round(mean(values), 4),
        "median": round(median(values), 4),
        "q1": round(quantile(values, 0.25), 4),
        "q3": round(quantile(values, 0.75), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "low_examples": low_examples,
        "high_examples": list(reversed(high_examples)),
    }


def summarize_binary_pairs(pairs: list[tuple[str, float]]) -> dict[str, Any]:
    if not pairs:
        return {"count": 0}
    positive = [(sid, value) for sid, value in pairs if value > 0]
    return {
        "count": len(pairs),
        "positive_count": len(positive),
        "positive_fraction": round(len(positive) / len(pairs), 4),
        "positive_examples": [sid for sid, _ in positive[:8]],
    }


def chunk_lines(lines: list[str], size: int) -> list[list[str]]:
    return [lines[index : index + size] for index in range(0, len(lines), size)]


def make_parent_doc(
    *,
    source_name: str,
    parent_id: str,
    title: str,
    family: str,
    entity_names: list[str],
    text_lines: list[str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return doc_record(
        doc_id=parent_id,
        knowledge_source=source_name,
        title=title,
        text="\n".join(line for line in text_lines if line),
        document_level="parent",
        parent_id=None,
        disease_tags=disease_tags(family),
        entity_names=entity_names,
        entity_ids=[],
        metadata=metadata,
    )


def make_child_doc(
    *,
    source_name: str,
    doc_id: str,
    title: str,
    family: str,
    parent_id: str,
    entity_names: list[str],
    text_lines: list[str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return doc_record(
        doc_id=doc_id,
        knowledge_source=source_name,
        title=title,
        text="\n".join(line for line in text_lines if line),
        document_level="chunk",
        parent_id=parent_id,
        disease_tags=disease_tags(family),
        entity_names=entity_names,
        entity_ids=[],
        metadata=metadata,
    )


def parse_tcga_matrix_rows(path: Path, kind: str) -> tuple[list[str], list[tuple[str, str]], int]:
    with open_text(path) as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        sample_ids = header[1:]
        rows: list[tuple[str, str]] = []
        for row in reader:
            if not row:
                continue
            feature = row[0].strip()
            values = [safe_float(value) for value in row[1:]]
            pairs = [(sample_ids[idx], value) for idx, value in enumerate(values) if value is not None]
            if kind == "mutation":
                summary = summarize_binary_pairs(pairs)
                line = (
                    f"{feature} | mutated_samples={summary['positive_count']}/{summary['count']} "
                    f"({summary['positive_fraction']}) | positive_examples={', '.join(summary['positive_examples']) or 'none'}"
                )
            else:
                summary = summarize_numeric_pairs(pairs)
                line = (
                    f"{feature} | n={summary['count']} | mean={summary['mean']} | median={summary['median']} "
                    f"| q1={summary['q1']} | q3={summary['q3']} | min={summary['min']} | max={summary['max']} "
                    f"| low={'; '.join(summary['low_examples'])} | high={'; '.join(summary['high_examples'])}"
                )
            rows.append((feature, line))
    return sample_ids, rows, len(rows)


def summarize_clinical_row(attribute: str, values: list[str]) -> str:
    cleaned = [value for value in values if value and value != "NA"]
    numeric = [safe_float(value) for value in cleaned]
    numeric_clean = [value for value in numeric if value is not None]
    if numeric_clean and len(numeric_clean) >= max(3, len(cleaned) // 2):
        summary = summarize_numeric_pairs([(str(index + 1), value) for index, value in enumerate(numeric_clean)])
        return (
            f"{attribute} | n={summary['count']} | mean={summary['mean']} | median={summary['median']} "
            f"| q1={summary['q1']} | q3={summary['q3']} | min={summary['min']} | max={summary['max']}"
        )
    counts = Counter(cleaned)
    top_levels = ", ".join(f"{name}:{count}" for name, count in counts.most_common(8))
    return f"{attribute} | non_missing={len(cleaned)} | category_counts={top_levels}"


def parse_tcga_clinical_rows(path: Path) -> tuple[list[str], list[tuple[str, str]], int]:
    with open_text(path) as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        sample_ids = header[1:]
        rows: list[tuple[str, str]] = []
        for row in reader:
            if not row:
                continue
            attribute = row[0].strip()
            rows.append((attribute, summarize_clinical_row(attribute, row[1:])))
    return sample_ids, rows, len(rows)


def locate_tcga_files(cohort: str) -> dict[str, Path]:
    cohort_dir = LINKEDOMICS_ROOT / cohort
    located: dict[str, Path] = {}
    for kind, predicate in TCGA_FILE_PATTERNS.items():
        path = find_file(cohort_dir, lambda name, pred=predicate: pred(name))
        if path:
            located[kind] = path
    return located


def build_tcga_corpus() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    stats_by_kind = Counter()
    stats_by_family = Counter()

    for config in PRIORITY_COHORTS:
        cohort = config["cohort"]
        family = config["family"]
        if family not in TARGET_FAMILIES:
            continue
        located = locate_tcga_files(cohort)
        for kind, path in located.items():
            if kind == "clinical":
                sample_ids, rows, row_count = parse_tcga_clinical_rows(path)
                entity_names = [cohort, family, "clinical", *disease_tags(family)]
            else:
                sample_ids, rows, row_count = parse_tcga_matrix_rows(path, kind)
                entity_names = [cohort, family, kind]
            parent_id = f"TCGARAW-{stable_hash(f'{cohort}|{kind}|{relpath(path)}')}"
            parent = make_parent_doc(
                source_name=TCGA_SOURCE,
                parent_id=parent_id,
                title=f"{cohort} | {kind} raw table",
                family=family,
                entity_names=entity_names,
                text_lines=[
                    "Raw LinkedOmics / TCGA table constrained only by same-source cohort and file coverage.",
                    f"Cohort: {cohort}",
                    f"Disease family: {family}",
                    f"File kind: {kind}",
                    f"Source file: {relpath(path)}",
                    f"Sample count: {len(sample_ids)}",
                    f"Row count: {row_count}",
                    f"Value orientation: rows are features/attributes, columns are TCGA samples.",
                ],
                metadata={
                    "source_family": "tcga_linkedomics",
                    "coverage_groups": coverage_groups_for_family(family),
                    "cohort": cohort,
                    "file_kind": kind,
                    "source_file": relpath(path),
                    "match_type": "exact_matched",
                },
            )
            docs.append(parent)
            for block_index, block in enumerate(chunk_lines([line for _, line in rows], TCGA_BLOCK_SIZE), start=1):
                block_features = [feature for feature, _ in rows[(block_index - 1) * TCGA_BLOCK_SIZE : block_index * TCGA_BLOCK_SIZE]]
                child_id = f"{parent_id}-BLK{block_index:04d}"
                docs.append(
                    make_child_doc(
                        source_name=TCGA_SOURCE,
                        doc_id=child_id,
                        title=f"{cohort} | {kind} rows {block_index}",
                        family=family,
                        parent_id=parent_id,
                        entity_names=block_features[:32] + [cohort, family],
                        text_lines=[
                            f"TCGA raw block for cohort {cohort}, file kind {kind}.",
                            f"Source file: {relpath(path)}",
                            f"Features in this block: {', '.join(block_features[:16])}" + (" ..." if len(block_features) > 16 else ""),
                            *block,
                        ],
                        metadata={
                            "source_family": "tcga_linkedomics",
                            "coverage_groups": coverage_groups_for_family(family),
                            "cohort": cohort,
                            "file_kind": kind,
                            "source_file": relpath(path),
                            "feature_names": block_features,
                            "match_type": "exact_matched",
                        },
                    )
                )
                edges.append(edge_record(parent_id, child_id, "HAS_BLOCK", 1.0))
                manifest.append(
                    {
                        "doc_id": child_id,
                        "source_family": "tcga_linkedomics",
                        "disease_family": family,
                        "coverage_groups": coverage_groups_for_family(family),
                        "source_files": [relpath(path)],
                        "cohort_or_screen": cohort,
                        "feature_layer": kind,
                        "comparison_context": None,
                        "entity_anchors": block_features[:32],
                        "match_type": "exact_matched",
                        "raw_unit_type": "table_row_block",
                    }
                )
            stats_by_kind[kind] += row_count
            stats_by_family[family] += row_count

    return docs, edges, manifest, {"row_count_by_kind": dict(sorted(stats_by_kind.items())), "row_count_by_family": dict(sorted(stats_by_family.items()))}


def strip_gene_name(column: str) -> str:
    return column.split(" (", 1)[0].strip()


def summarize_series(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"count": 0}
    ordered = values.sort_values()
    low_examples = [f"{idx}:{round(val, 4)}" for idx, val in ordered.head(TOP_EXAMPLE_COUNT).items()]
    high_examples = [f"{idx}:{round(val, 4)}" for idx, val in ordered.tail(TOP_EXAMPLE_COUNT).items()][::-1]
    return {
        "count": int(values.shape[0]),
        "mean": round(float(values.mean()), 4),
        "median": round(float(values.median()), 4),
        "q1": round(float(values.quantile(0.25)), 4),
        "q3": round(float(values.quantile(0.75)), 4),
        "min": round(float(values.min()), 4),
        "max": round(float(values.max()), 4),
        "low_examples": low_examples,
        "high_examples": high_examples,
    }


def summarize_binary_series(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").fillna(0)
    positive = values[values > 0]
    return {
        "count": int(values.shape[0]),
        "positive_count": int(positive.shape[0]),
        "positive_fraction": round(float(positive.shape[0] / values.shape[0]), 4) if values.shape[0] else 0.0,
        "positive_examples": [str(index) for index in positive.index[:8]],
    }


def build_depmap_matrix_blocks(
    *,
    source_name: str,
    family: str,
    kind: str,
    frame: pd.DataFrame,
    parent_id: str,
    source_file: str,
    docs: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
) -> int:
    lines: list[tuple[str, str]] = []
    for column in frame.columns:
        symbol = strip_gene_name(str(column))
        if kind in {"hotspot_mutation", "damaging_mutation"}:
            summary = summarize_binary_series(frame[column])
            line = (
                f"{symbol} | positive_models={summary['positive_count']}/{summary['count']} "
                f"({summary['positive_fraction']}) | example_models={', '.join(summary['positive_examples']) or 'none'}"
            )
        else:
            summary = summarize_series(frame[column])
            if summary["count"] == 0:
                continue
            line = (
                f"{symbol} | n={summary['count']} | mean={summary['mean']} | median={summary['median']} "
                f"| q1={summary['q1']} | q3={summary['q3']} | min={summary['min']} | max={summary['max']} "
                f"| low={'; '.join(summary['low_examples'])} | high={'; '.join(summary['high_examples'])}"
            )
        lines.append((symbol, line))
    for block_index, block in enumerate(chunk_lines([line for _, line in lines], DEP_BLOCK_SIZE), start=1):
        block_features = [feature for feature, _ in lines[(block_index - 1) * DEP_BLOCK_SIZE : block_index * DEP_BLOCK_SIZE]]
        child_id = f"{parent_id}-BLK{block_index:04d}"
        docs.append(
            make_child_doc(
                source_name=source_name,
                doc_id=child_id,
                title=f"{family} | {kind} block {block_index}",
                family=family,
                parent_id=parent_id,
                entity_names=block_features[:40] + [family, kind],
                text_lines=[
                    f"DepMap raw block for family {family}, file kind {kind}.",
                    f"Source file: {source_file}",
                    f"Feature count in block: {len(block_features)}",
                    *block,
                ],
                metadata={
                    "source_family": "depmap",
                    "coverage_groups": coverage_groups_for_family(family),
                    "family": family,
                    "file_kind": kind,
                    "source_file": source_file,
                    "feature_names": block_features,
                    "match_type": "exact_matched",
                },
            )
        )
        edges.append(edge_record(parent_id, child_id, "HAS_BLOCK", 1.0))
        manifest.append(
            {
                "doc_id": child_id,
                "source_family": "depmap",
                "disease_family": family,
                "coverage_groups": coverage_groups_for_family(family),
                "source_files": [source_file],
                "cohort_or_screen": family,
                "feature_layer": kind,
                "comparison_context": None,
                "entity_anchors": block_features[:40],
                "match_type": "exact_matched",
                "raw_unit_type": "family_feature_block",
            }
        )
    return len(lines)


def build_depmap_corpus() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    stats_by_kind = Counter()
    stats_by_family = Counter()

    model = load_model_metadata()
    family_to_models = {
        family: sorted(set(group["ModelID"]))
        for family, group in model.groupby("family")
        if family in TARGET_FAMILIES
    }

    model_meta_path = relpath((DEPMAP_ROOT / "model-condition-mapping/Model.csv"))
    for family, model_ids in family_to_models.items():
        family_models = model[model["family"] == family].copy()
        parent_id = f"DEPMAPRAW-MODEL-{stable_hash(family)}"
        docs.append(
            make_parent_doc(
                source_name=DEPMAP_SOURCE,
                parent_id=parent_id,
                title=f"{family} | model metadata raw table",
                family=family,
                entity_names=[family, *sorted(set(family_models["OncotreeCode"].dropna().astype(str)))[:12]],
                text_lines=[
                    "DepMap model metadata constrained only by same-source family coverage.",
                    f"Disease family: {family}",
                    f"Description: {FAMILY_DESCRIPTIONS.get(family, family)}",
                    f"Source file: {model_meta_path}",
                    f"Model count: {len(model_ids)}",
                    "Row orientation: one model per record from Model.csv.",
                ],
                metadata={
                    "source_family": "depmap",
                    "coverage_groups": coverage_groups_for_family(family),
                    "family": family,
                    "file_kind": "model_metadata",
                    "source_file": model_meta_path,
                    "match_type": "exact_matched",
                },
            )
        )
        model_lines = []
        for _, row in family_models.iterrows():
            rrid = row.get("RRID", "")
            primary_or_metastasis = row.get("PrimaryOrMetastasis", "")
            lineage = row.get("OncotreeLineage", "")
            subtype = row.get("OncotreeSubtype", "")
            primary_disease = row.get("OncotreePrimaryDisease", "")
            oncotree_code = row.get("OncotreeCode", "")
            cell_line_name = row.get("CellLineName", "")
            line = (
                f"{row['ModelID']} | cell_line={cell_line_name} | oncotree_code={oncotree_code} "
                f"| primary_disease={primary_disease} | subtype={subtype} "
                f"| lineage={lineage} | primary_or_metastasis={primary_or_metastasis} | rrid={rrid}"
            )
            model_lines.append((str(row["ModelID"]), line))
        for block_index, block in enumerate(chunk_lines([line for _, line in model_lines], MODEL_BLOCK_SIZE), start=1):
            block_models = [feature for feature, _ in model_lines[(block_index - 1) * MODEL_BLOCK_SIZE : block_index * MODEL_BLOCK_SIZE]]
            child_id = f"{parent_id}-BLK{block_index:04d}"
            docs.append(
                make_child_doc(
                    source_name=DEPMAP_SOURCE,
                    doc_id=child_id,
                    title=f"{family} | model metadata block {block_index}",
                    family=family,
                    parent_id=parent_id,
                    entity_names=block_models[:40] + [family],
                    text_lines=[
                        f"DepMap model metadata block for {family}.",
                        f"Source file: {model_meta_path}",
                        *block,
                    ],
                    metadata={
                        "source_family": "depmap",
                        "coverage_groups": coverage_groups_for_family(family),
                        "family": family,
                        "file_kind": "model_metadata",
                        "source_file": model_meta_path,
                        "model_ids": block_models,
                        "match_type": "exact_matched",
                    },
                )
            )
            edges.append(edge_record(parent_id, child_id, "HAS_BLOCK", 1.0))
            manifest.append(
                {
                    "doc_id": child_id,
                    "source_family": "depmap",
                    "disease_family": family,
                    "coverage_groups": coverage_groups_for_family(family),
                    "source_files": [model_meta_path],
                    "cohort_or_screen": family,
                    "feature_layer": "model_metadata",
                    "comparison_context": None,
                    "entity_anchors": block_models[:40],
                    "match_type": "exact_matched",
                    "raw_unit_type": "model_row_block",
                }
            )

    gene_effect, gene_dependency = load_dependency_matrices(model["ModelID"].tolist())
    dense_matrices = {
        "gene_effect": (gene_effect, relpath((DEPMAP_ROOT / "crispr-screen/CRISPRGeneEffect.csv"))),
        "gene_dependency_probability": (gene_dependency, relpath((DEPMAP_ROOT / "crispr-screen/CRISPRGeneDependency.csv"))),
        "expression": (load_dense_omics_matrix((DEPMAP_ROOT / "expression/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv")), relpath((DEPMAP_ROOT / "expression/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"))),
        "copy_number": (load_dense_omics_matrix((DEPMAP_ROOT / "copy-number/OmicsCNGeneWGS.csv")), relpath((DEPMAP_ROOT / "copy-number/OmicsCNGeneWGS.csv"))),
    }
    binary_matrices = {
        "hotspot_mutation": (load_binary_omics_matrix((DEPMAP_ROOT / "mutations/OmicsSomaticMutationsMatrixHotspot.csv")), relpath((DEPMAP_ROOT / "mutations/OmicsSomaticMutationsMatrixHotspot.csv"))),
        "damaging_mutation": (load_binary_omics_matrix((DEPMAP_ROOT / "mutations/OmicsSomaticMutationsMatrixDamaging.csv")), relpath((DEPMAP_ROOT / "mutations/OmicsSomaticMutationsMatrixDamaging.csv"))),
    }

    for kind, (frame, source_file) in dense_matrices.items():
        for family, model_ids in family_to_models.items():
            subset = frame.loc[frame.index.intersection(model_ids)]
            if subset.empty:
                continue
            parent_id = f"DEPMAPRAW-{stable_hash(f'{family}|{kind}|{source_file}')}"
            docs.append(
                make_parent_doc(
                    source_name=DEPMAP_SOURCE,
                    parent_id=parent_id,
                    title=f"{family} | {kind} raw matrix",
                    family=family,
                    entity_names=[family, kind],
                    text_lines=[
                        "DepMap raw matrix constrained only by same-source family and file coverage.",
                        f"Disease family: {family}",
                        f"File kind: {kind}",
                        f"Source file: {source_file}",
                        f"Model count: {subset.shape[0]}",
                        f"Feature count: {subset.shape[1]}",
                        "Row orientation after normalization: each summarized line below describes one feature column across family models.",
                    ],
                    metadata={
                        "source_family": "depmap",
                        "coverage_groups": coverage_groups_for_family(family),
                        "family": family,
                        "file_kind": kind,
                        "source_file": source_file,
                        "match_type": "exact_matched",
                    },
                )
            )
            feature_count = build_depmap_matrix_blocks(
                source_name=DEPMAP_SOURCE,
                family=family,
                kind=kind,
                frame=subset,
                parent_id=parent_id,
                source_file=source_file,
                docs=docs,
                edges=edges,
                manifest=manifest,
            )
            stats_by_kind[kind] += feature_count
            stats_by_family[family] += feature_count

    for kind, (frame, source_file) in binary_matrices.items():
        for family, model_ids in family_to_models.items():
            subset = frame.loc[frame.index.intersection(model_ids)]
            if subset.empty:
                continue
            parent_id = f"DEPMAPRAW-{stable_hash(f'{family}|{kind}|{source_file}')}"
            docs.append(
                make_parent_doc(
                    source_name=DEPMAP_SOURCE,
                    parent_id=parent_id,
                    title=f"{family} | {kind} raw matrix",
                    family=family,
                    entity_names=[family, kind],
                    text_lines=[
                        "DepMap raw binary mutation matrix constrained only by same-source family and file coverage.",
                        f"Disease family: {family}",
                        f"File kind: {kind}",
                        f"Source file: {source_file}",
                        f"Model count: {subset.shape[0]}",
                        f"Feature count: {subset.shape[1]}",
                        "Each summarized line below describes one mutation feature column across family models.",
                    ],
                    metadata={
                        "source_family": "depmap",
                        "coverage_groups": coverage_groups_for_family(family),
                        "family": family,
                        "file_kind": kind,
                        "source_file": source_file,
                        "match_type": "exact_matched",
                    },
                )
            )
            feature_count = build_depmap_matrix_blocks(
                source_name=DEPMAP_SOURCE,
                family=family,
                kind=kind,
                frame=subset,
                parent_id=parent_id,
                source_file=source_file,
                docs=docs,
                edges=edges,
                manifest=manifest,
            )
            stats_by_kind[kind] += feature_count
            stats_by_family[family] += feature_count

    signatures = load_global_signatures()
    signatures_path = relpath((DEPMAP_ROOT / "global-genomic-features/OmicsGlobalSignatures.csv"))
    for family, model_ids in family_to_models.items():
        subset = signatures.loc[signatures.index.intersection(model_ids)]
        if subset.empty:
            continue
        lines = []
        for column in subset.columns:
            summary = summarize_series(subset[column])
            if summary["count"] == 0:
                continue
            lines.append(
                f"{column} | n={summary['count']} | mean={summary['mean']} | median={summary['median']} | "
                f"q1={summary['q1']} | q3={summary['q3']} | min={summary['min']} | max={summary['max']}"
            )
        doc_id = f"DEPMAPRAW-SIG-{stable_hash(family)}"
        docs.append(
            doc_record(
                doc_id=doc_id,
                knowledge_source=DEPMAP_SOURCE,
                title=f"{family} | global genomic signatures",
                text="\n".join(
                    [
                        "DepMap raw global signature table summary by family.",
                        f"Disease family: {family}",
                        f"Source file: {signatures_path}",
                        f"Model count: {subset.shape[0]}",
                        *lines,
                    ]
                ),
                document_level="chunk",
                parent_id=None,
                disease_tags=disease_tags(family),
                entity_names=[family, "MSIScore", "LoHFraction", "WGD", "CIN", "Ploidy", "Aneuploidy"],
                entity_ids=[],
                metadata={
                    "source_family": "depmap",
                    "coverage_groups": coverage_groups_for_family(family),
                    "family": family,
                    "file_kind": "global_signatures",
                    "source_file": signatures_path,
                    "match_type": "exact_matched",
                },
            )
        )
        manifest.append(
            {
                "doc_id": doc_id,
                "source_family": "depmap",
                "disease_family": family,
                "coverage_groups": coverage_groups_for_family(family),
                "source_files": [signatures_path],
                "cohort_or_screen": family,
                "feature_layer": "global_signatures",
                "comparison_context": None,
                "entity_anchors": ["MSIScore", "LoHFraction", "WGD", "CIN", "Ploidy", "Aneuploidy"],
                "match_type": "exact_matched",
                "raw_unit_type": "family_signature_summary",
            }
        )

    subtypes = load_molecular_subtypes()
    subtypes_path = relpath((DEPMAP_ROOT / "global-genomic-features/OmicsInferredMolecularSubtypes.csv"))
    for family, model_ids in family_to_models.items():
        subset = subtypes.loc[subtypes.index.intersection(model_ids)]
        if subset.empty:
            continue
        subtype_lines = []
        for column in subset.columns:
            prevalence = float(pd.to_numeric(subset[column], errors="coerce").fillna(0).mean())
            if prevalence <= 0:
                continue
            subtype_lines.append(f"{column} | positive_models={int(round(prevalence * subset.shape[0]))}/{subset.shape[0]} | prevalence={round(prevalence, 4)}")
        parent_id = f"DEPMAPRAW-SUBTYPE-{stable_hash(family)}"
        docs.append(
            make_parent_doc(
                source_name=DEPMAP_SOURCE,
                parent_id=parent_id,
                title=f"{family} | inferred molecular subtypes",
                family=family,
                entity_names=[family],
                text_lines=[
                    "DepMap raw inferred molecular subtype table by family.",
                    f"Disease family: {family}",
                    f"Source file: {subtypes_path}",
                    f"Model count: {subset.shape[0]}",
                    f"Positive subtype feature count: {len(subtype_lines)}",
                ],
                metadata={
                    "source_family": "depmap",
                    "coverage_groups": coverage_groups_for_family(family),
                    "family": family,
                    "file_kind": "molecular_subtypes",
                    "source_file": subtypes_path,
                    "match_type": "exact_matched",
                },
            )
        )
        for block_index, block in enumerate(chunk_lines(subtype_lines, DEP_BLOCK_SIZE), start=1):
            child_id = f"{parent_id}-BLK{block_index:04d}"
            block_features = [line.split(" | ", 1)[0] for line in block]
            docs.append(
                make_child_doc(
                    source_name=DEPMAP_SOURCE,
                    doc_id=child_id,
                    title=f"{family} | molecular subtype block {block_index}",
                    family=family,
                    parent_id=parent_id,
                    entity_names=block_features[:40] + [family],
                    text_lines=[
                        f"DepMap molecular subtype block for {family}.",
                        f"Source file: {subtypes_path}",
                        *block,
                    ],
                    metadata={
                        "source_family": "depmap",
                        "coverage_groups": coverage_groups_for_family(family),
                        "family": family,
                        "file_kind": "molecular_subtypes",
                        "source_file": subtypes_path,
                        "feature_names": block_features,
                        "match_type": "exact_matched",
                    },
                )
            )
            edges.append(edge_record(parent_id, child_id, "HAS_BLOCK", 1.0))
            manifest.append(
                {
                    "doc_id": child_id,
                    "source_family": "depmap",
                    "disease_family": family,
                    "coverage_groups": coverage_groups_for_family(family),
                    "source_files": [subtypes_path],
                    "cohort_or_screen": family,
                    "feature_layer": "molecular_subtypes",
                    "comparison_context": None,
                    "entity_anchors": block_features[:40],
                    "match_type": "exact_matched",
                    "raw_unit_type": "family_feature_block",
                }
            )

    gdsc = load_gdsc_drug_response(model)
    for family, group in gdsc.groupby("family"):
        if family not in TARGET_FAMILIES:
            continue
        parent_id = f"DEPMAPRAW-GDSC-{stable_hash(family)}"
        source_files = [
            relpath((DEPMAP_ROOT / "drug-response/GDSC1_fitted_dose_response_24Jul22.csv")),
            relpath((DEPMAP_ROOT / "drug-response/GDSC2_fitted_dose_response_24Jul22.csv")),
            relpath((DEPMAP_ROOT / "model-condition-mapping/PortalCompounds.csv")),
            relpath((DEPMAP_ROOT / "drug-response/screened_compounds_rel_8.4.csv")),
        ]
        docs.append(
            make_parent_doc(
                source_name=DEPMAP_SOURCE,
                parent_id=parent_id,
                title=f"{family} | GDSC-aligned drug response",
                family=family,
                entity_names=[family],
                text_lines=[
                    "DepMap raw drug-response tables aligned to model family coverage.",
                    f"Disease family: {family}",
                    f"Source files: {' | '.join(source_files)}",
                    f"Response row count: {group.shape[0]}",
                ],
                metadata={
                    "source_family": "depmap",
                    "coverage_groups": coverage_groups_for_family(family),
                    "family": family,
                    "file_kind": "drug_response",
                    "source_file": " | ".join(source_files),
                    "match_type": "exact_matched",
                },
            )
        )
        drug_lines = []
        for compound_id, compound_group in group.groupby("CompoundID"):
            drug_label = str(compound_group["DrugLabel"].dropna().iloc[0]) if compound_group["DrugLabel"].notna().any() else str(compound_id)
            targets = str(compound_group["DrugTargets"].dropna().iloc[0]) if compound_group["DrugTargets"].notna().any() else ""
            mechanism = str(compound_group["DrugMechanism"].dropna().iloc[0]) if compound_group["DrugMechanism"].notna().any() else ""
            z_summary = summarize_series(compound_group.set_index("ModelID")["Z_SCORE"])
            auc_summary = summarize_series(compound_group.set_index("ModelID")["AUC"])
            ic50_summary = summarize_series(compound_group.set_index("ModelID")["LN_IC50"])
            drug_lines.append(
                f"{drug_label} [{compound_id}] | models={compound_group['ModelID'].nunique()} | targets={targets or 'NA'} | mechanism={mechanism or 'NA'} "
                f"| z_median={z_summary.get('median')} | auc_median={auc_summary.get('median')} | lnIC50_median={ic50_summary.get('median')}"
            )
        for block_index, block in enumerate(chunk_lines(drug_lines, DEP_BLOCK_SIZE), start=1):
            child_id = f"{parent_id}-BLK{block_index:04d}"
            block_drugs = [line.split(" | ", 1)[0] for line in block]
            docs.append(
                make_child_doc(
                    source_name=DEPMAP_SOURCE,
                    doc_id=child_id,
                    title=f"{family} | drug response block {block_index}",
                    family=family,
                    parent_id=parent_id,
                    entity_names=block_drugs[:40] + [family],
                    text_lines=[
                        f"DepMap drug-response block for {family}.",
                        *block,
                    ],
                    metadata={
                        "source_family": "depmap",
                        "coverage_groups": coverage_groups_for_family(family),
                        "family": family,
                        "file_kind": "drug_response",
                        "source_file": " | ".join(source_files),
                        "drug_labels": block_drugs,
                        "match_type": "exact_matched",
                    },
                )
            )
            edges.append(edge_record(parent_id, child_id, "HAS_BLOCK", 1.0))
            manifest.append(
                {
                    "doc_id": child_id,
                    "source_family": "depmap",
                    "disease_family": family,
                    "coverage_groups": coverage_groups_for_family(family),
                    "source_files": source_files,
                    "cohort_or_screen": family,
                    "feature_layer": "drug_response",
                    "comparison_context": None,
                    "entity_anchors": block_drugs[:40],
                    "match_type": "exact_matched",
                    "raw_unit_type": "family_drug_block",
                }
            )

    return docs, edges, manifest, {"feature_count_by_kind": dict(sorted(stats_by_kind.items())), "feature_count_by_family": dict(sorted(stats_by_family.items()))}


def build_pubmed_pdf_index() -> dict[str, dict[str, Any]]:
    graph = read_json(PUBMED_GRAPH_PATH)
    pdf_index: dict[str, dict[str, Any]] = defaultdict(lambda: {"disease": "", "title": "", "entity_names": set()})
    for node in graph["nodes"]:
        if node.get("node_type") != "evidence":
            continue
        citation = node.get("citation") or {}
        pdf_path = citation.get("pdf_path")
        if not pdf_path:
            continue
        entry = pdf_index[pdf_path]
        entry["disease"] = str(node.get("disease") or "")
        entry["title"] = citation.get("title") or Path(pdf_path).stem
        subject = (node.get("subject") or {}).get("label") or (node.get("subject") or {}).get("normalized_label")
        obj = (node.get("object") or {}).get("label") or (node.get("object") or {}).get("normalized_label")
        outcome = (node.get("outcome") or {}).get("label") or (node.get("outcome") or {}).get("normalized_label")
        for value in [subject, obj, outcome]:
            if value:
                entry["entity_names"].add(str(value))
    return {
        key: {
            "disease": value["disease"],
            "title": value["title"],
            "entity_names": sorted(value["entity_names"]),
        }
        for key, value in pdf_index.items()
    }


def match_pdf_meta(pdf_path: Path, pdf_index: dict[str, dict[str, Any]], family: str) -> tuple[dict[str, Any] | None, str]:
    relative = source_relpath(pdf_path)
    if relative in pdf_index:
        return pdf_index[relative], "exact_matched"
    for key, meta in pdf_index.items():
        if Path(key).name == pdf_path.name:
            disease = str(meta.get("disease") or "").lower()
            if family == "HCC" and "hcc" in disease:
                return meta, "exact_matched"
            if family == "CRC" and "crc" in disease:
                return meta, "exact_matched"
    return None, "near_matched_fulltext"


def build_paper_corpus() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    pdf_index = build_pubmed_pdf_index()
    paper_counts = Counter()
    match_counts = Counter()
    for family, pdf_dir in PAPER_DIRS.items():
        for pdf_path in sorted(pdf_dir.glob("*.pdf")):
            meta, match_type = match_pdf_meta(pdf_path, pdf_index, family)
            title = (meta or {}).get("title") or pdf_path.stem
            text, page_count = extract_pdf_text(pdf_path)
            if not text:
                continue
            parent_id = f"RAWPAPER-{stable_hash(relpath(pdf_path))}"
            entity_names = (meta or {}).get("entity_names") or []
            docs.append(
                make_parent_doc(
                    source_name=PAPER_SOURCE,
                    parent_id=parent_id,
                    title=title,
                    family=family,
                    entity_names=entity_names[:32] + [family],
                    text_lines=[
                        "Local paper full text used as the raw literature control.",
                        f"Disease family: {family}",
                        f"PDF path: {relpath(pdf_path)}",
                        f"Match type: {match_type}",
                        f"Page count: {page_count}",
                    ],
                    metadata={
                        "source_family": "pubmed_literature",
                        "coverage_groups": coverage_groups_for_family(family),
                        "file_kind": "paper_fulltext",
                        "source_file": relpath(pdf_path),
                        "match_type": match_type,
                    },
                )
            )
            for chunk_index, chunk in enumerate(chunk_text(text, size=PAPER_CHUNK_SIZE, overlap=PAPER_CHUNK_OVERLAP), start=1):
                child_id = f"{parent_id}-CH{chunk_index:04d}"
                docs.append(
                    make_child_doc(
                        source_name=PAPER_SOURCE,
                        doc_id=child_id,
                        title=f"{title} | fulltext chunk {chunk_index}",
                        family=family,
                        parent_id=parent_id,
                        entity_names=entity_names[:32] + [family],
                        text_lines=[chunk],
                        metadata={
                            "source_family": "pubmed_literature",
                            "coverage_groups": coverage_groups_for_family(family),
                            "file_kind": "paper_fulltext",
                            "source_file": relpath(pdf_path),
                            "chunk_index": chunk_index,
                            "match_type": match_type,
                        },
                    )
                )
                edges.append(edge_record(parent_id, child_id, "HAS_CHUNK", 1.0))
                if chunk_index > 1:
                    edges.append(edge_record(f"{parent_id}-CH{chunk_index - 1:04d}", child_id, "NEXT_CHUNK", 0.5))
                manifest.append(
                    {
                        "doc_id": child_id,
                        "source_family": "pubmed_literature",
                        "disease_family": family,
                        "coverage_groups": coverage_groups_for_family(family),
                        "source_files": [relpath(pdf_path)],
                        "cohort_or_screen": family,
                        "feature_layer": "paper_fulltext",
                        "comparison_context": None,
                        "entity_anchors": entity_names[:32],
                        "match_type": match_type,
                        "raw_unit_type": "fulltext_chunk",
                    }
                )
            paper_counts[family] += 1
            match_counts[match_type] += 1
    return docs, edges, manifest, {"paper_count_by_family": dict(sorted(paper_counts.items())), "match_type_counts": dict(sorted(match_counts.items()))}


def merged_stats(docs: list[dict[str, Any]], edges: list[dict[str, Any]], manifest: list[dict[str, Any]]) -> dict[str, Any]:
    source_counts = Counter(doc["knowledge_source"] for doc in docs)
    document_level_counts = Counter(doc["document_level"] for doc in docs)
    family_counts = Counter(row["disease_family"] for row in manifest)
    unit_counts = Counter(row["raw_unit_type"] for row in manifest)
    return {
        "doc_count": len(docs),
        "edge_count": len(edges),
        "manifest_count": len(manifest),
        "knowledge_source_counts": dict(sorted(source_counts.items())),
        "document_level_counts": dict(sorted(document_level_counts.items())),
        "disease_family_counts": dict(sorted(family_counts.items())),
        "raw_unit_type_counts": dict(sorted(unit_counts.items())),
    }


def main() -> None:
    tcga_docs, tcga_edges, tcga_manifest, tcga_stats = build_tcga_corpus()
    depmap_docs, depmap_edges, depmap_manifest, depmap_stats = build_depmap_corpus()
    paper_docs, paper_edges, paper_manifest, paper_stats = build_paper_corpus()

    # Preserve the paper's document identifiers when source roots are relocated.
    from paper_raw_ids import apply_paper_raw_ids
    tcga_stats["paper_id_compatibility"] = apply_paper_raw_ids(tcga_docs, tcga_edges, tcga_manifest)
    depmap_stats["paper_id_compatibility"] = apply_paper_raw_ids(depmap_docs, depmap_edges, depmap_manifest)
    paper_stats["paper_id_compatibility"] = apply_paper_raw_ids(paper_docs, paper_edges, paper_manifest)

    write_jsonl(TCGA_OUTPUT_DIR / "raw_tcga_multiomics_docs.jsonl", tcga_docs)
    write_jsonl(TCGA_OUTPUT_DIR / "raw_tcga_multiomics_edges.jsonl", tcga_edges)
    write_jsonl(TCGA_OUTPUT_DIR / "raw_tcga_multiomics_manifest.jsonl", tcga_manifest)
    write_json(TCGA_OUTPUT_DIR / "raw_tcga_multiomics_stats.json", tcga_stats)

    write_jsonl(DEPMAP_OUTPUT_DIR / "raw_depmap_docs.jsonl", depmap_docs)
    write_jsonl(DEPMAP_OUTPUT_DIR / "raw_depmap_edges.jsonl", depmap_edges)
    write_jsonl(DEPMAP_OUTPUT_DIR / "raw_depmap_manifest.jsonl", depmap_manifest)
    write_json(DEPMAP_OUTPUT_DIR / "raw_depmap_stats.json", depmap_stats)

    all_docs = tcga_docs + depmap_docs + paper_docs
    all_edges = tcga_edges + depmap_edges + paper_edges
    all_manifest = tcga_manifest + depmap_manifest + paper_manifest
    write_jsonl(RAW_OUTPUT_DIR / "raw_multisource_docs.jsonl", all_docs)
    write_jsonl(RAW_OUTPUT_DIR / "raw_multisource_edges.jsonl", all_edges)
    write_jsonl(RAW_OUTPUT_DIR / "raw_multisource_manifest.jsonl", all_manifest)
    write_json(
        RAW_OUTPUT_DIR / "raw_multisource_stats.json",
        {
            "tcga": tcga_stats,
            "depmap": depmap_stats,
            "papers_fulltext": paper_stats,
            "merged": merged_stats(all_docs, all_edges, all_manifest),
            "notes": [
                "TCGA docs are built directly from original LinkedOmics cohort tables, chunked by raw table rows.",
                "DepMap docs are built directly from original model metadata, dependency, omics, subtype, signature, and GDSC-aligned drug-response tables.",
                "Paper docs are built directly from local full-text PDFs; the PubMed aligned graph is used only to detect exact versus near full-text matching, not to define the raw retrieval unit.",
                "The evidence graph limits only coverage scope (allowed cohorts, families, and paper diseases), not the raw document unit content.",
            ],
        },
    )


if __name__ == "__main__":
    main()

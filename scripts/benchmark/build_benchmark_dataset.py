from __future__ import annotations

import json
import os
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "dataset" / "raw"
PROCESSED_DIR = ROOT / "dataset" / "processed"
DEFAULT_DEPMAP_MODEL_PATH = Path(
    os.environ.get(
        "DEPMAP_MODEL_PATH",
        str(RAW_DIR / "DepMap" / "model-condition-mapping" / "Model.csv"),
    )
)
CELLOSAURUS_PATH = RAW_DIR / "meta" / "cellosaurus.txt"

ONCOLOGY_PATTERN = re.compile(
    r"\b("
    r"cancer|carcinoma|tumou?r|neoplasm|oncology|adenocarcinoma|sarcoma|"
    r"leukemia|leukaemia|lymphoma|melanoma|glioma|blastoma|myeloma|"
    r"myelodysplastic|myeloproliferative|nsclc|sclc|aml|cll|cml|"
    r"myelofibrosis|mesothelioma|astrocytoma"
    r")\b",
    flags=re.IGNORECASE,
)
NON_CANCER_PATTERN = re.compile(r"\bnon-cancer|non cancer|normal\b", flags=re.IGNORECASE)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    ensure_dir(path.parent)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def is_oncology_text(text: str | None) -> bool:
    return bool(text and ONCOLOGY_PATTERN.search(text))


def trim_text(text: str | None, limit: int = 1600) -> str:
    if not text:
        return ""
    collapsed = " ".join(text.split())
    return collapsed[:limit]


def normalize_name(text: str | None) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(text or "").upper())


def build_gold_standard(
    answer_format: str,
    canonical_answer: Any,
    *,
    normalized_answer: Any = None,
    entity_ids: dict[str, Any] | None = None,
    acceptable_answers: list[str] | None = None,
    binary_answer: bool | None = None,
    class_label: str | None = None,
    evaluable: bool = True,
) -> dict[str, Any]:
    return {
        "answer_format": answer_format,
        "canonical_answer": canonical_answer,
        "normalized_answer": normalized_answer,
        "entity_ids": entity_ids or {},
        "acceptable_answers": acceptable_answers or [],
        "binary_answer": binary_answer,
        "class_label": class_label,
        "evaluable": evaluable,
    }


def attach_gold_standard(
    record: dict[str, Any],
    gold_standard: dict[str, Any],
) -> dict[str, Any]:
    record["gold_standard"] = gold_standard
    record["is_machine_evaluable"] = bool(gold_standard.get("evaluable"))
    return record


def parse_depmap_model(model_path: Path) -> dict[str, list[dict[str, Any]]]:
    if not model_path.exists():
        return {}

    df = pd.read_csv(model_path)
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    key_columns = [
        "CellLineName",
        "StrippedCellLineName",
        "CCLEName",
        "RRID",
        "ModelIDAlias",
    ]
    for row in df.to_dict(orient="records"):
        for column in key_columns:
            value = row.get(column)
            if pd.isna(value):
                continue
            parts = re.split(r"[;,|]", str(value))
            for part in parts:
                key = normalize_name(part.strip())
                if key:
                    index[key].append(row)
    return index


def parse_cellosaurus(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}

    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    content = path.read_text(errors="ignore")
    for block in content.split("\n//\n"):
        lines = block.splitlines()
        if not lines or not lines[0].startswith("ID   "):
            continue

        record = {
            "id": None,
            "accession": None,
            "synonyms": [],
            "diseases": [],
            "category": None,
        }
        for line in lines:
            if line.startswith("ID   "):
                record["id"] = line[5:].strip()
                record["synonyms"].append(record["id"])
            elif line.startswith("AC   ") and record["accession"] is None:
                record["accession"] = line[5:].strip()
            elif line.startswith("SY   "):
                record["synonyms"].extend(
                    [part.strip() for part in line[5:].split(";") if part.strip()]
                )
            elif line.startswith("DI   "):
                record["diseases"].append(line[5:].strip())
            elif line.startswith("CA   "):
                record["category"] = line[5:].strip()

        for synonym in record["synonyms"]:
            key = normalize_name(synonym)
            if key:
                index[key].append(record)

    return index


def choose_depmap_entry(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not entries:
        return None

    def sort_key(entry: dict[str, Any]) -> tuple[int, int, str]:
        disease = str(entry.get("OncotreePrimaryDisease") or "")
        subtype = str(entry.get("OncotreeSubtype") or "")
        non_cancer = bool(NON_CANCER_PATTERN.search(disease) or NON_CANCER_PATTERN.search(subtype))
        return (
            1 if non_cancer else 0,
            0 if entry.get("RRID") else 1,
            str(entry.get("ModelID") or ""),
        )

    return sorted(entries, key=sort_key)[0]


def parse_cellosaurus_disease(disease_line: str) -> dict[str, str | None]:
    parts = [part.strip() for part in disease_line.split(";")]
    return {
        "source_db": parts[0] if len(parts) > 0 else None,
        "disease_id": parts[1] if len(parts) > 1 else None,
        "disease_label": parts[2] if len(parts) > 2 else None,
    }


def choose_cellosaurus_entry(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not entries:
        return None

    def sort_key(entry: dict[str, Any]) -> tuple[int, int, str]:
        category = str(entry.get("category") or "")
        diseases = entry.get("diseases") or []
        return (
            0 if category == "Cancer cell line" else 1,
            -len(diseases),
            str(entry.get("accession") or ""),
        )

    return sorted(entries, key=sort_key)[0]


def resolve_cell_line_annotation(
    cell_line_name: str,
    depmap_index: dict[str, list[dict[str, Any]]],
    cellosaurus_index: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    key = normalize_name(cell_line_name)
    depmap_entry = choose_depmap_entry(depmap_index.get(key, []))
    if depmap_entry is not None:
        return {
            "mapping_source": "depmap",
            "cell_line_name": cell_line_name,
            "model_id": depmap_entry.get("ModelID"),
            "rrid": depmap_entry.get("RRID"),
            "oncotree_primary_disease": depmap_entry.get("OncotreePrimaryDisease"),
            "oncotree_subtype": depmap_entry.get("OncotreeSubtype"),
            "oncotree_lineage": depmap_entry.get("OncotreeLineage"),
            "oncotree_code": depmap_entry.get("OncotreeCode"),
            "ccle_name": depmap_entry.get("CCLEName"),
            "cellosaurus_accession": depmap_entry.get("RRID"),
            "cellosaurus_disease_id": None,
            "cellosaurus_disease_label": None,
        }

    cellosaurus_entry = choose_cellosaurus_entry(cellosaurus_index.get(key, []))
    if cellosaurus_entry is None:
        return None

    parsed_diseases = [
        parse_cellosaurus_disease(line) for line in (cellosaurus_entry.get("diseases") or [])
    ]
    primary = next(
        (
            item
            for item in parsed_diseases
            if item.get("source_db") == "NCIt" and item.get("disease_label")
        ),
        parsed_diseases[0] if parsed_diseases else None,
    )
    return {
        "mapping_source": "cellosaurus",
        "cell_line_name": cell_line_name,
        "model_id": None,
        "rrid": cellosaurus_entry.get("accession"),
        "oncotree_primary_disease": primary.get("disease_label") if primary else None,
        "oncotree_subtype": None,
        "oncotree_lineage": None,
        "oncotree_code": None,
        "ccle_name": None,
        "cellosaurus_accession": cellosaurus_entry.get("accession"),
        "cellosaurus_disease_id": primary.get("disease_id") if primary else None,
        "cellosaurus_disease_label": primary.get("disease_label") if primary else None,
    }


def build_tdc_cell_line_map() -> dict[str, dict[str, Any]]:
    depmap_index = parse_depmap_model(DEFAULT_DEPMAP_MODEL_PATH)
    cellosaurus_index = parse_cellosaurus(CELLOSAURUS_PATH)

    gdsc2 = pickle.load(open(RAW_DIR / "tdc" / "gdsc2.pkl", "rb"))
    drugcomb = pickle.load(open(RAW_DIR / "tdc" / "drugcomb.pkl", "rb"))
    unique_names = sorted(set(gdsc2["ID2"].astype(str)).union(set(drugcomb["Cell_Line_ID"].astype(str))))

    mapping = {}
    rows_for_csv = []
    for name in unique_names:
        annotation = resolve_cell_line_annotation(name, depmap_index, cellosaurus_index)
        if annotation is None:
            continue
        mapping[name] = annotation
        rows_for_csv.append(annotation)

    ensure_dir(RAW_DIR / "meta")
    pd.DataFrame(rows_for_csv).sort_values(["mapping_source", "cell_line_name"]).to_csv(
        RAW_DIR / "meta" / "tdc_cell_line_annotations.csv",
        index=False,
    )
    write_json(
        RAW_DIR / "meta" / "tdc_cell_line_annotation_stats.json",
        {
            "unique_cell_lines": len(unique_names),
            "mapped_cell_lines": len(mapping),
            "by_source": Counter(item["mapping_source"] for item in mapping.values()),
        },
    )
    return mapping


def normalize_civic_evidence() -> list[dict[str, Any]]:
    rows = load_jsonl(RAW_DIR / "civic" / "evidence_items.jsonl")
    normalized = []
    for row in rows:
        disease = row.get("disease") or {}
        therapies = [item["name"] for item in row.get("therapies") or [] if item.get("name")]
        therapy_text = ", ".join(therapies) if therapies else "the referenced clinical context"
        disease_name = disease.get("name") or "unknown disease"
        molecular_profile = (row.get("molecularProfile") or {}).get("name") or "unknown molecular profile"
        question = (
            f"In CIViC, what is the clinical implication of {molecular_profile} "
            f"for {therapy_text} in {disease_name}?"
        )
        answer = (
            f"{row.get('evidenceDirection', 'UNKNOWN').title()} "
            f"{str(row.get('significance', 'unknown')).lower()} "
            f"({str(row.get('evidenceType', 'unknown')).lower()} evidence, "
            f"level {row.get('evidenceLevel', 'NA')}). "
            f"{trim_text(row.get('description'))}"
        ).strip()
        record = {
            "id": f"civic-evidence-{row['id']}",
            "source": "civic_evidence",
            "task_type": "precision_oncology_evidence",
            "question": question,
            "answer": answer,
            "answer_type": "free_text",
            "oncology_focus": disease_name,
            "evidence_text": trim_text(row.get("description")),
            "entities": {
                "disease": disease_name,
                "molecular_profile": molecular_profile,
                "therapies": therapies,
                "source_title": (row.get("source") or {}).get("title"),
                "citation_id": (row.get("source") or {}).get("citationId"),
            },
            "provenance": {
                "source_record_id": row["id"],
                "source_url": row.get("link"),
                "source_dataset": "CIViC evidence items",
            },
            "metadata": {
                "evidence_type": row.get("evidenceType"),
                "evidence_level": row.get("evidenceLevel"),
                "evidence_direction": row.get("evidenceDirection"),
                "significance": row.get("significance"),
                "status": row.get("status"),
                "variant_origin": row.get("variantOrigin"),
                "therapy_interaction_type": row.get("therapyInteractionType"),
                "doid": disease.get("doid"),
            },
        }
        normalized.append(
            attach_gold_standard(
                record,
                build_gold_standard(
                    "categorical",
                    canonical_answer="|".join(
                        [
                            str(row.get("evidenceDirection") or "UNKNOWN"),
                            str(row.get("significance") or "UNKNOWN"),
                            str(row.get("evidenceType") or "UNKNOWN"),
                            str(row.get("evidenceLevel") or "UNKNOWN"),
                        ]
                    ),
                    normalized_answer={
                        "evidence_direction": row.get("evidenceDirection"),
                        "significance": row.get("significance"),
                        "evidence_type": row.get("evidenceType"),
                        "evidence_level": row.get("evidenceLevel"),
                    },
                    entity_ids={
                        "disease_doid": disease.get("doid"),
                        "citation_id": (row.get("source") or {}).get("citationId"),
                    },
                    acceptable_answers=[
                        f"{str(row.get('evidenceDirection', 'UNKNOWN')).lower()} {str(row.get('significance', 'unknown')).lower()}",
                        f"{str(row.get('evidenceDirection', 'UNKNOWN')).upper()}|{str(row.get('significance', 'UNKNOWN')).upper()}",
                    ],
                    class_label=f"{str(row.get('evidenceDirection', 'UNKNOWN')).upper()}_{str(row.get('significance', 'UNKNOWN')).upper()}",
                ),
            )
        )
    return normalized


def normalize_civic_assertions() -> list[dict[str, Any]]:
    rows = load_jsonl(RAW_DIR / "civic" / "assertions.jsonl")
    normalized = []
    for row in rows:
        disease = row.get("disease") or {}
        therapies = [item["name"] for item in row.get("therapies") or [] if item.get("name")]
        therapy_text = ", ".join(therapies) if therapies else "the referenced clinical context"
        disease_name = disease.get("name") or "unknown disease"
        molecular_profile = (row.get("molecularProfile") or {}).get("name") or "unknown molecular profile"
        amp_clause = f", AMP level {row.get('ampLevel')}" if row.get("ampLevel") else ""
        question = (
            f"In CIViC, what aggregated assertion is made about {molecular_profile} "
            f"for {therapy_text} in {disease_name}?"
        )
        answer = (
            f"{row.get('assertionDirection', 'UNKNOWN').title()} "
            f"{str(row.get('significance', 'unknown')).lower()} assertion"
            f"{amp_clause}. "
            f"{trim_text(row.get('description'))}"
        ).strip()
        record = {
            "id": f"civic-assertion-{row['id']}",
            "source": "civic_assertion",
            "task_type": "precision_oncology_assertion",
            "question": question,
            "answer": answer,
            "answer_type": "free_text",
            "oncology_focus": disease_name,
            "evidence_text": trim_text(row.get("description")),
            "entities": {
                "disease": disease_name,
                "molecular_profile": molecular_profile,
                "therapies": therapies,
            },
            "provenance": {
                "source_record_id": row["id"],
                "source_url": row.get("link"),
                "source_dataset": "CIViC assertions",
            },
            "metadata": {
                "assertion_type": row.get("assertionType"),
                "assertion_direction": row.get("assertionDirection"),
                "significance": row.get("significance"),
                "amp_level": row.get("ampLevel"),
                "status": row.get("status"),
                "evidence_items_count": row.get("evidenceItemsCount"),
                "doid": disease.get("doid"),
            },
        }
        normalized.append(
            attach_gold_standard(
                record,
                build_gold_standard(
                    "categorical",
                    canonical_answer="|".join(
                        [
                            str(row.get("assertionDirection") or "UNKNOWN"),
                            str(row.get("significance") or "UNKNOWN"),
                            str(row.get("assertionType") or "UNKNOWN"),
                            str(row.get("ampLevel") or "NA"),
                        ]
                    ),
                    normalized_answer={
                        "assertion_direction": row.get("assertionDirection"),
                        "significance": row.get("significance"),
                        "assertion_type": row.get("assertionType"),
                        "amp_level": row.get("ampLevel"),
                    },
                    entity_ids={"disease_doid": disease.get("doid")},
                    acceptable_answers=[
                        f"{str(row.get('assertionDirection', 'UNKNOWN')).lower()} {str(row.get('significance', 'unknown')).lower()} assertion",
                        f"{str(row.get('assertionDirection', 'UNKNOWN')).upper()}|{str(row.get('significance', 'UNKNOWN')).upper()}",
                    ],
                    class_label=f"{str(row.get('assertionDirection', 'UNKNOWN')).upper()}_{str(row.get('significance', 'UNKNOWN')).upper()}",
                ),
            )
        )
    return normalized


def normalize_disgenet() -> list[dict[str, Any]]:
    df = pd.read_csv(RAW_DIR / "disgenet" / "disgenet.csv")
    normalized = []
    for row in df.itertuples(index=False):
        disease_text = str(row.X2)
        if not is_oncology_text(disease_text):
            continue
        disease_name = disease_text.split(":", 1)[0].strip()
        question = f"According to DisGeNET, is Entrez gene {row.ID1} associated with {disease_name}?"
        answer = f"Yes. The reported association score is {float(row.Y):.3f}."
        record = {
            "id": f"disgenet-{row[0]}",
            "source": "disgenet",
            "task_type": "gene_disease_association",
            "question": question,
            "answer": answer,
            "answer_type": "free_text",
            "oncology_focus": disease_name,
            "evidence_text": trim_text(disease_text),
            "entities": {
                "gene_entrez_id": str(row.ID1),
                "disease_cui": row.ID2,
                "disease_name": disease_name,
            },
            "provenance": {
                "source_record_id": int(row[0]),
                "source_dataset": "DisGeNET via TDC Harvard Dataverse mirror",
            },
            "metadata": {
                "association_score": float(row.Y),
                "protein_sequence_length": len(str(row.X1)),
            },
        }
        normalized.append(
            attach_gold_standard(
                record,
                build_gold_standard(
                    "boolean",
                    canonical_answer="yes",
                    normalized_answer=True,
                    entity_ids={
                        "gene_entrez_id": str(row.ID1),
                        "disease_cui": row.ID2,
                    },
                    acceptable_answers=["yes", "true", "associated"],
                    binary_answer=True,
                    class_label="associated",
                ),
            )
        )
    return normalized


def select_best_mediqa_answer(answer_list: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = []
    for item in answer_list:
        answer = item.get("Answer") or {}
        candidates.append(answer)
    if not candidates:
        return None

    def sort_key(answer: dict[str, Any]) -> tuple[Any, Any, Any]:
        reference_rank = answer.get("ReferenceRank")
        system_rank = answer.get("SystemRank")
        reference_score = answer.get("ReferenceScore")
        return (
            reference_rank if reference_rank is not None else 10**9,
            system_rank if system_rank is not None else 10**9,
            -(reference_score if reference_score is not None else -10**9),
        )

    return sorted(candidates, key=sort_key)[0]


def normalize_mediqa() -> list[dict[str, Any]]:
    normalized = []
    mediqa_dir = RAW_DIR / "mediqa_qa"
    for parquet_path in sorted(mediqa_dir.glob("*.parquet")):
        df = pd.read_parquet(parquet_path)
        split_name = parquet_path.stem
        for payload in df["QUESTION"].tolist():
            question_text = trim_text(payload.get("QuestionText"))
            answer_list = payload.get("AnswerList")
            candidate_answers = [] if answer_list is None else list(answer_list)
            best_answer = select_best_mediqa_answer(candidate_answers)
            answer_text = trim_text((best_answer or {}).get("AnswerText"))
            combined = f"{question_text} {answer_text}"
            if not is_oncology_text(combined):
                continue
            answer_url = (best_answer or {}).get("AnswerURL")
            record = {
                "id": f"mediqa-{split_name}-{payload.get('QID')}",
                "source": "mediqa_qa",
                "task_type": "oncology_literature_qa",
                "question": question_text,
                "answer": answer_text,
                "answer_type": "free_text",
                "oncology_focus": "mixed_oncology",
                "evidence_text": answer_text,
                "entities": {
                    "qid": payload.get("QID"),
                    "answer_url": answer_url,
                },
                "provenance": {
                    "source_record_id": payload.get("QID"),
                    "source_dataset": "bigbio/mediqa_qa",
                    "split": split_name,
                    "source_url": answer_url,
                },
                "metadata": {
                    "reference_rank": (best_answer or {}).get("ReferenceRank"),
                    "reference_score": (best_answer or {}).get("ReferenceScore"),
                    "system_rank": (best_answer or {}).get("SystemRank"),
                },
            }
            normalized.append(
                attach_gold_standard(
                    record,
                    build_gold_standard(
                        "free_text",
                        canonical_answer=answer_text,
                        acceptable_answers=[answer_text] if answer_text else [],
                        evaluable=False,
                    ),
                )
            )
    return normalized


def build_tdc_cell_line_records(
    cell_line_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    records = []
    for cell_line_name, annotation in sorted(cell_line_map.items()):
        disease_name = annotation.get("oncotree_primary_disease") or annotation.get("cellosaurus_disease_label")
        if not disease_name or not is_oncology_text(disease_name):
            continue
        subtype = annotation.get("oncotree_subtype")
        answer = disease_name if not subtype else f"{disease_name} ({subtype})"
        record = {
            "id": f"tdc-cellline-{normalize_name(cell_line_name)}",
            "source": "tdc_cell_line",
            "task_type": "cancer_cell_line_annotation",
            "question": f"What cancer type is cell line {cell_line_name} annotated as in the harmonized TDC mapping?",
            "answer": answer,
            "answer_type": "free_text",
            "oncology_focus": disease_name,
            "evidence_text": trim_text(answer),
            "entities": {
                "cell_line_name": cell_line_name,
                "model_id": annotation.get("model_id"),
                "rrid": annotation.get("rrid"),
            },
            "provenance": {
                "source_record_id": cell_line_name,
                "source_dataset": "DepMap Model.csv + Cellosaurus fallback",
            },
            "metadata": {
                "mapping_source": annotation.get("mapping_source"),
                "oncotree_code": annotation.get("oncotree_code"),
                "oncotree_lineage": annotation.get("oncotree_lineage"),
                "oncotree_subtype": subtype,
                "cellosaurus_disease_id": annotation.get("cellosaurus_disease_id"),
            },
        }
        records.append(
            attach_gold_standard(
                record,
                build_gold_standard(
                    "entity",
                    canonical_answer=disease_name,
                    normalized_answer={
                        "primary_disease": disease_name,
                        "subtype": subtype,
                        "oncotree_code": annotation.get("oncotree_code"),
                    },
                    entity_ids={
                        "model_id": annotation.get("model_id"),
                        "rrid": annotation.get("rrid"),
                        "oncotree_code": annotation.get("oncotree_code"),
                        "cellosaurus_disease_id": annotation.get("cellosaurus_disease_id"),
                    },
                    acceptable_answers=[value for value in [disease_name, subtype, answer] if value],
                    class_label=annotation.get("oncotree_code") or annotation.get("cellosaurus_disease_id"),
                ),
            )
        )
    return records


def build_gdsc2_records(
    cell_line_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    df = pickle.load(open(RAW_DIR / "tdc" / "gdsc2.pkl", "rb")).copy()
    df["cell_line_name"] = df["ID2"].astype(str)
    df = df[df["cell_line_name"].isin(cell_line_map)]
    if df.empty:
        return []

    thresholds = df.groupby("ID1")["Y"].quantile([0.25, 0.75]).unstack()
    thresholds.columns = ["q25", "q75"]
    df = df.merge(thresholds, left_on="ID1", right_index=True, how="left")
    df["binary_answer"] = None
    df.loc[df["Y"] <= df["q25"], "binary_answer"] = True
    df.loc[df["Y"] >= df["q75"], "binary_answer"] = False
    df = df[df["binary_answer"].notna()].copy()

    records = []
    for idx, row in df.iterrows():
        annotation = cell_line_map[row["cell_line_name"]]
        disease_name = annotation.get("oncotree_primary_disease") or annotation.get("cellosaurus_disease_label")
        if not disease_name or not is_oncology_text(disease_name):
            continue
        binary = bool(row["binary_answer"])
        answer = (
            f"{'Yes' if binary else 'No'}. "
            f"The response score {float(row['Y']):.3f} is "
            f"{'at or below' if binary else 'at or above'} the drug-specific "
            f"{'25th' if binary else '75th'} percentile "
            f"({float(row['q25'] if binary else row['q75']):.3f})."
        )
        record = {
            "id": f"tdc-gdsc2-{idx}",
            "source": "tdc_gdsc2",
            "task_type": "drug_response_sensitivity",
            "question": f"In GDSC2, is {row['cell_line_name']} ({disease_name}) sensitive to {row['ID1']}?",
            "answer": answer,
            "answer_type": "free_text",
            "oncology_focus": disease_name,
            "evidence_text": trim_text(answer),
            "entities": {
                "drug_name": row["ID1"],
                "cell_line_name": row["cell_line_name"],
                "drug_smiles": row["X1"],
                "model_id": annotation.get("model_id"),
                "rrid": annotation.get("rrid"),
            },
            "provenance": {
                "source_record_id": int(idx),
                "source_dataset": "TDC GDSC2",
            },
            "metadata": {
                "response_score": float(row["Y"]),
                "drug_q25": float(row["q25"]),
                "drug_q75": float(row["q75"]),
                "label_rule": "sensitive_if_le_q25_else_resistant_if_ge_q75",
                "mapping_source": annotation.get("mapping_source"),
                "oncotree_code": annotation.get("oncotree_code"),
            },
        }
        records.append(
            attach_gold_standard(
                record,
                build_gold_standard(
                    "boolean",
                    canonical_answer="yes" if binary else "no",
                    normalized_answer=binary,
                    entity_ids={
                        "model_id": annotation.get("model_id"),
                        "rrid": annotation.get("rrid"),
                        "oncotree_code": annotation.get("oncotree_code"),
                    },
                    acceptable_answers=["yes", "sensitive"] if binary else ["no", "not sensitive", "resistant"],
                    binary_answer=binary,
                    class_label="sensitive" if binary else "not_sensitive",
                ),
            )
        )
    return records


def build_drugcomb_records(
    cell_line_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    df = pickle.load(open(RAW_DIR / "tdc" / "drugcomb.pkl", "rb")).copy()
    df["cell_line_name"] = df["Cell_Line_ID"].astype(str)
    df = df[df["cell_line_name"].isin(cell_line_map)]
    if df.empty:
        return []

    df = df[(df["Synergy_ZIP"] >= 10) | (df["Synergy_ZIP"] <= -10)].copy()
    df["binary_answer"] = df["Synergy_ZIP"] >= 10

    records = []
    for idx, row in df.iterrows():
        annotation = cell_line_map[row["cell_line_name"]]
        disease_name = annotation.get("oncotree_primary_disease") or annotation.get("cellosaurus_disease_label")
        if not disease_name or not is_oncology_text(disease_name):
            continue
        binary = bool(row["binary_answer"])
        answer = (
            f"{'Yes' if binary else 'No'}. "
            f"The ZIP synergy score is {float(row['Synergy_ZIP']):.3f}, "
            f"which is {'at least 10' if binary else 'at most -10'}."
        )
        record = {
            "id": f"tdc-drugcomb-{idx}",
            "source": "tdc_drugcomb",
            "task_type": "drug_combination_synergy",
            "question": (
                f"In DrugComb, is the combination of {row['Drug1_ID']} and {row['Drug2_ID']} "
                f"synergistic in {disease_name} cell line {row['cell_line_name']}?"
            ),
            "answer": answer,
            "answer_type": "free_text",
            "oncology_focus": disease_name,
            "evidence_text": trim_text(answer),
            "entities": {
                "drug_1": row["Drug1_ID"],
                "drug_2": row["Drug2_ID"],
                "cell_line_name": row["cell_line_name"],
                "model_id": annotation.get("model_id"),
                "rrid": annotation.get("rrid"),
            },
            "provenance": {
                "source_record_id": int(idx),
                "source_dataset": "TDC DrugComb",
            },
            "metadata": {
                "zip_score": float(row["Synergy_ZIP"]),
                "css": float(row["CSS"]),
                "loewe_score": float(row["Synergy_Loewe"]),
                "hsa_score": float(row["Synergy_HSA"]),
                "bliss_score": float(row["Synergy_Bliss"]),
                "label_rule": "synergistic_if_zip_ge_10_else_antagonistic_if_zip_le_-10",
                "mapping_source": annotation.get("mapping_source"),
                "oncotree_code": annotation.get("oncotree_code"),
            },
        }
        records.append(
            attach_gold_standard(
                record,
                build_gold_standard(
                    "boolean",
                    canonical_answer="yes" if binary else "no",
                    normalized_answer=binary,
                    entity_ids={
                        "model_id": annotation.get("model_id"),
                        "rrid": annotation.get("rrid"),
                        "oncotree_code": annotation.get("oncotree_code"),
                    },
                    acceptable_answers=["yes", "synergistic"] if binary else ["no", "not synergistic", "antagonistic"],
                    binary_answer=binary,
                    class_label="synergistic" if binary else "not_synergistic",
                ),
            )
        )
    return records


def build_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_source = Counter(record["source"] for record in records)
    by_task = Counter(record["task_type"] for record in records)
    by_focus = Counter(record["oncology_focus"] for record in records)
    by_format = Counter(record["gold_standard"]["answer_format"] for record in records)
    machine_evaluable = sum(1 for record in records if record["is_machine_evaluable"])

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["source"]].append(record)

    source_breakdown = {}
    for source, source_records in grouped.items():
        source_breakdown[source] = {
            "count": len(source_records),
            "task_types": dict(Counter(item["task_type"] for item in source_records)),
            "answer_formats": dict(Counter(item["gold_standard"]["answer_format"] for item in source_records)),
            "machine_evaluable": sum(1 for item in source_records if item["is_machine_evaluable"]),
            "top_oncology_focus": Counter(item["oncology_focus"] for item in source_records).most_common(10),
        }

    return {
        "total_records": len(records),
        "machine_evaluable_records": machine_evaluable,
        "by_source": dict(by_source),
        "by_task_type": dict(by_task),
        "by_answer_format": dict(by_format),
        "top_oncology_focus": by_focus.most_common(20),
        "source_breakdown": source_breakdown,
    }


def main() -> None:
    ensure_dir(PROCESSED_DIR)

    cell_line_map = build_tdc_cell_line_map()

    civic_evidence = normalize_civic_evidence()
    civic_assertions = normalize_civic_assertions()
    disgenet = normalize_disgenet()
    mediqa = normalize_mediqa()
    tdc_cell_lines = build_tdc_cell_line_records(cell_line_map)
    tdc_gdsc2 = build_gdsc2_records(cell_line_map)
    tdc_drugcomb = build_drugcomb_records(cell_line_map)

    per_source = {
        "civic_evidence": civic_evidence,
        "civic_assertion": civic_assertions,
        "disgenet": disgenet,
        "mediqa_qa": mediqa,
        "tdc_cell_line": tdc_cell_lines,
        "tdc_gdsc2": tdc_gdsc2,
        "tdc_drugcomb": tdc_drugcomb,
    }

    all_records = []
    machine_evaluable_records = []
    for source_name, rows in per_source.items():
        write_jsonl(PROCESSED_DIR / f"{source_name}.jsonl", rows)
        all_records.extend(rows)
        machine_evaluable_records.extend([row for row in rows if row["is_machine_evaluable"]])

    all_records = sorted(all_records, key=lambda item: item["id"])
    machine_evaluable_records = sorted(machine_evaluable_records, key=lambda item: item["id"])
    write_jsonl(PROCESSED_DIR / "benchmark_all.jsonl", all_records)
    write_jsonl(PROCESSED_DIR / "benchmark_machine_evaluable.jsonl", machine_evaluable_records)
    write_json(PROCESSED_DIR / "benchmark_stats.json", build_stats(all_records))


if __name__ == "__main__":
    main()

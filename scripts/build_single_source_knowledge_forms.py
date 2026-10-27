from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "dataset/rag/knowledge_sources/multi_cancer"
OUTPUT_DIR = ROOT / "dataset/rag/knowledge_sources/single_source_forms"
FORMS = {
    "document_corpus": "controlled_linearized_text",
    "semantic_kg": "controlled_triple_kg",
    "evidence_graph": "controlled_evidence_graph",
}
SOURCES = {
    "tcga": "tcga_linkedomics",
    "pubmed": "pubmed_literature",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected_ids: dict[tuple[str, str], set[str]] = {}
    report: dict[str, Any] = {"policy": "Direct metadata.source_family match; integrated and other-source evidence excluded.", "knowledge_bases": {}}

    for source_name, source_family in SOURCES.items():
        id_sets = {}
        for form_name, input_name in FORMS.items():
            input_docs = SOURCE_DIR / f"{input_name}_docs.jsonl"
            rows = []
            for row in read_jsonl(input_docs):
                if str((row.get("metadata") or {}).get("source_family") or "") != source_family:
                    continue
                item = dict(row)
                item["knowledge_source"] = f"{source_name}_{form_name}"
                rows.append(item)
            output_name = f"{source_name}_{form_name}"
            write_jsonl(OUTPUT_DIR / f"{output_name}_docs.jsonl", rows)
            ids = {str((row.get("metadata") or {}).get("source_evidence_id") or row["doc_id"]) for row in rows}
            id_sets[form_name] = ids
            selected_ids[(source_name, form_name)] = {row["doc_id"] for row in rows}
            report["knowledge_bases"][output_name] = {"doc_count": len(rows), "source_evidence_id_count": len(ids)}

        common = set.intersection(*id_sets.values())
        union = set.union(*id_sets.values())
        report["knowledge_bases"][f"{source_name}_alignment"] = {
            "common_source_evidence_ids": len(common),
            "union_source_evidence_ids": len(union),
            "exact_match": all(ids == common for ids in id_sets.values()),
        }

    evidence_edges = read_jsonl(SOURCE_DIR / "controlled_evidence_graph_edges.jsonl")
    for source_name in SOURCES:
        for form_name in FORMS:
            output_name = f"{source_name}_{form_name}"
            if form_name == "evidence_graph":
                ids = selected_ids[(source_name, form_name)]
                edges = [row for row in evidence_edges if row.get("source_doc_id") in ids and row.get("target_doc_id") in ids]
            else:
                edges = []
            write_jsonl(OUTPUT_DIR / f"{output_name}_edges.jsonl", edges)
            report["knowledge_bases"][output_name]["edge_count"] = len(edges)

    (OUTPUT_DIR / "single_source_forms_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

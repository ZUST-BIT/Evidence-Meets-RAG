from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from analyze_retrieval_evidence_quality import audit_retrieval
from build_cross_source_ablation_retrieval import source_requirements
from rag_eval_lib import (
    build_query_text,
    collect_record_gene_symbols,
    doc_matches_disease,
    doc_support_flags,
    load_prepared_corpus,
    read_jsonl,
    record_gene_match,
    retrieve_naive,
)
from run_rag_evaluation import serialize_doc


ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = ROOT / "dataset/rag/knowledge_sources"
BENCHMARK = ROOT / "dataset/evaluated/external_reasoning_benchmark_formal_v3_571.jsonl"
OUTPUT_DIR = ROOT / "dataset/rag/analysis/evidence_meets_rag_tcga_pubmed"
OUTPUT = OUTPUT_DIR / "tcga_pubmed_controlled_retrieval.jsonl"
PANEL = OUTPUT_DIR / "tcga_pubmed_panel.jsonl"
CANDIDATE_K = 384
TOP_K = 8
PANEL_SIZE = 30
TASKS = {
    "cross_source_corroboration_status",
    "cross_source_support_bundle",
    "evidence_signal_interpretation",
    "hierarchical_evidence_aggregation",
    "precision_oncology_evidence",
}


def family(record: dict[str, Any]) -> str:
    return str((record.get("metadata") or {}).get("priority_cancer_family") or record.get("oncology_focus") or "").lower()


def order_key(record: dict[str, Any]) -> str:
    return hashlib.sha256(f"tcga-pubmed-complementarity:{record['id']}".encode()).hexdigest()


def is_relevant(doc: Any, record: dict[str, Any]) -> bool:
    return doc_matches_disease(doc, record) and record_gene_match(doc, record)


def select_context(
    candidates: list[tuple[dict[str, Any], set[str], bool]],
    forbidden: str | None = None,
) -> list[dict[str, Any]]:
    eligible = [row for row in candidates if forbidden is None or forbidden not in row[1]]
    selected = []
    selected_ids = set()

    def add(rows: list[tuple[dict[str, Any], set[str], bool]], target: int) -> None:
        for doc, _requirements, _relevant in rows:
            if doc["doc_id"] in selected_ids:
                continue
            selected.append(doc)
            selected_ids.add(doc["doc_id"])
            if len(selected) >= target:
                return

    def pad() -> None:
        while len(selected) < TOP_K:
            index = len(selected) + 1
            selected.append({
                "doc_id": f"CONTROL-{forbidden or 'full'}-{index}",
                "knowledge_source": "ablation_control",
                "document_level": "control",
                "parent_id": None,
                "title": "Reserved evidence slot",
                "score": 0.0,
                "disease_tags": [],
                "entity_names": [],
                "entity_ids": [],
                "text": "No evidence is supplied in this reserved control slot.",
                "metadata": {"source_requirements": [], "ablation_control": True},
            })

    for source in ("tcga", "pubmed"):
        if source == forbidden:
            continue
        direct = [row for row in eligible if row[1] == {source} and row[2]]
        add(direct, min(TOP_K, len(selected) + 3))
    if forbidden is None:
        integrated = [row for row in eligible if len(row[1].intersection({"tcga", "pubmed"})) >= 2 and row[2]]
        add(integrated, min(TOP_K, len(selected) + 2))
        add([row for row in eligible if row[2]], TOP_K)
        add(eligible, TOP_K)
    else:
        # Keep the surviving source at the same three-slot budget instead of
        # letting it compensate for the removed source.
        background = [row for row in eligible if not row[1].intersection({"tcga", "pubmed"}) and row[2]]
        add(background, TOP_K)
        add([row for row in eligible if not row[1].intersection({"tcga", "pubmed"})], TOP_K)
        pad()
    return selected[:TOP_K]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = [
        row for row in read_jsonl(BENCHMARK)
        if family(row) in {"hcc", "crc"} and row.get("task_type") in TASKS
    ]
    records.sort(key=order_key)
    corpus = load_prepared_corpus(KNOWLEDGE_DIR, "multi_cancer", "evidence_graph_complete", use_cache=True)
    requirement_cache: dict[str, set[str]] = {}
    selected_records = []
    candidate_cache: dict[str, list[tuple[dict[str, Any], set[str], bool]]] = {}

    for index, record in enumerate(records, start=1):
        ranked = retrieve_naive(corpus, record, CANDIDATE_K)
        ranked_scores = {doc.doc_id: score for doc, score in ranked}
        candidate_ids = [doc.doc_id for doc, _score in ranked]
        seen = set(candidate_ids)
        scoped = set()
        for symbol in collect_record_gene_symbols(record):
            scoped.update(corpus.gene_symbol_to_doc_ids.get(symbol, []))
        candidate_ids.extend(doc_id for doc_id in sorted(scoped) if doc_id not in seen and is_relevant(corpus.doc_by_id[doc_id], record))
        candidates = []
        for doc_id in candidate_ids:
            doc = corpus.doc_by_id[doc_id]
            requirements = source_requirements(doc, corpus, requirement_cache) or {"other"}
            relations = sorted({rel for _neighbor, rel, _weight in corpus.graph_neighbors.get(doc_id, []) if rel not in {"HAS_CHUNK", "NEXT_CHUNK"}})[:10]
            serialized = serialize_doc(doc, ranked_scores.get(doc_id, 0.0), relations)
            serialized["metadata"]["source_requirements"] = sorted(requirements)
            serialized["metadata"]["source_role_flags"] = sorted(doc_support_flags(doc))
            candidates.append((serialized, requirements, is_relevant(doc, record)))
        direct = {
            source: sum(requirements == {source} and relevant for _doc, requirements, relevant in candidates)
            for source in ("tcga", "pubmed")
        }
        if min(direct.values()) < 2:
            continue
        if len(select_context(candidates, "tcga")) < TOP_K or len(select_context(candidates, "pubmed")) < TOP_K:
            continue
        selected_records.append(record)
        candidate_cache[str(record["id"])] = candidates
        if len(selected_records) >= PANEL_SIZE:
            break
        if index % 10 == 0:
            print(f"[scan] {index}/{len(records)} selected={len(selected_records)}", flush=True)

    with PANEL.open("w", encoding="utf-8") as handle:
        for record in selected_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    rows = []
    with OUTPUT.open("w", encoding="utf-8") as handle:
        for record in selected_records:
            rid = str(record["id"])
            candidates = candidate_cache[rid]
            for variant, forbidden in (("full", None), ("without_tcga", "tcga"), ("without_pubmed", "pubmed")):
                docs = select_context(candidates, forbidden)
                row = {
                    "record_id": rid,
                    "record": record,
                    "family": family(record),
                    "task_type": record.get("task_type"),
                    "variant": variant,
                    "query_text": build_query_text(record),
                    "retrieved_docs": docs,
                    "retrieval_audit": audit_retrieval(record, docs),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
    report = {
        "candidate_records": len(records),
        "selected_records": len(selected_records),
        "families": {name: sum(family(row) == name for row in selected_records) for name in ("hcc", "crc")},
        "candidate_k": CANDIDATE_K,
        "top_k": TOP_K,
        "rows": len(rows),
        "selection": "HCC/CRC task-semantic pool; >=2 direct disease-and-gene-matched TCGA and PubMed candidates; both ablations retain top-8; no outcomes used.",
    }
    (OUTPUT_DIR / "tcga_pubmed_build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from analyze_retrieval_evidence_quality import audit_retrieval
from rag_eval_lib import PreparedCorpus, PreparedDoc, build_query_text, read_jsonl, retrieve_naive
from run_rag_evaluation import serialize_doc


ROOT = Path(__file__).resolve().parents[1]
KB_DIR = ROOT / "dataset/rag/knowledge_sources/single_source_forms"
BENCHMARK = ROOT / "dataset/evaluated/external_reasoning_benchmark_formal_v3_571.jsonl"
OUTPUT_DIR = ROOT / "dataset/rag/analysis/evidence_meets_rag_single_source_forms"
OUTPUT = OUTPUT_DIR / "single_source_form_retrieval_571.jsonl"
SOURCES = ("tcga", "pubmed")
FORMS = ("document_corpus", "semantic_kg", "evidence_graph")
TOP_K = 8


def load_corpus(name: str) -> PreparedCorpus:
    docs = [PreparedDoc(**row) for row in read_jsonl(KB_DIR / f"{name}_docs.jsonl")]
    edges = read_jsonl(KB_DIR / f"{name}_edges.jsonl")
    return PreparedCorpus(docs, edges)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(BENCHMARK)
    row_count = 0
    with OUTPUT.open("w", encoding="utf-8") as handle:
        for source in SOURCES:
            for form in FORMS:
                name = f"{source}_{form}"
                print(f"[corpus] loading {name}", flush=True)
                corpus = load_corpus(name)
                for index, record in enumerate(records, start=1):
                    retrieved = retrieve_naive(corpus, record, TOP_K)
                    docs = []
                    for doc, score in retrieved:
                        relations = sorted({relation for _neighbor, relation, _weight in corpus.graph_neighbors.get(doc.doc_id, []) if relation not in {"HAS_CHUNK", "NEXT_CHUNK"}})[:10]
                        docs.append(serialize_doc(doc, score, relations))
                    row: dict[str, Any] = {
                        "record_id": str(record["id"]),
                        "record": record,
                        "source": source,
                        "form": form,
                        "knowledge_source": name,
                        "query_text": build_query_text(record),
                        "retrieved_docs": docs,
                        "retrieval_audit": audit_retrieval(record, docs),
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    row_count += 1
                    if index % 100 == 0 or index == len(records):
                        print(f"[retrieval] {name} {index}/{len(records)}", flush=True)
                del corpus
    report = {
        "benchmark_records": len(records),
        "sources": list(SOURCES),
        "forms": list(FORMS),
        "top_k": TOP_K,
        "retriever": "BM25 / Vanilla RAG",
        "rows": row_count,
    }
    (OUTPUT_DIR / "single_source_form_retrieval_run.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from analyze_retrieval_evidence_quality import audit_retrieval
from rag_eval_lib import build_query_text, load_prepared_corpus, read_jsonl, top_docs_from_scores
from run_rag_evaluation import serialize_doc


ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = ROOT / "dataset/rag/knowledge_sources"
PANEL = ROOT / "dataset/evaluated/external_reasoning_benchmark_formal_v3_571.jsonl"
OUTPUT_DIR = ROOT / "dataset/rag/analysis/evidence_meets_rag_hyperparameter_stability"
OUTPUT = OUTPUT_DIR / "retrieval_hyperparameter_stability_145.jsonl"
PARAMETER_PANEL_SIZE = 20
FORMS = {
    "document_corpus": "raw_observation_view",
    "semantic_kg": "semantic_kg",
    "evidence_graph": "evidence_graph_complete",
}
SETTINGS = [
    {"setting": "topk_2", "top_k": 2, "k1": 1.5, "b": 0.75, "factor": "top_k"},
    {"setting": "topk_4", "top_k": 4, "k1": 1.5, "b": 0.75, "factor": "top_k"},
    {"setting": "default", "top_k": 8, "k1": 1.5, "b": 0.75, "factor": "reference"},
    {"setting": "topk_16", "top_k": 16, "k1": 1.5, "b": 0.75, "factor": "top_k"},
    {"setting": "k1_0.9", "top_k": 8, "k1": 0.9, "b": 0.75, "factor": "k1"},
    {"setting": "k1_2.0", "top_k": 8, "k1": 2.0, "b": 0.75, "factor": "k1"},
    {"setting": "b_0.25", "top_k": 8, "k1": 1.5, "b": 0.25, "factor": "b"},
    {"setting": "b_1.0", "top_k": 8, "k1": 1.5, "b": 1.0, "factor": "b"},
]


def serialize(corpus: Any, ranked: list[tuple[Any, float]]) -> list[dict[str, Any]]:
    rows = []
    for doc, score in ranked:
        relations = sorted({relation for _neighbor, relation, _weight in corpus.graph_neighbors.get(doc.doc_id, []) if relation not in {"HAS_CHUNK", "NEXT_CHUNK"}})[:10]
        rows.append(serialize_doc(doc, score, relations))
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(PANEL)
    parameter_ids = {
        str(record["id"])
        for record in sorted(
            records,
            key=lambda row: hashlib.sha256(f"bm25-parameter-panel:{row['id']}".encode()).hexdigest(),
        )[:PARAMETER_PANEL_SIZE]
    }
    row_count = 0
    with OUTPUT.open("w", encoding="utf-8") as handle:
        for form, knowledge_source in FORMS.items():
            print(f"[corpus] {form} <- {knowledge_source}", flush=True)
            corpus = load_prepared_corpus(KNOWLEDGE_DIR, "multi_cancer", knowledge_source, use_cache=True)
            for index, record in enumerate(records, start=1):
                query = build_query_text(record)
                score_cache = {}
                doc_cache = {}
                for setting in SETTINGS:
                    if setting["factor"] in {"k1", "b"} and str(record["id"]) not in parameter_ids:
                        continue
                    score_key = (setting["k1"], setting["b"])
                    if score_key not in score_cache:
                        score_cache[score_key] = corpus.bm25_scores(query, k1=setting["k1"], b=setting["b"])
                    if score_key == (1.5, 0.75):
                        if score_key not in doc_cache:
                            doc_cache[score_key] = serialize(corpus, top_docs_from_scores(corpus, score_cache[score_key], 16))
                        docs = doc_cache[score_key][: setting["top_k"]]
                    else:
                        docs = serialize(corpus, top_docs_from_scores(corpus, score_cache[score_key], setting["top_k"]))
                    row = {
                        "record_id": str(record["id"]),
                        "form": form,
                        **setting,
                        "retrieval_audit": audit_retrieval(record, docs),
                        "doc_ids": [doc["doc_id"] for doc in docs],
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    row_count += 1
                if index % 25 == 0 or index == len(records):
                    print(f"[retrieval] {form} {index}/{len(records)}", flush=True)
            del corpus
    report = {
        "records": len(records),
        "bm25_parameter_panel_records": len(parameter_ids),
        "bm25_parameter_panel_selection": "fixed SHA-256 order independent of retrieval outcomes",
        "forms": list(FORMS),
        "settings": SETTINGS,
        "rows": row_count,
    }
    (OUTPUT_DIR / "retrieval_hyperparameter_run.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

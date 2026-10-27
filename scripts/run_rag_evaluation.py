from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any

from rag_eval_lib import (
    OpenAICompatibleClient,
    RETRIEVERS,
    build_answer_from_state_prompt,
    build_evidence_state_prompt,
    build_prompt,
    build_query_text,
    compute_answer_metrics,
    compute_retrieval_metrics,
    dense_rag_model_name,
    first_evidence_pack_doc,
    generation_stage_grounded,
    infer_state_demand_profile,
    inferred_evidence_state,
    load_prepared_corpus,
    map_answer_from_evidence_state,
    norm_text,
    normalize_prediction,
    read_json,
    read_jsonl,
    set_rag_profile,
    set_stage_config,
)


ROOT = Path(__file__).resolve().parents[1]
RAG_DIR = ROOT / "dataset" / "rag"
KNOWLEDGE_DIR = RAG_DIR / "knowledge_sources"
RESULTS_DIR = RAG_DIR / "results"
MATRIX_PATH = RAG_DIR / "experiment_matrix_latest.json"


def log_progress(message: str) -> None:
    print(message, flush=True)


def record_id(row: dict[str, Any]) -> str:
    return str((row.get("record") or {}).get("id") or "")


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: json_ready(item)
            for key, item in value.items()
            if key not in {"_scope_profile_cache"}
        }
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, set):
        return sorted(json_ready(item) for item in value)
    return value


def row_is_complete(row: dict[str, Any], retrieval_only: bool) -> bool:
    if not record_id(row):
        return False
    if retrieval_only:
        return bool(row.get("retrieved_docs"))
    return row.get("prediction") not in (None, "") and row.get("normalized_prediction") not in (None, "")


def load_existing_results(path: Path, retrieval_only: bool) -> dict[str, dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    existing: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rid = record_id(row)
            if rid and row_is_complete(row, retrieval_only):
                existing[rid] = row
    return existing


def append_result_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_ready(row), ensure_ascii=False) + "\n")


def serialize_doc(doc: Any, score: float, relation_hits: list[str]) -> dict[str, Any]:
    return {
        "doc_id": doc.doc_id,
        "knowledge_source": doc.knowledge_source,
        "document_level": doc.document_level,
        "parent_id": doc.parent_id,
        "title": doc.title,
        "score": round(score, 4),
        "disease_tags": doc.disease_tags,
        "entity_names": doc.entity_names[:12],
        "entity_ids": doc.entity_ids[:12],
        "text": doc.text[:1800],
        "metadata": {
            **doc.metadata,
            "relation_hits": relation_hits,
        },
    }


def retrieve_docs(corpus: Any, framework: str, record: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
    retrieved = RETRIEVERS[framework](corpus, record, top_k)
    serialized = []
    for doc, score in retrieved:
        relation_hits = []
        for neighbor_id, relation, _ in corpus.graph_neighbors.get(doc.doc_id, []):
            if relation not in {"HAS_CHUNK", "NEXT_CHUNK"}:
                relation_hits.append(relation)
        serialized.append(serialize_doc(doc, score, sorted(set(relation_hits))[:10]))
    return serialized


def validate_combo(matrix: dict[str, Any], view: str, knowledge_source: str, framework: str) -> dict[str, Any]:
    for experiment in matrix["experiments"]:
        if experiment.get("view") == view and experiment.get("knowledge_source") == knowledge_source and experiment.get("rag_framework") == framework:
            return experiment
    raise ValueError(f"unsupported experiment combination: {view} / {knowledge_source} / {framework}")


def answer_with_llm(records: list[dict[str, Any]], model: str | None) -> list[dict[str, Any]]:
    client = OpenAICompatibleClient(model=model)
    if not client.is_configured():
        raise RuntimeError("OPENAI_API_KEY is required for non-retrieval-only runs")
    outputs = []
    for row in records:
        prompt = build_prompt(row["record"], [(type("Doc", (), doc), doc["score"]) for doc in row["retrieved_docs"]])
        prediction = client.generate(prompt)
        row["prediction"] = prediction
        row["normalized_prediction"] = normalize_prediction(prediction, row["record"])
        outputs.append(row)
    return outputs


def build_context_blocks(row: dict[str, Any]) -> list[tuple[Any, float]]:
    context_blocks = []
    for doc in row["retrieved_docs"]:
        context_blocks.append(
            (
                type(
                    "Doc",
                    (),
                    {
                        "doc_id": doc["doc_id"],
                        "knowledge_source": doc["knowledge_source"],
                        "document_level": doc["document_level"],
                        "parent_id": doc["parent_id"],
                        "title": doc["title"],
                        "text": doc["text"],
                        "disease_tags": doc["disease_tags"],
                        "entity_names": doc["entity_names"],
                        "entity_ids": doc["entity_ids"],
                        "metadata": doc["metadata"],
                    },
                ),
                doc["score"],
            )
        )
    return context_blocks


def generate_prediction(client: OpenAICompatibleClient, row: dict[str, Any]) -> tuple[str, Any, dict[str, Any] | None, str | None]:
    context_blocks = build_context_blocks(row)
    if generation_stage_grounded() and first_evidence_pack_doc(context_blocks) is not None:
        demand_profile = infer_state_demand_profile(row["record"], context_blocks)
        if demand_profile.get("routing_strategy") == "direct_answer":
            prompt = build_prompt(row["record"], context_blocks)
            prediction = client.generate(prompt)
            normalized = normalize_prediction(prediction, row["record"])
            return prediction, normalized, {"routing_profile": demand_profile}, "adaptive_direct_prompt"

        state_prompt = build_evidence_state_prompt(row["record"], context_blocks, demand_profile)
        raw_state = client.generate(state_prompt)
        state = inferred_evidence_state(row["record"], context_blocks, raw_state)
        state["routing_profile"] = demand_profile
        mapped_answer = None
        if demand_profile.get("direct_mapping_preferred"):
            mapped_answer = map_answer_from_evidence_state(row["record"], state, demand_profile)
        if mapped_answer is None:
            answer_prompt = build_answer_from_state_prompt(row["record"], state, context_blocks, demand_profile)
            prediction = client.generate(answer_prompt)
            prediction_mode = "adaptive_state_then_llm_answer"
        else:
            prediction = mapped_answer
            prediction_mode = "adaptive_state_direct_mapping"
        normalized = normalize_prediction(prediction, row["record"])
        return prediction, normalized, state, prediction_mode
    prompt = build_prompt(row["record"], context_blocks)
    prediction = client.generate(prompt)
    normalized = normalize_prediction(prediction, row["record"])
    return prediction, normalized, None, "direct_prompt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unified RAG evaluation or retrieval-only previews.")
    parser.add_argument("--view", required=True, choices=["hcc_crc", "multi_cancer"])
    parser.add_argument(
        "--knowledge-source",
        required=True,
        choices=[
            "evidence_graph",
            "evidence_graph_complete",
            "evidence_graph_complete_label_masked",
            "evidence_graph_complete_no_structure",
            "evidence_graph_complete_no_evidence_text",
            "evidence_graph_complete_no_cross_source_links",
            "evidence_graph_complete_no_cross_layer_aggregation",
            "evidence_graph_complete_no_relation_types",
            "semantic_kg",
            "raw_multisource",
            "raw_observation_view",
            "pubmed_abstracts",
            "papers_fulltext",
            "primekg",
            "tarkg",
            "evidence_tarkg_hybrid",
            "raw_tarkg_hybrid",
        ],
    )
    parser.add_argument("--framework", required=True, choices=list(RETRIEVERS))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--rag-profile", choices=["tuned", "pure"], default="tuned")
    parser.add_argument("--retrieval-stage", choices=["base", "full"], default="full")
    parser.add_argument("--context-stage", choices=["flat", "structured"], default="structured")
    parser.add_argument("--generation-stage", choices=["plain", "grounded"], default="grounded")
    parser.add_argument("--no-cache", action="store_true", help="Disable persistent corpus cache loading.")
    parser.add_argument("--refresh-cache", action="store_true", help="Rebuild persistent corpus cache before running.")
    parser.add_argument("--dataset-path", default=None, help="Optional dataset JSONL path relative to repo root or absolute path.")
    parser.add_argument("--output-suffix", default=None, help="Optional suffix appended to output filenames to avoid overwriting other runs.")
    parser.add_argument("--generation-workers", type=int, default=1, help="Concurrent LLM generation workers. Defaults to 1.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing output JSONL by skipping completed records.")
    args = parser.parse_args()

    # The paper release supports the three controlled views without requiring
    # the development repository's generated experiment-matrix artifact.
    matrix = read_json(MATRIX_PATH) if MATRIX_PATH.exists() else {
        "views": {"multi_cancer": {"dataset_path": "dataset/evaluated/external_reasoning_benchmark_formal_v3_571.jsonl"}},
        "experiments": [{"view": "multi_cancer", "knowledge_source": source,
                         "rag_framework": framework, "status": "paper"}
                        for source in ("raw_observation_view", "semantic_kg", "evidence_graph_complete")
                        for framework in RETRIEVERS],
    }
    combo = validate_combo(matrix, args.view, args.knowledge_source, args.framework)
    set_rag_profile(args.rag_profile)
    set_stage_config(args.retrieval_stage, args.context_stage, args.generation_stage)
    output_stem = f"{args.view}__{args.knowledge_source}__{args.framework}"
    if args.output_suffix:
        safe_suffix = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in args.output_suffix).strip("_")
        if safe_suffix:
            output_stem = f"{output_stem}__{safe_suffix}"
    output_rows_path = RESULTS_DIR / f"{output_stem}.jsonl"
    output_summary_path = RESULTS_DIR / f"{output_stem}_summary.json"
    output_rows_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        if output_rows_path.exists():
            output_rows_path.unlink()
        if output_summary_path.exists():
            output_summary_path.unlink()

    corpus = load_prepared_corpus(
        KNOWLEDGE_DIR,
        args.view,
        args.knowledge_source,
        use_cache=not args.no_cache,
        refresh_cache=args.refresh_cache,
    )
    dataset_path = Path(args.dataset_path) if args.dataset_path else ROOT / matrix["views"][args.view]["dataset_path"]
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    records = read_jsonl(dataset_path)
    if args.limit > 0:
        records = records[: args.limit]
    existing_results = load_existing_results(output_rows_path, args.retrieval_only) if args.resume else {}
    pending_records = [record for record in records if str(record.get("id") or "") not in existing_results]
    log_progress(
        f"[INFO] view={args.view} knowledge_source={args.knowledge_source} framework={args.framework} "
        f"records={len(records)} retrieval_only={args.retrieval_only} rag_profile={args.rag_profile} "
        f"stages={args.retrieval_stage}/{args.context_stage}/{args.generation_stage} dataset={dataset_path}"
    )
    if args.resume:
        log_progress(f"[RESUME] completed={len(existing_results)} pending={len(pending_records)} output={output_rows_path}")

    results_by_id: dict[str, dict[str, Any]] = dict(existing_results)
    pending_results = []
    resolved_model = None
    for idx, record in enumerate(pending_records, start=1):
        retrieved_docs = retrieve_docs(corpus, args.framework, record, args.top_k)
        row = {
            "record": record,
            "query_text": build_query_text(record),
            "retrieved_docs": retrieved_docs,
            "prediction": None,
            "normalized_prediction": None,
            "inferred_evidence_state": None,
            "prediction_mode": None,
        }
        if args.retrieval_only:
            append_result_row(output_rows_path, row)
            results_by_id[str(record.get("id") or "")] = row
        else:
            pending_results.append(row)
        if idx % 10 == 0 or idx == len(pending_records):
            log_progress(f"[RETRIEVAL] {args.knowledge_source} {idx}/{len(pending_records)}")

    if not args.retrieval_only:
        client = OpenAICompatibleClient(model=args.model)
        resolved_model = client.model
        if not client.is_configured():
            raise RuntimeError("OPENAI_API_KEY is required for answer generation runs")
        if args.generation_workers <= 1:
            for idx, row in enumerate(pending_results, start=1):
                prediction, normalized, inferred_state, prediction_mode = generate_prediction(client, row)
                row["prediction"] = prediction
                row["normalized_prediction"] = normalized
                row["inferred_evidence_state"] = inferred_state
                row["prediction_mode"] = prediction_mode
                rid = str((row.get("record") or {}).get("id") or "")
                append_result_row(output_rows_path, row)
                results_by_id[rid] = row
                if idx % 5 == 0 or idx == len(pending_results):
                    log_progress(f"[GENERATION] {args.knowledge_source} {idx}/{len(pending_results)}")
        else:
            completed = 0
            with ThreadPoolExecutor(max_workers=args.generation_workers) as executor:
                future_to_idx = {
                    executor.submit(generate_prediction, client, row): idx
                    for idx, row in enumerate(pending_results)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    prediction, normalized, inferred_state, prediction_mode = future.result()
                    pending_results[idx]["prediction"] = prediction
                    pending_results[idx]["normalized_prediction"] = normalized
                    pending_results[idx]["inferred_evidence_state"] = inferred_state
                    pending_results[idx]["prediction_mode"] = prediction_mode
                    rid = str((pending_results[idx].get("record") or {}).get("id") or "")
                    append_result_row(output_rows_path, pending_results[idx])
                    results_by_id[rid] = pending_results[idx]
                    completed += 1
                    if completed % 5 == 0 or completed == len(pending_results):
                        log_progress(f"[GENERATION] {args.knowledge_source} {completed}/{len(pending_results)}")

    results = []
    for record in records:
        rid = str(record.get("id") or "")
        row = results_by_id.get(rid)
        if row is not None:
            results.append(row)
    if len(results) != len(records):
        missing = len(records) - len(results)
        raise RuntimeError(f"incomplete results: missing {missing} records; rerun with --resume to continue")

    summary = {
        "experiment": {
            "view": args.view,
            "knowledge_source": args.knowledge_source,
            "framework": args.framework,
            "matrix_status": combo.get("status"),
            "retrieval_only": args.retrieval_only,
            "rag_profile": args.rag_profile,
            "retrieval_stage": args.retrieval_stage,
            "context_stage": args.context_stage,
            "generation_stage": args.generation_stage,
            "top_k": args.top_k,
            "generation_workers": args.generation_workers,
            "model": resolved_model,
            "record_count": len(results),
            "resume_enabled": args.resume,
            "dataset_path": str(dataset_path),
        },
        "retrieval_metrics": compute_retrieval_metrics(results),
    }
    if args.framework == "dense_rag":
        summary["dense_retrieval"] = {
            "encoder": dense_rag_model_name(),
            "frozen": True,
            "embedding_dimension": 384,
            "max_sequence_length": 256,
            "normalization": "l2",
            "similarity": "cosine",
            "training_on_benchmark": False,
        }
    if not args.retrieval_only:
        summary["answer_metrics"] = compute_answer_metrics(results)
    with output_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    log_progress(f"[DONE] wrote {output_summary_path}")


if __name__ == "__main__":
    main()

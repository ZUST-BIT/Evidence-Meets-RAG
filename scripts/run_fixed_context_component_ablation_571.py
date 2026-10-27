from __future__ import annotations

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

from build_evidence_graph_ablation_suite import (
    is_aggregate_doc,
    is_summary_doc,
    minimal_evidence_text,
    node_type,
    plain_text_from_structured,
)
from rag_eval_lib import OpenAICompatibleClient, compute_answer_metrics, set_rag_profile, set_stage_config
from run_rag_evaluation import generate_prediction, json_ready


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "dataset/rag/results"
FRAMEWORKS = {
    "graph_rag": {
        "label": "Graph RAG",
        "source": RESULTS_DIR
        / "multi_cancer__evidence_graph_complete__graph_rag__formalv3_571_classic4_raw_vs_graph.jsonl",
        "output_prefix": "fixed_context_component_ablation_571",
    },
    "vanilla_rag": {
        "label": "Vanilla RAG",
        "source": RESULTS_DIR
        / "multi_cancer__evidence_graph_complete__naive_rag__formalv3_571_classic4_raw_vs_graph.jsonl",
        "output_prefix": "fixed_context_component_ablation_571_vanilla",
    },
}

CONDITIONS = [
    "full",
    "no_structure",
    "no_evidence_text",
    "no_cross_source_links",
    "no_cross_layer_aggregation",
    "no_relation_types",
]


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def record_id(row: dict[str, Any]) -> str:
    return str((row.get("record") or {}).get("id") or "")


def output_path(output_prefix: str, condition: str) -> Path:
    return RESULTS_DIR / f"{output_prefix}__{condition}.jsonl"


def summary_path(output_prefix: str) -> Path:
    return RESULTS_DIR / f"{output_prefix}__summary.json"


def remove_cross_source_cues(text: str) -> str:
    excluded = {
        "source families:",
        "source_families =",
        "cross_source_support",
        "cross-source support",
        "mechanism by source family",
    }
    return "\n".join(
        line for line in str(text or "").splitlines()
        if not any(cue in line.lower() for cue in excluded)
    )


def apply_condition(docs: list[dict[str, Any]], condition: str) -> list[dict[str, Any]]:
    if condition == "full":
        return deepcopy(docs)
    output: list[dict[str, Any]] = []
    for source in docs:
        doc = deepcopy(source)
        metadata = doc.get("metadata") or {}
        if condition == "no_structure":
            doc["text"] = plain_text_from_structured(doc)
            doc["parent_id"] = None
            doc["document_level"] = "flat"
            metadata["node_type"] = "flat_document"
            metadata.pop("_graph_rag_path_relations", None)
            metadata.pop("_graph_rag_causal_snippets", None)
            metadata.pop("relation_hits", None)
            metadata.pop("bundle_components", None)
            metadata.pop("summary_scope", None)
        elif condition == "no_evidence_text":
            if node_type(doc) == "evidence" or is_summary_doc(doc):
                doc["text"] = minimal_evidence_text(doc)
        elif condition == "no_cross_source_links":
            metadata["relation_hits"] = [
                relation
                for relation in metadata.get("relation_hits") or []
                if "SOURCE" not in str(relation).upper()
                and str(relation).upper() not in {"COMPLEMENTS", "SUPPORTS"}
            ]
            metadata.pop("source_family_counts", None)
            metadata.pop("source_target_counts", None)
            metadata.pop("source_mechanism_counts", None)
            doc["text"] = remove_cross_source_cues(str(doc.get("text") or ""))
        elif condition == "no_cross_layer_aggregation":
            if is_summary_doc(doc) or is_aggregate_doc(doc):
                continue
        elif condition == "no_relation_types":
            if metadata.get("relation_hits"):
                metadata["relation_hits"] = ["RELATED_TO"]
            if metadata.get("_graph_rag_path_relations"):
                metadata["_graph_rag_path_relations"] = ["RELATED_TO"]
            metadata.pop("_graph_rag_causal_snippets", None)
        else:
            raise ValueError(f"unknown condition: {condition}")
        metadata["fixed_context_ablation"] = condition
        doc["metadata"] = metadata
        output.append(doc)
    return output


def generate_one(
    client: OpenAICompatibleClient,
    source_row: dict[str, Any],
    condition: str,
) -> dict[str, Any]:
    row = {
        "record": deepcopy(source_row["record"]),
        "query_text": source_row.get("query_text"),
        "retrieved_docs": apply_condition(source_row.get("retrieved_docs") or [], condition),
        "prediction": None,
        "normalized_prediction": None,
        "inferred_evidence_state": None,
        "prediction_mode": None,
        "condition": condition,
    }
    prediction, normalized, state, mode = generate_prediction(client, row)
    row["prediction"] = prediction
    row["normalized_prediction"] = normalized
    row["inferred_evidence_state"] = state
    row["prediction_mode"] = mode
    row["model"] = client.model
    return json_ready(row)


def append_row(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fixed-context component ablation on all 571 records.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model", default=None)
    parser.add_argument("--framework", choices=FRAMEWORKS, default="graph_rag")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    parser.add_argument("--fresh", action="store_true", help="Discard completed rows for the selected conditions.")
    args = parser.parse_args()

    load_env_file(ROOT / ".env")
    framework = FRAMEWORKS[args.framework]
    source = framework["source"]
    output_prefix = str(framework["output_prefix"])
    source_rows = read_jsonl(source)
    if len(source_rows) != 571:
        raise ValueError(f"expected 571 source rows, found {len(source_rows)}")
    pack_sizes = [len(row.get("retrieved_docs") or []) for row in source_rows]
    if any(size < 1 or size > 8 for size in pack_sizes):
        raise ValueError("every source row must contain between one and eight retrieved units")

    set_rag_profile("tuned")
    set_stage_config("full", "structured", "grounded")
    client = OpenAICompatibleClient(model=args.model)
    if not client.is_configured():
        raise RuntimeError("OPENAI_API_KEY is required")

    locks = {condition: threading.Lock() for condition in args.conditions}
    completed: dict[str, dict[str, dict[str, Any]]] = {}
    pending: list[tuple[str, dict[str, Any]]] = []
    for condition in args.conditions:
        path = output_path(output_prefix, condition)
        if args.fresh and path.exists():
            path.unlink()
        existing = {record_id(row): row for row in read_jsonl(path) if record_id(row)}
        expected = {record_id(row): row for row in source_rows}
        if set(existing) - set(expected):
            raise ValueError(f"{condition}: cached output contains IDs outside the source cohort")
        for key, cached in existing.items():
            if (cached["record"] != expected[key]["record"]
                    or cached["retrieved_docs"] != apply_condition(expected[key]["retrieved_docs"], condition)
                    or cached.get("model", client.model) != client.model):
                raise ValueError(f"{condition}: cached input/model changed; select a clean output directory")
        completed[condition] = existing
        for source_row in source_rows:
            if record_id(source_row) not in existing:
                pending.append((condition, source_row))
        print(
            f"[condition] {condition}: completed={len(existing)} pending={571 - len(existing)}",
            flush=True,
        )

    print(
        f"[start] model={client.model} workers={args.workers} total_pending={len(pending)}",
        flush=True,
    )
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(generate_one, client, source_row, condition): (condition, record_id(source_row))
            for condition, source_row in pending
        }
        for future in as_completed(futures):
            condition, rid = futures[future]
            row = future.result()
            append_row(output_path(output_prefix, condition), row, locks[condition])
            completed[condition][rid] = row
            done += 1
            if done % 25 == 0 or done == len(pending):
                print(f"[generation] {done}/{len(pending)}", flush=True)

    summary: dict[str, Any] = {
        "protocol": {
            "source": str(source.relative_to(ROOT)),
            "record_count": 571,
            "retrieval_framework": framework["label"],
            "retrieval_depth": 8,
            "retrieval_policy": "fixed original retrieved evidence packs",
            "retrieved_pack_size_counts": {
                str(size): pack_sizes.count(size) for size in sorted(set(pack_sizes))
            },
            "rag_profile": "tuned",
            "retrieval_stage": "full",
            "context_stage": "structured",
            "generation_stage": "grounded",
            "model": client.model,
        },
        "conditions": {},
    }
    for condition in args.conditions:
        ordered = [completed[condition][record_id(source_row)] for source_row in source_rows]
        if len(ordered) != 571:
            raise RuntimeError(f"condition {condition} is incomplete")
        summary["conditions"][condition] = compute_answer_metrics(ordered)
    summary_path(output_prefix).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Gold-context oracle: generation upper bound with gold evidence injected.

For every benchmark record, the retrieved context is replaced by a single
synthetic document containing the record's gold ``evidence_text``. Retrieval
is thereby made perfect by construction, so the resulting Accuracy/Macro-F1
estimate the *use stage* ceiling of the pipeline: what the generator achieves
when evidence access is no longer a bottleneck.

The generator, prompts, and answer normalization are identical to the main
Vanilla RAG runs (GPT-5 mini via RAG_EVAL_MODEL). Because the synthetic
context is not an evidence pack, generation follows the same direct-prompt
path as the main-table Vanilla runs.

Output: dataset/rag/results/oracle_gold_context/oracle_gold_context_571.jsonl
        + summary JSON with overall and per-family metrics.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from rag_eval_lib import (
    OpenAICompatibleClient,
    gold_label,
    macro_f1_from_labels,
    set_rag_profile,
    set_stage_config,
)
from run_rag_evaluation import generate_prediction

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "dataset" / "evaluated" / "external_reasoning_benchmark_formal_v3_571.jsonl"
OUTPUT_DIR = ROOT / "dataset" / "rag" / "results" / "oracle_gold_context"
OUTPUT = OUTPUT_DIR / "oracle_gold_context_571.jsonl"
SUMMARY = OUTPUT_DIR / "oracle_gold_context_571_summary.json"

FAMILY_DIMENSIONS = [
    ("cross_source_mechanistic_tracing", "task_type", "mechanism_trace_intermediate_gene"),
    ("multi_evidence_logical_synthesis", "reasoning_type", "multi_evidence_logical_synthesis"),
    ("precision_oncology_evidence_judgment", "reasoning_type", "precision_oncology_evidence_judgment"),
    ("evidence_signal_interpretation", "reasoning_type", "evidence_signal_interpretation"),
    ("hierarchical_evidence_aggregation", "reasoning_type", "hierarchical_evidence_aggregation"),
    ("cross_source_corroboration", "reasoning_type", "cross_source_corroboration"),
]


def load_project_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def gold_context_row(record: dict[str, Any]) -> dict[str, Any]:
    entities = record.get("entities") or []
    if isinstance(entities, dict):
        entities = entities.values()
    entity_names = [str(e) for e in entities if isinstance(e, str)]
    doc = {
        "doc_id": f"GOLD-{record.get('id')}",
        "knowledge_source": "gold_evidence",
        "document_level": "parent",
        "parent_id": None,
        "title": "Gold evidence for this question",
        "score": 1.0,
        "disease_tags": [],
        "entity_names": entity_names,
        "entity_ids": [],
        "text": str(record.get("evidence_text") or ""),
        "metadata": {"oracle": True},
    }
    return {
        "record": record,
        "query_text": record.get("question"),
        "retrieved_docs": [doc],
    }


def family_of(record: dict[str, Any]) -> str | None:
    task_type = str(record.get("task_type") or (record.get("benchmark_annotation") or {}).get("task_type") or "")
    reasoning = str((record.get("benchmark_annotation") or {}).get("reasoning_type")
                    or (record.get("metadata") or {}).get("reasoning_type") or "")
    for key, group_by, group_value in FAMILY_DIMENSIONS:
        if group_by == "task_type" and task_type == group_value:
            return key
        if group_by == "reasoning_type" and reasoning == group_value:
            return key
    return None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
        gold = [gold_label(item["record"]) for item in items]
        pred = [str(item.get("normalized_prediction") or "") for item in items]
        exact = sum(1 for g, p in zip(gold, pred) if g == p)
        return {
            "count": len(items),
            "accuracy": round(exact / len(items), 4) if items else 0.0,
            "macro_f1": round(macro_f1_from_labels(gold, pred), 4) if items else 0.0,
        }

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        fam = family_of(row["record"])
        if fam:
            grouped[fam].append(row)
    return {
        "design": {
            "condition": "gold_context_oracle",
            "context": "single synthetic document containing the record's gold evidence_text",
            "generator": os.getenv("RAG_EVAL_MODEL", "gpt-5-mini"),
            "note": (
                "Diagnostic use-stage ceiling, not a deployable system: gold evidence text "
                "may state conclusions close to the answer, so this bounds the generation "
                "stage optimistically."
            ),
        },
        "overall": metrics(rows),
        "by_family": {fam: metrics(items) for fam, items in sorted(grouped.items())},
    }


def main() -> None:
    load_project_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    client = OpenAICompatibleClient()
    if not client.is_configured():
        raise RuntimeError("OPENAI_API_KEY is required")

    set_rag_profile("tuned")
    set_stage_config("full", "structured", "grounded")

    records = read_jsonl(BENCHMARK)
    expected_ids = {str(record["id"]) for record in records}
    if len(records) != 571 or len(expected_ids) != 571:
        raise ValueError("The oracle requires the complete 571-record benchmark with unique IDs")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # A failed or explicitly limited rerun must never leave a stale full summary.
    SUMMARY.unlink(missing_ok=True)

    completed: dict[str, dict[str, Any]] = {}
    if args.resume and OUTPUT.exists():
        for row in read_jsonl(OUTPUT):
            rid = str(row["record"]["id"])
            if rid not in expected_ids or rid in completed:
                raise ValueError("Oracle cache contains duplicate or unexpected record IDs")
            if row.get("normalized_prediction") not in (None, ""):
                completed[rid] = row
    elif OUTPUT.exists():
        OUTPUT.unlink()

    pending = [r for r in records if str(r["id"]) not in completed]
    if args.limit:
        pending = pending[: args.limit]
    print(f"total={len(records)} completed={len(completed)} pending={len(pending)}")

    def run_one(record: dict[str, Any]) -> dict[str, Any]:
        row = gold_context_row(record)
        prediction, normalized, state, mode = generate_prediction(client, row)
        return {
            "record": record,
            "query_text": row["query_text"],
            "retrieved_docs": row["retrieved_docs"],
            "prediction": prediction,
            "normalized_prediction": normalized,
            "inferred_evidence_state": json_ready(state),
            "prediction_mode": mode,
            "model": client.model,
        }

    results: list[dict[str, Any]] = list(completed.values())
    with OUTPUT.open("a", encoding="utf-8") as sink:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_one, record): record for record in pending}
            done = 0
            for future in as_completed(futures):
                record = futures[future]
                try:
                    row = future.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"FAIL {record.get('id')}: {exc}")
                    continue
                sink.write(json.dumps(json_ready(row), ensure_ascii=False) + "\n")
                sink.flush()
                results.append(row)
                done += 1
                if done % 25 == 0:
                    print(f"progress {done}/{len(pending)}")

    result_ids = [str(row["record"]["id"]) for row in results]
    complete = len(result_ids) == 571 and set(result_ids) == expected_ids
    if not args.limit and not complete:
        raise RuntimeError(f"Oracle incomplete: {len(set(result_ids))}/571 results; resume before reporting metrics")
    summary = summarize(results)
    summary["design"].update(record_count=len(results), expected_record_count=571, complete=complete)
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["overall"], indent=2))
    print(f"summary -> {SUMMARY}")


if __name__ == "__main__":
    main()

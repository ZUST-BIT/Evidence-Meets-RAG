from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

from rag_eval_lib import OpenAICompatibleClient, gold_label, set_rag_profile, set_stage_config
from run_rag_evaluation import generate_prediction, json_ready


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "dataset" / "evaluated" / "external_reasoning_benchmark_formal_v3_571.jsonl"
OUTPUT_DIR = ROOT / "dataset" / "rag" / "results" / "model_robustness"

SOURCES = {
    "doc_corpus": ROOT / "dataset/rag/results/multi_cancer__raw_observation_view__naive_rag__formalv3_571_classic4_raw_vs_graph.jsonl",
    "semantic_kg": ROOT / "dataset/rag/results/multi_cancer__semantic_kg__naive_rag__formalv3_571_classic4_semantic_kg.jsonl",
    "evidence_kg": ROOT / "dataset/rag/results/multi_cancer__evidence_graph_complete__naive_rag__formalv3_571_classic4_raw_vs_graph.jsonl",
}
MODEL_KEYS = ["RAG_EVAL_MODEL2", "RAG_EVAL_MODEL3", "RAG_EVAL_MODEL4", "RAG_EVAL_MODEL6"]
PROVIDER_ENV = {
    "RAG_EVAL_MODEL3": ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL"),
    "RAG_EVAL_MODEL6": ("QWEN_API_KEY", "QWEN_BASE_URL"),
}


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


def safe_name(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    completed = {}
    for row in read_jsonl(path):
        key = f"{row.get('knowledge_source')}::{row.get('record_id')}"
        if row.get("normalized_prediction") not in (None, ""):
            completed[key] = row
    return completed


def append_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_ready(row), ensure_ascii=False) + "\n")


def main() -> None:
    load_project_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True, choices=MODEL_KEYS)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Optional pending-row limit for endpoint health checks.")
    args = parser.parse_args()
    model_name = os.getenv(args.model_key)
    if not model_name:
        raise RuntimeError(f"{args.model_key} is not configured")
    if args.model_key in PROVIDER_ENV:
        key_env, url_env = PROVIDER_ENV[args.model_key]
        if not os.getenv(key_env) or not os.getenv(url_env):
            raise RuntimeError(f"{key_env} and {url_env} are required for {args.model_key}")
        os.environ["OPENAI_API_KEY"] = os.environ[key_env]
        os.environ["OPENAI_BASE_URL"] = os.environ[url_env]
    client = OpenAICompatibleClient(model=model_name)
    if not client.is_configured():
        raise RuntimeError("OPENAI_API_KEY is required")

    set_rag_profile("tuned")
    set_stage_config("full", "structured", "grounded")
    panel = read_jsonl(PANEL)
    panel_by_id = {str(row["id"]): row for row in panel}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / f"{safe_name(model_name)}.jsonl"
    if output.exists() and not args.resume:
        output.unlink()
    completed = load_completed(output) if args.resume else {}

    pending = []
    for source_key, path in SOURCES.items():
        base_rows = {str(row["record"]["id"]): row for row in read_jsonl(path)}
        for rid, record in panel_by_id.items():
            key = f"{source_key}::{rid}"
            if key in completed:
                continue
            row = deepcopy(base_rows[rid])
            row["record"] = record
            pending.append((key, source_key, row))
    if args.limit > 0:
        pending = pending[: args.limit]
    print(f"[start] model={model_name} completed={len(completed)} pending={len(pending)}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(generate_prediction, client, row): (key, source_key, row) for key, source_key, row in pending}
        done = 0
        failed = 0
        for future in as_completed(futures):
            key, source_key, row = futures[future]
            try:
                prediction, normalized, state, mode = future.result()
            except Exception as exc:
                failed += 1
                print(f"[error] {key}: {type(exc).__name__}: {exc}", flush=True)
                continue
            compact = {
                "record_id": str(row["record"]["id"]),
                "task_type": row["record"].get("task_type"),
                "oncology_focus": row["record"].get("oncology_focus"),
                "knowledge_source": source_key,
                "model_key": args.model_key,
                "model": model_name,
                "gold_label": gold_label(row["record"]),
                "prediction": prediction,
                "normalized_prediction": normalized,
                "correct": int(str(normalized) == gold_label(row["record"])),
                "prediction_mode": mode,
                "inferred_evidence_state": state,
            }
            append_row(output, compact)
            completed[key] = compact
            done += 1
            if done % 25 == 0 or done == len(pending):
                print(f"[generation] {done}/{len(pending)}", flush=True)
        if failed:
            print(f"[incomplete] failed={failed}; rerun with --resume", flush=True)
    if args.limit <= 0 and len(completed) != len(panel) * len(SOURCES):
        raise RuntimeError(f"Incomplete output: {len(completed)} rows")
    print(f"[done] {output} rows={len(completed)}", flush=True)


if __name__ == "__main__":
    main()

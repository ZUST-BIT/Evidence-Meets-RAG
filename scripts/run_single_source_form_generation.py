from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from rag_eval_lib import OpenAICompatibleClient, gold_label, read_jsonl, set_rag_profile, set_stage_config
from run_rag_evaluation import generate_prediction, json_ready


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "dataset/rag/analysis/evidence_meets_rag_single_source_forms/single_source_form_retrieval_571.jsonl"
OUTPUT = ROOT / "dataset/rag/results/single_source_form_generation.jsonl"


def load_env() -> None:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines() if (ROOT / ".env").exists() else []:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def family(row: dict[str, Any]) -> str:
    record = row["record"]
    return str((record.get("metadata") or {}).get("priority_cancer_family") or record.get("oncology_focus") or "").lower()


def key(row: dict[str, Any]) -> str:
    return f"{row['source']}::{row['form']}::{row['record_id']}"


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if OUTPUT.exists() and not args.resume:
        OUTPUT.unlink()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    completed = {}
    if args.resume and OUTPUT.exists():
        completed = {key(row): row for row in read_jsonl(OUTPUT) if row.get("normalized_prediction") not in {None, ""}}
    model = os.getenv("RAG_EVAL_MODEL", "gpt-5-mini")
    client = OpenAICompatibleClient(model=model)
    if not client.is_configured():
        raise RuntimeError("OPENAI_API_KEY is required")
    set_rag_profile("tuned")
    set_stage_config("full", "structured", "grounded")
    pending = []
    for row in read_jsonl(INPUT):
        if row["source"] == "pubmed" and family(row) not in {"hcc", "crc"}:
            continue
        if key(row) in completed:
            continue
        item = dict(row)
        item["model"] = model
        item["evaluation_scope"] = "pan_cancer_571" if row["source"] == "tcga" else "hcc_crc_127"
        pending.append(item)
    print(f"[generation] model={model} completed={len(completed)} pending={len(pending)}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(generate_prediction, client, row): row for row in pending}
        done = 0
        for future in as_completed(futures):
            row = futures[future]
            prediction, normalized, state, mode = future.result()
            row.update({
                "prediction": prediction,
                "normalized_prediction": normalized,
                "correct": int(str(normalized) == gold_label(row["record"])),
                "prediction_mode": mode,
                "inferred_evidence_state": state,
            })
            with OUTPUT.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(json_ready(row), ensure_ascii=False) + "\n")
            done += 1
            if done % 100 == 0 or done == len(pending):
                print(f"[generation] {done}/{len(pending)}", flush=True)


if __name__ == "__main__":
    main()

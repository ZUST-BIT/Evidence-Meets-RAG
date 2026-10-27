from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

from rag_eval_lib import OpenAICompatibleClient, gold_label, read_jsonl, set_rag_profile, set_stage_config
from run_rag_evaluation import generate_prediction, json_ready


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "dataset/rag/analysis/evidence_meets_rag_tcga_pubmed/tcga_pubmed_controlled_retrieval.jsonl"
OUTPUT = ROOT / "dataset/rag/results/tcga_pubmed_complementarity.jsonl"
GUIDANCE = (
    "\nEvidence-role procedure: first infer the cancer-specific molecular manifestation from TCGA patient-omics evidence; "
    "then infer the mechanism or causal interpretation from PubMed literature evidence; finally map the available role-specific "
    "evidence to the allowed external-knowledge answer label. A source may be absent under ablation, so do not invent its evidence."
)


def load_env() -> None:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines() if (ROOT / ".env").exists() else []:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def row_key(row: dict[str, Any]) -> str:
    return f"{row['variant']}::{row['record_id']}"


def main() -> None:
    print("[startup] loading environment", flush=True)
    load_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if OUTPUT.exists() and not args.resume:
        OUTPUT.unlink()
    done = {(r["record_id"], r["variant"]) for r in read_jsonl(OUTPUT)} if OUTPUT.exists() else set()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    model = os.getenv("RAG_EVAL_MODEL", "gpt-5-mini")
    client = OpenAICompatibleClient(model=model)
    if not client.is_configured():
        raise RuntimeError("OPENAI_API_KEY is required")
    set_rag_profile("tuned")
    set_stage_config("full", "structured", "grounded")
    print("[startup] loading retrieval rows", flush=True)
    pending = []
    for source in read_jsonl(INPUT):
        if (source["record_id"], source["variant"]) in done:
            continue
        row = deepcopy(source)
        row["record"]["original_question"] = row["record"]["question"]
        row["record"]["question"] = row["record"]["question"] + GUIDANCE
        row["model"] = model
        pending.append(row)
    print(f"[generation] model={model} pending={len(pending)}", flush=True)
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
                "generation_policy": "tcga_pubmed_role_aware_v1",
            })
            with OUTPUT.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(json_ready(row), ensure_ascii=False) + "\n")
            done += 1
            if done % 15 == 0 or done == len(pending):
                print(f"[generation] {done}/{len(pending)}", flush=True)


if __name__ == "__main__":
    main()

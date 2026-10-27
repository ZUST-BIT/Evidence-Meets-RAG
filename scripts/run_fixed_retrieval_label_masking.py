from __future__ import annotations

import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

from build_answer_label_masking_experiment import sanitize_mapping
from rag_eval_lib import OpenAICompatibleClient, compute_answer_metrics, compute_retrieval_metrics, set_rag_profile, set_stage_config
from run_rag_evaluation import generate_prediction, json_ready


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "dataset" / "evaluated" / "ecorag_label_masking_panel_120.jsonl"
ORIGINAL = ROOT / "dataset" / "rag" / "results" / "multi_cancer__evidence_graph_complete__naive_rag__formalv3_571_classic4_raw_vs_graph.jsonl"
MASKED_OUTPUT = ROOT / "dataset" / "rag" / "results" / "multi_cancer__evidence_graph_complete_label_masked__naive_rag__labelmask120.jsonl"
ORIGINAL_REPLAY_OUTPUT = ROOT / "dataset" / "rag" / "results" / "multi_cancer__evidence_graph_complete__naive_rag__labelmask120_original_replay.jsonl"

DROP_PREFIXES = {
    "claim",
    "predicate",
    "outcome",
    "summary",
    "direction",
    "polarity",
    "evidence category",
    "support flags",
    "example claims",
    "example linked semantic labels",
    "example linked evidence predicates",
    "example linked evidence claims",
    "association evidence inventory",
    "association components present",
    "bundle component inventory",
    "driver-role signal inventory",
    "predominant driver-role signal",
    "mechanism buckets",
    "mechanism by source family",
    "predicates",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def scrub_context_text(text: str) -> str:
    output = []
    skip_list = False
    for line in str(text or "").splitlines():
        stripped = line.strip()
        prefix = stripped.split(":", 1)[0].lower()
        if prefix in DROP_PREFIXES:
            skip_list = prefix in {"example claims", "example linked evidence claims"}
            continue
        if skip_list and stripped.startswith("-"):
            continue
        skip_list = False
        output.append(line)
    return "\n".join(output)


def mask_retrieved_doc(doc: dict[str, Any]) -> dict[str, Any]:
    masked = deepcopy(doc)
    original_len = len(str(masked.get("text") or ""))
    masked["title"] = "Evidence record: " + " | ".join(
        str(item) for item in (masked.get("entity_names") or [])[:3]
    )
    masked["text"] = scrub_context_text(masked.get("text") or "")[:original_len]
    removed = __import__("collections").Counter()
    masked["metadata"] = sanitize_mapping(masked.get("metadata") or {}, removed)
    if (doc.get("metadata") or {}).get("relation_hits"):
        masked["metadata"]["relation_hits"] = ["RELATED_TO"]
    masked["metadata"]["masking_variant"] = "fixed_retrieval_global_label_mask_v1"
    masked["knowledge_source"] = "evidence_graph_complete_label_masked"
    return masked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["original", "masked"], default="masked")
    # Defaults reproduce the original 120-record run; --panel/--tag let the same
    # fixed-retrieval procedure be applied to a different panel without forking
    # the script.
    parser.add_argument("--panel", type=Path, default=PANEL)
    parser.add_argument("--tag", default="labelmask120")
    args = parser.parse_args()
    if args.tag == "labelmask120":
        output_path = MASKED_OUTPUT if args.condition == "masked" else ORIGINAL_REPLAY_OUTPUT
    else:
        results_dir = MASKED_OUTPUT.parent
        output_path = results_dir / (
            f"multi_cancer__evidence_graph_complete_label_masked__naive_rag__{args.tag}.jsonl"
            if args.condition == "masked"
            else f"multi_cancer__evidence_graph_complete__naive_rag__{args.tag}_original_replay.jsonl"
        )
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    set_rag_profile("tuned")
    set_stage_config("full", "structured", "grounded")
    panel = read_jsonl(args.panel)
    panel_by_id = {str(row["id"]): row for row in panel}
    original = [row for row in read_jsonl(ORIGINAL) if str((row.get("record") or {}).get("id")) in panel_by_id]
    original.sort(key=lambda row: str((row.get("record") or {}).get("id")))
    rows = []
    for row in original:
        prepared = deepcopy(row)
        rid = str((prepared.get("record") or {}).get("id"))
        prepared["record"] = panel_by_id[rid]
        if args.condition == "masked":
            prepared["retrieved_docs"] = [mask_retrieved_doc(doc) for doc in prepared.get("retrieved_docs") or []]
        prepared["prediction"] = None
        prepared["normalized_prediction"] = None
        prepared["inferred_evidence_state"] = None
        prepared["prediction_mode"] = None
        rows.append(prepared)
    if len(rows) != len(panel):
        raise RuntimeError(f"Expected {len(panel)} original rows, found {len(rows)}")

    client = OpenAICompatibleClient()
    if not client.is_configured():
        raise RuntimeError("OPENAI_API_KEY is required")
    completed = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(generate_prediction, client, row): index for index, row in enumerate(rows)}
        done = 0
        for future in as_completed(futures):
            index = futures[future]
            prediction, normalized, state, mode = future.result()
            rows[index]["prediction"] = prediction
            rows[index]["normalized_prediction"] = normalized
            rows[index]["inferred_evidence_state"] = state
            rows[index]["prediction_mode"] = mode
            completed[index] = rows[index]
            done += 1
            if done % 10 == 0 or done == len(rows):
                print(f"[generation] {done}/{len(rows)}", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in completed:
            handle.write(json.dumps(json_ready(row), ensure_ascii=False) + "\n")
    summary = {
        "experiment": {
            "design": "fixed_retrieval_global_label_mask_v1",
            "condition": args.condition,
            "record_count": len(rows),
            "framework": "naive_rag",
            "top_k": 8,
            "retrieval_policy": "Reuse the exact Original Evidence KG top-8 doc IDs, scores, and order; mask answer-bearing content before regeneration.",
        },
        "retrieval_metrics": compute_retrieval_metrics(completed),
        "answer_metrics": compute_answer_metrics(completed),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[done] {output_path}", flush=True)


if __name__ == "__main__":
    main()

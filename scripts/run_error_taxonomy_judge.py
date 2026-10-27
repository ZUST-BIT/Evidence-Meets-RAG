from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from error_taxonomy_config import ERROR_TAXONOMY, TASK_CANDIDATES, all_taxonomy_ids, taxonomy_by_id
from run_faithfulness_judge import JUDGE_SYSTEM_PROMPT, JudgeClient, compact_context, read_jsonl, record_id
from run_significance_analysis import gold_answer, predicted_answer


ROOT = Path(__file__).resolve().parents[1]

SYSTEM_PLAN = {
    "doc_vanilla": {
        "framework": "Vanilla RAG",
        "knowledge_source": "Doc Corpus",
        "result_path": ROOT / "dataset" / "rag" / "results" / "multi_cancer__raw_observation_view__naive_rag__formalv3_571_classic4_raw_vs_graph.jsonl",
    },
    "semantic_vanilla": {
        "framework": "Vanilla RAG",
        "knowledge_source": "Semantic KG",
        "result_path": ROOT / "dataset" / "rag" / "results" / "multi_cancer__semantic_kg__naive_rag__formalv3_571_classic4_semantic_kg.jsonl",
    },
    "evidence_vanilla": {
        "framework": "Vanilla RAG",
        "knowledge_source": "Evidence KG",
        "result_path": ROOT / "dataset" / "rag" / "results" / "fixed_context_component_ablation_571_vanilla__full.jsonl",
    },
}


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(json.loads(line).get("id") or "") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def format_taxonomy(task_type: str) -> tuple[list[str], str]:
    allowed = TASK_CANDIDATES.get(task_type) or all_taxonomy_ids()
    by_id = taxonomy_by_id()
    lines = []
    for item_id in allowed:
        item = by_id[item_id]
        lines.append(f"- {item_id}: {item['name']} — {item['definition']}")
    return allowed, "\n".join(lines)


def build_prompt(row: dict[str, Any]) -> tuple[str, list[str]]:
    record = row["record"]
    task_type = str(record.get("task_type") or "")
    allowed, taxonomy_text = format_taxonomy(task_type)
    context = compact_context(row.get("retrieved_docs") or [], max_docs=6, max_chars=3200)
    return (
        "Assign the primary error type for a wrong biomedical RAG answer.\n"
        "Choose the single most important reason the answer is wrong.\n"
        "Use only the question, gold answer, model prediction, and retrieved context.\n"
        "If multiple issues exist, pick the earliest bottleneck.\n\n"
        "Allowed error types:\n"
        f"{taxonomy_text}\n\n"
        "Rules:\n"
        "- Prefer sparse_missing_evidence when a crucial fact is simply absent.\n"
        "- Prefer bundle_incompleteness when multiple evidence components are required but the context covers only part of the bundle.\n"
        "- Prefer conflict_resolution_failure when the context contains mixed or layered evidence and the model resolves it incorrectly.\n"
        "- Prefer multi_hop_reasoning_failure when the answer depends on connecting intermediate entities or steps.\n"
        "- Prefer label_granularity_mapping_failure when relevant evidence exists but the final label, tier, or format is mapped incorrectly.\n"
        "- Prefer scope_distractor_mismatch when the model follows off-target or off-disease evidence.\n"
        'Return JSON only: {"primary_error_type":"...", "secondary_error_type":"", "brief_reason":"..."}\n\n'
        f"task_type: {task_type}\n"
        f"question: {record.get('question')}\n"
        f"gold_answer: {gold_answer(record)}\n"
        f"model_prediction_label: {predicted_answer(row)}\n"
        f"model_prediction_text: {row.get('prediction')}\n\n"
        f"context:\n{context}"
    ), allowed


def parse_judgment(text: str, allowed: list[str]) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Judge response is not JSON: {text}")
    payload = json.loads(match.group(0))
    primary = str(payload.get("primary_error_type") or "").strip()
    secondary = str(payload.get("secondary_error_type") or "").strip()
    brief_reason = str(payload.get("brief_reason") or "").strip()
    allowed_set = set(allowed)
    if primary not in allowed_set:
        raise ValueError(f"Primary error type {primary!r} is not allowed: {allowed}")
    if secondary and secondary not in allowed_set:
        raise ValueError(f"Secondary error type {secondary!r} is not allowed: {allowed}")
    if secondary == primary:
        secondary = ""
    return {
        "primary_error_type": primary,
        "secondary_error_type": secondary,
        "brief_reason": brief_reason,
    }


def judge_with_repair(client: JudgeClient, row: dict[str, Any]) -> dict[str, Any]:
    prompt, allowed = build_prompt(row)
    raw = client.complete(JUDGE_SYSTEM_PROMPT, prompt)
    try:
        return parse_judgment(raw, allowed)
    except ValueError as exc:
        repaired = client.complete(
            JUDGE_SYSTEM_PROMPT,
            prompt
            + "\n\nYour previous answer violated the schema or used a disallowed label.\n"
            + f"Previous answer: {raw}\n"
            + f"Problem: {exc}\n"
            + "Return valid JSON only.",
        )
        return parse_judgment(repaired, allowed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM-based error taxonomy labeling over wrong RAG answers.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--systems", default="doc_vanilla,semantic_vanilla,evidence_vanilla")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    client = JudgeClient(model=args.model)
    systems = [item.strip() for item in args.systems.split(",") if item.strip()]
    for system_id in systems:
        if system_id not in SYSTEM_PLAN:
            raise RuntimeError(f"Unknown system_id: {system_id}")
        spec = SYSTEM_PLAN[system_id]
        rows = read_jsonl(spec["result_path"])
        wrong_rows = [row for row in rows if predicted_answer(row) != gold_answer(row["record"])]
        output_path = output_dir / f"{system_id}.jsonl"
        done_ids = load_existing_ids(output_path)
        for index, row in enumerate(wrong_rows, start=1):
            rid = record_id(row)
            if rid in done_ids:
                continue
            judgment = judge_with_repair(client, row)
            record = row["record"]
            out = {
                "id": rid,
                "system_id": system_id,
                "framework": spec["framework"],
                "knowledge_source": spec["knowledge_source"],
                "task_type": record.get("task_type"),
                "oncology_focus": record.get("oncology_focus"),
                "question": record.get("question"),
                "gold_label": gold_answer(record),
                "prediction_label": predicted_answer(row),
                "prediction_text": row.get("prediction"),
                "primary_error_type": judgment["primary_error_type"],
                "secondary_error_type": judgment["secondary_error_type"],
                "brief_reason": judgment["brief_reason"],
                "judge_model": client.model,
            }
            append_jsonl(output_path, out)
            print(
                f"[{system_id}] {index}/{len(wrong_rows)} {rid} -> {out['primary_error_type']}",
                flush=True,
            )


if __name__ == "__main__":
    main()

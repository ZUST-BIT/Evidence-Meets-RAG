from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests

from rag_eval_lib import compact_text_snippet, gold_label, norm_text, normalize_prediction


ROOT = Path(__file__).resolve().parents[1]

SYSTEM_PLAN = [
    {
        "system_id": "doc_vanilla",
        "panel": "knowledge_form",
        "framework": "Vanilla RAG",
        "knowledge_source": "Doc Corpus",
        "result_path": ROOT / "dataset" / "rag" / "results" / "multi_cancer__raw_observation_view__naive_rag__formalv3_571_classic4_raw_vs_graph.jsonl",
    },
    {
        "system_id": "semantic_vanilla",
        "panel": "knowledge_form",
        "framework": "Vanilla RAG",
        "knowledge_source": "Semantic KG",
        "result_path": ROOT / "dataset" / "rag" / "results" / "multi_cancer__semantic_kg__naive_rag__formalv3_571_classic4_semantic_kg.jsonl",
    },
    {
        "system_id": "evidence_vanilla",
        "panel": "knowledge_form",
        "framework": "Vanilla RAG",
        "knowledge_source": "Evidence KG",
        "result_path": ROOT / "dataset" / "rag" / "results" / "multi_cancer__evidence_graph_complete__naive_rag__formalv3_571_classic4_raw_vs_graph.jsonl",
    },
    {
        "system_id": "evidence_dense",
        "panel": "framework",
        "framework": "Dense RAG",
        "knowledge_source": "Evidence KG",
        "result_path": ROOT / "dataset" / "rag" / "results" / "multi_cancer__evidence_graph_complete__dense_rag__formalv3_571_classic4_raw_vs_graph.jsonl",
    },
    {
        "system_id": "evidence_graph",
        "panel": "framework",
        "framework": "Graph RAG",
        "knowledge_source": "Evidence KG",
        "result_path": ROOT / "dataset" / "rag" / "results" / "multi_cancer__evidence_graph_complete__graph_rag__formalv3_571_classic4_raw_vs_graph.jsonl",
    },
    {
        "system_id": "evidence_crag",
        "panel": "framework",
        "framework": "CRAG",
        "knowledge_source": "Evidence KG",
        "result_path": ROOT / "dataset" / "rag" / "results" / "multi_cancer__evidence_graph_complete__crag_rag__formalv3_571_classic4_raw_vs_graph.jsonl",
    },
    {
        "system_id": "evidence_raptor",
        "panel": "framework",
        "framework": "RAPTOR",
        "knowledge_source": "Evidence KG",
        "result_path": ROOT / "dataset" / "rag" / "results" / "multi_cancer__evidence_graph_complete__raptor_rag__formalv3_571_cross_raptor_kg2rag.jsonl",
    },
    {
        "system_id": "evidence_kg2rag",
        "panel": "framework",
        "framework": "KG2RAG",
        "knowledge_source": "Evidence KG",
        "result_path": ROOT / "dataset" / "rag" / "results" / "multi_cancer__evidence_graph_complete__kg2rag__formalv3_571_cross_raptor_kg2rag.jsonl",
    },
]

JUDGE_SYSTEM_PROMPT = (
    "You are a strict biomedical evidence-grounding judge. "
    "Use only the provided context. Do not use outside knowledge. "
    "Return compact JSON only."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def record_id(row: dict[str, Any]) -> str:
    return str((row.get("record") or row).get("id") or "")


def predicted_answer(row: dict[str, Any]) -> str:
    value = row.get("normalized_prediction")
    if value not in (None, ""):
        return str(value)
    return normalize_prediction(str(row.get("prediction") or ""), row.get("record") or {})


class JudgeClient:
    def __init__(self, model: str | None = None) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = model or os.getenv("JUDGE_MODEL") or "gpt-5.4-mini"
        self.disable_proxy = os.getenv("OPENAI_DISABLE_PROXY", "").strip().lower() in {"1", "true", "yes", "on"}
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        session = requests.Session()
        if self.disable_proxy:
            session.trust_env = False
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
        }
        transient = {429, 500, 502, 503, 504}
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = session.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=120,
                )
                if response.status_code in transient:
                    response.raise_for_status()
                response.raise_for_status()
                body = response.json()
                return body["choices"][0]["message"]["content"].strip()
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status not in transient or attempt == 2:
                    raise
                last_error = exc
                time.sleep(2 * (attempt + 1))
            except requests.RequestException as exc:
                if attempt == 2:
                    raise
                last_error = exc
                time.sleep(2 * (attempt + 1))
        if last_error is not None:
            raise last_error
        raise RuntimeError("Judge request failed without a response")


def compact_context(retrieved_docs: list[dict[str, Any]], *, max_docs: int = 6, max_chars: int = 3600) -> str:
    blocks: list[str] = []
    for idx, doc in enumerate(retrieved_docs[:max_docs], start=1):
        metadata = doc.get("metadata") or {}
        relation_hits = metadata.get("relation_hits") or []
        block = (
            f"[{idx}] source={doc.get('knowledge_source') or 'unknown'} "
            f"level={doc.get('document_level') or 'unknown'}\n"
            f"title: {compact_text_snippet(str(doc.get('title') or ''), 180)}\n"
            f"diseases: {', '.join((doc.get('disease_tags') or [])[:6])}\n"
            f"entities: {', '.join((doc.get('entity_names') or [])[:10])}\n"
            f"relation_hits: {', '.join(relation_hits[:8])}\n"
            f"text: {compact_text_snippet(str(doc.get('text') or ''), 520)}"
        )
        blocks.append(block)
    context = "\n\n".join(blocks).strip()
    if len(context) <= max_chars:
        return context
    return context[: max_chars - 3] + "..."


def build_prompt(row: dict[str, Any]) -> str:
    record = row["record"]
    gold = record.get("gold_standard") or {}
    context = compact_context(row.get("retrieved_docs") or [])
    return (
        "Judge three binary labels for a biomedical RAG answer.\n"
        "Definitions:\n"
        "1. retrieval_sufficient=1 only if the context is sufficient to justify the gold answer.\n"
        "2. answer_supported=1 only if the context supports the model answer's main claim.\n"
        "3. answer_contradicted=1 only if the context explicitly conflicts with the model answer.\n"
        "Rules:\n"
        "- Be conservative.\n"
        "- If the context is insufficient, answer_supported should usually be 0.\n"
        "- Missing evidence is not contradiction.\n"
        "- answer_supported and answer_contradicted cannot both be 1.\n"
        "- contradiction=1 only when the context supports an opposite conclusion to the model answer.\n"
        "- If the model answer is contradicted, then answer_supported must be 0.\n"
        "- Use only the provided context.\n"
        'Return JSON only: {"retrieval_sufficient":0,"answer_supported":0,"answer_contradicted":0,"note":"short phrase"}\n\n'
        f"task_type: {record.get('task_type')}\n"
        f"question: {record.get('question')}\n"
        f"gold_answer: {gold.get('normalized_answer') or record.get('answer')}\n"
        f"gold_canonical: {gold_label(record)}\n"
        f"model_prediction_text: {row.get('prediction')}\n"
        f"model_prediction_label: {predicted_answer(row)}\n\n"
        f"context:\n{context}"
    )


def parse_judgment(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Judge response is not JSON: {text}")
    payload = json.loads(match.group(0))
    result = {
        "retrieval_sufficient": int(payload["retrieval_sufficient"]),
        "answer_supported": int(payload["answer_supported"]),
        "answer_contradicted": int(payload["answer_contradicted"]),
        "note": str(payload.get("note") or "").strip(),
    }
    for key in ("retrieval_sufficient", "answer_supported", "answer_contradicted"):
        if result[key] not in {0, 1}:
            raise ValueError(f"Judge field {key} must be 0/1: {payload}")
    if result["answer_supported"] == 1 and result["answer_contradicted"] == 1:
        raise ValueError(f"Judge returned incompatible labels: {payload}")
    return result


def judge_with_repair(client: JudgeClient, row: dict[str, Any]) -> dict[str, Any]:
    prompt = build_prompt(row)
    raw = client.complete(JUDGE_SYSTEM_PROMPT, prompt)
    try:
        return parse_judgment(raw)
    except ValueError as exc:
        repair_prompt = (
            prompt
            + "\n\nYour previous answer violated the schema or logic constraints.\n"
            + f"Previous answer: {raw}\n"
            + f"Problem: {exc}\n"
            + "Re-evaluate and return valid JSON only."
        )
        repaired = client.complete(JUDGE_SYSTEM_PROMPT, repair_prompt)
        return parse_judgment(repaired)


def load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(json.loads(line).get("id") or "") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM-based faithfulness judgments over existing RAG result files.")
    parser.add_argument("--subset", type=Path, required=True, help="Subset jsonl built from the 571 benchmark")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for judged jsonl files")
    parser.add_argument("--model", default=None, help="Judge model override; defaults to JUDGE_MODEL")
    parser.add_argument("--systems", default=None, help="Comma-separated system ids to run; defaults to all")
    args = parser.parse_args()

    subset_path = args.subset if args.subset.is_absolute() else ROOT / args.subset
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    subset_rows = read_jsonl(subset_path)
    subset_ids = [record_id(row) for row in subset_rows]
    subset_id_set = set(subset_ids)
    client = JudgeClient(model=args.model)
    requested = {item.strip() for item in (args.systems or "").split(",") if item.strip()}
    systems = [item for item in SYSTEM_PLAN if not requested or item["system_id"] in requested]
    for item in systems:
        result_rows = {record_id(row): row for row in read_jsonl(item["result_path"]) if record_id(row) in subset_id_set}
        output_path = output_dir / f"{item['system_id']}.jsonl"
        done_ids = load_existing_ids(output_path)
        for index, rid in enumerate(subset_ids, start=1):
            if rid in done_ids:
                continue
            if rid not in result_rows:
                raise RuntimeError(f"Missing record {rid} in {item['result_path']}")
            row = result_rows[rid]
            judgment = judge_with_repair(client, row)
            record = row["record"]
            out = {
                "id": rid,
                "system_id": item["system_id"],
                "panel": item["panel"],
                "framework": item["framework"],
                "knowledge_source": item["knowledge_source"],
                "task_type": record.get("task_type"),
                "oncology_focus": record.get("oncology_focus"),
                "question": record.get("question"),
                "gold_answer": (record.get("gold_standard") or {}).get("normalized_answer") or record.get("answer"),
                "gold_label": gold_label(record),
                "prediction_text": row.get("prediction"),
                "prediction_label": predicted_answer(row),
                "correct": int(predicted_answer(row) == gold_label(record)),
                "retrieved_doc_count": len(row.get("retrieved_docs") or []),
                "judge_model": client.model,
                "judge_note": judgment["note"],
                "retrieval_sufficient": judgment["retrieval_sufficient"],
                "answer_supported": judgment["answer_supported"],
                "answer_contradicted": judgment["answer_contradicted"],
            }
            append_jsonl(output_path, out)
            print(
                f"[{item['system_id']}] {index}/{len(subset_ids)} "
                f"{rid} -> S{out['retrieval_sufficient']} A{out['answer_supported']} C{out['answer_contradicted']}",
                flush=True,
            )


if __name__ == "__main__":
    main()

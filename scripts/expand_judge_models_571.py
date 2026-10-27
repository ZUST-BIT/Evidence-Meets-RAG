"""Extend the original faithfulness and generator panels to all 571 records.

Old result files are immutable.  The canonical Evidence Graph Vanilla run is
the already-completed Full condition of the fixed-context component ablation.
``prepare`` validates and copies only cache entries with matching inputs; it
does not make network requests.  ``judge`` and ``models`` perform resumable API
work, beginning with one real pending example as an endpoint health check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests

from analyze_multi_model_robustness import CORE_TASKS, MODELS, SOURCES, accuracy, compare, exact_mcnemar, summarize
from rag_eval_lib import OpenAICompatibleClient, gold_label, set_rag_profile, set_stage_config
from run_faithfulness_judge import (
    JUDGE_SYSTEM_PROMPT, JudgeClient, SYSTEM_PLAN, build_prompt as judge_prompt,
    judge_with_repair, predicted_answer, read_jsonl,
)
from run_multi_model_robustness import MODEL_KEYS, PROVIDER_ENV, load_project_env, safe_name
from run_rag_evaluation import build_context_blocks, generate_prediction
from rag_eval_lib import build_prompt as generator_prompt
from summarize_faithfulness_results import summarize_rows

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "dataset/rag/results"
OUTPUT = RESULTS / "unified_571"
CANONICAL = RESULTS / "fixed_context_component_ablation_571_vanilla__full.jsonl"
SOURCE_PATHS = {
    "doc_corpus": RESULTS / "multi_cancer__raw_observation_view__naive_rag__formalv3_571_classic4_raw_vs_graph.jsonl",
    "semantic_kg": RESULTS / "multi_cancer__semantic_kg__naive_rag__formalv3_571_classic4_semantic_kg.jsonl",
    "evidence_kg": CANONICAL,
}
SYSTEM_SOURCE = {"doc_vanilla": "doc_corpus", "semantic_vanilla": "semantic_kg", "evidence_vanilla": "evidence_kg"}
MODEL_LABELS = {"gpt_5_mini_existing": "GPT-5 mini", **{k: v for k, v in MODELS.items() if k != "gpt_5_mini_existing"}}
MODEL_LABELS["gemini_3_5_flash"] = "Gemini 3.5 Flash"
MODEL_LABELS["gemini_3_8_flash"] = "Gemini 3.8 Flash"
_HEALTH_CHECK = False


def apply_generator_plan() -> None:
    path = OUTPUT / "models" / "generator_plan.json"
    if not path.exists():
        return
    plan = json.loads(path.read_text(encoding="utf-8"))
    for key, model in plan.get("model_overrides", {}).items():
        if key not in MODEL_KEYS or not isinstance(model, str) or not model.strip():
            raise ValueError("Invalid saved generator model override")
        os.environ[key] = model


def provider_vars(model_key: str) -> tuple[str, str]:
    # Gemini can retain its exact model ID while using a separately approved
    # provider.  Never send a credential from the original gateway to a new
    # provider merely because only its URL was configured.
    if model_key == "RAG_EVAL_MODEL2" and os.getenv("GEMINI_API_KEY") and os.getenv("GEMINI_BASE_URL"):
        return "GEMINI_API_KEY", "GEMINI_BASE_URL"
    return PROVIDER_ENV.get(model_key, ("OPENAI_API_KEY", "OPENAI_BASE_URL"))


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_if_present(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_sources() -> tuple[list[str], dict[str, dict[str, dict[str, Any]]]]:
    data = {source: {str(row["record"]["id"]): row for row in read_jsonl(path)} for source, path in SOURCE_PATHS.items()}
    ids = list(data["evidence_kg"])
    if len(ids) != 571 or any(set(rows) != set(ids) for rows in data.values()):
        raise ValueError("The canonical source files must each contain the same 571 distinct record IDs")
    if any(rows[rid]["record"] != data["evidence_kg"][rid]["record"] for rows in data.values() for rid in ids):
        raise ValueError("Benchmark record contents differ across knowledge sources")
    return ids, data


def judge_hash(row: dict[str, Any], model: str) -> str:
    return digest({"model": model, "system": JUDGE_SYSTEM_PROMPT, "user": judge_prompt(row), "temperature": 0})


def model_hash(row: dict[str, Any], model: str) -> str:
    # All original Vanilla top-8 contexts take the direct-prompt path.  Abort if
    # a future context introduces evidence-pack routing instead of silently
    # hashing a different request from the one generate_prediction will send.
    context = build_context_blocks(row)
    from rag_eval_lib import first_evidence_pack_doc
    if first_evidence_pack_doc(context) is not None:
        raise ValueError("Unexpected evidence-pack context in fixed Vanilla model panel")
    return digest({"model": model, "prompt": generator_prompt(row["record"], context), "profile": "tuned/full/structured/grounded"})


def judge_row(system_id: str, row: dict[str, Any], judgment: dict[str, Any], model: str, **extra: Any) -> dict[str, Any]:
    item = next(item for item in SYSTEM_PLAN if item["system_id"] == system_id)
    record = row["record"]
    return {
        "id": str(record["id"]), "system_id": system_id, "panel": item["panel"],
        "framework": item["framework"], "knowledge_source": item["knowledge_source"],
        "task_type": record.get("task_type"), "oncology_focus": record.get("oncology_focus"),
        "question": record.get("question"),
        "gold_answer": (record.get("gold_standard") or {}).get("normalized_answer") or record.get("answer"),
        "gold_label": gold_label(record), "prediction_text": row.get("prediction"),
        "prediction_label": predicted_answer(row), "correct": int(predicted_answer(row) == gold_label(record)),
        "retrieved_doc_count": len(row.get("retrieved_docs") or []), "judge_model": model,
        "judge_note": judgment.get("note", judgment.get("judge_note", "")),
        **{key: int(judgment[key]) for key in ("retrieval_sufficient", "answer_supported", "answer_contradicted")},
        "input_sha256": judge_hash(row, model), **extra,
    }


def model_row(source: str, row: dict[str, Any], model: str, model_key: str, prediction: str,
              normalized: str, state: Any = None, mode: str | None = None, **extra: Any) -> dict[str, Any]:
    return {
        "record_id": str(row["record"]["id"]), "task_type": row["record"].get("task_type"),
        "oncology_focus": row["record"].get("oncology_focus"), "knowledge_source": source,
        "model_key": model_key, "model": model, "gold_label": gold_label(row["record"]),
        "prediction": prediction, "normalized_prediction": normalized,
        "correct": int(str(normalized) == gold_label(row["record"])),
        "prediction_mode": mode, "inferred_evidence_state": state,
        "input_sha256": model_hash(row, model), **extra,
    }


def valid_existing(path: Path, id_key: str, expected: dict[str, str], *, source_key: bool = False) -> dict[str, dict[str, Any]]:
    found = {}
    for row in read_if_present(path):
        key = (f"{row['knowledge_source']}::" if source_key else "") + str(row[id_key])
        if key in found:
            raise ValueError(f"Duplicate cached row in {path.name}: {key}")
        if key not in expected or row.get("input_sha256") != expected[key]:
            raise ValueError(f"Stale or mismatched cached row in {path.name}: {key}")
        found[key] = row
    return found


def prepare(judge_model: str) -> dict[str, Any]:
    ids, sources = load_sources()
    old_sources = {
        source: {str(row["record"]["id"]): row for row in read_jsonl(next(item["result_path"] for item in SYSTEM_PLAN if item["system_id"] == system))}
        for system, source in SYSTEM_SOURCE.items()
    }
    panel = {str(row["id"]): row for row in read_if_present(ROOT / "dataset/evaluated/external_reasoning_benchmark_formal_v3_571.jsonl")}
    manifest = {"record_count": 571, "record_ids_sha256": digest(ids), "canonical_full": str(CANONICAL.relative_to(ROOT)),
                "judge_model": judge_model, "sources": {}, "judges": {}, "models": {},
                "transmitted_fields": {
                    "judge": ["question", "gold_answer", "gold_canonical", "model_prediction_text", "model_prediction_label",
                              "retrieved_context_top6_max3600chars:source/level/title/diseases/entities/relation_hits/text"],
                    "generator": ["question", "answer_format_instructions", "existing_top8_retrieved_context"],
                    "authentication": "Bearer token sent only to the selected configured provider; never written to artifacts",
                }}
    for source, path in SOURCE_PATHS.items():
        manifest["sources"][source] = {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    for system_id, source in SYSTEM_SOURCE.items():
        output = OUTPUT / "faithfulness" / f"{system_id}.jsonl"
        expected = {rid: judge_hash(sources[source][rid], judge_model) for rid in ids}
        done = valid_existing(output, "id", expected)
        item = next(item for item in SYSTEM_PLAN if item["system_id"] == system_id)
        old_source = old_sources[source]
        for cached in read_if_present(RESULTS / "faithfulness/formal" / f"{system_id}.jsonl"):
            rid = str(cached["id"])
            old = old_source.get(rid)
            current = sources[source].get(rid)
            if rid in done or old is None or current is None or cached.get("judge_model") != judge_model:
                continue
            # Original judge files do not store context hashes, so establish
            # provenance from their recorded answer/question and frozen source.
            if (cached.get("prediction_text") != old.get("prediction")
                    or cached.get("prediction_label") != predicted_answer(old)
                    or cached.get("question") != old["record"].get("question")
                    or cached.get("gold_label") != gold_label(old["record"])
                    or judge_hash(old, judge_model) != expected[rid]):
                continue
            copied = judge_row(system_id, current, cached, judge_model,
                               provenance="reused_exact_judge_input", legacy_source=str(item["result_path"].relative_to(ROOT)))
            append(output, copied)
            done[rid] = copied
        manifest["judges"][system_id] = {"completed": len(done), "pending": 571 - len(done), "expected": 571}
    for key in ["existing", *MODEL_KEYS]:
        model = os.getenv("RAG_EVAL_MODEL", "gpt-5-mini") if key == "existing" else os.getenv(key)
        if not model:
            manifest["models"][key] = {"configuration": "model_missing"}
            continue
        name = "gpt_5_mini_existing" if key == "existing" else safe_name(model)
        path = OUTPUT / "models" / f"{name}.jsonl"
        expected = {f"{source}::{rid}": model_hash(sources[source][rid], model) for source in sources for rid in ids}
        done = valid_existing(path, "record_id", expected, source_key=True)
        if key == "existing":
            for source in sources:
                for rid in ids:
                    cache_key = f"{source}::{rid}"
                    if cache_key in done:
                        continue
                    row = sources[source][rid]
                    copied = model_row(source, row, model, "RAG_EVAL_MODEL", str(row.get("prediction") or ""),
                                       predicted_answer(row), row.get("inferred_evidence_state"), row.get("prediction_mode"),
                                       provenance="existing_canonical_571_generation")
                    append(path, copied)
                    done[cache_key] = copied
        else:
            # Legacy generator runs replaced each record with this exact panel
            # record.  Verify those real generation inputs before cache reuse.
            for cached in read_if_present(RESULTS / "model_robustness" / f"{name}.jsonl"):
                source, rid = cached["knowledge_source"], str(cached["record_id"])
                cache_key = f"{source}::{rid}"
                if cache_key in done or cached.get("model") != model or cached.get("normalized_prediction") in (None, ""):
                    continue
                old = deepcopy(old_sources[source][rid])
                old["record"] = panel[rid]
                if model_hash(old, model) != expected[cache_key] or cached.get("gold_label") != gold_label(old["record"]):
                    continue
                copied = dict(cached, input_sha256=expected[cache_key], provenance="reused_exact_generation_input")
                append(path, copied)
                done[cache_key] = copied
        provider_key, provider_url = provider_vars(key)
        manifest["models"][name] = {"model": model, "model_key": key, "completed": len(done), "pending": 1713 - len(done),
                                    "expected": 1713, "provider_credentials_present": bool(os.getenv(provider_key)),
                                    "provider_url_present": bool(os.getenv(provider_url)),
                                    "provider_key_variable": provider_key, "provider_url_variable": provider_url}
    manifest["minimum_pending_api_calls"] = {
        "judge": sum(row["pending"] for row in manifest["judges"].values()),
        "generator": sum(row.get("pending", 0) for row in manifest["models"].values()),
        "note": "One request per pending record. A judge schema repair can add one request; no endpoint substitution is performed.",
    }
    write_json(OUTPUT / "judge_models_manifest.json", manifest)
    return manifest


class SafeAPIError(RuntimeError):
    """Contains no credentials, response bodies, hostnames or request payloads."""


def request_once(client: Any, payload: dict[str, Any]) -> str:
    # Preserve the original runner's three-attempt transport retry policy for
    # an already healthy queue.  The initial endpoint probe has one attempt.
    attempts = 1 if _HEALTH_CHECK else 3
    for attempt in range(attempts):
        with requests.Session() as session:
            if client.disable_proxy:
                session.trust_env = False
            try:
                response = session.post(f"{client.base_url}/chat/completions",
                                        headers={"Authorization": f"Bearer {client.api_key}", "Content-Type": "application/json"},
                                        json=payload, timeout=(10, 120))
            except requests.RequestException as exc:
                if attempt + 1 < attempts:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise SafeAPIError(type(exc).__name__) from None
            if response.status_code in {429, 500, 502, 503, 504} and attempt + 1 < attempts:
                time.sleep(2 * (attempt + 1))
                continue
            break
    if response.status_code != 200:
        error = f"HTTP_{response.status_code}"
        try:
            detail = response.json().get("error", {})
            code = str(detail.get("code") or "") if isinstance(detail, dict) else ""
            message = str(detail.get("message") or "").lower() if isinstance(detail, dict) else ""
            if re.fullmatch(r"[A-Za-z0-9_:-]{1,80}", code):
                error += f":{code}"
            if "no available channel" in message or "无可用渠道" in message:
                error += ":provider_no_available_channel"
        except (ValueError, TypeError, AttributeError):
            pass
        raise SafeAPIError(error)
    try:
        return response.json()["choices"][0]["message"]["content"].strip()
    except (ValueError, KeyError, TypeError, IndexError, AttributeError):
        raise SafeAPIError("invalid_response_schema") from None


class ResumableJudge(JudgeClient):
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return request_once(self, {"model": self.model, "messages": [{"role": "system", "content": system_prompt},
                                  {"role": "user", "content": user_prompt}], "temperature": 0})


class ResumableGenerator(OpenAICompatibleClient):
    def generate(self, prompt: str) -> str:
        payload = {"model": self.model, "messages": [
            {"role": "system", "content": "You are a precise biomedical QA assistant. Return the shortest answer consistent with the requested answer format."},
            {"role": "user", "content": prompt}]}
        if not self.model.lower().startswith("gpt-5"):
            payload["temperature"] = 0
        if self.model.lower().startswith("qwen"):
            payload.update({"enable_thinking": False, "max_tokens": 128})
        return request_once(self, payload)


def run_pending(label: str, jobs: list[Any], execute: Callable, save: Callable, workers: int) -> bool:
    global _HEALTH_CHECK
    if not jobs:
        print(f"[{label}] already complete", flush=True)
        return True
    print(f"[{label}] pending={len(jobs)}; one real pending example is the endpoint health check", flush=True)
    try:
        _HEALTH_CHECK = True
        save(execute(jobs[0]))
    except Exception as exc:
        error = str(exc) if isinstance(exc, SafeAPIError) else type(exc).__name__
        append(OUTPUT / "errors.jsonl", {"phase": label, "stage": "single_health_check", "error": error,
                                       "utc": datetime.now(timezone.utc).isoformat()})
        print(f"[{label}] blocked={error}; no remaining API jobs submitted", flush=True)
        return False
    finally:
        _HEALTH_CHECK = False
    done, failed = 1, False
    iterator = iter(jobs[1:])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        active = {pool.submit(execute, job) for _, job in zip(range(workers), iterator)}
        while active:
            try:
                completed, active = wait(active, return_when=FIRST_COMPLETED)
            except KeyboardInterrupt:
                failed = True
                print(f"[{label}] stop requested; saving all in-flight results before exit", flush=True)
                completed, active = wait(active)
            for future in completed:
                try:
                    save(future.result())
                    done += 1
                except Exception as exc:
                    error = str(exc) if isinstance(exc, SafeAPIError) else type(exc).__name__
                    append(OUTPUT / "errors.jsonl", {"phase": label, "stage": "generation", "error": error,
                                                   "utc": datetime.now(timezone.utc).isoformat()})
                    failed = True
                    print(f"[{label}] halted={error}; allowing only in-flight requests to finish", flush=True)
            if not failed:
                for _ in completed:
                    try:
                        job = next(iterator)
                    except StopIteration:
                        break
                    active.add(pool.submit(execute, job))
            if done % 25 == 0 or not active:
                print(f"[{label}] newly_completed={done}/{len(jobs)}", flush=True)
    return not failed


def run_judges(model: str, workers: int, limit: int) -> None:
    ids, sources = load_sources()
    client = ResumableJudge(model)
    jobs = []
    for system, source in SYSTEM_SOURCE.items():
        done = {row["id"] for row in read_if_present(OUTPUT / "faithfulness" / f"{system}.jsonl")}
        jobs.extend((system, sources[source][rid]) for rid in ids if rid not in done)
    if limit:
        jobs = jobs[:limit]
    def execute(job):
        system, row = job
        return judge_row(system, row, judge_with_repair(client, row), model, provenance="new_571_extension_judgment")
    run_pending("faithfulness", jobs, execute,
                lambda row: append(OUTPUT / "faithfulness" / f"{row['system_id']}.jsonl", row), workers)


def run_models(workers: int, limit: int, requested: list[str]) -> None:
    ids, sources = load_sources()
    for key in requested:
        model = os.getenv(key)
        if not model:
            print(f"[{key}] model configuration missing", flush=True)
            continue
        client = ResumableGenerator(model)
        if key in PROVIDER_ENV or (key == "RAG_EVAL_MODEL2" and provider_vars(key)[0] == "GEMINI_API_KEY"):
            key_name, url_name = provider_vars(key)
            client.api_key, client.base_url = os.getenv(key_name), os.getenv(url_name, "").rstrip("/")
        if not client.api_key or not client.base_url:
            print(f"[{key}] provider configuration missing", flush=True)
            continue
        output = OUTPUT / "models" / f"{safe_name(model)}.jsonl"
        done = {f"{row['knowledge_source']}::{row['record_id']}" for row in read_if_present(output)}
        jobs = [(source, sources[source][rid]) for source in sources for rid in ids if f"{source}::{rid}" not in done]
        if limit:
            jobs = jobs[:limit]
        def execute(job):
            source, row = job
            prediction, normalized, state, mode = generate_prediction(client, row)
            if normalized in (None, ""):
                raise SafeAPIError("empty_normalized_prediction")
            return model_row(source, row, model, key, prediction, normalized, state, mode, provenance="new_571_extension_generation")
        run_pending(safe_name(model), jobs, execute, lambda row: append(output, row), workers)


def summaries() -> None:
    ids, _ = load_sources()
    faith = {"record_count": 571, "systems": {}, "incomplete": {}}
    for system in SYSTEM_SOURCE:
        rows = read_if_present(OUTPUT / "faithfulness" / f"{system}.jsonl")
        if {row["id"] for row in rows} == set(ids) and len(rows) == 571:
            faith["systems"][system] = summarize_rows(rows)
        else:
            faith["incomplete"][system] = {"completed": len(rows), "expected": 571}
    write_json(OUTPUT / "faithfulness" / "summary.json", faith)
    models = {"record_count": 571, "sources": list(SOURCES), "models": {}, "incomplete": {}}
    core_path = ROOT / "dataset/rag/analysis/unified_571/core_significance/core_significance_summary.json"
    core_effects = {}
    if core_path.exists():
        core = json.loads(core_path.read_text(encoding="utf-8"))
        if core["design"]["record_count"] != 571:
            raise ValueError("Core significance source is not the 571-record evaluation")
        core_effects = {row["baseline"]: row for row in core["overall"] if row["framework"] == "Vanilla RAG"}
    active_labels = {"gpt_5_mini_existing": "GPT-5 mini"}
    for key in MODEL_KEYS:
        if model := os.getenv(key):
            name = safe_name(model)
            active_labels[name] = MODEL_LABELS.get(name, model)
    for model_index, (name, label) in enumerate(active_labels.items()):
        rows = read_if_present(OUTPUT / "models" / f"{name}.jsonl")
        groups = {source: [row for row in rows if row["knowledge_source"] == source] for source in SOURCES}
        if not all(len(group) == 571 and {row["record_id"] for row in group} == set(ids) for group in groups.values()):
            models["incomplete"][name] = {"completed": len(rows), "expected": 1713}
            continue
        summary = {"label": label, "sources": {source: summarize(group) for source, group in groups.items()},
                   "effects": {}, "by_task_type": {}}
        # Keep each model's original seed slot even while earlier queues are
        # incomplete.  Intermediate summaries must not change another model's CI.
        seed = 20260628 + 2 * model_index
        for baseline in ("doc_corpus", "semantic_kg"):
            effect = compare(groups["evidence_kg"], groups[baseline], seed)
            effect["mcnemar_p_unrounded"] = exact_mcnemar(effect["wins"], effect["losses"])
            if effect["mcnemar_p"] == 0:
                effect["mcnemar_p"] = effect["mcnemar_p_unrounded"]
            if name == "gpt_5_mini_existing" and core_effects:
                authority = core_effects[SOURCES[baseline]]
                for key in ("wins", "ties", "losses", "accuracy_delta", "macro_f1_delta"):
                    if effect[key] != authority[key]:
                        raise ValueError(f"Canonical GPT-5 mini effect disagrees with core significance: {baseline}/{key}")
                for key in ("accuracy_ci95", "macro_f1_ci95", "accuracy_delta", "macro_f1_delta"):
                    effect[key] = authority[key]
                effect["mcnemar_p"] = authority["mcnemar_p"] or effect["mcnemar_p_unrounded"]
                effect["bootstrap_authority"] = str(core_path.relative_to(ROOT))
            summary["effects"][f"evidence_vs_{baseline}"] = effect
            seed += 1
        for task in CORE_TASKS:
            subsets = {source: [row for row in group if row["task_type"] == task] for source, group in groups.items()}
            evidence_accuracy = accuracy(subsets["evidence_kg"])
            summary["by_task_type"][task] = {
                "count": len(subsets["evidence_kg"]), "evidence_accuracy": round(evidence_accuracy, 4),
                "evidence_minus_doc_accuracy": round(evidence_accuracy - accuracy(subsets["doc_corpus"]), 4),
                "evidence_minus_semantic_accuracy": round(evidence_accuracy - accuracy(subsets["semantic_kg"]), 4),
            }
        models["models"][name] = summary
    write_json(OUTPUT / "models" / "summary.json", models)


def validate_completed(judge_model: str) -> dict[str, Any]:
    """Verify every published row against its frozen 571-record input."""
    ids, sources = load_sources()
    id_set = set(ids)
    validation = {"record_count": 571, "faithfulness": {}, "models": {}}
    for system, source in SYSTEM_SOURCE.items():
        path = OUTPUT / "faithfulness" / f"{system}.jsonl"
        rows = read_if_present(path)
        if len(rows) != 571 or {row["id"] for row in rows} != id_set:
            raise ValueError(f"Incomplete or duplicate faithfulness rows: {system}")
        for row in rows:
            original = sources[source][row["id"]]
            if row["judge_model"] != judge_model or row["input_sha256"] != judge_hash(original, judge_model):
                raise ValueError(f"Faithfulness input hash mismatch: {system}/{row['id']}")
            if any(row[key] not in (0, 1) for key in ("retrieval_sufficient", "answer_supported", "answer_contradicted")):
                raise ValueError(f"Nonbinary judge labels: {system}/{row['id']}")
            if row["correct"] != int(predicted_answer(original) == gold_label(original["record"])):
                raise ValueError(f"Faithfulness correctness mismatch: {system}/{row['id']}")
            if row["answer_supported"] and row["answer_contradicted"]:
                raise ValueError(f"Incompatible judge labels: {system}/{row['id']}")
        validation["faithfulness"][system] = {
            "rows": 571, "unique_ids": 571, "verified_input_hashes": 571,
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    active = [("gpt_5_mini_existing", os.getenv("RAG_EVAL_MODEL", "gpt-5-mini"))]
    active.extend((safe_name(os.environ[key]), os.environ[key]) for key in MODEL_KEYS)
    for name, model in active:
        path = OUTPUT / "models" / f"{name}.jsonl"
        rows = read_if_present(path)
        keys = {(row["knowledge_source"], row["record_id"]) for row in rows}
        expected_keys = {(source, rid) for source in sources for rid in ids}
        if len(rows) != 1713 or keys != expected_keys:
            raise ValueError(f"Incomplete or duplicate generator rows: {name}")
        for row in rows:
            original = sources[row["knowledge_source"]][row["record_id"]]
            if row.get("model") != model or row.get("normalized_prediction") in (None, ""):
                raise ValueError(f"Invalid model or empty prediction: {name}/{row['record_id']}")
            if row["input_sha256"] != model_hash(original, model):
                raise ValueError(f"Generator input hash mismatch: {name}/{row['record_id']}")
            if row["task_type"] != original["record"]["task_type"]:
                raise ValueError(f"Generator task type mismatch: {name}/{row['record_id']}")
            if (row["gold_label"] != gold_label(original["record"])
                    or row["correct"] != int(row["normalized_prediction"] == row["gold_label"])):
                raise ValueError(f"Generator correctness mismatch: {name}/{row['record_id']}")
        validation["models"][name] = {
            "model": model, "rows": 1713, "unique_source_id_keys": 1713,
            "empty_predictions": 0, "verified_input_hashes": 1713,
            "per_source_count": {source: 571 for source in sources},
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    validation["all_checks_passed"] = True
    write_json(OUTPUT / "judge_models_validation.json", validation)
    return validation


def main() -> None:
    load_project_env()
    apply_generator_plan()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["prepare", "judge", "models", "summary", "validate", "all"], default="prepare")
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", "gpt-5.4-mini"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model-keys", default=",".join(MODEL_KEYS))
    parser.add_argument("--skip-maintenance", action="store_true",
                        help="Use already-prepared caches without rewriting shared manifests/summaries; for independent parallel queues.")
    args = parser.parse_args()
    if args.workers < 1 or args.limit < 0:
        parser.error("workers must be positive and limit nonnegative")
    requested = args.model_keys.split(",")
    if any(key not in MODEL_KEYS for key in requested):
        parser.error("unsupported model key")
    set_rag_profile("tuned")
    set_stage_config("full", "structured", "grounded")
    if args.phase == "validate":
        print(json.dumps(validate_completed(args.judge_model), ensure_ascii=False), flush=True)
        return
    if args.phase != "summary" and not args.skip_maintenance:
        manifest = prepare(args.judge_model)
        print(json.dumps({"judges": manifest["judges"], "models": manifest["models"]}, ensure_ascii=False), flush=True)
    if args.phase in ("judge", "all"):
        run_judges(args.judge_model, args.workers, args.limit)
    if args.phase in ("models", "all"):
        run_models(args.workers, args.limit, requested)
    if args.phase in ("judge", "models", "all") and not args.skip_maintenance:
        prepare(args.judge_model)
    if args.phase != "prepare" and not args.skip_maintenance:
        summaries()


if __name__ == "__main__":
    main()

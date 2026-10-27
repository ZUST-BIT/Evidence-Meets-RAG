#!/usr/bin/env python3
"""Run or analyze one publication experiment. No API calls occur in plan/smoke."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from paper_paths import (ROOT, BENCHMARK, RESULTS, ANALYSIS, FROZEN, FULL, GRAPH_FULL,
                         SOURCES, FRAMEWORKS, checked_rows, dump, read_rows, result_path, suffix)

MODEL_DEFAULTS = {"RAG_EVAL_MODEL": "gpt-5-mini", "RAG_EVAL_MODEL2": "gemini-3.8-flash",
                  "RAG_EVAL_MODEL3": "deepseek-v4-flash", "RAG_EVAL_MODEL4": "gpt-5.4-mini",
                  "RAG_EVAL_MODEL6": "qwen3.5-flash", "JUDGE_MODEL": "gpt-5.4-mini"}


def call(script: str, *args: str) -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, args)], cwd=ROOT, check=True)


def generate_contexts(input_path: Path, output: Path, workers: int) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from rag_eval_lib import OpenAICompatibleClient, set_rag_profile, set_stage_config, compute_answer_metrics
    from run_rag_evaluation import generate_prediction, json_ready
    rows = checked_rows(input_path)
    set_rag_profile("tuned"); set_stage_config("full", "structured", "grounded")
    existing = {str(r["record"]["id"]): r for r in read_rows(output)} if output.exists() else {}
    expected = {r["record"]["id"] for r in rows}
    if set(existing) - expected:
        raise ValueError("Output contains IDs outside the frozen benchmark; use a clean output directory")
    def input_hash(row):
        payload = {"record": row["record"], "retrieved_docs": row["retrieved_docs"],
                   "model": os.environ["RAG_EVAL_MODEL"], "profile": "tuned/full/structured/grounded"}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    for row in rows:
        prior = existing.get(row["record"]["id"])
        if prior and prior.get("input_sha256") != input_hash(row):
            raise ValueError("Cached generation has different input/model; use a clean output directory")
    pending = [r for r in rows if not existing.get(r["record"]["id"], {}).get("prediction")]
    output.parent.mkdir(parents=True, exist_ok=True)
    if pending:
        client = OpenAICompatibleClient(model=os.environ["RAG_EVAL_MODEL"])
        if not client.is_configured():
            raise RuntimeError("Configure the generation API before running this command")
        def one(source):
            row = {k: source[k] for k in ("record", "query_text", "retrieved_docs") if k in source}
            prediction, normalized, state, mode = generate_prediction(client, row)
            row.update(prediction=prediction, normalized_prediction=normalized,
                       inferred_evidence_state=state, prediction_mode=mode,
                       model=client.model, input_sha256=input_hash(source))
            return json_ready(row)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = {pool.submit(one, r): r["record"]["id"] for r in pending}
            for future in as_completed(jobs):
                row = future.result()
                with output.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                existing[row["record"]["id"]] = row
    final = checked_rows(output, predictions=True)
    dump(output.with_name(output.stem + "_summary.json"), {"experiment": {"record_count": 571,
        "model": os.environ["RAG_EVAL_MODEL"], "input_context": str(input_path.relative_to(ROOT))},
        "answer_metrics": compute_answer_metrics(final)})


def stages(args) -> None:
    flags = ["--workers", str(args.workers)]
    if args.input_mode == "frozen":
        flags += ["--frozen-input-dir", str(FROZEN / "stage")]
    if args.phase in ("all", "retrieve"):
        call("expand_stage_571.py", "--phase", "retrieve", *flags)
    if args.phase in ("all", "generate"):
        call("expand_stage_571.py", "--phase", "generate", *flags)
    if args.phase in ("all", "summary"):
        call("expand_stage_571.py", "--phase", "summarize", *flags)


def main_experiment(args) -> None:
    for framework in args.frameworks:
        for source in args.sources:
            if framework == "graph_rag" and source == "evidence_graph_complete":
                # Stage Full is authoritative; never make a second Full sample.
                flags = ["--workers", str(args.workers)]
                if args.input_mode == "frozen":
                    flags += ["--frozen-input-dir", str(FROZEN / "stage")]
                call("expand_stage_571.py", "--phase", "retrieve", *flags)
                call("expand_stage_571.py", "--phase", "generate", "--configs", "R1C1G1", *flags)
                continue
            target = result_path(framework, source)
            if args.input_mode == "frozen":
                generate_contexts(FROZEN / f"main/{framework}__{source}.jsonl", target, args.workers)
                if target == FULL:
                    legacy_name = result_path(framework, source, canonical=False)
                    shutil.copyfile(target, legacy_name)
            else:
                call("run_rag_evaluation.py", "--view", "multi_cancer", "--knowledge-source", source,
                     "--framework", framework, "--dataset-path", str(BENCHMARK), "--top-k", "8",
                     "--generation-workers", str(args.workers), "--resume", "--output-suffix", suffix(framework, source))
                if target == FULL:
                    shutil.copyfile(result_path(framework, source, canonical=False), FULL)


def analyze(args) -> None:
    from paper_analysis import analyze_publication
    analyze_publication()


def restore_context_file(relative: str, target: Path) -> None:
    from analyze_retrieval_evidence_quality import audit_retrieval
    rows = read_rows(FROZEN / relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as h:
        for row in rows:
            row["retrieval_audit"] = audit_retrieval(row["record"], row["retrieved_docs"])
            h.write(json.dumps(row, ensure_ascii=False) + "\n")


def smoke() -> None:
    """Exercise retrieval, scoring, masking, judge parsing, and paired analysis offline."""
    import ast
    from rag_eval_lib import PreparedCorpus, PreparedDoc, RETRIEVERS, compute_answer_metrics, normalize_prediction
    from run_fixed_context_component_ablation_571 import apply_condition
    from run_faithfulness_judge import compact_context, parse_judgment
    from summarize_faithfulness_results import summarize_rows
    from analyze_core_paired_significance import compare
    record = {"id": "fixture", "question": "Does EGFR support NSCLC?", "task_type": "cross_source_corroboration_status",
              "entity_anchors": ["EGFR"], "oncology_focus": "NSCLC", "answer": "yes",
              "gold_standard": {"answer_format": "boolean", "binary_answer": True, "normalized_answer": "yes"}}
    doc = PreparedDoc(doc_id="fixture-doc", knowledge_source="evidence_graph_complete", document_level="evidence",
                      parent_id=None, title="EGFR NSCLC evidence", text="EGFR supports NSCLC pathways.",
                      disease_tags=["NSCLC"], entity_names=["EGFR"], entity_ids=[], metadata={})
    corpus = PreparedCorpus([doc], [])
    ranked = RETRIEVERS["naive_rag"](corpus, record, 8)
    assert ranked and ranked[0][0].doc_id == doc.doc_id
    assert normalize_prediction("yes", record) == "positive"
    row = {"record": record, "prediction": "yes", "normalized_prediction": "positive", "retrieved_docs": []}
    assert compute_answer_metrics([row])["accuracy"] == 1.0
    item = dict(doc.__dict__, score=1.0)
    assert apply_condition([item], "full") == [item]
    assert len(compact_context([item] * 8)) <= 3600
    flags = parse_judgment('{"retrieval_sufficient":0,"answer_supported":1,"answer_contradicted":0}')
    assert summarize_rows([dict(flags, correct=1)])["fcr"] == 1.0
    with tempfile.TemporaryDirectory(prefix="evidencemeetsrag-smoke-") as td:
        a, b = Path(td)/"a.jsonl", Path(td)/"b.jsonl"
        a.write_text(json.dumps(row)+"\n"); b.write_text(json.dumps(dict(row, normalized_prediction="negative"))+"\n")
        effect = compare(a, b, None, 64, 20260628)
        assert effect["accuracy_delta"] == 1.0 and effect["count"] == 1
    for path in (ROOT / "scripts").rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    cohort = len(read_rows(BENCHMARK)) if BENCHMARK.exists() else None
    if cohort is not None:
        assert cohort == 571 and len({r["id"] for r in read_rows(BENCHMARK)}) == 571
    print(json.dumps({"status": "pass", "api_calls": 0, "benchmark_records": cohort,
                      "checks": ["BM25", "normalization", "answer metrics", "Full identity", "judge schema/compaction", "FCR", "paired bootstrap/McNemar", "script syntax"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["plan", "smoke", "main", "ablation", "stages", "grounding", "models", "masking",
        "retrieval-controls", "analyze", "figures", "errors", "long-context", "oracle", "source-forms", "dual-source", "human-review", "corpus-audit", "profile", "case"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--phase", default="all", help="Experiment phase (see docs/experiments.md)")
    parser.add_argument("--input-mode", choices=["frozen", "retrieve"], default="retrieve",
                        help="Retrieve from locally constructed corpora (default), or reuse restored frozen contexts")
    parser.add_argument("--frameworks", nargs="+", choices=FRAMEWORKS, default=list(FRAMEWORKS))
    parser.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES))
    parser.add_argument("--stage", choices=["all", "budget", "hyperparam"], default="all")
    args = parser.parse_args()
    from run_fixed_context_component_ablation_571 import load_env_file
    load_env_file(ROOT / ".env")
    for key, value in MODEL_DEFAULTS.items(): os.environ.setdefault(key, value)
    if args.task == "plan":
        print(json.dumps({"count": 571, "frameworks": FRAMEWORKS, "sources": SOURCES,
            "canonical_vanilla": str(FULL.relative_to(ROOT)), "canonical_graph": str(GRAPH_FULL.relative_to(ROOT)),
            "model_ids": MODEL_DEFAULTS, "api_calls": 0}, indent=2)); return
    if args.task == "smoke": smoke(); return
    if args.task == "main": main_experiment(args)
    elif args.task == "ablation": call("run_fixed_context_component_ablation_571.py", "--framework", "vanilla_rag", "--workers", str(args.workers))
    elif args.task == "stages": stages(args)
    elif args.task in ("grounding", "models"):
        phase = args.phase if args.phase != "all" else ("judge" if args.task == "grounding" else "models")
        call("expand_judge_models_571.py", "--phase", phase, "--workers", str(args.workers))
    elif args.task == "masking": call("expand_masking_571.py", "--mode", "run" if args.phase == "all" else args.phase, "--workers", str(args.workers))
    elif args.task == "retrieval-controls":
        extra = ["--recovered-source-dir", str(ROOT / "dataset/frozen_corpora/retrieval/multi_cancer")] if (ROOT / "dataset/frozen_corpora/retrieval/multi_cancer").exists() else []
        call("expand_retrieval_controls_571.py", "--stage", args.stage, *extra)
    elif args.task == "analyze": analyze(args)
    elif args.task == "figures":
        call("build_unified_571_figures.py")
        from paper_analysis import figure7
        figure7()
    elif args.task == "errors":
        call("run_error_taxonomy_judge.py", "--output-dir", "dataset/rag/results/unified_571/error_taxonomy")
        from paper_analysis import error_summary
        error_summary()
    elif args.task == "long-context":
        for source in ("raw_observation_view", "evidence_graph_complete"):
            target = RESULTS / f"multi_cancer__{source}__naive_rag__longcontext_k100.jsonl"
            if args.input_mode == "frozen": generate_contexts(FROZEN / f"long_context/{source}.jsonl", target, args.workers)
            else: call("run_rag_evaluation.py", "--view", "multi_cancer", "--knowledge-source", source,
                "--framework", "naive_rag", "--top-k", "100", "--dataset-path", str(BENCHMARK),
                "--output-suffix", "longcontext_k100", "--resume", "--generation-workers", str(args.workers))
        import analyze_longcontext_baseline as long
        long.RUNS["eg_k8"] = FULL; long.OUT = ANALYSIS / "longcontext"; long.main()
    elif args.task == "oracle": call("run_gold_context_oracle.py", "--workers", str(args.workers), "--resume")
    elif args.task == "source-forms":
        target = ROOT / "dataset/rag/analysis/evidence_meets_rag_single_source_forms/single_source_form_retrieval_571.jsonl"
        if args.input_mode == "frozen": restore_context_file("source_forms/single_source_form_retrieval_571.jsonl", target)
        else: call("run_single_source_form_retrieval.py")
        call("run_single_source_form_generation.py", "--workers", str(args.workers), "--resume"); call("analyze_single_source_knowledge_forms.py")
    elif args.task == "dual-source":
        target = ROOT / "dataset/rag/analysis/evidence_meets_rag_tcga_pubmed/tcga_pubmed_controlled_retrieval.jsonl"
        if args.input_mode == "frozen": restore_context_file("dual_source/tcga_pubmed_controlled_retrieval.jsonl", target)
        else: call("build_tcga_pubmed_complementarity.py")
        call("run_tcga_pubmed_complementarity.py", "--workers", str(args.workers), "--resume"); call("analyze_tcga_pubmed_complementarity.py")
    elif args.task == "human-review": call("analyze_benchmark_human_review.py")
    elif args.task == "corpus-audit": call("audit_ecorag_coverability_leakage.py"); call("analyze_retrieval_evidence_quality.py")
    elif args.task == "profile":
        from paper_analysis import profile
        profile()
    elif args.task == "case":
        from paper_analysis import case_trace
        case_trace()


if __name__ == "__main__":
    main()

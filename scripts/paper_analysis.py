"""Offline analysis of newly generated paper outputs; no archived predictions."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import sys

from paper_paths import (ROOT, BENCHMARK, RESULTS, ANALYSIS, FULL, GRAPH_FULL,
                         FRAMEWORKS, SOURCES, checked_rows, dump, read_rows, result_path)


def analyze_publication() -> None:
    from rag_eval_lib import compute_answer_metrics, gold_label, macro_f1_from_labels
    from analyze_retrieval_evidence_quality import audit_retrieval
    import analyze_core_paired_significance as core
    import analyze_template_cluster_bootstrap as template
    import analyze_cross_cancer_performance as cancer
    from run_fixed_context_component_ablation_571 import CONDITIONS
    from expand_masking_571 import prepare
    # All inputs are strict 571-ID cohorts before any paired intersection.
    metrics = {}
    for framework in FRAMEWORKS:
        metrics[framework] = {source: compute_answer_metrics(checked_rows(result_path(framework, source), predictions=True)) for source in SOURCES}
    dump(ANALYSIS / "main_metrics.json", {"count_per_system": 571, "frameworks": metrics})
    prepare()  # canonical aliases, original masking input, and retrieval_571.json
    def exact_metrics(rows):
        gold = [gold_label(r["record"]) for r in rows]
        pred = [r["normalized_prediction"] for r in rows]
        return {"count": len(rows), "accuracy": sum(g == p for g, p in zip(gold, pred))/len(rows),
                "macro_f1": macro_f1_from_labels(gold, pred)}
    full = exact_metrics(checked_rows(FULL, predictions=True))
    ablation = {}
    for condition in CONDITIONS:
        path = RESULTS / f"fixed_context_component_ablation_571_vanilla__{condition}.jsonl"
        value = exact_metrics(checked_rows(path, predictions=True))
        ablation[condition] = {**value, "accuracy_delta_pp": 100*(value["accuracy"]-full["accuracy"]),
                              "macro_f1_delta_pp": 100*(value["macro_f1"]-full["macro_f1"])}
    core.OUTPUT_DIR = ANALYSIS / "core_significance"
    core.main()  # original overall seeds 20260628/35/42/49; six tasks between pairs
    paired = json.loads((core.OUTPUT_DIR / "core_significance_summary.json").read_text())
    dump(ANALYSIS / "main_and_ablation.json", {
        "canonical_full": str(FULL.relative_to(ROOT)), "same_full_output_for_table3_and_figure5": True,
        "vanilla": {"document": metrics["naive_rag"][SOURCES[0]], "semantic": metrics["naive_rag"][SOURCES[1]], "evidence": full},
        "ablation": ablation, "paired_comparisons": {row["baseline"]: row for row in paired["overall"] if row["framework"] == "Vanilla RAG"},
        "full_residual_errors": sum(r["normalized_prediction"] != gold_label(r["record"]) for r in checked_rows(FULL))})
    template.OUTPUT_DIR = ANALYSIS / "template_cluster_bootstrap"
    template.main()
    # Family performance is independent of the old auxiliary mechanism panel.
    cancer.OUTPUT_DIR = ANALYSIS / "cross_cancer"
    cancer.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    performance, _, normalized = cancer.load_performance()
    dump(cancer.OUTPUT_DIR / "cross_cancer_summary.json", {"design": {"benchmark_records": 571,
        "families": cancer.FAMILY_ORDER, "bootstrap_iterations": 10000}, "performance": performance,
        "task_decomposition": cancer.build_task_decomposition(normalized)})
    graph_alias = RESULTS / "unified_571/canonical/evidence_graph_rag.jsonl"
    graph_alias.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(GRAPH_FULL, graph_alias)
    subprocess.run([sys.executable, str(ROOT/"scripts/run_significance_batch.py"), "--plan",
        str(ROOT/"configs/significance_plan_571_main.json"), "--outdir", str(ANALYSIS/"significance"),
        "--bootstrap-iters", "5000", "--seed", "42"], cwd=ROOT, check=True)
    dump(ANALYSIS / "benchmark_composition.json", {"records": 571,
        "by_task": dict(Counter(r.get("task_type") for r in read_rows(BENCHMARK))),
        "by_reasoning_family": dict(Counter((r.get("metadata") or {}).get("reasoning_type") for r in read_rows(BENCHMARK)))})
    print("Paper main metrics, paired inference, template clusters, cancer effects, and ablations refreshed.")


def error_summary() -> None:
    import run_error_taxonomy_judge as taxonomy
    from rag_eval_lib import gold_label
    systems = {}
    for name, spec in taxonomy.SYSTEM_PLAN.items():
        rows = read_rows(RESULTS / f"unified_571/error_taxonomy/{name}.jsonl")
        originals = checked_rows(spec["result_path"], predictions=True)
        wrong = {r["record"]["id"] for r in originals if r["normalized_prediction"] != gold_label(r["record"])}
        if len(rows) != len({r["id"] for r in rows}) or {r["id"] for r in rows} != wrong:
            raise ValueError(f"{name}: error labels do not cover exactly the current wrong answers")
        counts = Counter(r["primary_error_type"] for r in rows)
        access = sum(counts[k] for k in ("sparse_missing_evidence", "bundle_incompleteness"))
        systems[name] = {"classified_errors": len(rows), "by_error": dict(counts), "access_errors": access,
            "use_errors": len(rows)-access, "use_share": (len(rows)-access)/len(rows) if rows else None}
    dump(ANALYSIS / "residual_errors.json", {"evaluation_count": 571, "evidence_classification_complete": True,
        "evidence_expected_errors": systems["evidence_vanilla"]["classified_errors"], "systems": systems})


def figure7() -> None:
    import analyze_cross_cancer_performance as cancer
    payload = json.loads((ANALYSIS/"cross_cancer/cross_cancer_summary.json").read_text())
    cancer.OUTPUT_DIR = ROOT / "outputs/figures"; cancer.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cancer.plot_effect_matrix(payload["performance"])


def profile() -> None:
    """Original differential timing and title+text chars/4 context estimate."""
    from rag_eval_lib import prepared_corpus_paths
    report = {"design": {"timing": "(T_limit101-T_limit1)/100; naive_rag top8, single process",
        "tokens": "mean sum(len(title)+len(text))/4 across 571 Vanilla contexts",
        "hardware_dependent": True}, "forms": {}}
    for source in SOURCES:
        timings = []
        for limit in (1, 101):
            result = subprocess.run([sys.executable, str(ROOT/"scripts/profile_retrieval_worker.py"),
                "--source", source, "--limit", str(limit)], cwd=ROOT, text=True, capture_output=True, check=True)
            timings.append(json.loads(result.stdout.strip().splitlines()[-1]))
        paths = prepared_corpus_paths(ROOT/"dataset/rag/knowledge_sources", "multi_cancer", source)
        rows = checked_rows(result_path("naive_rag", source))
        tokens = sum(sum(len(d["title"])+len(d["text"]) for d in r["retrieved_docs"])/4 for r in rows)/571
        report["forms"][source] = {"retrieval_units": timings[0]["units"], "edges": timings[0]["edges"],
            "disk_docs_mb": paths[0].stat().st_size/2**20, "disk_edges_mb": paths[1].stat().st_size/2**20,
            "index_load_s": timings[0]["load_seconds"], "differential_per_query_s": (timings[1]["total_seconds"]-timings[0]["total_seconds"])/100,
            "avg_context_tokens_k8": round(tokens), "raw_timings": timings}
    dump(ANALYSIS / "cost_profile.json", report)


def case_trace() -> None:
    from analyze_retrieval_evidence_quality import audit_retrieval
    rid = "cross-source-support-bundle-v2-aml-ABL1"
    record = next(r for r in read_rows(BENCHMARK) if r["id"] == rid)
    result = {"record": record, "views": {}}
    for source, judge_id in zip(SOURCES, ("doc_vanilla", "semantic_vanilla", "evidence_vanilla")):
        row = next(r for r in checked_rows(result_path("naive_rag", source), predictions=True) if r["record"]["id"] == rid)
        item = {"prediction": row["prediction"], "normalized_prediction": row["normalized_prediction"],
                "retrieval_diagnostics": audit_retrieval(record, row["retrieved_docs"]), "retrieved_docs": row["retrieved_docs"]}
        judge = RESULTS / f"unified_571/faithfulness/{judge_id}.jsonl"
        if judge.exists():
            item["grounding_judgment"] = next(r for r in read_rows(judge) if r["id"] == rid)
        result["views"][source] = item
    dump(ANALYSIS / "abl1_aml_case_trace.json", result)

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from rag_eval_lib import (
    bundle_component_clues,
    collect_record_gene_symbols,
    doc_matches_disease,
    doc_metadata_field,
    doc_support_flags,
    load_prepared_corpus,
    record_gene_match,
    render_context,
    set_rag_profile,
    set_stage_config,
)


ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = ROOT / "dataset" / "rag" / "knowledge_sources"

STRUCTURED_MARKERS = [
    "retrieval_diagnostic_summary:",
    "hierarchical_summary:",
    "bundle_component_summary:",
    "support_strength_summary:",
    "evidence_facet_statuses:",
    "evidence_state_summary:",
    "causal_subgraph_summary:",
    "graph_path_clues:",
    "crag_support_summary:",
    "raptor_hierarchy_summary:",
    "kg2rag_graph_summary:",
    "kg2rag_relation_paths:",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def stratum_of(record: dict[str, Any]) -> str:
    return str((record.get("metadata") or {}).get("stage_probe_stratum") or "unlabeled")


def reconstruct_retrieved(corpus, row: dict[str, Any]):
    retrieved = []
    for item in row.get("retrieved_docs") or []:
        doc_id = str(item.get("doc_id") or "")
        doc = corpus.doc_by_id.get(doc_id)
        if doc is None:
            continue
        retrieved.append((doc, float(item.get("score") or 0.0)))
    return retrieved


def retrieval_row_metrics(record: dict[str, Any], retrieved) -> dict[str, float]:
    docs = [doc for doc, _ in retrieved]
    source_families = {str(doc_metadata_field(doc, "source_family") or doc.knowledge_source) for doc in docs}
    levels = {str(doc.document_level or "") for doc in docs}
    relation_like = 0
    path_like = 0
    disease_matches = 0
    gene_matches = 0
    anchor_hit = 0
    summary_node_hit = 0
    multi_level_hit = 1.0 if len(levels) >= 2 else 0.0
    bundle_coverage = 0.0
    if docs:
        for doc in docs:
            disease_match = doc_matches_disease(doc, record)
            gene_match = record_gene_match(doc, record)
            disease_matches += 1 if disease_match else 0
            gene_matches += 1 if gene_match else 0
            if disease_match and gene_match:
                anchor_hit = 1
            flags = doc_support_flags(doc)
            if flags or (doc.metadata or {}).get("relation_hits") or (doc.metadata or {}).get("_graph_rag_path_relations"):
                relation_like = 1
            if (doc.metadata or {}).get("_graph_rag_path_relations") or (doc.metadata or {}).get("relation_hits"):
                path_like = 1
            if getattr(doc, "_node_type", "") in {
                "mechanism_role_summary",
                "target_support_summary",
                "evidence_profile",
                "observation_bundle_summary",
                "cross_source_summary",
            }:
                summary_node_hit = 1
        components = bundle_component_clues(record, retrieved)
        bundle_coverage = sum(1 for value in components.values() if value) / max(1, len(components))
    return {
        "anchor_hit": float(anchor_hit),
        "disease_match_rate": disease_matches / len(docs) if docs else 0.0,
        "gene_match_rate": gene_matches / len(docs) if docs else 0.0,
        "source_diversity": float(len(source_families)),
        "layer_diversity": float(len(levels)),
        "relation_clue_hit": float(relation_like),
        "path_clue_hit": float(path_like),
        "summary_node_hit": float(summary_node_hit),
        "multi_level_hit": float(multi_level_hit),
        "bundle_component_coverage": float(bundle_coverage),
        "query_gene_count": float(len(collect_record_gene_symbols(record))),
    }


def context_row_metrics(record: dict[str, Any], retrieved, context_text: str) -> dict[str, float]:
    structured_count = sum(1 for marker in STRUCTURED_MARKERS if marker in context_text)
    return {
        "context_chars": float(len(context_text)),
        "structured_section_count": float(structured_count),
        "structured_section_hit": 1.0 if structured_count > 0 else 0.0,
        "bundle_summary_visible": 1.0 if "bundle_component_summary:" in context_text else 0.0,
        "hierarchical_summary_visible": 1.0 if "hierarchical_summary:" in context_text else 0.0,
        "causal_summary_visible": 1.0 if ("causal_subgraph_summary:" in context_text or "graph_path_clues:" in context_text) else 0.0,
        "support_summary_visible": 1.0 if "support_strength_summary:" in context_text else 0.0,
    }


def aggregate_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted(rows[0].keys())
    return {key: mean([float(row.get(key) or 0.0) for row in rows]) for key in keys}


def shapley_values(metric_by_config: dict[tuple[int, int, int], float]) -> dict[str, float]:
    players = ["R", "C", "G"]
    factorial = math.factorial
    n = len(players)
    values: dict[str, float] = {}
    for idx, player in enumerate(players):
        total = 0.0
        for mask in range(1 << n):
            if mask & (1 << idx):
                continue
            subset = tuple(1 if mask & (1 << bit) else 0 for bit in range(n))
            superset = list(subset)
            superset[idx] = 1
            superset = tuple(superset)
            subset_size = sum(subset)
            weight = factorial(subset_size) * factorial(n - subset_size - 1) / factorial(n)
            total += weight * (metric_by_config[superset] - metric_by_config[subset])
        values[player] = round(total, 4)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze stage-probe retrieval/context metrics and QA Shapley values.")
    parser.add_argument("--results-glob", required=True, help="Glob under repo root for stage-probe result jsonl files.")
    parser.add_argument("--view", default="multi_cancer")
    parser.add_argument("--knowledge-source", default="evidence_graph_complete")
    parser.add_argument("--output-path", default="dataset/rag/results/stage_probe_96_stage_analysis.json")
    args = parser.parse_args()

    corpus = load_prepared_corpus(KNOWLEDGE_DIR, args.view, args.knowledge_source)
    results_paths = sorted(ROOT.glob(args.results_glob))
    if not results_paths:
        raise RuntimeError(f"no result files matched {args.results_glob}")

    report: dict[str, Any] = {"runs": []}
    qa_metric_by_config: dict[str, dict[tuple[int, int, int], float]] = {"accuracy": {}, "macro_f1": {}}

    for result_path in results_paths:
        summary_path = result_path.with_name(result_path.stem + "_summary.json")
        summary = read_json(summary_path)
        experiment = summary.get("experiment") or {}
        retrieval_stage = str(experiment.get("retrieval_stage") or "full")
        context_stage = str(experiment.get("context_stage") or "structured")
        generation_stage = str(experiment.get("generation_stage") or "grounded")
        set_rag_profile(str(experiment.get("rag_profile") or "tuned"))
        set_stage_config(retrieval_stage, context_stage, generation_stage)

        result_rows = read_jsonl(result_path)
        retrieval_rows_by_stratum: dict[str, list[dict[str, float]]] = defaultdict(list)
        context_rows_by_stratum: dict[str, list[dict[str, float]]] = defaultdict(list)

        for row in result_rows:
            record = row.get("record") or {}
            stratum = stratum_of(record)
            retrieved = reconstruct_retrieved(corpus, row)
            retrieval_metrics = retrieval_row_metrics(record, retrieved)
            context_text = render_context(record, retrieved)
            context_metrics = context_row_metrics(record, retrieved, context_text)
            retrieval_rows_by_stratum[stratum].append(retrieval_metrics)
            context_rows_by_stratum[stratum].append(context_metrics)

        config_bits = (
            1 if retrieval_stage == "full" else 0,
            1 if context_stage == "structured" else 0,
            1 if generation_stage == "grounded" else 0,
        )
        label = f"R{config_bits[0]}C{config_bits[1]}G{config_bits[2]}"
        run_report = {
            "label": label,
            "result_path": str(result_path.relative_to(ROOT)),
            "experiment": experiment,
            "retrieval_metrics": {
                "overall": aggregate_metric_rows([metric for rows in retrieval_rows_by_stratum.values() for metric in rows]),
                "by_stratum": {name: aggregate_metric_rows(rows) for name, rows in sorted(retrieval_rows_by_stratum.items())},
            },
            "context_metrics": {
                "overall": aggregate_metric_rows([metric for rows in context_rows_by_stratum.values() for metric in rows]),
                "by_stratum": {name: aggregate_metric_rows(rows) for name, rows in sorted(context_rows_by_stratum.items())},
            },
        }
        if "answer_metrics" in summary:
            answer_metrics = summary["answer_metrics"]
            run_report["qa_metrics"] = {
                "count": answer_metrics.get("count"),
                "accuracy": answer_metrics.get("accuracy"),
                "macro_f1": answer_metrics.get("macro_f1"),
            }
            if answer_metrics.get("accuracy") is not None:
                qa_metric_by_config["accuracy"][config_bits] = float(answer_metrics["accuracy"])
            if answer_metrics.get("macro_f1") is not None:
                qa_metric_by_config["macro_f1"][config_bits] = float(answer_metrics["macro_f1"])
        report["runs"].append(run_report)

    report["runs"] = sorted(report["runs"], key=lambda item: item["label"])
    if len(qa_metric_by_config["accuracy"]) == 8 and len(qa_metric_by_config["macro_f1"]) == 8:
        full_cfg = (1, 1, 1)
        loo = {
            "R": round(qa_metric_by_config["macro_f1"][full_cfg] - qa_metric_by_config["macro_f1"][(0, 1, 1)], 4),
            "C": round(qa_metric_by_config["macro_f1"][full_cfg] - qa_metric_by_config["macro_f1"][(1, 0, 1)], 4),
            "G": round(qa_metric_by_config["macro_f1"][full_cfg] - qa_metric_by_config["macro_f1"][(1, 1, 0)], 4),
        }
        report["qa_contribution"] = {
            "macro_f1_shapley": shapley_values(qa_metric_by_config["macro_f1"]),
            "accuracy_shapley": shapley_values(qa_metric_by_config["accuracy"]),
            "macro_f1_leave_one_out_drop": loo,
        }

    output_path = ROOT / args.output_path
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved analysis -> {output_path}")


if __name__ == "__main__":
    main()

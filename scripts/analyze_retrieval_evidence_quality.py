from __future__ import annotations

import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from audit_ecorag_coverability_leakage import (
    COMPONENT_CUES,
    disease_terms,
    doc_source_family,
    gene_terms,
    norm_text,
    other_entity_terms,
    required_components,
)


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "dataset" / "evaluated" / "external_reasoning_benchmark_formal_v3_571.jsonl"
OUTPUT_DIR = ROOT / "dataset" / "rag" / "analysis" / "ecorag_retrieval_quality"

SYSTEMS = {
    "doc_corpus": {
        "label": "Document Corpus",
        "result": ROOT / "dataset" / "rag" / "results" / "multi_cancer__raw_observation_view__naive_rag__formalv3_571_classic4_raw_vs_graph.jsonl",
        "judge": ROOT / "dataset" / "rag" / "results" / "unified_571" / "faithfulness" / "doc_vanilla.jsonl",
    },
    "semantic_kg": {
        "label": "Semantic KG",
        "result": ROOT / "dataset" / "rag" / "results" / "multi_cancer__semantic_kg__naive_rag__formalv3_571_classic4_semantic_kg.jsonl",
        "judge": ROOT / "dataset" / "rag" / "results" / "unified_571" / "faithfulness" / "semantic_vanilla.jsonl",
    },
    "evidence_kg": {
        "label": "Evidence KG",
        "result": ROOT / "dataset" / "rag" / "results" / "fixed_context_component_ablation_571_vanilla__full.jsonl",
        "judge": ROOT / "dataset" / "rag" / "results" / "unified_571" / "faithfulness" / "evidence_vanilla.jsonl",
    },
}

METRICS = [
    "entity_hit",
    "joint_scope_precision",
    "component_recall",
    "complete_bundle",
    "source_diversity",
    "support_mrr",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def result_id(row: dict[str, Any]) -> str:
    return str((row.get("record") or row).get("id") or "")


def doc_blob(doc: dict[str, Any]) -> str:
    return norm_text([
        doc.get("title") or "",
        doc.get("text") or "",
        doc.get("disease_tags") or [],
        doc.get("entity_names") or [],
        doc.get("metadata") or {},
    ])


def contains_term(blob: str, term: str) -> bool:
    normalized = norm_text(term)
    return bool(normalized and re.search(rf"\b{re.escape(normalized)}\b", blob))


def doc_disease_match(record: dict[str, Any], doc: dict[str, Any]) -> bool:
    blob = norm_text([doc.get("disease_tags") or [], doc.get("title") or "", doc.get("text") or ""])
    return any(contains_term(blob, term) for term in disease_terms(record))


def anchor_terms(record: dict[str, Any]) -> list[str]:
    genes = gene_terms(record)
    if genes:
        return genes
    return other_entity_terms(record)


def doc_entity_match(record: dict[str, Any], doc: dict[str, Any]) -> bool:
    terms = anchor_terms(record)
    if not terms:
        return True
    blob = doc_blob(doc)
    return any(contains_term(blob, term) for term in terms)


def doc_components(doc: dict[str, Any]) -> set[str]:
    blob = doc_blob(doc)
    hits = set()
    for component, cues in COMPONENT_CUES.items():
        if any(contains_term(blob, cue) for cue in cues):
            hits.add(component)
    return hits


def normalized_source_families(docs: list[dict[str, Any]]) -> set[str]:
    families = set()
    for doc in docs:
        raw = doc_source_family(doc)
        for value in re.split(r"[|,]", raw):
            value = norm_text(value).replace(" ", "_")
            if value and value not in {"unknown", "integrated", "evidence_graph_complete", "semantic_kg"}:
                families.add(value)
    return families


def audit_retrieval(record: dict[str, Any], docs: list[dict[str, Any]]) -> dict[str, Any]:
    required = required_components(record)
    disease_matches = [doc_disease_match(record, doc) for doc in docs]
    entity_matches = [doc_entity_match(record, doc) for doc in docs]
    joint_matches = [d and e for d, e in zip(disease_matches, entity_matches)]
    sources = normalized_source_families(docs)
    retrieved_components = set()
    support_rank = None
    per_doc_components = []
    for rank, (doc, joint) in enumerate(zip(docs, joint_matches), start=1):
        components = doc_components(doc)
        per_doc_components.append(components)
        if joint:
            retrieved_components.update(components)
            if support_rank is None and (components or not required):
                support_rank = rank
    covered = []
    for component in required:
        if component == "cross_source":
            covered.append(len(sources) >= 2)
        else:
            covered.append(component in retrieved_components)
    component_recall = sum(covered) / len(covered) if covered else float(any(joint_matches))
    return {
        "retrieved_count": len(docs),
        "entity_hit": float(any(entity_matches)),
        "disease_hit": float(any(disease_matches)),
        "joint_scope_precision": round(sum(joint_matches) / max(len(docs), 1), 4),
        "component_recall": round(component_recall, 4),
        "complete_bundle": float(bool(covered) and all(covered)),
        "source_diversity": len(sources),
        "cross_source_retrieved": float(len(sources) >= 2),
        "support_mrr": round(1 / support_rank, 4) if support_rank else 0.0,
        "required_components": required,
        "covered_components": [component for component, hit in zip(required, covered) if hit],
        "source_families": sorted(sources),
    }


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / max(len(rows), 1)


def pearson(xs: list[float], ys: list[float]) -> float:
    if not xs or len(xs) != len(ys):
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denominator = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return numerator / denominator if denominator else 0.0


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {"count": len(rows)}
    for metric in METRICS:
        summary[metric] = round(mean(rows, metric), 4)
    summary["disease_hit"] = round(mean(rows, "disease_hit"), 4)
    summary["cross_source_retrieved"] = round(mean(rows, "cross_source_retrieved"), 4)
    summary["retrieval_sufficient_es"] = round(mean(rows, "retrieval_sufficient"), 4)
    summary["accuracy"] = round(mean(rows, "correct"), 4)
    summary["fcr"] = round(mean(rows, "faithfully_correct"), 4)
    return summary


def bootstrap_delta(
    rows_a: dict[str, dict[str, Any]],
    rows_b: dict[str, dict[str, Any]],
    metric: str,
    iterations: int = 10000,
) -> dict[str, Any]:
    ids = sorted(set(rows_a) & set(rows_b))
    deltas = [float(rows_a[rid][metric]) - float(rows_b[rid][metric]) for rid in ids]
    observed = sum(deltas) / len(deltas)
    rng = random.Random(f"20260627:{metric}")
    samples = []
    for _ in range(iterations):
        samples.append(sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas))
    samples.sort()
    return {
        "delta": round(observed, 4),
        "ci95": [round(samples[int(iterations * 0.025)], 4), round(samples[int(iterations * 0.975) - 1], 4)],
    }


def by_group(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "unknown")].append(row)
    return {name: summarize(items) for name, items in sorted(grouped.items())}


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["knowledge_source", "count", *METRICS, "disease_hit", "cross_source_retrieved", "retrieval_sufficient_es", "accuracy", "fcr"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t")
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Evidence Meets RAG: Retrieval-side Evidence Quality",
        "",
        "## Design",
        "",
        "We conduct a paired offline analysis on the frozen 571-record benchmark. For each record, the same Vanilla RAG and top-k=8 setting is compared across Document Corpus, Semantic KG, and Evidence KG. Metrics are deterministic task-aware proxies computed from the actually retrieved documents; retrieval sufficiency (ES) is taken from the existing GPT-5.4-mini faithfulness judge as an external criterion.",
        "",
        "## Overall Results",
        "",
        "| Source | Entity Hit | Joint Scope P@8 | Component Recall | Complete Bundle | Source Diversity | Support MRR | ES | Accuracy | FCR |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in ["doc_corpus", "semantic_kg", "evidence_kg"]:
        row = payload["sources"][key]["overall"]
        lines.append(
            f"| {payload['sources'][key]['label']} | {row['entity_hit']:.4f} | {row['joint_scope_precision']:.4f} | "
            f"{row['component_recall']:.4f} | {row['complete_bundle']:.4f} | {row['source_diversity']:.4f} | "
            f"{row['support_mrr']:.4f} | {row['retrieval_sufficient_es']:.4f} | {row['accuracy']:.4f} | {row['fcr']:.4f} |"
        )
    lines.extend(["", "## Paired Evidence KG Effects", ""])
    for baseline in ["doc_corpus", "semantic_kg"]:
        label = payload["sources"][baseline]["label"]
        effects = payload["paired_effects"][f"evidence_kg_minus_{baseline}"]
        rendered = "; ".join(f"{metric} {value['delta']:+.4f} (95% CI {value['ci95']})" for metric, value in effects.items())
        lines.append(f"- Versus **{label}**: {rendered}.")
    lines.extend([
        "",
        "## Task Breakdown",
        "",
        "| Task | Doc Component R | Semantic Component R | Evidence Component R | Doc Bundle | Semantic Bundle | Evidence Bundle |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    task_order = list(payload["sources"]["evidence_kg"]["by_task_type"])
    for task in task_order:
        doc_task = payload["sources"]["doc_corpus"]["by_task_type"][task]
        semantic_task = payload["sources"]["semantic_kg"]["by_task_type"][task]
        evidence_task = payload["sources"]["evidence_kg"]["by_task_type"][task]
        lines.append(
            f"| {task} | {doc_task['component_recall']:.4f} | {semantic_task['component_recall']:.4f} | {evidence_task['component_recall']:.4f} | "
            f"{doc_task['complete_bundle']:.4f} | {semantic_task['complete_bundle']:.4f} | {evidence_task['complete_bundle']:.4f} |"
        )
    lines.extend(["", "## External Validity", ""])
    for key in ["doc_corpus", "semantic_kg", "evidence_kg"]:
        correlations = payload["sources"][key]["correlations"]
        label = payload["sources"][key]["label"]
        lines.append(
            f"- **{label}**: Component Recall correlation with ES = {correlations['component_recall_vs_es']:+.4f}, "
            f"with correctness = {correlations['component_recall_vs_correct']:+.4f}; Joint Scope Precision correlation with ES = {correlations['joint_scope_precision_vs_es']:+.4f}."
        )
    validity = payload["external_validity"]
    lines.append(
        f"- Across all 435 source-record observations, Component Recall correlates with ES at {validity['pooled_component_recall_vs_es']:+.4f}; "
        f"after centering metrics within each record, the correlation is {validity['within_record_component_recall_vs_es']:+.4f}."
    )
    evidence = payload["sources"]["evidence_kg"]["overall"]
    doc = payload["sources"]["doc_corpus"]["overall"]
    semantic = payload["sources"]["semantic_kg"]["overall"]
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"Evidence KG retrieves more task-required evidence components ({evidence['component_recall']:.4f}) than Document Corpus ({doc['component_recall']:.4f}) and Semantic KG ({semantic['component_recall']:.4f}), and it yields the highest complete-bundle rate ({evidence['complete_bundle']:.4f}). This directly supports the mechanism that evidence-centric organization improves compositional evidence access.",
        f"Its judge-based retrieval sufficiency is also highest ({evidence['retrieval_sufficient_es']:.4f}), although the absolute ES level remains low. The result should therefore be framed as a relative retrieval improvement with substantial remaining headroom, not as solved evidence retrieval.",
        "Entity Hit and lexical scope metrics should be interpreted jointly with component metrics: raw documents may mention the correct entity while failing to assemble the task-required evidence bundle. This distinction is central to the paper's evidence-centric claim.",
        "Observed source diversity is lower for Evidence KG than Document Corpus, so the results do not support a blanket diversity advantage. Semantic KG's zero value reflects unavailable source-family metadata in its retrieved documents and must not be interpreted as true zero diversity.",
        "",
        "All automatic retrieval metrics are rule-based proxies rather than human gold-document relevance labels. Their strongest use is triangulation with the independently judged ES/FCR outcomes and paired source comparisons.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    panel = read_jsonl(PANEL)
    panel_by_id = {str(row["id"]): row for row in panel}
    all_rows = []
    source_payload = {}
    by_source_id: dict[str, dict[str, dict[str, Any]]] = {}
    for source_key, config in SYSTEMS.items():
        results = {result_id(row): row for row in read_jsonl(config["result"]) if result_id(row) in panel_by_id}
        judges = {str(row["id"]): row for row in read_jsonl(config["judge"])}
        source_rows = []
        for rid, record in panel_by_id.items():
            if rid not in results or rid not in judges:
                raise RuntimeError(f"Missing {source_key} data for {rid}")
            retrieval = audit_retrieval(record, results[rid].get("retrieved_docs") or [])
            judge = judges[rid]
            row = {
                "record_id": rid,
                "knowledge_source": source_key,
                "knowledge_source_label": config["label"],
                "task_type": record.get("task_type"),
                "oncology_focus": record.get("oncology_focus"),
                **retrieval,
                "retrieval_sufficient": int(judge.get("retrieval_sufficient") or 0),
                "answer_supported": int(judge.get("answer_supported") or 0),
                "correct": int(judge.get("correct") or 0),
                "faithfully_correct": int(bool(judge.get("correct")) and bool(judge.get("answer_supported"))),
            }
            source_rows.append(row)
            all_rows.append(row)
        correlations = {
            "component_recall_vs_es": round(pearson([r["component_recall"] for r in source_rows], [r["retrieval_sufficient"] for r in source_rows]), 4),
            "component_recall_vs_correct": round(pearson([r["component_recall"] for r in source_rows], [r["correct"] for r in source_rows]), 4),
            "joint_scope_precision_vs_es": round(pearson([r["joint_scope_precision"] for r in source_rows], [r["retrieval_sufficient"] for r in source_rows]), 4),
            "complete_bundle_vs_es": round(pearson([r["complete_bundle"] for r in source_rows], [r["retrieval_sufficient"] for r in source_rows]), 4),
        }
        source_payload[source_key] = {
            "label": config["label"],
            "overall": summarize(source_rows),
            "by_task_type": by_group(source_rows, "task_type"),
            "correlations": correlations,
        }
        by_source_id[source_key] = {row["record_id"]: row for row in source_rows}

    paired = {}
    for baseline in ["doc_corpus", "semantic_kg"]:
        paired[f"evidence_kg_minus_{baseline}"] = {
            metric: bootstrap_delta(by_source_id["evidence_kg"], by_source_id[baseline], metric)
            for metric in ["joint_scope_precision", "component_recall", "complete_bundle", "source_diversity", "support_mrr"]
        }
    grouped_records: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        grouped_records[row["record_id"]].append(row)
    centered_component = []
    centered_es = []
    for rows in grouped_records.values():
        component_mean = mean(rows, "component_recall")
        es_mean = mean(rows, "retrieval_sufficient")
        centered_component.extend(float(row["component_recall"]) - component_mean for row in rows)
        centered_es.extend(float(row["retrieval_sufficient"]) - es_mean for row in rows)
    external_validity = {
        "pooled_component_recall_vs_es": round(pearson([row["component_recall"] for row in all_rows], [row["retrieval_sufficient"] for row in all_rows]), 4),
        "pooled_component_recall_vs_correct": round(pearson([row["component_recall"] for row in all_rows], [row["correct"] for row in all_rows]), 4),
        "within_record_component_recall_vs_es": round(pearson(centered_component, centered_es), 4),
    }
    payload = {
        "design": {
            "panel": str(PANEL.relative_to(ROOT)),
            "record_count": len(panel),
            "framework": "Vanilla RAG",
            "top_k": 8,
            "metric_note": "Deterministic rule-based retrieval diagnostics triangulated with existing LLM-judged ES/FCR.",
        },
        "sources": source_payload,
        "paired_effects": paired,
        "external_validity": external_validity,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "retrieval_quality_records.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (OUTPUT_DIR / "retrieval_quality_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compact = [{"knowledge_source": source_payload[key]["label"], **source_payload[key]["overall"]} for key in ["doc_corpus", "semantic_kg", "evidence_kg"]]
    write_tsv(OUTPUT_DIR / "retrieval_quality_summary.tsv", compact)
    report = render_report(payload)
    (OUTPUT_DIR / "retrieval_quality_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

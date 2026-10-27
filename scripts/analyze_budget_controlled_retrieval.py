from __future__ import annotations

import csv
import hashlib
import heapq
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from analyze_retrieval_evidence_quality import METRICS, audit_retrieval, bootstrap_delta
from audit_ecorag_coverability_leakage import DISEASE_ALIASES, family_of, gene_terms, norm_text
from rag_eval_lib import PreparedCorpus, PreparedDoc, RETRIEVERS


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "dataset" / "rag" / "knowledge_sources" / "multi_cancer"
PANEL = ROOT / "dataset" / "evaluated" / "external_reasoning_benchmark_formal_v3_571.jsonl"
OUTPUT_DIR = ROOT / "dataset" / "rag" / "analysis" / "ecorag_budget_control"
TARGET_DOCS = 18251
TOP_K = 8
CONTEXT_CHAR_BUDGET = 8000

SOURCES = {
    "doc_corpus": {
        "label": "Document Corpus",
        "docs": SOURCE_DIR / "raw_observation_view_docs.jsonl",
        "full_results": ROOT / "dataset/rag/results/multi_cancer__raw_observation_view__naive_rag__formalv3_571_classic4_raw_vs_graph.jsonl",
    },
    "semantic_kg": {
        "label": "Semantic KG",
        "docs": SOURCE_DIR / "semantic_kg_docs.jsonl",
        "full_results": ROOT / "dataset/rag/results/multi_cancer__semantic_kg__naive_rag__formalv3_571_classic4_semantic_kg.jsonl",
    },
    "evidence_kg": {
        "label": "Evidence KG",
        "docs": SOURCE_DIR / "evidence_graph_complete_docs.jsonl",
        "full_results": ROOT / "dataset/rag/results/multi_cancer__evidence_graph_complete__naive_rag__formalv3_571_classic4_raw_vs_graph.jsonl",
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_hash(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big")


def push_smallest(heap: list[tuple[int, int, dict[str, Any]]], doc: dict[str, Any], limit: int, counter: int) -> None:
    item = (-stable_hash(str(doc.get("doc_id") or counter)), counter, doc)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item[0] > heap[0][0]:
        heapq.heapreplace(heap, item)


def doc_gene_hits(doc: dict[str, Any], query_genes: set[str]) -> set[str]:
    metadata = doc.get("metadata") or {}
    raw = metadata.get("raw_node") if isinstance(metadata.get("raw_node"), dict) else {}
    candidates = [doc.get("title") or "", *(doc.get("entity_names") or [])]
    for value in [metadata.get("gene_symbol"), raw.get("gene_symbol"), (raw.get("subject") or {}).get("label") if isinstance(raw.get("subject"), dict) else None]:
        if value:
            candidates.append(str(value))
    tokens = set(re.findall(r"\b[A-Z0-9-]{2,18}\b", " ".join(candidates).upper()))
    return tokens & query_genes


def doc_family(doc: dict[str, Any], alias_to_family: dict[str, str]) -> str:
    metadata = doc.get("metadata") or {}
    raw = metadata.get("raw_node") if isinstance(metadata.get("raw_node"), dict) else {}
    values = [*(doc.get("disease_tags") or []), metadata.get("disease"), raw.get("disease"), raw.get("disease_family")]
    for value in values:
        normalized = norm_text(value)
        if normalized in alias_to_family:
            return alias_to_family[normalized]
    return "unknown"


def deterministic_sample(path: Path, panel: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    query_genes = {gene for record in panel for gene in gene_terms(record)}
    panel_families = {family_of(record) for record in panel}
    alias_to_family = {}
    for family, aliases in DISEASE_ALIASES.items():
        alias_to_family[norm_text(family)] = family
        for alias in aliases:
            alias_to_family[norm_text(alias)] = family

    per_gene: defaultdict[str, list[tuple[int, int, dict[str, Any]]]] = defaultdict(list)
    per_family: defaultdict[str, list[tuple[int, int, dict[str, Any]]]] = defaultdict(list)
    global_heap: list[tuple[int, int, dict[str, Any]]] = []
    source_count = 0
    with path.open(encoding="utf-8") as handle:
        for counter, line in enumerate(handle):
            if not line.strip():
                continue
            doc = json.loads(line)
            source_count += 1
            push_smallest(global_heap, doc, TARGET_DOCS, counter)
            for gene in doc_gene_hits(doc, query_genes):
                push_smallest(per_gene[gene], doc, 30, counter)
            family = doc_family(doc, alias_to_family)
            if family in panel_families:
                push_smallest(per_family[family], doc, 200, counter)

    selected: dict[str, dict[str, Any]] = {}
    for heaps in [per_gene.values(), per_family.values()]:
        for heap in heaps:
            for _priority, _counter, doc in heap:
                selected[str(doc["doc_id"])] = doc
    if len(selected) > TARGET_DOCS:
        selected = dict(sorted(selected.items(), key=lambda item: stable_hash(item[0]))[:TARGET_DOCS])
    for _priority, _counter, doc in sorted(global_heap, reverse=True):
        if len(selected) >= min(TARGET_DOCS, source_count):
            break
        selected.setdefault(str(doc["doc_id"]), doc)
    docs = [selected[key] for key in sorted(selected)]
    node_types = Counter(str((doc.get("metadata") or {}).get("node_type") or doc.get("document_level") or "unknown") for doc in docs)
    return docs, {
        "source_doc_count": source_count,
        "sampled_doc_count": len(docs),
        "query_gene_count": len(query_genes),
        "genes_with_retained_docs": sum(bool(heap) for heap in per_gene.values()),
        "node_type_counts": dict(node_types.most_common()),
    }


def prepared_docs(rows: list[dict[str, Any]]) -> list[PreparedDoc]:
    return [PreparedDoc(**row) for row in rows]


def serialize_budgeted(docs_with_scores: list[tuple[PreparedDoc, float]]) -> tuple[list[dict[str, Any]], int]:
    remaining = CONTEXT_CHAR_BUDGET
    output = []
    used = 0
    for doc, score in docs_with_scores[:TOP_K]:
        header = " ".join([doc.title, *doc.disease_tags, *doc.entity_names[:12]])
        header_cost = min(len(header), remaining)
        remaining -= header_cost
        text = doc.text[:remaining]
        remaining -= len(text)
        used += header_cost + len(text)
        metadata = doc.metadata or {}
        raw = metadata.get("raw_node") if isinstance(metadata.get("raw_node"), dict) else {}
        source_family = metadata.get("source_family") or raw.get("source_family") or metadata.get("source_graph")
        output.append({
            "doc_id": doc.doc_id,
            "knowledge_source": doc.knowledge_source,
            "document_level": doc.document_level,
            "parent_id": doc.parent_id,
            "title": doc.title[:header_cost],
            "score": round(score, 4),
            "disease_tags": doc.disease_tags,
            "entity_names": doc.entity_names[:12],
            "entity_ids": doc.entity_ids[:12],
            "text": text,
            "metadata": {"source_family": source_family} if source_family else {},
        })
        if remaining <= 0:
            break
    return output, used


def budget_existing_docs(docs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    prepared = []
    for doc in docs[:TOP_K]:
        prepared.append((PreparedDoc(
            doc_id=str(doc.get("doc_id") or ""),
            knowledge_source=str(doc.get("knowledge_source") or ""),
            document_level=str(doc.get("document_level") or ""),
            parent_id=doc.get("parent_id"),
            title=str(doc.get("title") or ""),
            text=str(doc.get("text") or ""),
            disease_tags=list(doc.get("disease_tags") or []),
            entity_names=list(doc.get("entity_names") or []),
            entity_ids=list(doc.get("entity_ids") or []),
            metadata=dict(doc.get("metadata") or {}),
        ), float(doc.get("score") or 0.0)))
    return serialize_budgeted(prepared)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {"count": len(rows)}
    for metric in [*METRICS, "disease_hit", "cross_source_retrieved"]:
        summary[metric] = round(sum(float(row[metric]) for row in rows) / len(rows), 4)
    summary["average_context_chars"] = round(sum(row["context_chars"] for row in rows) / len(rows), 1)
    return summary


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Evidence Meets RAG: Knowledge-scale and Retrieval-budget Control",
        "",
        "## Design",
        "",
        "We compare three knowledge shapes under a common corpus budget of 18,251 documents, Vanilla BM25 retrieval, top-k=8, and an 8,000-character context budget on the balanced 145-record faithfulness panel. Semantic KG and Evidence KG are deterministically downsampled with coverage-preserving strata (up to 30 documents per benchmark gene and 200 per cancer family), followed by fixed-hash filling. Selection uses entity and disease anchors only, never gold answers. Full-corpus retrievals are also truncated to the same context budget for a sensitivity reference.",
        "",
        "## Size-matched Results",
        "",
        "| Source | Corpus Docs | Entity Hit | Joint Scope P@8 | Component Recall | Complete Bundle | Support MRR | Avg Context Chars |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for source in ["doc_corpus", "semantic_kg", "evidence_kg"]:
        row = payload["matched"][source]
        lines.append(
            f"| {row['label']} | {row['sampled_doc_count']} | {row['entity_hit']:.4f} | {row['joint_scope_precision']:.4f} | "
            f"{row['component_recall']:.4f} | {row['complete_bundle']:.4f} | {row['support_mrr']:.4f} | {row['average_context_chars']:.1f} |"
        )
    lines.extend(["", "## Evidence KG Paired Effects Under Equal Size", ""])
    for baseline in ["doc_corpus", "semantic_kg"]:
        effects = payload["paired_effects"][f"evidence_kg_minus_{baseline}"]
        label = payload["matched"][baseline]["label"]
        lines.append(
            f"- Versus **{label}**: Component Recall {effects['component_recall']['delta']:+.4f} "
            f"(95% CI {effects['component_recall']['ci95']}), Complete Bundle {effects['complete_bundle']['delta']:+.4f} "
            f"(95% CI {effects['complete_bundle']['ci95']})."
        )
    lines.extend(["", "## Full-to-matched Sensitivity", ""])
    for source in ["semantic_kg", "evidence_kg"]:
        full = payload["full_budgeted"][source]
        matched = payload["matched"][source]
        lines.append(
            f"- **{matched['label']}**: Component Recall {full['component_recall']:.4f} -> {matched['component_recall']:.4f}; "
            f"Complete Bundle {full['complete_bundle']:.4f} -> {matched['complete_bundle']:.4f}."
        )
    evidence = payload["matched"]["evidence_kg"]
    doc = payload["matched"]["doc_corpus"]
    semantic = payload["matched"]["semantic_kg"]
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"At the same 18,251-document and 8,000-character budgets, Evidence KG obtains Component Recall {evidence['component_recall']:.4f} and Complete Bundle {evidence['complete_bundle']:.4f}, compared with {doc['component_recall']:.4f}/{doc['complete_bundle']:.4f} for Document Corpus and {semantic['component_recall']:.4f}/{semantic['complete_bundle']:.4f} for Semantic KG. This is the primary test of whether the retrieval advantage survives corpus-size control.",
        "Coverage-preserving sampling is benchmark-aware at the entity/disease level, so this experiment controls scale without pretending to be a benchmark-independent corpus construction. It is best described as a matched-budget sensitivity analysis.",
        "The experiment evaluates retrieval composition rather than regenerated answer accuracy. It supports a representation-efficiency claim only when the paired component and bundle intervals remain positive; answer-level confirmation would require new generation runs on the matched contexts.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    panel = read_jsonl(PANEL)
    panel_by_id = {str(row["id"]): row for row in panel}
    matched_rows: dict[str, list[dict[str, Any]]] = {}
    matched_summary = {}
    full_summary = {}
    build_reports = {}
    all_records = []

    for source_key, config in SOURCES.items():
        print(f"[sample] {config['label']}", flush=True)
        sampled, build_report = deterministic_sample(config["docs"], panel)
        build_reports[source_key] = build_report
        sample_path = OUTPUT_DIR / f"{source_key}_matched_{TARGET_DOCS}_docs.jsonl"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with sample_path.open("w", encoding="utf-8") as handle:
            for doc in sampled:
                handle.write(json.dumps(doc, ensure_ascii=False) + "\n")
        corpus = PreparedCorpus(prepared_docs(sampled), [])
        source_rows = []
        for record in panel:
            retrieved = RETRIEVERS["naive_rag"](corpus, record, TOP_K)
            docs, chars = serialize_budgeted(retrieved)
            row = {
                "record_id": str(record["id"]),
                "knowledge_source": source_key,
                "condition": "size_matched",
                "task_type": record.get("task_type"),
                "context_chars": chars,
                **audit_retrieval(record, docs),
            }
            source_rows.append(row)
            all_records.append(row)
        matched_rows[source_key] = source_rows
        matched_summary[source_key] = {"label": config["label"], **build_report, **summarize_rows(source_rows)}

        existing = {str((row.get("record") or {}).get("id")): row for row in read_jsonl(config["full_results"])}
        full_rows = []
        for rid, record in panel_by_id.items():
            docs, chars = budget_existing_docs(existing[rid].get("retrieved_docs") or [])
            row = {
                "record_id": rid,
                "knowledge_source": source_key,
                "condition": "full_corpus_budgeted",
                "task_type": record.get("task_type"),
                "context_chars": chars,
                **audit_retrieval(record, docs),
            }
            full_rows.append(row)
            all_records.append(row)
        full_summary[source_key] = {"label": config["label"], **summarize_rows(full_rows)}

    by_id = {source: {row["record_id"]: row for row in rows} for source, rows in matched_rows.items()}
    paired = {}
    for baseline in ["doc_corpus", "semantic_kg"]:
        paired[f"evidence_kg_minus_{baseline}"] = {
            metric: bootstrap_delta(by_id["evidence_kg"], by_id[baseline], metric, iterations=10000)
            for metric in ["entity_hit", "joint_scope_precision", "component_recall", "complete_bundle", "support_mrr"]
        }
    payload = {
        "design": {
            "panel": str(PANEL.relative_to(ROOT)),
            "record_count": len(panel),
            "target_doc_count": TARGET_DOCS,
            "top_k": TOP_K,
            "context_char_budget": CONTEXT_CHAR_BUDGET,
            "sampling": "Deterministic entity/disease coverage strata plus fixed-hash fill; no answer labels used.",
        },
        "matched": matched_summary,
        "full_budgeted": full_summary,
        "paired_effects": paired,
        "build_reports": build_reports,
    }
    with (OUTPUT_DIR / "budget_control_records.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (OUTPUT_DIR / "budget_control_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = render_report(payload)
    (OUTPUT_DIR / "budget_control_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

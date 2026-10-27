from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from analyze_retrieval_evidence_quality import audit_retrieval
from rag_eval_lib import build_query_text, load_prepared_corpus, read_jsonl, retrieve_naive
from run_rag_evaluation import serialize_doc


ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = ROOT / "dataset/rag/knowledge_sources"
BENCHMARK = ROOT / "dataset/evaluated/external_reasoning_benchmark_formal_v3_571.jsonl"
OUTPUT_DIR = ROOT / "dataset/rag/analysis/ecorag_cross_source"
OUTPUT = OUTPUT_DIR / "cross_source_retrieval_127.jsonl"
CANDIDATE_K = 256
TOP_K = 8

SOURCE_MAP = {
    "tcga_linkedomics": "tcga",
    "depmap": "depmap",
    "pubmed_literature": "pubmed",
}
TARGET_SOURCES = {"tcga", "depmap", "pubmed"}
VARIANTS = {
    "full": {"mode": "full"},
    "without_tcga": {"mode": "without", "source": "tcga"},
    "without_depmap": {"mode": "without", "source": "depmap"},
    "without_pubmed": {"mode": "without", "source": "pubmed"},
    "only_tcga": {"mode": "only", "source": "tcga"},
    "only_depmap": {"mode": "only", "source": "depmap"},
    "only_pubmed": {"mode": "only", "source": "pubmed"},
}


def family(record: dict[str, Any]) -> str:
    return str((record.get("metadata") or {}).get("priority_cancer_family") or record.get("oncology_focus") or "").lower()


def normalize_sources(values: Any) -> set[str]:
    if isinstance(values, str):
        values = [values]
    output = set()
    for value in values or []:
        normalized = SOURCE_MAP.get(str(value), str(value).lower())
        output.add(normalized if normalized in TARGET_SOURCES else "other")
    return output


def source_requirements(doc: Any, corpus: Any, cache: dict[str, set[str]], visiting: set[str] | None = None) -> set[str]:
    if doc.doc_id in cache:
        return cache[doc.doc_id]
    visiting = set(visiting or set())
    if doc.doc_id in visiting:
        return set()
    visiting.add(doc.doc_id)
    metadata = doc.metadata or {}
    raw_node = metadata.get("raw_node") or {}
    node_type = str(metadata.get("node_type") or raw_node.get("node_type") or "")
    direct = normalize_sources(raw_node.get("source_family"))
    if direct and "other" not in direct and str(raw_node.get("source_family")) != "integrated":
        cache[doc.doc_id] = direct
        return direct

    context = raw_node.get("source_context") or {}
    sources = normalize_sources(context.get("member_source_families") or [])
    member_ids = list(context.get("member_evidence_ids") or []) + list(raw_node.get("aggregation_parent_ids") or [])
    if node_type in {"target_support_summary", "mechanism_role_summary"}:
        member_ids.extend(metadata.get("derived_from_node_ids") or [])
        sources.update(normalize_sources([key for key in (metadata.get("source_family_counts") or {}) if key != "integrated"]))
    for member_id in dict.fromkeys(member_ids):
        member = corpus.doc_by_id.get(str(member_id))
        if member is not None:
            sources.update(source_requirements(member, corpus, cache, visiting))
    if not sources and direct:
        sources = direct
    cache[doc.doc_id] = sources
    return sources


def eligible(requirements: set[str], spec: dict[str, str]) -> bool:
    mode = spec["mode"]
    if mode == "full":
        return True
    source = spec["source"]
    if mode == "without":
        return source not in requirements
    return bool(requirements) and requirements <= {source}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = [record for record in read_jsonl(BENCHMARK) if family(record) in {"hcc", "crc"}]
    if len(panel) != 127:
        raise RuntimeError(f"Expected 127 HCC/CRC records, found {len(panel)}")
    corpus = load_prepared_corpus(KNOWLEDGE_DIR, "multi_cancer", "evidence_graph_complete", use_cache=True)
    requirement_cache: dict[str, set[str]] = {}
    exposure_counts = Counter()
    changed_counts = Counter()
    rows = []
    with OUTPUT.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(panel, start=1):
            candidates = retrieve_naive(corpus, record, CANDIDATE_K)
            candidate_rows = []
            for doc, score in candidates:
                requirements = source_requirements(doc, corpus, requirement_cache)
                relation_hits = sorted({relation for _neighbor, relation, _weight in corpus.graph_neighbors.get(doc.doc_id, []) if relation not in {"HAS_CHUNK", "NEXT_CHUNK"}})[:10]
                serialized = serialize_doc(doc, score, relation_hits)
                serialized["metadata"]["source_requirements"] = sorted(requirements)
                candidate_rows.append((serialized, requirements))
            full_ids = [doc["doc_id"] for doc, _requirements in candidate_rows[:TOP_K]]
            for source in TARGET_SOURCES:
                if any(source in requirements for _doc, requirements in candidate_rows[:TOP_K]):
                    exposure_counts[source] += 1
            for variant, spec in VARIANTS.items():
                selected = [doc for doc, requirements in candidate_rows if eligible(requirements, spec)][:TOP_K]
                selected_ids = [doc["doc_id"] for doc in selected]
                if selected_ids != full_ids:
                    changed_counts[variant] += 1
                audit = audit_retrieval(record, selected)
                row = {
                    "record_id": str(record["id"]),
                    "family": family(record),
                    "task_type": record.get("task_type"),
                    "variant": variant,
                    "query_text": build_query_text(record),
                    "record": record,
                    "retrieved_docs": selected,
                    "retrieval_audit": audit,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
            if index % 10 == 0 or index == len(panel):
                print(f"[retrieval] {index}/{len(panel)}", flush=True)
    report = {
        "design": {
            "records": len(panel),
            "families": dict(Counter(family(record) for record in panel)),
            "candidate_k": CANDIDATE_K,
            "top_k": TOP_K,
            "ranking_control": "All variants filter the same full-index BM25 candidate ranking; IDF and score calibration are fixed.",
            "mixed_derived_unit_policy": "An integrated aggregate or summary is eligible only when all resolved source requirements satisfy the variant.",
        },
        "full_top8_exposure": {source: {"count": exposure_counts[source], "rate": round(exposure_counts[source] / len(panel), 4)} for source in sorted(TARGET_SOURCES)},
        "changed_contexts": {variant: {"count": changed_counts[variant], "rate": round(changed_counts[variant] / len(panel), 4)} for variant in VARIANTS},
        "row_count": len(rows),
    }
    (OUTPUT_DIR / "cross_source_retrieval_build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

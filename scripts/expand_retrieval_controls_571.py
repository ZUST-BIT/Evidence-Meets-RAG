"""Extend the existing retrieval controls to the unchanged 571-question benchmark.

The size-matched corpora are reused verbatim; they are never resampled. A compact
query-vocabulary BM25 index implements the original formula and token order while
avoiding loading the large evidence graph (and all graph neighbors) into memory.
Every ranked setting is checkpointed individually. Existing 145/20-question
controls are checked for exact retrieved IDs and audit values before final output.
"""
from __future__ import annotations

import argparse
from array import array
from collections import Counter, defaultdict
from functools import lru_cache
import gc
import hashlib
import json
import math
from pathlib import Path
import pickle
from typing import Any

import numpy as np

import analyze_retrieval_evidence_quality as retrieval_quality
from analyze_budget_controlled_retrieval import (
    SOURCES, TARGET_DOCS, budget_existing_docs, serialize_budgeted, summarize_rows,
)
from analyze_retrieval_evidence_quality import audit_retrieval, bootstrap_delta
from analyze_retrieval_hyperparameter_stability import (
    PRIMARY_METRICS, paired_delta, summarize_rows as summarize_hyper,
)
from rag_eval_lib import PreparedDoc, build_query_text, prepared_corpus_paths, tokenize
from run_rag_evaluation import serialize_doc
from run_retrieval_hyperparameter_stability import FORMS, SETTINGS


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "dataset/evaluated/external_reasoning_benchmark_formal_v3_571.jsonl"
BASE = ROOT / "dataset/rag/analysis/unified_571"
OLD_BUDGET = ROOT / "dataset/rag/analysis/ecorag_budget_control"
OLD_HYPER = ROOT / "dataset/rag/analysis/evidence_meets_rag_hyperparameter_stability/retrieval_hyperparameter_stability_145.jsonl"
EXPECTED_N = 571
SOURCES["evidence_kg"]["full_results"] = ROOT / "dataset/rag/results/fixed_context_component_ablation_571_vanilla__full.jsonl"


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def checked_rows(path: Path, key_fields: tuple[str, ...], ids: set[str]) -> dict[tuple[str, ...], dict[str, Any]]:
    result = {}
    for row in read_rows(path):
        key = tuple(str(row[field]) for field in key_fields)
        assert str(row["record_id"]) in ids, (path, "unexpected ID", row["record_id"])
        assert key not in result, (path, "duplicate key", key)
        result[key] = row
    return result


class CompactBM25:
    """Same BM25 calculation as PreparedCorpus, retaining only query postings."""

    def __init__(self, path: Path, queries: list[str], cache_dir: Path) -> None:
        self.path = path
        words = {word for query in queries for word in tokenize(query)}
        stat = path.stat()
        self.signature = {
            "path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "query_vocabulary_sha256": hashlib.sha256("\n".join(sorted(words)).encode()).hexdigest(),
            "format": 1,
        }
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = cache_dir / f"{path.stem}.compact_bm25.pkl"
        if cache.exists():
            with cache.open("rb") as handle:
                payload = pickle.load(handle)
            if payload["signature"] == self.signature:
                self.__dict__.update(payload["index"])
                print(f"[index-cache] {path.name}: {len(self.ids)} docs", flush=True)
                return
        ids, lengths, offsets = [], array("I"), array("Q")
        posting_ids, posting_tfs = defaultdict(lambda: array("I")), defaultdict(lambda: array("I"))
        with path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                doc = PreparedDoc(**json.loads(line))
                index = len(ids)
                ids.append(doc.doc_id)
                offsets.append(offset)
                counts = Counter(tokenize(doc.searchable_text))
                lengths.append(sum(counts.values()))
                for word in words.intersection(counts):
                    posting_ids[word].append(index)
                    posting_tfs[word].append(counts[word])
                if len(ids) % 25000 == 0:
                    print(f"[index-build] {path.name}: {len(ids)} docs", flush=True)
        assert len(set(ids)) == len(ids), (path, "duplicate doc IDs")
        self.ids = ids
        self.offsets = np.asarray(offsets, dtype=np.uint64)
        self.lengths = np.asarray(lengths, dtype=np.float64)
        self.avg_doc_len = float(sum(lengths) / max(len(lengths), 1))
        self.postings = {
            word: (np.asarray(posting_ids[word], dtype=np.uint32), np.asarray(tfs, dtype=np.float64))
            for word, tfs in posting_tfs.items()
        }
        del posting_ids, posting_tfs
        self.id_to_index = {doc_id: i for i, doc_id in enumerate(ids)}
        fields = ["ids", "offsets", "lengths", "avg_doc_len", "postings", "id_to_index"]
        with cache.with_suffix(".tmp").open("wb") as handle:
            pickle.dump({"signature": self.signature, "index": {field: getattr(self, field) for field in fields}}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        cache.with_suffix(".tmp").replace(cache)
        print(f"[index-ready] {path.name}: {len(ids)} docs; {sum(len(x[0]) for x in self.postings.values())} query postings", flush=True)

    def rank(self, query: str, limit: int, k1: float = 1.5, b: float = 0.75) -> list[tuple[str, float]]:
        scores = np.zeros(len(self.ids), dtype=np.float64)
        # Query-token repetitions and accumulation order match bm25_scores exactly.
        for word in tokenize(query):
            if word not in self.postings:
                continue
            indexes, tfs = self.postings[word]
            df = len(indexes)
            idf = math.log(1 + (len(self.ids) - df + 0.5) / (df + 0.5))
            denominator = tfs + k1 * (1 - b + b * self.lengths[indexes] / max(self.avg_doc_len, 1))
            scores[indexes] += idf * ((tfs * (k1 + 1)) / np.maximum(denominator, 1e-9))
        positive = np.flatnonzero(scores > 0)
        if len(positive) > limit:
            cutoff = np.partition(scores[positive], -limit)[-limit]
            positive = positive[scores[positive] >= cutoff]
        ranked = sorted(((self.ids[int(i)], float(scores[i])) for i in positive), key=lambda item: (item[1], item[0]), reverse=True)
        return ranked[:limit]

    @lru_cache(maxsize=1024)
    def doc(self, doc_id: str) -> PreparedDoc:
        with self.path.open("rb") as handle:
            handle.seek(int(self.offsets[self.id_to_index[doc_id]]))
            row = json.loads(handle.readline())
        assert row["doc_id"] == doc_id
        return PreparedDoc(**row)


def validate_original_panel(records: list[dict[str, Any]]) -> None:
    assert len(records) == len({str(row["id"]) for row in records}) == EXPECTED_N


def ensure_matched_corpora() -> None:
    """Reconstruct the fixed published sample, never resample a new panel."""
    plan = json.loads((ROOT / "configs/matched_corpus_ids.json").read_text())
    OLD_BUDGET.mkdir(parents=True, exist_ok=True)
    reports = {}
    for source, item in plan["sources"].items():
        target = OLD_BUDGET / f"{source}_matched_{TARGET_DOCS}_docs.jsonl"
        expected = item["doc_ids"]
        if not target.exists():
            wanted = set(expected)
            selected = {}
            with SOURCES[source]["docs"].open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        doc = json.loads(line)
                        if doc["doc_id"] in wanted:
                            selected[doc["doc_id"]] = line
            if set(selected) != wanted:
                raise ValueError(f"{source}: {len(wanted-set(selected))} published matched IDs missing; restore the archived corpus rather than resampling")
            temporary = target.with_suffix(".tmp")
            temporary.write_text("".join(selected[key] for key in expected), encoding="utf-8")
            temporary.replace(target)
        actual = [row["doc_id"] for row in read_rows(target)]
        if actual != expected:
            raise ValueError(f"{source}: fixed matched-corpus IDs/order differ from the publication")
        reports[source] = item["build_report"]
    write_json(OLD_BUDGET / "budget_control_summary.json", {"build_reports": reports})


def run_budget(records: list[dict[str, Any]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    ids = {str(row["id"]) for row in records}
    rows_path = output / "budget_control_records.jsonl"
    existing = checked_rows(rows_path, ("knowledge_source", "condition", "record_id"), ids)
    ensure_matched_corpora()
    original_summary = json.loads((OLD_BUDGET / "budget_control_summary.json").read_text())
    matched, full = {}, {}
    queries = [build_query_text(row) for row in records]
    for source, config in SOURCES.items():
        remaining = [r for r in records if (source, "size_matched", str(r["id"])) not in existing]
        sample = OLD_BUDGET / f"{source}_matched_{TARGET_DOCS}_docs.jsonl"
        if remaining:
            print(f"[budget-corpus] {source}", flush=True)
            corpus = CompactBM25(sample, queries, output / "indexes")
            assert len(corpus.ids) == TARGET_DOCS
            for index, record in enumerate(remaining, 1):
                rid = str(record["id"])
                ranked = corpus.rank(build_query_text(record), 8)
                docs, chars = serialize_budgeted([(corpus.doc(doc_id), score) for doc_id, score in ranked])
                row = {
                    "record_id": rid, "knowledge_source": source, "condition": "size_matched",
                    "task_type": record.get("task_type"), "context_chars": chars,
                    "doc_ids": [doc["doc_id"] for doc in docs], **audit_retrieval(record, docs),
                }
                append_row(rows_path, row)
                existing[(source, "size_matched", rid)] = row
                if index % 25 == 0 or index == len(remaining):
                    print(f"[budget] {source}: {index}/{len(remaining)}", flush=True)
            corpus.doc.cache_clear()
            del corpus
            gc.collect()
        source_results = {}
        for result in read_rows(config["full_results"]):
            rid = str(result["record"]["id"])
            assert rid not in source_results, (source, "duplicate full-result ID", rid)
            source_results[rid] = result
        assert set(source_results) == ids, (source, "full-result IDs differ from benchmark")
        for record in records:
            rid = str(record["id"])
            key = (source, "full_corpus_budgeted", rid)
            if key in existing:
                continue
            docs, chars = budget_existing_docs(source_results[rid].get("retrieved_docs") or [])
            row = {
                "record_id": rid, "knowledge_source": source, "condition": "full_corpus_budgeted",
                "task_type": record.get("task_type"), "context_chars": chars, **audit_retrieval(record, docs),
            }
            append_row(rows_path, row)
            existing[key] = row
        del source_results
        for condition, target in [("size_matched", matched), ("full_corpus_budgeted", full)]:
            subset = [existing[(source, condition, str(r["id"]))] for r in records]
            target[source] = {"label": config["label"], **summarize_rows(subset)}
            if condition == "size_matched":
                target[source].update(original_summary["build_reports"][source])
    assert len(existing) == 3 * 2 * EXPECTED_N
    # Exact audit agreement on every pre-existing panel record, including budgeted full contexts.
    reference_n = 0
    for old in read_rows(OLD_BUDGET / "budget_control_records.jsonl"):
        key = (old["knowledge_source"], old["condition"], str(old["record_id"]))
        new = existing[key]
        assert all(new[field] == value for field, value in old.items()), ("budget reference mismatch", key, {f: (old[f], new[f]) for f in old if new[f] != old[f]})
        reference_n += 1
    by_id = {
        source: {str(r["id"]): existing[(source, "size_matched", str(r["id"]))] for r in records}
        for source in SOURCES
    }
    effects = {
        f"evidence_kg_minus_{baseline}": {
            metric: {"n": EXPECTED_N, **bootstrap_delta(by_id["evidence_kg"], by_id[baseline], metric, iterations=10000)}
            for metric in ["entity_hit", "joint_scope_precision", "component_recall", "complete_bundle", "support_mrr"]
        }
        for baseline in ["doc_corpus", "semantic_kg"]
    }
    payload = {
        "design": {
            "panel": str(PANEL.relative_to(ROOT)), "record_count": EXPECTED_N,
            "target_doc_count": TARGET_DOCS, "top_k": 8, "context_char_budget": 8000,
            "sampling": "Unchanged original 18,251-document matched corpus files; no resampling.",
            "queries": "Unmodified original build_query_text(record); original metadata retained.",
            "ci": "Paired percentile bootstrap, 10000 iterations; original seed 20260627:<metric>.",
        },
        "matched": matched, "full_budgeted": full, "paired_effects": effects,
        "build_reports": original_summary["build_reports"],
        "original_records_exactly_verified": reference_n,
    }
    write_json(output / "budget_control_summary.json", payload)
    print("[budget-complete] " + json.dumps(effects, ensure_ascii=False), flush=True)


def relevant_relations(edges_path: Path, requested: set[str]) -> dict[str, list[str]]:
    relations: dict[str, set[str]] = defaultdict(set)
    if edges_path.exists():
        with edges_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                edge = json.loads(line)
                relation = edge["relation"]
                if relation in {"HAS_CHUNK", "NEXT_CHUNK"}:
                    continue
                for doc_id in [edge["source_doc_id"], edge["target_doc_id"]]:
                    if doc_id in requested:
                        relations[doc_id].add(relation)
    return {doc_id: sorted(values)[:10] for doc_id, values in relations.items()}


def cache_hyper_audit_features():
    """Memoize unchanged original helpers for repeated documents within a form."""
    names = ["doc_components", "doc_disease_match", "doc_entity_match"]
    originals = {name: getattr(retrieval_quality, name) for name in names}
    components, disease, entity = {}, {}, {}

    def cached_components(doc):
        key = str(doc["doc_id"])
        if key not in components:
            components[key] = originals["doc_components"](doc)
        return components[key]

    def cached_disease(record, doc):
        key = (str(record["id"]), str(doc["doc_id"]))
        if key not in disease:
            disease[key] = originals["doc_disease_match"](record, doc)
        return disease[key]

    def cached_entity(record, doc):
        key = (str(record["id"]), str(doc["doc_id"]))
        if key not in entity:
            entity[key] = originals["doc_entity_match"](record, doc)
        return entity[key]

    retrieval_quality.doc_components = cached_components
    retrieval_quality.doc_disease_match = cached_disease
    retrieval_quality.doc_entity_match = cached_entity

    def restore():
        for name, original in originals.items():
            setattr(retrieval_quality, name, original)

    return restore


def run_hyper(records: list[dict[str, Any]], output: Path, only_forms: list[str] | None = None,
              recovered_source_dir: Path | None = None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    ids = {str(r["id"]) for r in records}
    by_id = {str(r["id"]): r for r in records}
    queries = [build_query_text(row) for row in records]
    ranks_path = output / "retrieval_rank_checkpoint.jsonl"
    rows_path = output / "retrieval_hyperparameter_stability_571.jsonl"
    ranks = checked_rows(ranks_path, ("form", "setting", "record_id"), ids)
    existing = checked_rows(rows_path, ("form", "setting", "record_id"), ids)
    for form, source in FORMS.items():
        if only_forms and form not in only_forms:
            continue
        if all((form, s["setting"], rid) in existing for s in SETTINGS for rid in ids):
            continue
        docs_path, edges_path = prepared_corpus_paths(ROOT / "dataset/rag/knowledge_sources", "multi_cancer", source)
        if recovered_source_dir is not None:
            restored = recovered_source_dir / docs_path.name
            assert restored.exists(), ("recovered source missing", restored)
            print(f"[recovered-source] {docs_path} <- {restored}", flush=True)
            docs_path = restored
        print(f"[hyper-corpus] {form}", flush=True)
        corpus = CompactBM25(docs_path, queries, output / "indexes")
        for index, record in enumerate(records, 1):
            rid = str(record["id"])
            score_cache = {}
            for setting in SETTINGS:
                key = (form, setting["setting"], rid)
                if key in ranks:
                    continue
                score_key = (setting["k1"], setting["b"])
                if score_key not in score_cache:
                    limit = 16 if score_key == (1.5, 0.75) else 8
                    score_cache[score_key] = corpus.rank(build_query_text(record), limit, *score_key)
                ranked = score_cache[score_key][:setting["top_k"]]
                row = {"record_id": rid, "form": form, **setting, "ranked": ranked}
                append_row(ranks_path, row)
                ranks[key] = row
            if index % 25 == 0 or index == len(records):
                print(f"[hyper-rank] {form}: {index}/{len(records)}", flush=True)
        requested = {doc_id for key, row in ranks.items() if key[0] == form for doc_id, score in row["ranked"]}
        print(f"[hyper-relations] {form}: {len(requested)} retrieved unique docs", flush=True)
        relations = relevant_relations(edges_path, requested)
        restore_audit_helpers = cache_hyper_audit_features()
        for index, record in enumerate(records, 1):
            rid = str(record["id"])
            for setting in SETTINGS:
                key = (form, setting["setting"], rid)
                if key in existing:
                    continue
                ranked = ranks[key]["ranked"]
                docs = [serialize_doc(corpus.doc(doc_id), score, relations.get(doc_id, [])) for doc_id, score in ranked]
                row = {
                    "record_id": rid, "form": form, **setting,
                    "retrieval_audit": audit_retrieval(record, docs),
                    "doc_ids": [doc["doc_id"] for doc in docs],
                }
                append_row(rows_path, row)
                existing[key] = row
            if index % 25 == 0 or index == len(records):
                print(f"[hyper-audit] {form}: {index}/{len(records)}", flush=True)
        restore_audit_helpers()
        del restore_audit_helpers
        corpus.doc.cache_clear()
        del corpus, relations
        gc.collect()
    reference_n = 0
    for old in read_rows(OLD_HYPER):
        key = (old["form"], old["setting"], str(old["record_id"]))
        if only_forms and key[0] not in only_forms:
            continue
        new = existing[key]
        assert new == old, ("hyper reference mismatch", key, {f: (old[f], new[f]) for f in old if new[f] != old[f]})
        reference_n += 1
    if only_forms:
        write_json(output / ("partial_verification_" + "_".join(only_forms) + ".json"), {"original_records_exactly_verified": reference_n})
        return
    assert len(existing) == len(FORMS) * len(SETTINGS) * EXPECTED_N
    rows = list(existing.values())
    summaries, effects = {}, {}
    for setting in SETTINGS:
        name = setting["setting"]
        summaries[name] = {}
        for form in FORMS:
            subset = [existing[(form, name, str(r["id"]))] for r in records]
            assert len(subset) == EXPECTED_N
            summaries[name][form] = summarize_hyper(subset)
        effects[name] = {
            f"evidence_graph_minus_{baseline}": {
                metric: paired_delta(rows, name, "evidence_graph", baseline, metric)
                for metric in PRIMARY_METRICS
            }
            for baseline in ["document_corpus", "semantic_kg"]
        }
    stability = {}
    for label, settings in [("topk", ["topk_2", "topk_4", "default", "topk_16"]), ("bm25", ["k1_0.9", "k1_2.0", "b_0.25", "b_1.0"])]:
        for baseline, short in [("document_corpus", "document"), ("semantic_kg", "semantic")]:
            stability[f"{label}_min_delta_vs_{short}"] = {
                metric: min(effects[name][f"evidence_graph_minus_{baseline}"][metric]["delta"] for name in settings)
                for metric in PRIMARY_METRICS
            }
    payload = {
        "input": str(rows_path.relative_to(ROOT)), "rows": len(rows),
        "design": {"records": EXPECTED_N, "bm25_parameter_panel_records": EXPECTED_N, "forms": FORMS, "settings": SETTINGS,
                   "queries": "Unmodified original build_query_text(record); original metadata retained.",
                   "serialization": "Original serialize_doc: per-document text[:1800], original metadata, relation_hits.",
                   "ci": "Paired percentile bootstrap, 10000 iterations; original seed 20260629:<setting>:<form_a>:<form_b>:<metric>."},
        "summaries": summaries, "paired_effects": effects, "stability": stability,
        "original_records_exactly_verified": reference_n,
    }
    write_json(output / "retrieval_hyperparameter_summary.json", payload)
    print("[hyper-complete] " + json.dumps(stability, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["budget", "hyperparam", "all"], default="all")
    parser.add_argument("--forms", nargs="+", choices=list(FORMS))
    parser.add_argument("--recovered-source-dir", type=Path,
                        help="Use an original source recovered from archive if its configured path is unavailable.")
    args = parser.parse_args()
    records = read_rows(PANEL)
    assert len(records) == EXPECTED_N
    ids = [str(row["id"]) for row in records]
    assert len(set(ids)) == EXPECTED_N
    validate_original_panel(records)
    if args.stage in {"budget", "all"}:
        run_budget(records, BASE / "budget")
    if args.stage in {"hyperparam", "all"}:
        run_hyper(records, BASE / "hyperparam", args.forms, args.recovered_source_dir)


if __name__ == "__main__":
    main()

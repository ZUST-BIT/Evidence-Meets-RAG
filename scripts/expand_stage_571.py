"""Expand the paper's eight R/C/G stage settings to the formal 571 IDs.

The default phase is offline retrieval preparation. Generation is explicit and
resumable, uses the project's unchanged generation routine, and never rewrites
the original experiment files. Historical 96-row generations are audited for
prompt equality but are not reused because their summaries omit the model.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import CancelledError, FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP
import gc
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import re
import subprocess
import sys
import time
from typing import Any, Iterator

from rag_eval_lib import (
    OpenAICompatibleClient, PreparedCorpus, PreparedDoc, build_prompt,
    build_query_text, collect_doc_gene_symbols, compute_answer_metrics,
    compute_retrieval_metrics, disease_term_set, doc_support_flags, norm_text,
    first_evidence_pack_doc, gold_label, macro_f1_from_labels,
    normalize_prediction, prepared_corpus_paths,
    set_rag_profile, set_stage_config, tokenize, top_docs_from_scores,
)
from run_rag_evaluation import build_context_blocks, generate_prediction, json_ready, retrieve_docs


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "dataset/evaluated/external_reasoning_benchmark_formal_v3_571.jsonl"
DEFAULT_OUTPUT = ROOT / "dataset/rag/results/unified_571/stage"
LEGACY_DIR = ROOT / "dataset/rag/results"
LEGACY = {
    "R0C0G0": "stage_probe96qa_base_flat_plain",
    "R0C0G1": "stage_probe96qa_base_flat_grounded",
    "R0C1G0": "stage_probe96qa_base_structured_plain",
    "R0C1G1": "stage_probe96qa_base_structured_grounded",
    "R1C0G0": "stage_probe96_r1probe2_full_flat_plain",
    "R1C0G1": "stage_probe96_r1probe2_full_flat_grounded",
    "R1C1G0": "stage_probe96_r1probe2_full_structured_plain",
    "R1C1G1": "stage_probe96_r1probe2_full_structured_grounded",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(json_ready(value), ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def save_json(path: Path, value: Any) -> None:
    if path.name == "manifest.json":
        with path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            latest = json.loads(path.read_text()) if path.exists() else {}
            def merge(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
                merged = dict(old)
                for key, item in new.items():
                    merged[key] = merge(merged[key], item) if isinstance(item, dict) and isinstance(merged.get(key), dict) else item
                return merged
            merged = merge(latest, value)
            value.clear()
            value.update(merged)
            temp = path.with_suffix(path.suffix + ".tmp")
            temp.write_text(json.dumps(json_ready(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temp.replace(path)
            return
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(json_ready(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def append_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_ready(row), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def row_id(row: dict[str, Any]) -> str:
    return str((row.get("record") or {}).get("id") or "")


def index_rows(path: Path, allowed: set[str], *, complete: bool = False) -> dict[str, dict[str, Any]]:
    result = {}
    for row in read_jsonl(path):
        rid = row_id(row)
        if rid not in allowed:
            raise RuntimeError(f"unexpected record ID in {path.name}: {rid}")
        if rid in result:
            raise RuntimeError(f"duplicate record ID in {path.name}: {rid}")
        if not complete or (row.get("prediction") not in (None, "") and row.get("normalized_prediction") not in (None, "")):
            result[rid] = row
    return result


def stage_values(label: str) -> tuple[str, str, str]:
    return ("full" if label[1] == "1" else "base",
            "structured" if label[3] == "1" else "flat",
            "grounded" if label[5] == "1" else "plain")


def memory_mb() -> float:
    # ru_maxrss is bytes on macOS and KiB on Linux.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(value / (1024 ** 2 if sys.platform == "darwin" else 1024), 1)


def log(message: str) -> None:
    print(message, flush=True)


class QueryIndexedCorpus(PreparedCorpus):
    """The same full corpus, with postings restricted to certified query terms.

    Graph RAG calls retrieve_hybrid(build_query_text(record)) for seeds/base;
    its causal expansion does not issue any additional BM25 query. Restricting
    inverted postings to the union of those query terms leaves every score,
    document length, corpus size, IDF, and average document length unchanged.
    A runtime guard rejects any unexpected BM25 term instead of approximating.
    """

    def __init__(self, docs: list[PreparedDoc], edges: Any, query_terms: set[str], progress: bool = False) -> None:
        self.query_terms = query_terms
        self.progress = progress
        self.edge_count = 0
        super().__init__(docs, edges)
        import numpy as np
        self._np = np
        self._ids = list(self.doc_by_id)
        self._id_to_index = {doc_id: index for index, doc_id in enumerate(self._ids)}
        self._lengths = np.asarray([self.doc_len[doc_id] for doc_id in self._ids], dtype=np.float64)
        self._array_postings = {
            word: (np.asarray([self._id_to_index[doc_id] for doc_id, _ in postings], dtype=np.uint32),
                   np.asarray([tf for _, tf in postings], dtype=np.float64))
            for word, postings in self.postings.items()
        }
        self._last_query = None
        self._last_scores = None

    def _build(self) -> None:
        total_length = 0
        entity_nodes = {"entity", "cohort", "Gene", "Disease", "Drug", "Phenotype", "Pathway", "Cohort", "MolecularFeature", "Compound", "Symptom", "Anatomy", "TCM_CMM"}
        for index, doc in enumerate(self.docs, 1):
            if doc.parent_id:
                self.parent_to_children[doc.parent_id].append(doc.doc_id)
            doc._disease_terms = set()
            for tag in doc.disease_tags:
                doc._disease_terms.update(disease_term_set(tag))
            doc._normalized_entity_names = {norm_text(name) for name in doc.entity_names if norm_text(name)}
            doc._gene_symbols = collect_doc_gene_symbols(doc)
            doc._node_type = (doc.metadata or {}).get("node_type") or ""
            doc._has_disease_tags = bool(doc._disease_terms)
            doc._is_entity_like = doc.document_level == "node" or doc._node_type in entity_nodes
            doc._is_evidence_like = doc._node_type == "evidence" or doc.document_level == "parent"
            doc._support_flags = doc_support_flags(doc)
            for symbol in doc._gene_symbols:
                self.gene_symbol_to_doc_ids[symbol].append(doc.doc_id)
            tokens = tokenize(doc.searchable_text)
            self.doc_len[doc.doc_id] = len(tokens)
            total_length += len(tokens)
            term_freq = Counter(token for token in tokens if token in self.query_terms)
            for token, tf in term_freq.items():
                self.postings[token].append((doc.doc_id, tf))
                self.doc_freq[token] += 1
            if self.progress and index % 25000 == 0:
                log(f"[CORPUS_INDEX] docs={index}/{len(self.docs)} max_rss_mb={memory_mb()}")
        self.avg_doc_len = total_length / max(len(self.docs), 1)
        for edge in self.edges:
            source, target, relation = edge["source_doc_id"], edge["target_doc_id"], edge["relation"]
            weight = float(edge.get("weight") or 1.0)
            self.graph_neighbors[source].append((target, relation, weight))
            self.graph_neighbors[target].append((source, relation, weight))
            self.edge_count += 1
            if self.progress and self.edge_count % 500000 == 0:
                log(f"[CORPUS_INDEX] edges={self.edge_count} max_rss_mb={memory_mb()}")

    def bm25_scores(self, query_text: str, k1: float = 1.5, b: float = 0.75) -> dict[str, float]:
        unexpected = set(tokenize(query_text)) - self.query_terms
        if unexpected:
            raise RuntimeError("Unexpected BM25 query outside the certified 571-query vocabulary")
        key = query_text, k1, b
        if key == self._last_query:
            return dict(self._last_scores)
        np = self._np
        scores = np.zeros(len(self._ids), dtype=np.float64)
        for word in tokenize(query_text):
            if word not in self._array_postings:
                continue
            indexes, tfs = self._array_postings[word]
            df = len(indexes)
            idf = math.log(1 + (len(self._ids) - df + 0.5) / (df + 0.5))
            # Preserve Python's left-to-right operation order, including
            # b * dl / avgdl, and one accumulation per repeated query token.
            denominator = tfs + k1 * (1 - b + b * self._lengths[indexes] / max(self.avg_doc_len, 1))
            scores[indexes] += idf * ((tfs * (k1 + 1)) / np.maximum(denominator, 1e-9))
        result = {self._ids[int(index)]: float(scores[index]) for index in np.flatnonzero(scores > 0)}
        self._last_query, self._last_scores = key, result
        return dict(result)


def validate_query_index(records: list[dict[str, Any]], terms: set[str]) -> dict[str, Any]:
    import random
    rng = random.Random(571)
    words = sorted(terms)[:80] + ["unqueriednoisea", "unqueriednoiseb", "unqueriednoisec"]
    docs = [PreparedDoc(doc_id=f"fixture_{index:03d}", knowledge_source="evidence_graph_complete",
                        document_level="node", parent_id=None, title=f"BRCA1 evidence {index}",
                        text=" ".join(rng.choices(words, k=20 + index % 29)), disease_tags=["colorectal cancer"],
                        entity_names=["BRCA1"], entity_ids=["HGNC:1100"], metadata={"node_type": "evidence"})
            for index in range(101)]
    edges = [{"source_doc_id": docs[index].doc_id, "target_doc_id": docs[index + 1].doc_id,
              "relation": "SUPPORTS", "weight": 1} for index in range(100)]
    baseline = PreparedCorpus(deepcopy(docs), deepcopy(edges))
    restricted = QueryIndexedCorpus(deepcopy(docs), iter(deepcopy(edges)), terms)
    assert baseline.doc_len == restricted.doc_len
    assert baseline.avg_doc_len == restricted.avg_doc_len
    assert dict(baseline.graph_neighbors) == dict(restricted.graph_neighbors)
    for record in records:
        query = build_query_text(record)
        left, right = baseline.bm25_scores(query), restricted.bm25_scores(query)
        assert left == right, "query-restricted BM25 differs from baseline"
        assert [(doc.doc_id, score) for doc, score in top_docs_from_scores(baseline, left, 8)] == [
            (doc.doc_id, score) for doc, score in top_docs_from_scores(restricted, right, 8)]
    examples = {}
    for record in records:
        examples.setdefault(record["task_type"], record)
    graph_cases = 0
    for retrieval_stage in ("base", "full"):
        set_stage_config(retrieval_stage, "structured", "grounded")
        for record in examples.values():
            assert json_digest(retrieve_docs(baseline, "graph_rag", deepcopy(record), 8)) == json_digest(
                retrieve_docs(restricted, "graph_rag", deepcopy(record), 8)), "Graph RAG fixture differs"
            graph_cases += 1
    return {"fixture_docs": 101, "fixture_edges": 100, "queries_tested": len(records),
            "exact_float_score_equality": True, "top8_id_score_order_equality": True,
            "exact_serialized_graph_retrieval_cases": graph_cases,
            "document_lengths_equal": True, "average_document_length_equal": True,
            "graph_adjacency_equal": True}


def load_corpus_streaming(knowledge_dir: Path, manifest: dict[str, Any], output: Path,
                          records: list[dict[str, Any]]) -> PreparedCorpus:
    docs_path, edges_path = prepared_corpus_paths(knowledge_dir, "multi_cancer", "evidence_graph_complete")
    started = time.time()
    terms = {token for record in records for token in tokenize(build_query_text(record))}
    validation = validate_query_index(records, terms)
    log(f"[CORPUS] certified_query_terms={len(terms)} exact_score_fixture_queries={len(records)}")
    # JSON-derived corpus objects are acyclic. Full-generation GC scans become
    # extremely costly on this multi-GB object graph; use reference counting
    # during construction/retrieval and free the entire corpus after preparation.
    gc.disable()
    log(f"[CORPUS] docs={docs_path} edges={edges_path} max_rss_mb={memory_mb()}")
    docs = [PreparedDoc(**row) for row in iter_jsonl(docs_path)]
    log(f"[CORPUS] loaded_docs={len(docs)} max_rss_mb={memory_mb()}")
    # PreparedCorpus consumes edges once during _build. Streaming them avoids a
    # second, multi-GB copy without changing the corpus or scoring algorithm.
    corpus = QueryIndexedCorpus(docs, iter_jsonl(edges_path), terms, progress=True)
    full_validation = []
    for index in (0, len(records) // 2, len(records) - 1):
        query = build_query_text(records[index])
        check_started = time.time()
        reference = PreparedCorpus.bm25_scores(corpus, query)
        vectorized = corpus.bm25_scores(query)
        assert reference == vectorized, f"Full-corpus exact BM25 mismatch for question {index}"
        full_validation.append({"record_id": str(records[index]["id"]), "matched_scores": len(reference),
                                "exact_float_equality": True, "seconds": round(time.time() - check_started, 2)})
        log(f"[CORPUS_VALIDATE] full_query={index} exact_scores={len(reference)} seconds={full_validation[-1]['seconds']}")
    del reference, vectorized
    corpus.postings.clear()  # Array postings now serve the same exact scoring formula.
    corpus.edges = []
    corpus.doc_term_freq.clear()  # bm25_scores uses postings, not these duplicate counters.
    corpus.source_docs_path = docs_path
    manifest["corpus"] = {
        "docs_path": str(docs_path), "edges_path": str(edges_path),
        "docs_bytes": docs_path.stat().st_size, "edges_bytes": edges_path.stat().st_size,
        "docs_sha256": sha256(docs_path), "edges_sha256": sha256(edges_path),
        "document_count": len(docs), "load_seconds": round(time.time() - started, 2),
        "max_rss_mb": memory_mb(), "loader": "exact query-restricted BM25 index, full docs and graph, streamed edges",
        "query_terms": len(terms), "query_terms_sha256": json_digest(sorted(terms)),
        "index_equivalence_validation": validation, "edge_count": corpus.edge_count,
        "full_corpus_vectorized_validation": full_validation,
        "python_version": sys.version, "numpy_version": corpus._np.__version__,
    }
    recovery_path = docs_path.parent / "recovery_manifest.json"
    if recovery_path.exists():
        manifest["corpus"]["recovery"] = json.loads(recovery_path.read_text())
    save_json(output / "manifest.json", manifest)
    log(f"[CORPUS] ready seconds={manifest['corpus']['load_seconds']} max_rss_mb={memory_mb()}")
    return corpus


def prepare(records: list[dict[str, Any]], allowed: set[str], args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    corpus = None
    for state in (0, 1):
        path = args.output_dir / f"R{state}_retrieval.jsonl"
        rows = index_rows(path, allowed)
        pending = [record for record in records if str(record["id"]) not in rows]
        log(f"[RETRIEVAL] R{state} existing={len(rows)} pending={len(pending)}")
        frozen = {}
        if getattr(args, "frozen_input_dir", None):
            frozen = {row_id(row): row for row in read_jsonl(args.frozen_input_dir / f"R{state}_retrieval.jsonl")}
            if set(frozen) != allowed:
                raise ValueError("Released stage contexts must contain exactly the frozen 571 IDs")
        if pending and not frozen and corpus is None:
            corpus = load_corpus_streaming(args.knowledge_dir, manifest, args.output_dir, records)
        set_stage_config("full" if state else "base", "structured", "grounded")
        for index, record in enumerate(pending, 1):
            row = {
                "record": deepcopy(record), "query_text": build_query_text(record),
                "retrieved_docs": deepcopy(frozen[str(record["id"])]["retrieved_docs"]) if frozen else retrieve_docs(corpus, "graph_rag", record, 8),
                "prediction": None, "normalized_prediction": None,
                "inferred_evidence_state": None, "prediction_mode": None,
            }
            row = json_ready(row)
            row["retrieval_sha256"] = json_digest(row["retrieved_docs"])
            append_row(path, row)
            rows[str(record["id"])] = row
            if index % 10 == 0 or index == len(pending):
                log(f"[RETRIEVAL] R{state} {len(rows)}/571 max_rss_mb={memory_mb()}")
        if set(rows) != allowed:
            raise RuntimeError(f"R{state} retrieval does not cover exactly the formal 571 IDs")
        manifest.setdefault("retrieval", {})[f"R{state}"] = {
            "count": len(rows), "path": str(path), "sha256": sha256(path),
            "retrieval_metrics": compute_retrieval_metrics(list(rows.values())),
        }
        save_json(args.output_dir / "manifest.json", manifest)
    del corpus
    gc.enable()
    gc.collect()
    audit_legacy(allowed, args.output_dir, manifest)
    audit_main_full(allowed, args.output_dir, manifest)


def audit_legacy(allowed: set[str], output: Path, manifest: dict[str, Any]) -> None:
    retrievals = {state: index_rows(output / f"R{state}_retrieval.jsonl", allowed) for state in (0, 1)}
    audit = {}
    seen_inputs = set()
    planned = {}
    for label, suffix in LEGACY.items():
        set_stage_config(*stage_values(label))
        legacy_path = LEGACY_DIR / f"multi_cancer__evidence_graph_complete__graph_rag__{suffix}.jsonl"
        summary_path = legacy_path.with_name(legacy_path.stem + "_summary.json")
        old = read_jsonl(legacy_path)
        matching = []
        for row in old:
            rid = row_id(row)
            if rid not in allowed:
                continue
            current = retrievals[int(label[1])][rid]
            if build_prompt(row["record"], build_context_blocks(row)) == build_prompt(current["record"], build_context_blocks(current)):
                matching.append(rid)
        experiment = json.loads(summary_path.read_text()).get("experiment", {}) if summary_path.exists() else {}
        audit[label] = {
            "legacy_path": str(legacy_path), "legacy_count": len(old),
            "identical_current_prompts": len(matching), "identical_prompt_ids": matching,
            "legacy_model": experiment.get("model"), "reused_predictions": 0,
            "reason": "Historical summary omits the model; generation is rerun rather than assuming identical model provenance.",
        }
        direct_keys = set()
        state_questions = 0
        for current in retrievals[int(label[1])].values():
            blocks = build_context_blocks(current)
            if first_evidence_pack_doc(blocks) is None:
                direct_keys.add(json_digest({"model": "gpt-5-mini", "prompt": build_prompt(current["record"], blocks),
                                             "run_signature": manifest["run_signature"]}))
            else:
                state_questions += 1
        planned[label] = {"questions": 571, "new_distinct_direct_inputs": len(direct_keys - seen_inputs),
                          "identical_previous_direct_inputs": len(direct_keys & seen_inputs),
                          "state_questions": state_questions}
        seen_inputs.update(direct_keys)
    save_json(output / "legacy_audit.json", audit)
    manifest["planned_generation"] = {
        "configs": planned, "distinct_direct_inputs": len(seen_inputs),
        "maximum_unshared_questions": 571 * 8,
        "state_calls_note": "state routes can need one or two calls; no evidence-pack docs means exactly one call per new direct input",
    }
    manifest["legacy_audit_path"] = str(output / "legacy_audit.json")
    manifest["status"] = "retrieval_prepared_generation_pending"
    save_json(output / "manifest.json", manifest)


def audit_main_full(allowed: set[str], output: Path, manifest: dict[str, Any]) -> None:
    """Distinguish the current stage Full input from the paper's saved main run."""
    path = LEGACY_DIR / "multi_cancer__evidence_graph_complete__graph_rag__formalv3_571_classic4_raw_vs_graph.jsonl"
    summary_path = path.with_name(path.stem + "_summary.json")
    previous = index_rows(path, allowed, complete=True)
    current = index_rows(output / "R1_retrieval.jsonl", allowed)
    set_stage_config("full", "structured", "grounded")
    counts = {"docid_order": 0, "text_order": 0, "score_order": 0,
              "docid_text_score_order": 0, "all_serialized_retrieval_fields": 0, "full_generation_prompt": 0}
    differences = []
    for rid, row in current.items():
        old = previous.get(rid)
        if old is None:
            differences.append({"id": rid, "reason": "missing previous completed prediction"})
            continue
        old_docs, new_docs = old["retrieved_docs"], row["retrieved_docs"]
        matches = {}
        for field, metric in (("doc_id", "docid_order"), ("text", "text_order"), ("score", "score_order")):
            matches[metric] = [doc.get(field) for doc in old_docs] == [doc.get(field) for doc in new_docs]
        matches["docid_text_score_order"] = all(matches.values())
        matches["all_serialized_retrieval_fields"] = json_digest(old_docs) == json_digest(new_docs)
        matches["full_generation_prompt"] = build_prompt(old["record"], build_context_blocks(old)) == build_prompt(row["record"], build_context_blocks(row))
        for metric, matched in matches.items():
            counts[metric] += int(matched)
        if not matches["full_generation_prompt"]:
            differences.append({"id": rid, "matching_fields": matches,
                                "old_doc_ids": [doc["doc_id"] for doc in old_docs],
                                "new_doc_ids": [doc["doc_id"] for doc in new_docs]})
    experiment = json.loads(summary_path.read_text()).get("experiment", {}) if summary_path.exists() else {}
    report = {"count": 571, "source_main_result": str(path), "main_experiment": experiment,
              "main_model_recorded": experiment.get("model"), "matching_question_counts": counts,
              "differences": differences, "reused_predictions": 0,
              "decision": "Input consistency audit only; the original main Graph RAG results are not changed."}
    save_json(output / "main_full_input_audit.json", report)
    manifest["main_full_input_audit_path"] = str(output / "main_full_input_audit.json")
    save_json(output / "manifest.json", manifest)


def generate(records: list[dict[str, Any]], allowed: set[str], args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    client = OpenAICompatibleClient(model="gpt-5-mini")
    if not client.is_configured():
        raise RuntimeError("OPENAI_API_KEY is not configured")
    for label in args.configs:
        set_stage_config(*stage_values(label))
        source = index_rows(args.output_dir / f"R{label[1]}_retrieval.jsonl", allowed)
        if set(source) != allowed:
            raise RuntimeError(f"R{label[1]} retrieval is incomplete; run --phase retrieve first")
        path = args.output_dir / f"{label}.jsonl"
        completed = index_rows(path, allowed, complete=True)
        for rid, row in completed.items():
            if row.get("retrieval_sha256") != source[rid].get("retrieval_sha256"):
                raise RuntimeError(f"stale retrieval in resumed generation: {label}/{rid}")
        pending = [deepcopy(source[str(record["id"])]) for record in records if str(record["id"]) not in completed]
        # Same model + same direct-prompt input receives one shared prediction
        # across stage settings. This removes gratuitous sampling differences
        # for settings whose actual inputs coincide. Only this run's proven
        # model/input provenance is reusable, never undocumented legacy rows.
        cache = {}
        for previous_label in LEGACY:
            for row in index_rows(args.output_dir / f"{previous_label}.jsonl", allowed, complete=True).values():
                key = row.get("generation_input_sha256")
                if key and row.get("model") == client.model and row.get("run_signature") == manifest["run_signature"]:
                    cache.setdefault(key, row)
        calls = []
        reused_count = 0
        for row in pending:
            blocks = build_context_blocks(row)
            if first_evidence_pack_doc(blocks) is None:
                key = json_digest({"model": client.model, "prompt": build_prompt(row["record"], blocks),
                                   "run_signature": manifest["run_signature"]})
                row["generation_input_sha256"] = key
            else:
                key = None
            previous = cache.get(key) if key else None
            if previous is None:
                calls.append(row)
                continue
            row.update(prediction=previous["prediction"],
                       normalized_prediction=normalize_prediction(previous["prediction"], row["record"]),
                       inferred_evidence_state=None, prediction_mode=previous["prediction_mode"],
                       stage_config=label, model=client.model, run_signature=manifest["run_signature"],
                       reused_from={"config": previous["stage_config"], "id": row_id(previous),
                                    "reason": "identical model and generation input hash"})
            append_row(path, row)
            completed[row_id(row)] = row
            reused_count += 1
        log(f"[GENERATION] {label} existing={len(completed)} pending={len(pending)} workers={args.workers}")
        log(f"[GENERATION] {label} new_api_calls={len(calls)} reused_identical_inputs={reused_count}")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(generate_prediction, client, row): row for row in calls}
            failure = None
            for future in as_completed(futures):
                row = futures[future]
                try:
                    pred, normalized, inferred_state, mode = future.result()
                except CancelledError:
                    continue
                except Exception as exc:
                    if failure is None:
                        status = getattr(getattr(exc, "response", None), "status_code", None)
                        failure = {"config": label, "error_type": type(exc).__name__, "http_status": status,
                                   "failed_id": row_id(row), "resume_command": "--phase generate"}
                        for queued in futures:
                            queued.cancel()
                    continue
                row.update(prediction=pred, normalized_prediction=normalized,
                           inferred_evidence_state=inferred_state, prediction_mode=mode,
                           stage_config=label, model=client.model,
                           run_signature=manifest["run_signature"])
                append_row(path, row)
                completed[row_id(row)] = row
                if len(completed) % 10 == 0 or len(completed) == 571:
                    log(f"[GENERATION] {label} {len(completed)}/571")
            if failure is not None:
                failure["completed"] = len(completed)
                save_json(args.output_dir / "generation_failure.json", failure)
                log("[GENERATION_FAILED] " + json.dumps(failure))
                raise RuntimeError(f"Generation stopped: {failure['error_type']}, HTTP {failure['http_status']}") from None
        if set(completed) != allowed:
            raise RuntimeError(f"{label} generation incomplete")
        rows = [completed[str(record["id"])] for record in records]
        summary = {
            "experiment": {**manifest["configuration"], "stage_config": label,
                           "retrieval_stage": stage_values(label)[0], "context_stage": stage_values(label)[1],
                           "generation_stage": stage_values(label)[2], "model": client.model,
                           "record_count": 571, "generation_workers": args.workers,
                           "reused_identical_inputs_this_invocation": reused_count,
                           "api_questions_this_invocation": len(calls),
                           "total_reused_identical_inputs": sum(bool(row.get("reused_from")) for row in rows),
                           "total_api_generated_questions": sum(not row.get("reused_from") for row in rows),
                           "run_signature": manifest["run_signature"]},
            "answer_metrics": compute_answer_metrics(rows), "retrieval_metrics": compute_retrieval_metrics(rows),
        }
        save_json(args.output_dir / f"{label}_summary.json", summary)
        manifest.setdefault("generation", {})[label] = {"count": 571, "sha256": sha256(path),
                                                       "answer_metrics": summary["answer_metrics"]}
        save_json(args.output_dir / "manifest.json", manifest)
    summarize(args.output_dir, manifest)


def follow_generation(records: list[dict[str, Any]], allowed: set[str], args: argparse.Namespace,
                      manifest: dict[str, Any]) -> None:
    """Consume one retrieval JSONL as it grows; finalize only at exactly 571."""
    if len(args.configs) != 1:
        raise RuntimeError("--phase follow requires exactly one --configs label")
    label = args.configs[0]
    set_stage_config(*stage_values(label))
    client = OpenAICompatibleClient(model="gpt-5-mini")
    if not client.is_configured():
        raise RuntimeError("OPENAI_API_KEY is not configured")
    path = args.output_dir / f"{label}.jsonl"
    completed = index_rows(path, allowed, complete=True)
    queued = set(completed)
    cache = {}
    for previous_label in LEGACY:
        for row in index_rows(args.output_dir / f"{previous_label}.jsonl", allowed, complete=True).values():
            key = row.get("generation_input_sha256")
            if key and row.get("model") == client.model and row.get("run_signature") == manifest["run_signature"]:
                cache.setdefault(key, row)
    source_path = args.output_dir / f"R{label[1]}_retrieval.jsonl"
    futures = {}
    failure = None
    last_read = 0.0
    source = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        while len(completed) < 571:
            if time.time() - last_read >= 2:
                source = {}
                if source_path.exists():
                    # Ignore a last line currently being appended by retrieval.
                    for line in source_path.read_text(encoding="utf-8").splitlines(keepends=True):
                        if line.endswith("\n") and line.strip():
                            row = json.loads(line)
                            rid = row_id(row)
                            if rid not in allowed or rid in source:
                                raise RuntimeError("Unexpected or duplicate source ID in followed retrieval")
                            source[rid] = row
                last_read = time.time()
            if failure is None:
                for record in records:
                    rid = str(record["id"])
                    if rid in queued or rid not in source:
                        continue
                    row = deepcopy(source[rid])
                    blocks = build_context_blocks(row)
                    if first_evidence_pack_doc(blocks) is None:
                        row["generation_input_sha256"] = json_digest({"model": client.model,
                            "prompt": build_prompt(row["record"], blocks), "run_signature": manifest["run_signature"]})
                    previous = cache.get(row.get("generation_input_sha256"))
                    if previous is not None:
                        row.update(prediction=previous["prediction"],
                            normalized_prediction=normalize_prediction(previous["prediction"], row["record"]),
                            inferred_evidence_state=None, prediction_mode=previous["prediction_mode"],
                            stage_config=label, model=client.model, run_signature=manifest["run_signature"],
                            reused_from={"config": previous["stage_config"], "id": row_id(previous),
                                         "reason": "identical model and generation input hash"})
                        append_row(path, row)
                        completed[rid] = row
                        queued.add(rid)
                        continue
                    futures[executor.submit(generate_prediction, client, row)] = row
                    queued.add(rid)
                    if len(futures) >= args.workers * 2:
                        break
            if not futures:
                if failure:
                    break
                time.sleep(1)
                continue
            done, _ = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
            for future in done:
                row = futures.pop(future)
                try:
                    prediction, normalized, state, mode = future.result()
                except CancelledError:
                    continue
                except Exception as exc:
                    if failure is None:
                        failure = {"config": label, "error_type": type(exc).__name__,
                            "http_status": getattr(getattr(exc, "response", None), "status_code", None),
                            "failed_id": row_id(row)}
                        for pending in futures:
                            pending.cancel()
                    continue
                row.update(prediction=prediction, normalized_prediction=normalized,
                           inferred_evidence_state=state, prediction_mode=mode, stage_config=label,
                           model=client.model, run_signature=manifest["run_signature"])
                append_row(path, row)
                completed[row_id(row)] = row
                if len(completed) % 10 == 0 or len(completed) == 571:
                    log(f"[FOLLOW_GENERATION] {label} {len(completed)}/571 retrieval_ready={len(source)} workers={args.workers}")
        if failure:
            failure["completed"] = len(completed)
            save_json(args.output_dir / "generation_failure.json", failure)
            raise RuntimeError(f"Generation stopped: {failure['error_type']}, HTTP {failure['http_status']}")
    generate(records, allowed, args, manifest)


def follow_direct_configs(records: list[dict[str, Any]], allowed: set[str], args: argparse.Namespace,
                          manifest: dict[str, Any]) -> None:
    """Queue immutable direct prompts for multiple configs while retrieval grows.

    Only the main thread reads/changes STAGE_CONFIG. Workers receive the exact
    already-rendered prompt and never call a stage-dependent helper.
    """
    client = OpenAICompatibleClient(model="gpt-5-mini")
    if not client.is_configured():
        raise RuntimeError("OPENAI_API_KEY is not configured")
    completed = {label: index_rows(args.output_dir / f"{label}.jsonl", allowed, complete=True) for label in args.configs}
    queued = {label: set(rows) for label, rows in completed.items()}
    cache = {}
    for label in LEGACY:
        for row in index_rows(args.output_dir / f"{label}.jsonl", allowed, complete=True).values():
            key = row.get("generation_input_sha256")
            if key and row.get("model") == client.model and row.get("run_signature") == manifest["run_signature"]:
                cache.setdefault(key, row)
    source, sizes, futures, key_to_future = {}, {}, {}, {}
    last_read = 0.0
    failure = None
    target = 571 * len(args.configs)

    def save_prediction(row: dict[str, Any], prediction: str, origin: dict[str, Any] | None) -> None:
        label, rid = row["stage_config"], row_id(row)
        row.update(prediction=prediction, normalized_prediction=normalize_prediction(prediction, row["record"]),
                   inferred_evidence_state=None, prediction_mode="direct_prompt", model=client.model,
                   run_signature=manifest["run_signature"])
        if origin is not None:
            row["reused_from"] = {"config": origin["stage_config"], "id": row_id(origin),
                                  "reason": "identical model and generation input hash"}
        append_row(args.output_dir / f"{label}.jsonl", row)
        completed[label][rid] = row
        cache.setdefault(row["generation_input_sha256"], row)
        total = sum(len(rows) for rows in completed.values())
        if len(completed[label]) % 50 == 0 or len(completed[label]) == 571:
            log(f"[DIRECT_QUEUE] {label} {len(completed[label])}/571 total={total}/{target} workers={args.workers}")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        while sum(len(rows) for rows in completed.values()) < target:
            if time.time() - last_read >= 2:
                for state in {int(label[1]) for label in args.configs}:
                    path = args.output_dir / f"R{state}_retrieval.jsonl"
                    size = path.stat().st_size if path.exists() else 0
                    if size == sizes.get(state):
                        continue
                    rows = {}
                    if path.exists():
                        for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
                            if line.endswith("\n") and line.strip():
                                row = json.loads(line)
                                rid = row_id(row)
                                if rid not in allowed or rid in rows:
                                    raise RuntimeError("Unexpected or duplicate source ID in followed retrieval")
                                rows[rid] = row
                    source[state], sizes[state] = rows, size
                last_read = time.time()
            if failure is None:
                for label in args.configs:
                    if len(futures) >= args.workers * 2:
                        break
                    set_stage_config(*stage_values(label))
                    for record in records:
                        rid = str(record["id"])
                        if rid in queued[label] or rid not in source.get(int(label[1]), {}):
                            continue
                        row = deepcopy(source[int(label[1])][rid])
                        blocks = build_context_blocks(row)
                        if first_evidence_pack_doc(blocks) is not None:
                            raise RuntimeError("Multi-config direct queue requires the Graph RAG direct-prompt branch")
                        prompt = build_prompt(row["record"], blocks)
                        key = json_digest({"model": client.model, "prompt": prompt,
                                           "run_signature": manifest["run_signature"]})
                        row.update(stage_config=label, generation_input_sha256=key)
                        queued[label].add(rid)
                        if key in cache:
                            save_prediction(row, cache[key]["prediction"], cache[key])
                        elif key in key_to_future:
                            futures[key_to_future[key]][1].append(row)
                        else:
                            future = executor.submit(client.generate, prompt)
                            futures[future] = (key, [row])
                            key_to_future[key] = future
                        if len(futures) >= args.workers * 2:
                            break
            if not futures:
                if failure:
                    break
                time.sleep(1)
                continue
            done, _ = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
            for future in done:
                key, rows = futures.pop(future)
                key_to_future.pop(key, None)
                try:
                    prediction = future.result()
                except CancelledError:
                    continue
                except Exception as exc:
                    if failure is None:
                        failure = {"error_type": type(exc).__name__, "http_status":
                            getattr(getattr(exc, "response", None), "status_code", None),
                            "config": rows[0]["stage_config"], "failed_id": row_id(rows[0])}
                        for pending in futures:
                            pending.cancel()
                    continue
                first = rows[0]
                save_prediction(first, prediction, None)
                for row in rows[1:]:
                    save_prediction(row, prediction, first)
        if failure:
            failure["completed_by_config"] = {label: len(rows) for label, rows in completed.items()}
            save_json(args.output_dir / "generation_failure.json", failure)
            raise RuntimeError(f"Generation stopped: {failure['error_type']}, HTTP {failure['http_status']}")
    generate(records, allowed, args, manifest)


def summarize(output: Path, manifest: dict[str, Any]) -> None:
    report = {}
    for label in LEGACY:
        path = output / f"{label}_summary.json"
        if path.exists():
            report[label] = json.loads(path.read_text())["answer_metrics"]
            rows = read_jsonl(output / f"{label}.jsonl")
            if len(rows) != 571 or len({row_id(row) for row in rows}) != 571:
                raise RuntimeError(f"Cannot finalize {label}: expected 571 unique results")
            gl = [gold_label(row["record"]) for row in rows]
            pl = [row["normalized_prediction"] for row in rows]
            correct = sum(g == p for g, p in zip(gl, pl))
            report[label].update(accuracy=correct / 571, macro_f1=macro_f1_from_labels(gl, pl), correct_count=correct)
    if len(report) == 8:
        manifest["status"] = "complete"
        accuracy = {label: metrics["accuracy"] for label, metrics in report.items()}
        macro = {label: metrics["macro_f1"] for label, metrics in report.items()}
        save_json(output / "figure6_metrics.json", {"count_per_config": 571, "metrics": report,
                   "conditional_differences": {module: [
                       {"off": label, "on": label[:idx] + "1" + label[idx + 1:],
                        "accuracy_difference": accuracy[label[:idx] + "1" + label[idx + 1:]] - accuracy[label],
                        "macro_f1_difference": macro[label[:idx] + "1" + label[idx + 1:]] - macro[label]}
                       for label in LEGACY if label[idx] == "0"] for module, idx in (("R", 1), ("C", 3), ("G", 5))}})
        plot_figure6(output, report)
    save_json(output / "manifest.json", manifest)


def format_effect(delta: float) -> tuple[str, str]:
    value = Decimal(str(delta * 100)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    if value == 0:
        return "•", "0.0"
    return ("↑" if value > 0 else "↓"), f"{value:+.1f}"


def plot_figure6(output: Path, report: dict[str, Any]) -> None:
    """Portable rendering of the same twelve conditional stage effects."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.6))
    cells = []
    for ax, (band, idx) in zip(axes, (("Retrieval", 1), ("Context", 3), ("Generation", 5))):
        off_labels = [key for key in LEGACY if key[idx] == "0"]
        values = []
        for k, off in enumerate(off_labels):
            on = off[:idx] + "1" + off[idx+1:]
            da = report[on]["accuracy"] - report[off]["accuracy"]
            df = report[on]["macro_f1"] - report[off]["macro_f1"]
            values.append(da * 100)
            cells.append({"band": band.lower(), "off": off, "on": on,
                          "accuracy_raw_difference": da, "macro_f1_raw_difference": df})
            ax.text(k % 2, k // 2, f"{off}→{on}\nAcc {da*100:+.1f} pp\nF1 {df*100:+.1f} pp",
                    ha="center", va="center", fontsize=7)
        ax.imshow([values[:2], values[2:]], cmap="RdYlGn", vmin=-10, vmax=10)
        ax.set_title(band)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    for ext in ("pdf", "png", "svg"):
        fig.savefig(output / f"figure6_stage_ablation_571.{ext}", bbox_inches="tight")
    plt.close(fig)
    save_json(output / "figure6_effect_cells.json", {"count_per_config": 571, "cells": cells})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("retrieve", "generate", "follow", "follow-direct", "all", "summarize"), default="retrieve")
    parser.add_argument("--knowledge-dir", type=Path, default=ROOT / "dataset/rag/knowledge_sources")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--configs", nargs="+", choices=list(LEGACY), default=list(LEGACY))
    parser.add_argument("--frozen-input-dir", type=Path, help="Use released R0/R1 context inputs instead of rerunning retrieval")
    args = parser.parse_args()
    env_path = ROOT / ".env"
    for line in env_path.read_text().splitlines() if env_path.exists() else []:
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    records = read_jsonl(args.dataset)
    ids = [str(record.get("id") or "") for record in records]
    if len(records) != 571 or len(set(ids)) != 571 or not all(ids):
        raise RuntimeError("Input must contain exactly the 571 unique formal benchmark IDs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configuration = {"dataset_path": str(args.dataset), "dataset_sha256": sha256(args.dataset),
                     "ids_sha256": json_digest(sorted(ids)), "count": 571, "view": "multi_cancer",
                     "knowledge_source": "evidence_graph_complete", "framework": "graph_rag",
                     "top_k": 8, "rag_profile": "tuned", "model": "gpt-5-mini",
                     "gold_metadata_policy": "unchanged existing configuration",
                     "library_sha256": sha256(ROOT / "scripts/rag_eval_lib.py"),
                     "generation_runner_sha256": sha256(ROOT / "scripts/run_rag_evaluation.py")}
    if args.frozen_input_dir:
        configuration["frozen_context_sha256"] = {f"R{r}": sha256(args.frozen_input_dir / f"R{r}_retrieval.jsonl") for r in (0, 1)}
    signature = json_digest(configuration)
    manifest_path = args.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "configuration": configuration, "run_signature": signature, "status": "initialized"}
    if manifest["run_signature"] != signature:
        raise RuntimeError("Existing run has a different dataset/config/library; use another output directory")
    set_rag_profile("tuned")
    save_json(manifest_path, manifest)
    if args.phase in ("retrieve", "all"):
        prepare(records, set(ids), args, manifest)
    if args.phase in ("generate", "all"):
        generate(records, set(ids), args, manifest)
    if args.phase == "follow":
        follow_generation(records, set(ids), args, manifest)
    if args.phase == "follow-direct":
        follow_direct_configs(records, set(ids), args, manifest)
    if args.phase == "summarize":
        summarize(args.output_dir, manifest)


if __name__ == "__main__":
    main()

"""Validate the complete frozen benchmark using only the Python standard library."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "dataset/evaluated/external_reasoning_benchmark_formal_v3_571.jsonl"
EXPECTED_TASKS = {"cross_source_corroboration_status":100,"cross_source_support_bundle":81,
    "evidence_signal_interpretation":85,"hierarchical_evidence_aggregation":86,
    "mechanism_bioasq_factoid":7,"mechanism_bioasq_list":7,"mechanism_bioasq_yesno":5,
    "mechanism_trace_intermediate_gene":100,"precision_oncology_evidence":100}
EXPECTED_REASONING = {"cross_source_corroboration":100,"multi_evidence_logical_synthesis":81,
    "evidence_signal_interpretation":97,"hierarchical_evidence_aggregation":93,
    "multi_hop_evidence_reasoning":100,"precision_oncology_evidence_judgment":100}
REQUIRED = {"id","source","task_type","question","answer","answer_type","oncology_focus",
    "evidence_text","entities","provenance","metadata","gold_standard","is_machine_evaluable","benchmark_annotation"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT)
    parser.add_argument("--reference", type=Path, help="Compare rebuilt records by ID and full content against this frozen benchmark.")
    args = parser.parse_args()
    raw = args.benchmark.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    assert len(rows) == 571, "Expected all 571 records"
    assert len({row["id"] for row in rows}) == 571, "Duplicate IDs"
    assert all(REQUIRED <= row.keys() for row in rows), "Missing full record fields"
    assert all(row["is_machine_evaluable"] and row["gold_standard"].get("evaluable") for row in rows)
    assert all(row["question"] and row["answer"] and row["gold_standard"].get("canonical_answer") for row in rows)
    tasks = Counter(row["task_type"] for row in rows)
    reasoning = Counter(row["benchmark_annotation"]["reasoning_type"] for row in rows)
    assert dict(tasks) == EXPECTED_TASKS, "Task-type count mismatch"
    assert dict(reasoning) == EXPECTED_REASONING, "Six-family count mismatch"
    assert len({row["metadata"]["priority_cancer_family"] for row in rows}) == 11
    assert b"/Users/" not in raw and b"/Volumes/" not in raw, "Local absolute path found"
    manifest = json.loads((ROOT / "dataset/evaluated/benchmark_manifest.json").read_text())
    if args.benchmark.resolve() == DEFAULT.resolve():
        assert hashlib.sha256(raw).hexdigest() == manifest["benchmark"]["sha256"], "Frozen benchmark bytes changed"
    result = {"records":571,"unique_ids":571,"reasoning_families":6,"cancer_families":11,
              "sha256":hashlib.sha256(raw).hexdigest(),"task_counts":dict(tasks),"reasoning_counts":dict(reasoning)}
    if args.reference:
        reference = {row["id"]:row for line in args.reference.read_text().splitlines() if (row:=json.loads(line))}
        current = {row["id"]:row for row in rows}
        missing = sorted(reference.keys()-current.keys())
        extra = sorted(current.keys()-reference.keys())
        changed = sorted(key for key in reference.keys() & current.keys() if reference[key] != current[key])
        result["comparison"] = {"missing_ids":missing,"extra_ids":extra,"changed_records":changed}
        print(json.dumps(result,indent=2))
        if missing or extra or changed:
            raise SystemExit(1)
    else:
        print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()

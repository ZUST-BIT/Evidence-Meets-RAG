"""Export publication inputs from an author's archive, excluding model outputs.

This maintainer tool takes a source checkout explicitly. It exports only record,
query, and retrieved evidence, with a per-file checksum manifest. It never copies
predictions, normalized predictions, judged labels, credentials, or result metrics.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from paper_paths import FRAMEWORKS, SOURCES, suffix


def fingerprints(row: dict, profiles: list[tuple[str, str, str]]) -> dict:
    """Hash every consumed generation/judging input, with no model invocation."""
    from rag_eval_lib import (build_prompt, build_evidence_state_prompt, build_answer_from_state_prompt,
        first_evidence_pack_doc, infer_state_demand_profile, inferred_evidence_state,
        map_answer_from_evidence_state, set_rag_profile, set_stage_config)
    from run_rag_evaluation import build_context_blocks, json_ready
    from run_faithfulness_judge import build_prompt as judge_prompt
    payload = {}
    for profile in profiles:
        set_rag_profile("tuned"); set_stage_config(*profile)
        current = deepcopy(row)
        current.setdefault("prediction", None)
        contexts = build_context_blocks(current)
        values = {"direct_prompt": build_prompt(current["record"], contexts), "judge_prompt": judge_prompt(current)}
        if first_evidence_pack_doc(contexts) is not None:
            demand = infer_state_demand_profile(current["record"], contexts)
            values["demand_profile"] = json_ready(demand)
            values["state_prompt"] = build_evidence_state_prompt(current["record"], contexts, demand)
            state = inferred_evidence_state(current["record"], contexts, "{}")
            values["state_fixture"] = json_ready(state)
            values["mapped_answer"] = map_answer_from_evidence_state(current["record"], state, demand)
            values["answer_from_state_prompt"] = build_answer_from_state_prompt(current["record"], state, contexts, demand)
        payload["/".join(profile)] = hashlib.sha256(json.dumps(values, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compact-unused-neighbors", action="store_true", help="Drop only retrieved_docs[].metadata.neighbor_names and verify consumed prompts")
    args = parser.parse_args()
    archive = args.archive_root.resolve()
    plans = {}
    for framework in FRAMEWORKS:
        for source in SOURCES:
            name = f"multi_cancer__{source}__{framework}__{suffix(framework, source)}.jsonl"
            path = archive / "dataset/rag/results" / name
            if source == "evidence_graph_complete" and framework == "naive_rag":
                path = archive / "dataset/rag/results/fixed_context_component_ablation_571_vanilla__full.jsonl"
            if source == "evidence_graph_complete" and framework == "graph_rag":
                path = archive / "dataset/rag/results/unified_571/stage/R1C1G1.jsonl"
            plans[f"main/{framework}__{source}.jsonl"] = path
    for r in (0, 1):
        plans[f"stage/R{r}_retrieval.jsonl"] = archive / f"dataset/rag/results/unified_571/stage/R{r}_retrieval.jsonl"
    for source in ("raw_observation_view", "evidence_graph_complete"):
        plans[f"long_context/{source}.jsonl"] = archive / f"dataset/rag/results/multi_cancer__{source}__naive_rag__longcontext_k100.jsonl"
    plans["source_forms/single_source_form_retrieval_571.jsonl"] = archive / "dataset/rag/analysis/evidence_meets_rag_single_source_forms/single_source_form_retrieval_571.jsonl"
    plans["dual_source/tcga_pubmed_controlled_retrieval.jsonl"] = archive / "dataset/rag/analysis/evidence_meets_rag_tcga_pubmed/tcga_pubmed_controlled_retrieval.jsonl"
    manifest = {"format": 1, "contains_predictions": False,
                "archived_provenance_literal": True, "prompt_text_unchanged": True, "files": {}}
    manifest["removed_unused_field"] = "retrieved_docs[].metadata.neighbor_names" if args.compact_unused_neighbors else None
    for name, source in plans.items():
        rows = []
        removed = 0
        verified = 0
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                original = json.loads(line)
                item = {k: original[k] for k in ("record_id", "record", "query_text", "retrieved_docs", "source", "form", "knowledge_source", "family", "task_type", "variant", "condition", "setting") if k in original}
                if args.compact_unused_neighbors:
                    previous = item
                    item = deepcopy(item)
                    for doc in item["retrieved_docs"]:
                        if "neighbor_names" in (doc.get("metadata") or {}):
                            del doc["metadata"]["neighbor_names"]
                            removed += 1
                    profiles = [("full", "structured", "grounded")]
                    if name.startswith("stage/"):
                        profiles = [("full" if "R1_" in name else "base", c, g)
                                    for c in ("flat", "structured") for g in ("plain", "grounded")]
                    if fingerprints(previous, profiles) != fingerprints(item, profiles):
                        raise ValueError(f"Compaction changed a consumed generation/judge input in {name}")
                    verified += len(profiles)
                    if name == "main/naive_rag__evidence_graph_complete.jsonl":
                        from run_fixed_context_component_ablation_571 import apply_condition, CONDITIONS
                        for condition in CONDITIONS:
                            before = dict(previous, retrieved_docs=apply_condition(previous["retrieved_docs"], condition))
                            after = dict(item, retrieved_docs=apply_condition(item["retrieved_docs"], condition))
                            if fingerprints(before, profiles) != fingerprints(after, profiles):
                                raise ValueError(f"Compaction changed {condition} ablation input")
                            verified += 1
                    if name.startswith("dual_source/"):
                        from run_tcga_pubmed_complementarity import GUIDANCE
                        before, after = deepcopy(previous), deepcopy(item)
                        for value in (before, after): value["record"]["question"] += GUIDANCE
                        if fingerprints(before, profiles) != fingerprints(after, profiles):
                            raise ValueError("Compaction changed source-role-aware prompt")
                        verified += 1
                # Archived provenance strings are descriptive scientific input,
                # never paths opened by a runner. Preserve the published prompt.
                def check_fields(value):
                    if isinstance(value, dict):
                        for key, val in value.items():
                            if key.lower() in {"api_key", "authorization", "access_token", "refresh_token"}:
                                raise ValueError(f"Unexpected authentication field in {name}")
                            check_fields(val)
                    elif isinstance(value, list):
                        for val in value: check_fields(val)
                check_fields(item)
                serialized = json.dumps(item, ensure_ascii=False)
                rows.append(serialized)
        target = args.output_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(rows) + "\n", encoding="utf-8")
        manifest["files"][name] = {"rows": len(rows), "bytes": target.stat().st_size,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "removed_neighbor_fields": removed,
            "verified_prompt_profile_cases": verified}
        print(f"[export] {name}: rows={len(rows)} removed={removed} verified={verified}", flush=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"files": len(plans), "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()

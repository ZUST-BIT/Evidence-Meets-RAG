"""Template-clustered bootstrap for the main knowledge-shape comparison.

The 571-record benchmark contains 129 entity-normalized question templates,
the largest holding 100 records. Record-level bootstrap treats records as
independent, which template structure could violate. This analysis re-estimates
the main Evidence-vs-Document and Evidence-vs-Semantic Accuracy deltas with:

1. cluster bootstrap: templates resampled with replacement (all records of a
   sampled template enter the replicate),
2. leave-out-largest sensitivity: deltas recomputed after dropping the single
   largest template family.

No LLM calls: operates on the saved prediction files used for Table 2.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_benchmark_validity import normalized_template  # noqa: E402
from rag_eval_lib import gold_label  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "dataset" / "rag" / "analysis" / "ecorag_template_cluster_bootstrap"
ITERATIONS = 10_000
SEED = 20260820

RESULTS = {
    "vanilla": {
        "label": "Vanilla RAG",
        "evidence": ROOT / "dataset/rag/results/fixed_context_component_ablation_571_vanilla__full.jsonl",
        "doc": ROOT / "dataset/rag/results/multi_cancer__raw_observation_view__naive_rag__formalv3_571_classic4_raw_vs_graph.jsonl",
        "semantic": ROOT / "dataset/rag/results/multi_cancer__semantic_kg__naive_rag__formalv3_571_classic4_semantic_kg.jsonl",
    },
    "graph": {
        "label": "Graph RAG",
        "evidence": ROOT / "dataset/rag/results/unified_571/stage/R1C1G1.jsonl",
        "doc": ROOT / "dataset/rag/results/multi_cancer__raw_observation_view__graph_rag__formalv3_571_classic4_raw_vs_graph.jsonl",
        "semantic": ROOT / "dataset/rag/results/multi_cancer__semantic_kg__graph_rag__formalv3_571_classic4_semantic_kg.jsonl",
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def correct_map(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in read_jsonl(path):
        record = row["record"]
        rid = str(record["id"])
        out[rid] = int(gold_label(record) == str(row.get("normalized_prediction") or ""))
    return out


def cluster_bootstrap(
    clusters: list[list[str]],
    correct_a: dict[str, int],
    correct_b: dict[str, int],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    deltas = []
    n_clusters = len(clusters)
    for _ in range(iterations):
        num = 0
        den = 0
        for _ in range(n_clusters):
            cluster = clusters[rng.randrange(n_clusters)]
            for rid in cluster:
                num += correct_a[rid] - correct_b[rid]
                den += 1
        deltas.append(num / den if den else 0.0)
    deltas.sort()

    def pct(q: float) -> float:
        idx = min(len(deltas) - 1, max(0, int(q * (len(deltas) - 1))))
        return deltas[idx]

    return {
        "mean_delta": round(sum(deltas) / len(deltas), 4),
        "ci95": [round(pct(0.025), 4), round(pct(0.975), 4)],
    }


def point_delta(ids: list[str], correct_a: dict[str, int], correct_b: dict[str, int]) -> float:
    return round(sum(correct_a[r] - correct_b[r] for r in ids) / len(ids), 4)


def main() -> None:
    benchmark = read_jsonl(ROOT / "dataset/evaluated/external_reasoning_benchmark_formal_v3_571.jsonl")
    template_of = {str(r["id"]): normalized_template(r) for r in benchmark}

    clusters_map: dict[str, list[str]] = defaultdict(list)
    for rid, tpl in template_of.items():
        clusters_map[tpl].append(rid)
    clusters = list(clusters_map.values())
    largest_tpl = max(clusters_map, key=lambda t: len(clusters_map[t]))
    largest_ids = set(clusters_map[largest_tpl])
    print(f"templates={len(clusters)} largest={len(largest_ids)} records={len(template_of)}")

    report: dict[str, Any] = {
        "design": {
            "iterations": ITERATIONS,
            "seed": SEED,
            "unit": "normalized question template (cluster bootstrap)",
            "templates": len(clusters),
            "largest_template_records": len(largest_ids),
        },
        "frameworks": {},
    }

    for key, spec in RESULTS.items():
        ev = correct_map(spec["evidence"])
        doc = correct_map(spec["doc"])
        sem = correct_map(spec["semantic"])
        ids = sorted(set(ev) & set(doc) & set(sem) & set(template_of))
        id_clusters = [
            [rid for rid in cluster if rid in set(ids)]
            for cluster in clusters
        ]
        id_clusters = [c for c in id_clusters if c]
        drop_ids = [rid for rid in ids if rid not in largest_ids]

        entry: dict[str, Any] = {"label": spec["label"], "n_records": len(ids)}
        for name, base in (("evidence_vs_doc", doc), ("evidence_vs_semantic", sem)):
            entry[name] = {
                "point_delta_full": point_delta(ids, ev, base),
                "cluster_bootstrap": cluster_bootstrap(id_clusters, ev, base, ITERATIONS, SEED),
                "point_delta_drop_largest_template": point_delta(drop_ids, ev, base),
                "n_after_drop": len(drop_ids),
            }
        report["frameworks"][key] = entry
        print(json.dumps(entry, indent=2))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "template_cluster_bootstrap_summary.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

from rag_eval_lib import read_jsonl


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "dataset/rag/results/tcga_pubmed_complementarity.jsonl"
OUTPUT_DIR = ROOT / "dataset/rag/analysis/evidence_meets_rag_tcga_pubmed"
SOURCES = {"tcga": "without_tcga", "pubmed": "without_pubmed"}


def exact_mcnemar(wins: int, losses: int) -> float:
    n = wins + losses
    if not n:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / (2**n))


def bootstrap(values: list[float], seed: str) -> list[float]:
    rng = random.Random(seed)
    samples = [sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(10000)]
    samples.sort()
    return [round(samples[250], 4), round(samples[9749], 4)]


def contrast(rows: list[dict[str, Any]], source: str, task: str | None) -> dict[str, Any]:
    variant = SOURCES[source]
    full = {row["record_id"]: row for row in rows if row["variant"] == "full" and (task is None or row["task_type"] == task)}
    ablated = {row["record_id"]: row for row in rows if row["variant"] == variant and row["record_id"] in full}
    pairs = [(full[rid], ablated[rid]) for rid in sorted(full)]
    deltas = [float(a["correct"]) - float(b["correct"]) for a, b in pairs]
    wins = sum(delta > 0 for delta in deltas)
    losses = sum(delta < 0 for delta in deltas)
    retrieval = {}
    for metric in ("component_recall", "complete_bundle", "joint_scope_precision", "support_mrr"):
        metric_deltas = [float(a["retrieval_audit"][metric]) - float(b["retrieval_audit"][metric]) for a, b in pairs]
        retrieval[metric] = {
            "delta": round(sum(metric_deltas) / len(metric_deltas), 4),
            "ci95": bootstrap(metric_deltas, f"{source}:{task}:{metric}"),
        }
    return {
        "source": source,
        "scope": task or "all",
        "count": len(pairs),
        "full_accuracy": round(sum(a["correct"] for a, _b in pairs) / len(pairs), 4),
        "ablated_accuracy": round(sum(b["correct"] for _a, b in pairs) / len(pairs), 4),
        "accuracy_delta": round(sum(deltas) / len(deltas), 4),
        "accuracy_ci95": bootstrap(deltas, f"{source}:{task}:accuracy"),
        "wins": wins,
        "ties": len(pairs) - wins - losses,
        "losses": losses,
        "mcnemar_p": round(exact_mcnemar(wins, losses), 6),
        "retrieval": retrieval,
    }


def main() -> None:
    rows = read_jsonl(INPUT)
    results = []
    for scope in (None, "precision_oncology_evidence"):
        for source in SOURCES:
            results.append(contrast(rows, source, scope))
    payload = {
        "design": {
            "status": "exploratory task-specific complementarity diagnostic",
            "top_k": 8,
            "fixed_surviving_source_quota": 3,
            "source_aware_generation": True,
            "bootstrap_iterations": 10000,
        },
        "contrasts": results,
    }
    (OUTPUT_DIR / "tcga_pubmed_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Evidence Meets RAG: TCGA-PubMed Complementarity",
        "",
        "## Paired Answer Effects",
        "",
        "| Scope | Source removed | N | Accuracy Full/Ablated | Contribution (95% CI) | McNemar p | W/T/L | Component Recall | Complete Bundle |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            f"| {row['scope']} | {row['source']} | {row['count']} | {row['full_accuracy']:.4f}/{row['ablated_accuracy']:.4f} | "
            f"{row['accuracy_delta']:+.4f} {row['accuracy_ci95']} | {row['mcnemar_p']:.6f} | {row['wins']}/{row['ties']}/{row['losses']} | "
            f"{row['retrieval']['component_recall']['delta']:+.4f} | {row['retrieval']['complete_bundle']['delta']:+.4f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Interpret the mixed-task panel and precision-oncology subset separately using their current source-removal differences, intervals, and exact tests. These are exploratory task-specific diagnostics rather than population-level causal estimates.",
        "",
    ])
    report = "\n".join(lines)
    (OUTPUT_DIR / "tcga_pubmed_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

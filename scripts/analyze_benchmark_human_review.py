#!/usr/bin/env python3
"""Analyze two independent benchmark review forms."""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "dataset/evaluated/benchmark_validity_audit"
FORM1 = AUDIT_DIR / "blinded_review_form_120_1.csv"
FORM2 = AUDIT_DIR / "blinded_review_form_120_2.csv"
PANEL = AUDIT_DIR / "blinded_review_panel_120.jsonl"
OUT_JSON = AUDIT_DIR / "human_review_metrics.json"
OUT_MD = AUDIT_DIR / "human_review_metrics.md"
OUT_DISAGREE = AUDIT_DIR / "human_review_disagreements.csv"

CRITERIA = [
    "scientifically_well_formed_0_1",
    "gold_supported_0_1",
    "answer_identifiable_0_1",
    "requires_evidence_integration_0_1",
    "representation_neutral_0_1",
    "leakage_risk_0_1",
]
LABELS = {
    "scientifically_well_formed_0_1": "Scientific well-formedness",
    "gold_supported_0_1": "Gold support",
    "answer_identifiable_0_1": "Answer identifiability",
    "requires_evidence_integration_0_1": "Evidence-integration demand",
    "representation_neutral_0_1": "Representation neutrality",
    "leakage_risk_0_1": "Leakage risk",
}
TARGETS = {
    "scientifically_well_formed_0_1": (">=", 0.95),
    "gold_supported_0_1": (">=", 0.95),
    "answer_identifiable_0_1": (">=", 0.90),
    "requires_evidence_integration_0_1": (">=", 0.80),
    "representation_neutral_0_1": (">=", 0.90),
    "leakage_risk_0_1": ("<=", 0.05),
}


def read_form(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        rid = row["review_id"].strip()
        if not rid or rid in result:
            raise ValueError(f"Missing or duplicate review_id in {path}: {rid!r}")
        result[rid] = row
    return result


def kappa(a: list[int], b: list[int]) -> float | None:
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa = sum(a) / n
    pb = sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if math.isclose(1 - pe, 0.0) else (po - pe) / (1 - pe)


def wilson(successes: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [center - half, center + half]


def bootstrap_kappa(a: list[int], b: list[int], reps: int = 10000) -> list[float | None]:
    rng = random.Random(20260701)
    values = []
    n = len(a)
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        value = kappa([a[i] for i in idx], [b[i] for i in idx])
        if value is not None:
            values.append(value)
    if not values:
        return [None, None]
    values.sort()
    return [values[int(0.025 * len(values))], values[min(len(values) - 1, int(0.975 * len(values)))]]


def fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.3f}"


def main() -> None:
    r1, r2 = read_form(FORM1), read_form(FORM2)
    expected = {f"BR-{i:03d}" for i in range(1, 121)}
    common = sorted(set(r1) & set(r2))
    panel = {row["review_id"]: row for row in (json.loads(line) for line in PANEL.read_text().splitlines() if line.strip())}

    invalid = []
    for reviewer, rows in (("reviewer1", r1), ("reviewer2", r2)):
        for rid, row in rows.items():
            for criterion in CRITERIA:
                if row.get(criterion, "").strip() not in {"0", "1"}:
                    invalid.append([reviewer, rid, criterion, row.get(criterion, "")])
            if row.get("confidence_1_5", "").strip() not in {"1", "2", "3", "4", "5"}:
                invalid.append([reviewer, rid, "confidence_1_5", row.get("confidence_1_5", "")])
    if invalid:
        raise ValueError(f"Invalid values: {invalid[:10]}")

    metrics = {}
    disagreements = []
    for criterion in CRITERIA:
        a = [int(r1[rid][criterion]) for rid in common]
        b = [int(r2[rid][criterion]) for rid in common]
        n = len(common)
        s1, s2 = sum(a), sum(b)
        agreement_n = sum(x == y for x, y in zip(a, b))
        kap = kappa(a, b)
        direction, threshold = TARGETS[criterion]
        pooled_rate = (s1 + s2) / (2 * n)
        target_met = pooled_rate >= threshold if direction == ">=" else pooled_rate <= threshold
        metrics[criterion] = {
            "label": LABELS[criterion],
            "n_paired": n,
            "reviewer1_rate": s1 / n,
            "reviewer1_wilson95": wilson(s1, n),
            "reviewer2_rate": s2 / n,
            "reviewer2_wilson95": wilson(s2, n),
            "pooled_rating_rate": pooled_rate,
            "both_positive_rate": sum(x == 1 and y == 1 for x, y in zip(a, b)) / n,
            "both_negative_rate": sum(x == 0 and y == 0 for x, y in zip(a, b)) / n,
            "raw_agreement": agreement_n / n,
            "cohen_kappa": kap,
            "cohen_kappa_bootstrap95": bootstrap_kappa(a, b),
            "target": f"{direction}{threshold:.0%}",
            "target_met_on_pooled_rate": target_met,
            "disagreements": n - agreement_n,
        }
        for rid, x, y in zip(common, a, b):
            if x != y:
                info = panel.get(rid, {})
                disagreements.append({
                    "review_id": rid,
                    "criterion": criterion,
                    "reviewer1": x,
                    "reviewer2": y,
                    "task_type": (info.get("stratum") or {}).get("task_type", ""),
                    "cancer_family": (info.get("stratum") or {}).get("cancer_family", ""),
                    "question": info.get("question", ""),
                    "reviewer1_notes": r1[rid].get("notes", ""),
                    "reviewer2_notes": r2[rid].get("notes", ""),
                })

    confidence1 = [int(r1[rid]["confidence_1_5"]) for rid in common]
    confidence2 = [int(r2[rid]["confidence_1_5"]) for rid in common]
    task_disagreements = defaultdict(Counter)
    for item in disagreements:
        task_disagreements[item["task_type"]][item["criterion"]] += 1

    report = {
        "input": {"reviewer1": FORM1.name, "reviewer2": FORM2.name},
        "expected_items": 120,
        "reviewer1_items": len(r1),
        "reviewer2_items": len(r2),
        "paired_items": len(common),
        "missing_reviewer1": sorted(expected - set(r1)),
        "missing_reviewer2": sorted(expected - set(r2)),
        "unpaired_reviewer1_only": sorted(set(r1) - set(r2)),
        "unpaired_reviewer2_only": sorted(set(r2) - set(r1)),
        "metrics": metrics,
        "confidence": {
            "reviewer1_mean": sum(confidence1) / len(confidence1),
            "reviewer2_mean": sum(confidence2) / len(confidence2),
            "reviewer1_distribution": dict(Counter(confidence1)),
            "reviewer2_distribution": dict(Counter(confidence2)),
        },
        "task_disagreement_counts": {task: dict(counts) for task, counts in task_disagreements.items()},
        "total_criterion_disagreements": len(disagreements),
    }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    with OUT_DISAGREE.open("w", newline="") as handle:
        fields = list(disagreements[0]) if disagreements else ["review_id", "criterion"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(disagreements)

    lines = [
        "# Human Benchmark Review Metrics", "",
        f"- Paired valid items: {len(common)}/120",
        f"- Missing from reviewer 1: {', '.join(report['missing_reviewer1']) or 'none'}",
        f"- Missing from reviewer 2: {', '.join(report['missing_reviewer2']) or 'none'}",
        f"- Mean confidence: reviewer 1 {report['confidence']['reviewer1_mean']:.2f}/5; reviewer 2 {report['confidence']['reviewer2_mean']:.2f}/5", "",
        "| Criterion | R1 rate | R2 rate | Pooled | Agreement | Cohen kappa (95% bootstrap CI) | Target |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for criterion in CRITERIA:
        m = metrics[criterion]
        ci = m["cohen_kappa_bootstrap95"]
        lines.append(
            f"| {m['label']} | {m['reviewer1_rate']:.1%} | {m['reviewer2_rate']:.1%} | "
            f"{m['pooled_rating_rate']:.1%} | {m['raw_agreement']:.1%} | {fmt(m['cohen_kappa'])} "
            f"[{fmt(ci[0])}, {fmt(ci[1])}] | {m['target']} ({'met' if m['target_met_on_pooled_rate'] else 'not met'}) |"
        )
    lines.extend([
        "", "## Interpretation", "",
        "Rates are the prevalence of score=1. For leakage, lower is better. Pooled rates average the two independent ratings; they are not adjudicated final acceptance rates.",
        "Cohen's kappa can be low or undefined when nearly all ratings are in one category, even when raw agreement is high. Report both statistics.",
        f"There are {len(disagreements)} criterion-level disagreements; adjudication is required before reporting final accepted-item rates.",
    ])
    OUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

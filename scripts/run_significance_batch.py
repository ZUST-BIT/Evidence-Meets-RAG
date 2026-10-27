from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from run_significance_analysis import ROOT, compare_results


DEFAULT_PLAN = ROOT / "configs/significance_plan_571_main.json"
DEFAULT_OUTDIR = ROOT / "dataset" / "rag" / "results" / "significance"


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    header = [
        "Framework",
        "Comparison",
        "Acc Δ",
        "Acc p-value",
        "Wins/Ties/Losses",
        "F1 Δ",
        "F1 95% CI",
        "F1 p-value",
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join([":---", ":---", "---:", "---:", ":---:", "---:", ":---:", "---:"]) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["framework"],
                    row["comparison"],
                    f"{row['acc_delta']:+.4f}",
                    f"{row['acc_p_value']:.6f}",
                    row["wins_ties_losses"],
                    f"{row['f1_delta']:+.4f}",
                    f"[{row['f1_ci_low']:+.4f}, {row['f1_ci_high']:+.4f}]",
                    f"{row['f1_p_value']:.6f}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch significance analysis for predefined result comparisons.")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--bootstrap-iters", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    plan_path = args.plan if args.plan.is_absolute() else ROOT / args.plan
    outdir = args.outdir if args.outdir.is_absolute() else ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    summary_rows: list[dict[str, Any]] = []
    detailed_results: list[dict[str, Any]] = []
    for index, item in enumerate(plan, start=1):
        result = compare_results(
            path_a=ROOT / item["result_a"],
            path_b=ROOT / item["result_b"],
            label_a=item["label_a"],
            label_b=item["label_b"],
            task_type=item.get("task_type"),
            bootstrap_iters=args.bootstrap_iters,
            seed=args.seed + index,
        )
        safe_name = f"{index:02d}_{item['framework'].lower().replace(' ', '_')}_{item['knowledge_form_b'].lower().replace(' ', '_')}"
        (outdir / f"{safe_name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        detailed_results.append(result)
        summary_rows.append(
            {
                "framework": item["framework"],
                "comparison": f"{item['knowledge_form_a']} vs {item['knowledge_form_b']}",
                "acc_delta": result["accuracy"]["delta"],
                "acc_p_value": result["accuracy"]["p_value"],
                "wins_ties_losses": f"{result['accuracy']['wins']}/{result['accuracy']['ties']}/{result['accuracy']['losses']}",
                "f1_delta": result["macro_f1"]["delta"],
                "f1_ci_low": result["macro_f1"]["ci95"][0],
                "f1_ci_high": result["macro_f1"]["ci95"][1],
                "f1_p_value": result["macro_f1"]["p_value"],
            }
        )

    with (outdir / "significance_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "framework",
                "comparison",
                "acc_delta",
                "acc_p_value",
                "wins_ties_losses",
                "f1_delta",
                "f1_ci_low",
                "f1_ci_high",
                "f1_p_value",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    write_markdown(summary_rows, outdir / "significance_summary.md")
    (outdir / "significance_summary.json").write_text(json.dumps(detailed_results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {outdir / 'significance_summary.json'}")
    print(f"Saved {outdir / 'significance_summary.csv'}")
    print(f"Saved {outdir / 'significance_summary.md'}")


if __name__ == "__main__":
    main()

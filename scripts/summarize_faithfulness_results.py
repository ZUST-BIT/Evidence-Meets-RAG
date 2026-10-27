from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_faithfulness_judge import ROOT, SYSTEM_PLAN, read_jsonl


KNOWLEDGE_ORDER = ["doc_vanilla", "semantic_vanilla", "evidence_vanilla"]
FRAMEWORK_ORDER = [
    "evidence_vanilla",
    "evidence_dense",
    "evidence_graph",
    "evidence_crag",
    "evidence_raptor",
    "evidence_kg2rag",
]


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    total = len(rows)
    if not total:
        return {
            "count": 0,
            "accuracy": 0.0,
            "es": 0.0,
            "as": 0.0,
            "cr": 0.0,
            "fcr": 0.0,
            "ucr": 0.0,
        }
    return {
        "count": total,
        "accuracy": round(sum(int(row["correct"]) for row in rows) / total, 4),
        "es": round(sum(int(row["retrieval_sufficient"]) for row in rows) / total, 4),
        "as": round(sum(int(row["answer_supported"]) for row in rows) / total, 4),
        "cr": round(sum(int(row["answer_contradicted"]) for row in rows) / total, 4),
        "fcr": round(sum(1 for row in rows if int(row["correct"]) and int(row["answer_supported"])) / total, 4),
        "ucr": round(sum(1 for row in rows if int(row["correct"]) and not int(row["answer_supported"])) / total, 4),
    }


def write_markdown_table(title: str, rows: list[dict[str, Any]], key: str) -> str:
    header_name = "Knowledge Source" if key == "knowledge_source" else "Framework"
    lines = [
        f"### {title}",
        "",
        f"| {header_name} | Accuracy | ES | AS | CR | FCR | UCR |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row[key],
                    f"{row['accuracy']:.4f}",
                    f"{row['es']:.4f}",
                    f"{row['as']:.4f}",
                    f"{row['cr']:.4f}",
                    f"{row['fcr']:.4f}",
                    f"{row['ucr']:.4f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize faithfulness judge outputs into paper-ready tables.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    input_dir = args.input_dir if args.input_dir.is_absolute() else ROOT / args.input_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_system: dict[str, list[dict[str, Any]]] = {}
    for item in SYSTEM_PLAN:
        path = input_dir / f"{item['system_id']}.jsonl"
        rows_by_system[item["system_id"]] = read_jsonl(path)

    knowledge_rows: list[dict[str, Any]] = []
    framework_rows: list[dict[str, Any]] = []
    for system_id in KNOWLEDGE_ORDER:
        item = next(entry for entry in SYSTEM_PLAN if entry["system_id"] == system_id)
        summary = summarize_rows(rows_by_system[system_id])
        knowledge_rows.append({"knowledge_source": item["knowledge_source"], **summary})
    for system_id in FRAMEWORK_ORDER:
        item = next(entry for entry in SYSTEM_PLAN if entry["system_id"] == system_id)
        summary = summarize_rows(rows_by_system[system_id])
        framework_rows.append({"framework": item["framework"], **summary})

    payload: dict[str, Any] = {
        "knowledge_source_panel": knowledge_rows,
        "framework_panel": framework_rows,
    }
    (output_dir / "faithfulness_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    knowledge_md = write_markdown_table("Faithfulness by knowledge source under Vanilla RAG", knowledge_rows, "knowledge_source")
    framework_md = write_markdown_table("Faithfulness by framework under Evidence KG", framework_rows, "framework")
    combined = (
        "## Faithfulness / evidence-grounding summary\n\n"
        "Metrics: `ES` = evidence sufficiency, `AS` = answer supportedness, `CR` = contradiction rate, "
        "`FCR` = faithful correct rate, `UCR` = unsupported correct rate.\n\n"
        + knowledge_md
        + "\n\n"
        + framework_md
        + "\n"
    )
    (output_dir / "faithfulness_knowledge_source_table.md").write_text(knowledge_md + "\n", encoding="utf-8")
    (output_dir / "faithfulness_framework_table.md").write_text(framework_md + "\n", encoding="utf-8")
    (output_dir / "faithfulness_summary.md").write_text(combined, encoding="utf-8")
    print(f"Saved {output_dir / 'faithfulness_summary.json'}")
    print(f"Saved {output_dir / 'faithfulness_summary.md'}")


if __name__ == "__main__":
    main()

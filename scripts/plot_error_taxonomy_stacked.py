from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "dataset"
    / "rag"
    / "results"
    / "error_taxonomy"
    / "vanilla_firstpass"
    / "error_taxonomy_summary.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "dataset" / "rag" / "figures"

SYSTEM_ORDER = [
    ("doc_vanilla", "Document Corpus"),
    ("semantic_vanilla", "Semantic KG"),
    ("evidence_vanilla", "Evidence Graph"),
]

ERROR_SPECS = [
    ("sparse_missing_evidence", "Sparse Missing", "#4C78A8"),
    ("bundle_incompleteness", "Bundle Incompleteness", "#F58518"),
    ("label_granularity_mapping_failure", "Label Mapping", "#54A24B"),
    ("scope_distractor_mismatch", "Scope Mismatch", "#E45756"),
    ("conflict_resolution_failure", "Conflict Resolution", "#B279A2"),
    ("multi_hop_reasoning_failure", "Multi-hop Reasoning", "#9D755D"),
]


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_percentage_matrix(summary: dict) -> tuple[list[str], np.ndarray]:
    labels = [label for _system_id, label in SYSTEM_ORDER]
    matrix = np.zeros((len(ERROR_SPECS), len(SYSTEM_ORDER)), dtype=float)
    for col, (system_id, _label) in enumerate(SYSTEM_ORDER):
        system_summary = summary["system_summaries"][system_id]
        total = system_summary["count"]
        for row, (error_id, _short_label, _color) in enumerate(ERROR_SPECS):
            count = system_summary["by_error"].get(error_id, 0)
            matrix[row, col] = count / total * 100.0 if total else 0.0
    return labels, matrix


def annotate_sparse(ax, positions: np.ndarray, sparse_values: np.ndarray) -> None:
    for x, value in zip(positions, sparse_values):
        ax.text(
            x,
            value - 4.0,
            f"{value:.1f}%",
            ha="center",
            va="top",
            fontsize=10.5,
            color="white",
            fontweight="bold",
        )


def annotate_minor(ax, positions: np.ndarray, minor_matrix: np.ndarray) -> None:
    for col, x in enumerate(positions):
        cumulative = 0.0
        for row in range(minor_matrix.shape[0]):
            value = minor_matrix[row, col]
            if value < 2.0:
                cumulative += value
                continue
            ax.text(
                x,
                cumulative + value / 2.0,
                f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=9.0,
                color="#1F2937",
            )
            cumulative += value


def plot_taxonomy_stacked(summary: dict, output_dir: Path) -> tuple[Path, Path]:
    labels, matrix = build_percentage_matrix(summary)
    sparse_values = matrix[0]
    minor_matrix = matrix[1:]
    minor_totals = minor_matrix.sum(axis=0)
    normalized_minor = np.divide(
        minor_matrix,
        minor_totals,
        out=np.zeros_like(minor_matrix),
        where=minor_totals > 0,
    ) * 100.0

    fig = plt.figure(figsize=(10.2, 5.6), constrained_layout=False)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.24)
    left_ax = fig.add_subplot(grid[0, 0])
    right_ax = fig.add_subplot(grid[0, 1])
    x = np.arange(len(labels))
    width = 0.58

    handles = []
    legend_labels = []
    sparse_bar = left_ax.bar(
        x,
        sparse_values,
        width=width,
        color=ERROR_SPECS[0][2],
        edgecolor="white",
        linewidth=0.8,
    )
    handles.append(sparse_bar[0])
    legend_labels.append(ERROR_SPECS[0][1])

    bottom = np.zeros(len(labels), dtype=float)
    for row, (_error_id, short_label, color) in enumerate(ERROR_SPECS[1:]):
        values = normalized_minor[row]
        bars = right_ax.bar(
            x,
            values,
            width=width,
            bottom=bottom,
            color=color,
            edgecolor="white",
            linewidth=0.8,
        )
        handles.append(bars[0])
        legend_labels.append(short_label)
        bottom += values

    annotate_sparse(left_ax, x, sparse_values)
    annotate_minor(right_ax, x, normalized_minor)

    left_ax.set_ylim(0, 100)
    left_ax.set_xticks(x)
    left_ax.set_xticklabels(labels, fontsize=10.2)
    left_ax.set_ylabel("Error share (%)", fontsize=11)
    left_ax.set_title(
        "Sparse / Missing Evidence",
        fontsize=12.5,
        fontweight="bold",
        pad=10,
    )
    left_ax.grid(axis="y", color="#D1D5DB", linewidth=0.8, alpha=0.8)
    left_ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        left_ax.spines[spine].set_visible(False)

    right_ax.set_ylim(0, 100)
    right_ax.set_xticks(x)
    right_ax.set_xticklabels(labels, fontsize=10.2)
    right_ax.set_ylabel("Re-normalized non-Sparse share (%)", fontsize=11)
    right_ax.set_title(
        "Distribution of remaining error types",
        fontsize=12.5,
        fontweight="bold",
        pad=10,
    )
    right_ax.grid(axis="y", color="#D1D5DB", linewidth=0.8, alpha=0.8)
    right_ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        right_ax.spines[spine].set_visible(False)

    fig.suptitle(
        "Primary error-type composition under Vanilla RAG",
        fontsize=13.5,
        fontweight="bold",
        y=0.98,
    )

    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.09),
        ncol=6,
        frameon=False,
        fontsize=9.5,
        columnspacing=1.2,
        handlelength=1.4,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.21)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "error_taxonomy_stacked_inset.png"
    pdf_path = output_dir / "error_taxonomy_stacked_inset.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot stacked error taxonomy bars with inset zoom.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to error_taxonomy_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the figure outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = load_summary(args.input)
    png_path, pdf_path = plot_taxonomy_stacked(summary, args.output_dir)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()

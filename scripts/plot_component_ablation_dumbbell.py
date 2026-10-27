from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PDF = ROOT / "outputs" / "figures" / "component_ablation_dumbbell.pdf"
OUTPUT_PNG = ROOT / "outputs" / "figures" / "component_ablation_dumbbell.png"
SUMMARY_PATH = ROOT / "dataset/rag/results/fixed_context_component_ablation_571_vanilla__summary.json"

LABELS = [
    "w/o structure",
    "w/o evidence text",
    "w/o cross-source links",
    "w/o aggregation",
    "w/o relation types",
]

CONDITION_ORDER = [
    "no_structure",
    "no_evidence_text",
    "no_cross_source_links",
    "no_cross_layer_aggregation",
    "no_relation_types",
]

LOWER = "#356A9A"
HIGHER = "#D27A3D"
BASELINE = "#8B949E"
CONNECTOR = "#C7CDD3"
GRID = "#E2E6EA"
TEXT = "#263238"


def draw_panel(
    ax: plt.Axes,
    title: str,
    full: float,
    values: np.ndarray,
    xlim: tuple[float, float],
) -> None:
    y = np.arange(len(LABELS)) * 1.16
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.42, y[-1] + 0.72)
    ax.axvline(full, color=BASELINE, linewidth=1.0, linestyle=(0, (2.2, 2.2)), zorder=1)

    for row, value in enumerate(values):
        color = LOWER if value < full else HIGHER
        ax.plot([full, value], [y[row], y[row]], color=CONNECTOR, linewidth=1.7, solid_capstyle="round", zorder=2)
        ax.scatter(full, y[row], s=22, facecolor="white", edgecolor=BASELINE, linewidth=0.9, zorder=3)
        ax.scatter(value, y[row], s=28, facecolor=color, edgecolor="white", linewidth=0.6, zorder=4)

        delta_pp = (value - full) * 100
        label_x = value
        label_ha = "center"
        if abs(value - full) < 0.007:
            offset = -0.0015 if value < full else 0.0015
            label_x = value + offset
            label_ha = "right" if value < full else "left"
        ax.text(
            label_x,
            y[row] + 0.40,
            f"{value:.3f} ({delta_pp:+.1f})",
            ha=label_ha,
            va="center",
            fontsize=5.7,
            color=color,
            fontweight="medium",
            clip_on=False,
        )

    ax.set_title(f"{title}\nFull = {full:.3f}", fontsize=7.0, fontweight="bold", color=TEXT, pad=4)
    ax.set_yticks(y)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=6.2, colors=TEXT, length=2.2)
    ax.tick_params(axis="y", length=0, pad=3)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#7C8791")
    ax.spines["bottom"].set_linewidth(0.7)


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))["conditions"]
    panels = [
        (
            "Accuracy",
            float(summary["full"]["accuracy"]),
            np.array([float(summary[key]["accuracy"]) for key in CONDITION_ORDER]),
            (0.405, 0.450),
        ),
        (
            "Macro-F1",
            float(summary["full"]["macro_f1"]),
            np.array([float(summary[key]["macro_f1"]) for key in CONDITION_ORDER]),
            (0.425, 0.475),
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.90), sharey=True)
    for ax, panel in zip(axes, panels):
        draw_panel(ax, *panel)

    axes[0].invert_yaxis()
    axes[0].set_yticklabels(LABELS, fontsize=6.6, color=TEXT)
    axes[1].tick_params(axis="y", labelleft=False)
    fig.subplots_adjust(left=0.34, right=0.985, top=0.66, bottom=0.16, wspace=0.28)

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUTPUT_PNG, dpi=260, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()

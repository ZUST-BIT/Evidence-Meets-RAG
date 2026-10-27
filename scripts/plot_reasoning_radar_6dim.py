from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "dataset" / "rag" / "results"
DEFAULT_DATA_JSON = ROOT / "dataset" / "rag" / "figures" / "formalv3_571_reasoning_radar_6dim_metrics.json"
DEFAULT_OUTPUT_DIR = ROOT / "dataset" / "rag" / "figures"

sys.path.append(str(Path(__file__).resolve().parent))
from rag_eval_lib import gold_label, macro_f1_from_labels  # noqa: E402


DIMENSIONS = [
    {
        "key": "cross_source_mechanistic_tracing",
        "label": "Cross-source\nmechanistic tracing",
        "group_by": "task_type",
        "group_value": "mechanism_trace_intermediate_gene",
    },
    {
        "key": "multi_evidence_logical_synthesis",
        "label": "Multi-evidence\nsynthesis",
        "group_by": "reasoning_type",
        "group_value": "multi_evidence_logical_synthesis",
    },
    {
        "key": "precision_oncology_evidence_judgment",
        "label": "Precision oncology\njudgment",
        "group_by": "reasoning_type",
        "group_value": "precision_oncology_evidence_judgment",
    },
    {
        "key": "evidence_signal_interpretation",
        "label": "Evidence signal\ninterpretation",
        "group_by": "reasoning_type",
        "group_value": "evidence_signal_interpretation",
    },
    {
        "key": "hierarchical_evidence_aggregation",
        "label": "Hierarchical evidence\naggregation",
        "group_by": "reasoning_type",
        "group_value": "hierarchical_evidence_aggregation",
    },
    {
        "key": "cross_source_corroboration",
        "label": "Cross-source\ncorroboration",
        "group_by": "reasoning_type",
        "group_value": "cross_source_corroboration",
    },
]

# Two candidate arrangements for the six framework panels.
#   "row"  -> 1x6, spans both columns (figure*), shortest vertically
#   "grid" -> 2x3, also spans both columns, roughly double the radar diameter
# Font sizes are tuned per layout because the row form gives each panel about
# half the width of the grid form.
# Drawn at final print size (a two-column ACM \textwidth is about 7 inches), so
# the point sizes below are what the reader actually sees. Drawing large and
# letting LaTeX shrink the result is what made the earlier version unreadable.
LAYOUTS = {
    "row": (1, 6, (7.0, 2.05), {
        "label": 3.6, "tick": 2.8, "title": 5.5, "title_y": 1.20,
        "legend": 6.0, "legend_y": 1.00, "rect": (0.0, 0.0, 1.0, 0.84), "wspace": 0.62,
    }),
    "grid": (2, 3, (7.0, 3.55), {
        "label": 5.2, "tick": 3.6, "title": 7.0, "title_y": 1.16,
        "legend": 7.0, "legend_y": 1.005, "rect": (0.0, 0.0, 1.0, 0.965), "wspace": 0.72,
        "hspace": 0.62, "labelpad": -1.0,
    }),
    # Axis labels are identical in all six panels, so printing them 36 times is
    # pure redundancy. Labelling only the first panel lets every circle grow and
    # cuts the figure height further; the caption states that axes are shared.
    # Main-text form: two panels side by side across both columns. With only two
    # circles to place, each is far larger than in the six-panel grid while the
    # figure is barely a third as tall.
    # Single-column form: an ACM sigconf column is about 3.33 in, so two panels
    # leave roughly 1.6 in each and every point size has to come down with them.
    # Same per-panel geometry and point sizes as quad1col, one row instead of
    # two, so the appendix panels are visually identical to the main-text ones.
    "duo1col": (1, 2, (3.33, 1.95), {
        "label": 4.0, "tick": 2.9, "title": 5.6, "title_y": 1.20,
        "legend": 5.2, "legend_y": 1.0, "rect": (0.0, 0.0, 1.0, 0.90), "wspace": 1.05,
        "labelpad": -1.5, "lw": 1.0, "legend_lw": 1.8,
    }),
    "quad1col": (2, 2, (3.33, 3.25), {
        "label": 4.0, "tick": 2.9, "title": 5.6, "title_y": 1.22,
        "legend": 5.2, "legend_y": 1.005, "rect": (0.0, 0.0, 1.0, 0.975), "wspace": 1.05,
        "hspace": 0.42, "labelpad": -1.5, "lw": 1.0, "legend_lw": 1.8,
    }),
    # Single-column 3x2 form: all six frameworks in one column-width figure.
    # Same per-panel geometry as quad1col with one more row.
    "grid1col": (3, 2, (3.33, 4.22), {
        "label": 4.0, "tick": 2.9, "title": 5.6, "title_y": 1.12,
        "legend": 5.2, "legend_y": 0.995, "rect": (0.0, 0.0, 1.0, 0.99), "wspace": 0.78,
        "hspace": 0.56, "labelpad": -2.0, "lw": 1.0, "legend_lw": 1.8,
    }),
    "pair1col": (1, 2, (3.33, 1.78), {
        "label": 4.4, "tick": 3.2, "title": 6.0, "title_y": 1.20,
        "legend": 5.4, "legend_y": 1.0, "rect": (0.0, 0.0, 1.0, 0.965), "wspace": 0.95,
        "labelpad": 0.0, "lw": 1.1, "legend_lw": 1.8,
    }),
    # Stacked single-column form: each panel gets the full column width, so the
    # circles and their labels are far larger than in the side-by-side variant
    # while the total page area is smaller than the two-column version.
    "pair1col_stack": (2, 1, (3.33, 3.95), {
        "label": 5.4, "tick": 4.0, "title": 7.0, "title_y": 1.14,
        "legend": 6.5, "legend_y": 1.015, "rect": (0.0, 0.0, 1.0, 0.955), "wspace": 0.20,
        "hspace": 0.72, "labelpad": -1.0,
    }),
    "pair": (1, 2, (7.0, 2.55), {
        "label": 6.4, "tick": 4.6, "title": 8.5, "title_y": 1.14,
        "legend": 8.0, "legend_y": 1.02, "rect": (0.0, 0.0, 1.0, 0.955), "wspace": 0.55,
        "labelpad": -1.0,
    }),
    "keyed": (2, 3, (7.0, 2.85), {
        "label": 5.4, "tick": 3.6, "title": 7.0, "title_y": 1.10,
        "legend": 7.0, "legend_y": 1.01, "rect": (0.0, 0.0, 1.0, 0.965), "wspace": 0.30,
        "hspace": 0.40, "labelpad": -1.0, "label_first_only": True,
    }),
}
LAYOUT = "row"
NAME_SUFFIX = ""

ALL_FRAMEWORKS = [
    ("naive_rag", "Vanilla RAG"),
    ("dense_rag", "Dense RAG"),
    ("graph_rag", "Graph RAG"),
    ("crag_rag", "CRAG"),
    ("raptor_rag", "RAPTOR"),
    ("kg2rag", "KG2RAG"),
]
# Main text shows four frameworks: Vanilla and Graph RAG carry the paired
# significance tests and the degeneracy audit, KG2RAG is the only framework
# significant against both baselines, and CRAG is the one case where Semantic KG
# collapses. Dense RAG and RAPTOR add complementary retrieval comparisons.
MAIN_FRAMEWORKS = [f for f in ALL_FRAMEWORKS
                   if f[0] in {"naive_rag", "graph_rag", "crag_rag", "kg2rag"}]
APPENDIX_FRAMEWORKS = [f for f in ALL_FRAMEWORKS if f[0] in {"dense_rag", "raptor_rag"}]
FRAMEWORKS = ALL_FRAMEWORKS

SOURCES = [
    ("raw_observation_view", "Document Corpus", "#4C78A8"),
    ("semantic_kg", "Semantic KG", "#F58518"),
    ("evidence_graph_complete", "Evidence Graph", "#54A24B"),
]


def summary_suffix(framework: str, source: str) -> str:
    if framework in {"raptor_rag", "kg2rag"}:
        return "formalv3_571_cross_raptor_kg2rag"
    if source == "semantic_kg":
        return "formalv3_571_classic4_semantic_kg"
    return "formalv3_571_classic4_raw_vs_graph"


def result_path(framework: str, source: str) -> Path:
    suffix = summary_suffix(framework, source)
    return RESULTS_DIR / f"multi_cancer__{source}__{framework}__{suffix}.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task_type = str((row.get("record") or {}).get("task_type") or "")
        reasoning_type = ((row.get("record") or {}).get("benchmark_annotation") or {}).get("reasoning_type")
        if task_type:
            grouped[("task_type", task_type)].append(row)
        if reasoning_type:
            grouped[("reasoning_type", str(reasoning_type))].append(row)

    sample_counts = {
        item["key"]: len(grouped.get((item["group_by"], item["group_value"]), []))
        for item in DIMENSIONS
    }
    accuracy: dict[str, float] = {}
    macro_f1: dict[str, float] = {}
    for item in DIMENSIONS:
        key = item["key"]
        group_rows = grouped.get((item["group_by"], item["group_value"]), [])
        if not group_rows:
            accuracy[key] = 0.0
            macro_f1[key] = 0.0
            continue
        gold_labels = [gold_label(item["record"]) for item in group_rows]
        pred_labels = [str(item.get("normalized_prediction") or "") for item in group_rows]
        exact = sum(1 for gold, pred in zip(gold_labels, pred_labels) if gold == pred)
        accuracy[key] = round(exact / len(group_rows), 4)
        macro_f1[key] = round(macro_f1_from_labels(gold_labels, pred_labels), 4)
    return {
        "sample_counts": sample_counts,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


def collect_metrics() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "benchmark": "formal_v3_571",
        "view": "multi_cancer",
        "dimensions": [
            {
                "key": item["key"],
                "label": item["label"].replace("\n", " "),
                "group_by": item["group_by"],
                "group_value": item["group_value"],
            }
            for item in DIMENSIONS
        ],
        "frameworks": [key for key, _label in FRAMEWORKS],
        "framework_labels": {key: label for key, label in FRAMEWORKS},
        "sources": [label for _key, label, _color in SOURCES],
        "source_map": {key: label for key, label, _color in SOURCES},
        "source_colors": {label: color for _key, label, color in SOURCES},
        "metrics": {
            "accuracy": defaultdict(dict),
            "macro_f1": defaultdict(dict),
        },
        "sample_counts": {},
    }

    canonical_counts: Counter[str] = Counter()
    for framework, _framework_label in FRAMEWORKS:
        for source_key, source_label, _color in SOURCES:
            rows = load_jsonl(result_path(framework, source_key))
            summary = summarize_rows(rows)
            payload["metrics"]["accuracy"][framework][source_label] = summary["accuracy"]
            payload["metrics"]["macro_f1"][framework][source_label] = summary["macro_f1"]
            canonical_counts.update(summary["sample_counts"])

    payload["metrics"]["accuracy"] = dict(payload["metrics"]["accuracy"])
    payload["metrics"]["macro_f1"] = dict(payload["metrics"]["macro_f1"])
    payload["sample_counts"] = {
        item["key"]: canonical_counts.get(item["key"], 0) // max(len(FRAMEWORKS) * len(SOURCES), 1)
        for item in DIMENSIONS
    }
    return payload


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_or_build_data(data_json: Path, rebuild_data: bool) -> dict[str, Any]:
    if rebuild_data or not data_json.exists():
        payload = collect_metrics()
        ensure_parent(data_json)
        data_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return payload
    return json.loads(data_json.read_text(encoding="utf-8"))


def close_radar(values: list[float]) -> list[float]:
    return values + values[:1]


def metric_axis_upper_bound(data: dict[str, Any], metric_key: str) -> float:
    values: list[float] = []
    for framework_metrics in data["metrics"][metric_key].values():
        for source_metrics in framework_metrics.values():
            values.extend(float(value) for value in source_metrics.values())
    if not values:
        return 1.0
    peak = max(values)
    upper = math.ceil((peak + 0.02) / 0.05) * 0.05
    return max(0.30, min(1.0, upper))


def normalized_metric_payload(data: dict[str, Any], metric_key: str) -> tuple[dict[str, Any], dict[str, tuple[float, float]]]:
    axis_ranges: dict[str, tuple[float, float]] = {}
    normalized = json.loads(json.dumps(data))
    for dimension in data["dimensions"]:
        key = dimension["key"]
        values: list[float] = []
        for framework_metrics in data["metrics"][metric_key].values():
            for source_metrics in framework_metrics.values():
                values.append(float(source_metrics[key]))
        dim_min = min(values) if values else 0.0
        dim_max = max(values) if values else 1.0
        axis_ranges[key] = (dim_min, dim_max)
        span = dim_max - dim_min
        for framework_key in data["frameworks"]:
            for source_label in data["sources"]:
                raw_value = float(data["metrics"][metric_key][framework_key][source_label][key])
                normalized_value = 1.0 if span <= 1e-12 else (raw_value - dim_min) / span
                normalized["metrics"][metric_key][framework_key][source_label][key] = round(normalized_value, 4)
    return normalized, axis_ranges


def task_relative_metric_payload(
    data: dict[str, Any], metric_key: str
) -> tuple[dict[str, Any], dict[str, tuple[float, float]]]:
    """Scale each task by its best observed score without shifting its zero."""
    axis_ranges: dict[str, tuple[float, float]] = {}
    relative = json.loads(json.dumps(data))
    for dimension in data["dimensions"]:
        key = dimension["key"]
        values = [
            float(source_metrics[key])
            for framework_metrics in data["metrics"][metric_key].values()
            for source_metrics in framework_metrics.values()
        ]
        task_max = max(values) if values else 1.0
        axis_ranges[key] = (0.0, task_max)
        for framework_key in data["frameworks"]:
            for source_label in data["sources"]:
                raw_value = float(data["metrics"][metric_key][framework_key][source_label][key])
                relative["metrics"][metric_key][framework_key][source_label][key] = round(
                    raw_value / task_max if task_max > 1e-12 else 0.0, 4
                )
    return relative, axis_ranges


def contrast_enhanced_metric_payload(
    data: dict[str, Any],
    metric_key: str,
    *,
    target_low: float = 0.22,
    target_high: float = 0.92,
    min_span: float = 0.08,
    pad_ratio: float = 0.12,
) -> tuple[dict[str, Any], dict[str, tuple[float, float]]]:
    axis_ranges: dict[str, tuple[float, float]] = {}
    enhanced = json.loads(json.dumps(data))
    for dimension in data["dimensions"]:
        key = dimension["key"]
        values: list[float] = []
        for framework_metrics in data["metrics"][metric_key].values():
            for source_metrics in framework_metrics.values():
                values.append(float(source_metrics[key]))
        if not values:
            axis_ranges[key] = (0.0, 1.0)
            continue
        raw_min = min(values)
        raw_max = max(values)
        span = raw_max - raw_min
        padded_span = max(span * (1 + 2 * pad_ratio), min_span)
        center = (raw_min + raw_max) / 2
        scaled_min = center - padded_span / 2
        scaled_max = center + padded_span / 2
        axis_ranges[key] = (scaled_min, scaled_max)
        denom = scaled_max - scaled_min
        for framework_key in data["frameworks"]:
            for source_label in data["sources"]:
                raw_value = float(data["metrics"][metric_key][framework_key][source_label][key])
                base = 0.5 if denom <= 1e-12 else (raw_value - scaled_min) / denom
                clamped = min(1.0, max(0.0, base))
                enhanced_value = target_low + clamped * (target_high - target_low)
                enhanced["metrics"][metric_key][framework_key][source_label][key] = round(enhanced_value, 4)
    return enhanced, axis_ranges


def plot_metric_grid(
    data: dict[str, Any],
    metric_key: str,
    output_dir: Path,
    *,
    normalized: bool = False,
    task_relative: bool = False,
    contrast_enhanced: bool = False,
    axis_ranges: dict[str, tuple[float, float]] | None = None,
) -> None:
    framework_labels = data["framework_labels"]
    source_colors = data["source_colors"]
    dimension_keys = [item["key"] for item in data["dimensions"]]
    dimension_labels = [item["label"] for item in DIMENSIONS]
    radial_upper = 1.0 if normalized or task_relative or contrast_enhanced else metric_axis_upper_bound(data, metric_key)
    radial_ticks = [round(radial_upper * ratio, 2) for ratio in (0.2, 0.4, 0.6, 0.8, 1.0)]

    angles = [2 * math.pi * idx / len(dimension_keys) for idx in range(len(dimension_keys))]
    angles = close_radar(angles)

    nrows, ncols, figsize, fs = LAYOUTS[LAYOUT]
    fig, axes = plt.subplots(nrows, ncols, subplot_kw={"projection": "polar"}, figsize=figsize)
    axes = axes.flatten()
    legend_handles = []
    legend_labels = []

    for idx, (ax, (framework_key, framework_label)) in enumerate(zip(axes, FRAMEWORKS)):
        ax.set_theta_offset(math.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        show_labels = not (fs.get("label_first_only") and idx > 0)
        ax.set_xticklabels(dimension_labels if show_labels else [""] * len(dimension_keys),
                           fontsize=fs["label"])
        ax.set_ylim(0.0, radial_upper)
        ax.set_yticks(radial_ticks)
        ax.set_yticklabels([f"{tick:.2f}".rstrip("0").rstrip(".") for tick in radial_ticks], fontsize=fs["tick"], color="#666666")
        ax.grid(color="#CCCCCC", alpha=0.7)
        # Pull the wrap-around labels in toward the circle: that ring of blank
        # space, not the plot itself, is what makes a radar grid tall.
        if "labelpad" in fs:
            ax.tick_params(axis="x", pad=fs["labelpad"])
        ax.set_title(framework_label, y=fs["title_y"], fontsize=fs["title"], fontweight="bold")

        for _source_key, source_label, _color in SOURCES:
            metric_values = data["metrics"][metric_key][framework_key][source_label]
            values = [float(metric_values[key]) for key in dimension_keys]
            color = source_colors[source_label]
            line, = ax.plot(angles, close_radar(values), color=color, linewidth=fs.get("lw", 2.0))
            ax.fill(angles, close_radar(values), color=color, alpha=0.10)
            if source_label not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(source_label)


    # No suptitle and no footer: the LaTeX caption carries the title, the
    # per-axis scaling disclosure, and the sample sizes, so repeating them
    # inside the artwork only steals vertical space.
    legend = fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, fs["legend_y"]),
        ncol=3,
        frameon=False,
        fontsize=fs["legend"],
        handlelength=2.8,
        handletextpad=0.6,
        columnspacing=2.0,
        borderaxespad=0.0,
    )
    for handle in legend.legend_handles:
        handle.set_linewidth(fs.get("legend_lw", 3.0))
    fig.tight_layout(rect=fs["rect"])
    fig.subplots_adjust(wspace=fs["wspace"], **({"hspace": fs["hspace"]} if "hspace" in fs else {}))

    output_dir.mkdir(parents=True, exist_ok=True)
    if task_relative:
        name_suffix = f"{metric_key}_task_relative"
    elif contrast_enhanced:
        name_suffix = f"{metric_key}_contrast_enhanced"
    elif normalized:
        name_suffix = f"{metric_key}_axis_normalized"
    else:
        name_suffix = metric_key
    png_path = output_dir / f"formalv3_571_reasoning_radar_6dim{NAME_SUFFIX}_{name_suffix}.png"
    svg_path = output_dir / f"formalv3_571_reasoning_radar_6dim{NAME_SUFFIX}_{name_suffix}.svg"
    # Write PDF straight from matplotlib so the figure stays vector in the
    # paper; the 5-axis PDFs were rasterised through PNG.
    pdf_path = output_dir / f"formalv3_571_reasoning_radar_6dim{NAME_SUFFIX}_{name_suffix}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build editable radar-metric JSON and plot reasoning radar figures.")
    parser.add_argument("--data-json", type=Path, default=DEFAULT_DATA_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rebuild-data", action="store_true", help="Recompute the editable JSON from result JSONL files before plotting.")
    parser.add_argument(
        "--mode",
        choices=["all", "contrast-only"],
        default="all",
        help="Choose whether to generate all figure variants or only contrast-enhanced figures.",
    )
    parser.add_argument("--layout", choices=sorted(LAYOUTS), default="row")
    parser.add_argument("--suffix", default="", help="Appended to output filenames, for side-by-side comparison.")
    parser.add_argument("--frameworks", choices=["all", "main", "appendix"], default="all")
    args = parser.parse_args()
    global LAYOUT, NAME_SUFFIX, FRAMEWORKS
    LAYOUT = args.layout
    NAME_SUFFIX = args.suffix
    FRAMEWORKS = {"main": MAIN_FRAMEWORKS, "appendix": APPENDIX_FRAMEWORKS}.get(args.frameworks, ALL_FRAMEWORKS)

    data = load_or_build_data(args.data_json, rebuild_data=args.rebuild_data)
    relative_accuracy, relative_accuracy_ranges = task_relative_metric_payload(data, "accuracy")
    relative_macro_f1, relative_macro_f1_ranges = task_relative_metric_payload(data, "macro_f1")
    enhanced_accuracy, enhanced_accuracy_ranges = contrast_enhanced_metric_payload(data, "accuracy")
    enhanced_macro_f1, enhanced_macro_f1_ranges = contrast_enhanced_metric_payload(data, "macro_f1")
    if args.mode == "all":
        plot_metric_grid(data, "accuracy", args.output_dir)
        plot_metric_grid(data, "macro_f1", args.output_dir)
        normalized_accuracy, accuracy_ranges = normalized_metric_payload(data, "accuracy")
        normalized_macro_f1, macro_f1_ranges = normalized_metric_payload(data, "macro_f1")
        plot_metric_grid(normalized_accuracy, "accuracy", args.output_dir, normalized=True, axis_ranges=accuracy_ranges)
        plot_metric_grid(normalized_macro_f1, "macro_f1", args.output_dir, normalized=True, axis_ranges=macro_f1_ranges)
    plot_metric_grid(relative_accuracy, "accuracy", args.output_dir, task_relative=True, axis_ranges=relative_accuracy_ranges)
    plot_metric_grid(relative_macro_f1, "macro_f1", args.output_dir, task_relative=True, axis_ranges=relative_macro_f1_ranges)
    plot_metric_grid(enhanced_accuracy, "accuracy", args.output_dir, contrast_enhanced=True, axis_ranges=enhanced_accuracy_ranges)
    plot_metric_grid(enhanced_macro_f1, "macro_f1", args.output_dir, contrast_enhanced=True, axis_ranges=enhanced_macro_f1_ranges)


if __name__ == "__main__":
    main()

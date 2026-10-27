"""Regenerate existing figure designs from the unified 571 outputs."""
from __future__ import annotations

import json
from pathlib import Path
import runpy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from expand_masking_571 import ROOT, FULL, ANALYSIS, dump
import plot_component_ablation_dumbbell as components
import plot_reasoning_radar_6dim as radar

OUT = ROOT / 'outputs/figures'
GRAPH_FULL = ROOT/'dataset/rag/results/unified_571/stage/R1C1G1.jsonl'


def figure3():
    original_path = radar.result_path
    def result_path(framework, source):
        if source == 'evidence_graph_complete':
            if framework == 'naive_rag':
                return FULL
            if framework == 'graph_rag':
                if not GRAPH_FULL.exists() or sum(1 for _ in GRAPH_FULL.open()) != 571:
                    raise RuntimeError('Graph main/radar requires the completed shared Figure6 Full')
                return GRAPH_FULL
        return original_path(framework, source)
    radar.result_path = result_path
    data = radar.collect_metrics()
    dump(ANALYSIS/'figure3_radar_metrics.json', data)
    radar.LAYOUT = 'grid1col'
    radar.NAME_SUFFIX = ''
    relative, ranges = radar.task_relative_metric_payload(data, 'macro_f1')
    radar.plot_metric_grid(relative, 'macro_f1', OUT, task_relative=True, axis_ranges=ranges)
    radar.result_path = original_path


def figure4():
    retrieval = json.loads((ANALYSIS/'retrieval_571.json').read_text())['sources']
    main = json.loads((ANALYSIS/'main_and_ablation.json').read_text())['vanilla']
    errors = json.loads((ANALYSIS/'residual_errors.json').read_text())
    oracle = json.loads((ROOT/'dataset/rag/results/oracle_gold_context/oracle_gold_context_571_summary.json').read_text())
    if oracle['overall']['count'] != 571:
        raise ValueError('Figure 4 requires the complete 571-question oracle summary')
    ceiling = oracle['overall']['accuracy']
    assert errors['evidence_classification_complete']
    # Reuse the original plotting recipe, changing only its data constants and
    # output directory; the geometry, colors, labels and layout stay intact.
    source = (ROOT/'scripts/plot_mechanism_access_use.py').read_text()
    cr = [retrieval[k]['component_recall'] for k in ('doc_corpus','semantic_kg','evidence_kg')]
    acc = [main[k]['accuracy'] for k in ('document','semantic','evidence')]
    systems = [errors['systems'][k] for k in ('doc_vanilla','semantic_vanilla','evidence_vanilla')]
    use_shares = [f"{100 * value['use_share']:.1f}%" if value['use_share'] is not None else 'n/a' for value in systems]
    replacements = {
        'OUT = ROOT / "dataset" / "rag" / "figures"': f'OUT = Path({str(OUT)!r})',
        'CR = [0.2437, 0.2828, 0.6540]      # 145-record balanced panel': f'CR = {cr!r}  # full 571-question benchmark',
        'ACC = [0.3517, 0.4069, 0.4621]     # same panel, same runs': f'ACC = {acc!r}  # same 571 IDs and canonical Full',
        'CEILING = 0.8476                    # gold-evidence ceiling, 571 records': f'CEILING = {ceiling!r}  # current 571-question oracle',
        '"gold-evidence\\nceiling 0.85"': f'f"gold-evidence\\nceiling {{CEILING:.2f}}"',
        'ax1.text(2.42, CEILING - 0.005,': 'if not any(abs(CEILING - value) < 0.06 for value in CR + ACC): ax1.text(2.42, CEILING - 0.005,',
        'ax1.set_title("(a) Access and answer quality", fontsize=6.4, pad=3)': 'ax1.set_title("(a) Access and answer quality" + (f"\\ngold-evidence ceiling {CEILING:.2f}" if any(abs(CEILING - value) < 0.06 for value in CR + ACC) else ""), fontsize=6.4, pad=3)',
        'ERR_TOTAL = [376, 334, 327]': f"ERR_TOTAL = {[v['classified_errors'] for v in systems]!r}",
        'ERR_ACCESS = [375, 288, 271]': f"ERR_ACCESS = {[v['access_errors'] for v in systems]!r}",
        'ERR_USE = [1, 46, 56]': f"ERR_USE = {[v['use_errors'] for v in systems]!r}",
        'USE_SHARE = ["0.3%", "13.8%", "17.1%"]': f"USE_SHARE = {use_shares!r}",
        'ax2.set_ylim(0, 455)': f"ax2.set_ylim(0, {max(455, max(v['classified_errors'] for v in systems) * 1.15)!r})",
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise RuntimeError(f'Original figure4 recipe changed: {old}')
        source = source.replace(old, new)
    # This is local, versioned plotting code, never document-supplied code.
    exec(compile(source, str(ROOT/'scripts/plot_mechanism_access_use.py'), 'exec'), {'__file__': str(ROOT/'scripts/plot_mechanism_access_use.py'), '__name__': '__unified_571_figure4__'})
    plt.close('all')


def figure5():
    summary = json.loads((ANALYSIS/'main_and_ablation.json').read_text())['ablation']
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.90), sharey=True)
    for ax, title, metric, bounds in [(axes[0], 'Accuracy', 'accuracy', (0.405, 0.450)),
                                       (axes[1], 'Macro-F1', 'macro_f1', (0.425, 0.475))]:
        values = [summary['full'][metric], *(summary[key][metric] for key in components.CONDITION_ORDER)]
        if min(values) < bounds[0] or max(values) > bounds[1]:
            padding = max(0.01, (max(values) - min(values)) * 0.25)
            bounds = (min(values) - padding, max(values) + padding)
        components.draw_panel(ax, title, summary['full'][metric],
            np.array([summary[key][metric] for key in components.CONDITION_ORDER]), bounds)
        for label,key in zip(ax.texts,components.CONDITION_ORDER):
            value=summary[key][metric]
            change=100*(value-summary['full'][metric])
            label.set_text(f'{value:.4f} ({change:+.1f})')
        ax.set_title(f"{title}\nFull = {summary['full'][metric]:.4f}", fontsize=7.0,
                     fontweight='bold', color=components.TEXT, pad=4)
    axes[0].invert_yaxis()
    axes[0].set_yticklabels(components.LABELS, fontsize=6.6, color=components.TEXT)
    axes[1].tick_params(axis='y', labelleft=False)
    fig.subplots_adjust(left=0.34, right=0.985, top=0.66, bottom=0.16, wspace=0.28)
    fig.savefig(OUT/'component_ablation_dumbbell.pdf', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(OUT/'component_ablation_dumbbell.png', dpi=260, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    figure3()
    figure4()
    figure5()
    print(f'Updated existing Figure 3/4/5 designs: {OUT}')


if __name__ == '__main__':
    main()

"""Render completed reviewed readout comparisons; no new evaluation or resampling."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry', type=Path, required=True)
    p.add_argument('--registry-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert sha(a.registry) == a.registry_sha256
    d = json.loads(a.registry.read_text())
    assert d['status'] == 'reviewed_completed_results_for_document_only'
    assert not d['new_AP_or_statistics_computed']
    for name, digest in d['files'].items():
        assert sha(ROOT / name) == digest, name
    constants = d['all_twelve_constant_contrasts']
    trained = d['all_eight_trained_readout_contrasts']
    assert len(constants) == 12 and len(trained) == 8
    c = {(x['modality'], x['metric'], x['baseline']): x for x in constants}
    t = {(x['modality'], x['metric'], x['baseline']): x for x in trained}
    m = d['readout_metrics']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'svg.hashsalt': 'CURE-V10-completed-figure4-v6'})
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.45),
                            gridspec_kw={'width_ratios': [1.05, 1.0, 1.35]})
    names = ['CURE', 'CURE-feature head', 'Task-feature head', 'Fit mean', 'Fit median', 'Zero']
    keys = ['original_saved_CURE_head', 'fresh_head_on_CURE_features', 'fresh_head_on_Task_features']
    baselines = ['fit_global_mean', 'fit_global_median', 'zero']
    colors = ['#b34938', '#55795d', '#35658d', '#71889b', '#526970', '#999999']
    absolute_rows = []
    for j, modality in enumerate(['camera_removal', 'radar_thinning']):
        ax = axes[j]
        for row, name in enumerate(names):
            values = [m[keys[row]][metric][j] if row < 3 else c[modality, metric, baselines[row-3]]['baseline_error']
                      for metric in ['RMSE', 'MAE']]
            absolute_rows.append({'modality': modality, 'readout': name, 'RMSE': values[0], 'MAE': values[1]})
            ax.scatter(values[0], row, s=40, marker='o', color=colors[row], zorder=3)
            ax.scatter(values[1], row, s=42, marker='s', facecolor='white', edgecolor=colors[row], linewidth=1.5, zorder=4)
            label = f'{values[0]:.2f}' if j == 0 else f'{values[0]:.4f}'
            ax.annotate(label, (values[0], row), xytext=(6, 0), textcoords='offset points',
                        fontsize=9.2, va='center', color=colors[row])
        ax.set_yticks(range(6)); ax.set_yticklabels(names if j == 0 else [], fontsize=10)
        ax.set_ylim(5.45, -.45)
        ax.grid(axis='x', color='#e5e5e5', lw=.8); ax.set_axisbelow(True)
        ax.set_title(['(a) Camera error', '(b) Radar error'][j], fontsize=13, pad=12)
        ax.set_xlabel('Teacher-loss units', fontsize=11)
        ax.tick_params(axis='y', length=0)
        if j == 0:
            ax.set_xlim(0, 14.4); ax.set_xticks([0, 4, 8, 12])
        else:
            ax.set_xlim(0, .036); ax.set_xticks([0, .01, .02, .03])
            ax.ticklabel_format(axis='x', style='plain')
    ax = axes[2]
    selections = [
        ('Task-feature\nhead', t['camera_removal', 'RMSE', 'trained_on_Task_features']['percent_error_reduction'], t['camera_removal', 'RMSE', 'trained_on_Task_features']['relative_interval']['percentile95'], colors[2]),
        ('Fit mean', c['camera_removal', 'RMSE', 'fit_global_mean']['relative_error_reduction_percent'], c['camera_removal', 'RMSE', 'fit_global_mean']['relative_reduction_bootstrap']['percentile_95'], colors[3]),
        ('CURE-feature\nhead', t['camera_removal', 'RMSE', 'trained_on_CURE_features']['percent_error_reduction'], t['camera_removal', 'RMSE', 'trained_on_CURE_features']['relative_interval']['percentile95'], colors[1]),
    ]
    for y, (name, value, interval, color) in enumerate(selections):
        lo, hi = interval
        ax.errorbar(value, y, xerr=np.array([[value-lo], [hi-value]]), fmt='o', color=color,
                    markersize=5, capsize=3, linewidth=1.5)
        ax.annotate(f'{value:.2f}%', (hi, y), xytext=(6, 0), textcoords='offset points', fontsize=10, va='center', color=color)
    ax.set_yticks(range(3)); ax.set_yticklabels([x[0] for x in selections], fontsize=10)
    ax.set_ylim(2.55, -.55); ax.set_xlim(-8, 76); ax.set_xticks([0, 20, 40, 60])
    ax.axvline(0, color='#555555', ls='--', lw=1)
    ax.grid(axis='x', color='#e5e5e5', lw=.8); ax.set_axisbelow(True); ax.tick_params(axis='y', length=0)
    ax.set_title('(c) CURE vs comparator', fontsize=13, pad=12)
    ax.set_xlabel('Camera RMSE reduction (%)', fontsize=11)
    handles = [Line2D([], [], color='#555555', marker='o', linestyle='none', label='RMSE (labels)'),
               Line2D([], [], color='#555555', marker='s', markerfacecolor='white', linestyle='none', label='MAE')]
    fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(.395, 1.0), fontsize=11)
    fig.subplots_adjust(left=.155, right=.985, top=.80, bottom=.25, wspace=.72)
    fig.text(.155, .105, '2,521 objects · 8 reused recordings · 2,000 post-hoc paired recording draws', fontsize=10.5, color='#555555')
    fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    for axis in axes:
        for label in axis.get_yticklabels():
            assert label.get_window_extent(renderer).x0 >= 0
    a.output.mkdir(parents=True, exist_ok=False)
    stem = a.output / 'fig4_completed_utility_comparison_v6'
    for ext in ['png', 'pdf', 'svg']:
        metadata = {'Creator': 'CURE completed reviewed utility comparisons'}
        if ext == 'svg': metadata['Date'] = None
        if ext == 'pdf': metadata.update(CreationDate=None, ModDate=None)
        fig.savefig(stem.with_suffix('.'+ext), dpi=240, metadata=metadata)
    plt.close(fig)
    data = dict(status='rendered_completed_reviewed_utility_comparisons_only',
                registry_sha256=a.registry_sha256, source_sha256=sha(Path(__file__)),
                absolute_rows=absolute_rows, all_12_constant_contrasts=constants,
                all_8_trained_readout_contrasts=trained, all_readout_metrics=m,
                interval_panel='Camera RMSE against both trained comparators and historical fitting mean; all 20 contrasts retained in source data and appendix',
                new_evaluation_or_resampling=False)
    stem.with_suffix('.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    with stem.with_suffix('.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['modality', 'metric', 'baseline', 'CURE_error', 'baseline_error', 'reduction_percent', 'pointwise_low_percent', 'pointwise_high_percent'])
        for x in constants:
            w.writerow([x['modality'], x['metric'], x['baseline'], x['CURE'], x['baseline_error'], x['relative_error_reduction_percent'], *x['relative_reduction_bootstrap']['percentile_95']])
        for x in trained:
            w.writerow([x['modality'], x['metric'], x['baseline'], x['CURE_error'], x['baseline_error'], x['percent_error_reduction'], *x['relative_interval']['percentile95']])
    print(json.dumps({'figure': str(stem), 'source_sha256': data['source_sha256']}))


if __name__ == '__main__':
    main()

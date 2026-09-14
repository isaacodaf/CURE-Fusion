"""Scientific plots from an already validated result model, never learned output generation."""
from pathlib import Path
import csv
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ARMS = ('frozen_unadapted_pretrained_reference', 'task_only', 'cure', 'modality_dropout')
LABELS = ('Unadapted task extension', 'Task-only', 'CURE', 'Modality dropout')
COLORS = ('#85858A', '#376A9F', '#B04A3D', '#48806B')
MARKERS = ('D', 's', 'o', '^')
CONDITIONS = ('clean', 'camera_removed', 'radar_removed', 'radar_thinned_fresh')
CONDITION_LABELS = ('Clean', 'Camera removed', 'Radar removed', 'Radar thinned')
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':8,
    'axes.labelsize':8, 'axes.titlesize':9, 'xtick.labelsize':7.5, 'ytick.labelsize':8,
    'legend.fontsize':7.5, 'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none', 'svg.hashsalt':'CURE_Fusion_V6_20260913',
    'axes.spines.top':False, 'axes.spines.right':False, 'axes.linewidth':.65,
    'savefig.facecolor':'white', 'figure.facecolor':'white'})



def export(fig, destination, name, rows, caption, synthetic=False):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if synthetic:
        fig.text(.5, .985, 'SYNTHETIC QA — NOT RESEARCH RESULTS', ha='center', va='top', color='#A02020', weight='bold', fontsize=9)
    for extension in ('pdf', 'svg', 'png'):
        kwargs = {'dpi':600} if extension == 'png' else {}
        if extension == 'pdf':
            kwargs['metadata'] = {'CreationDate':None, 'ModDate':None, 'Creator':'CURE evidence-bound scientific figure builder'}
        if extension == 'svg':
            kwargs['metadata'] = {'Date':None, 'Creator':'CURE evidence-bound scientific figure builder'}
        fig.savefig(destination / (name + '.' + extension), **kwargs)
    plt.close(fig)
    with (destination / (name + '.csv')).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (destination / (name + '.json')).write_text(json.dumps({'synthetic_QA_only':synthetic, 'rows':rows}, indent=2, allow_nan=False) + '\n')
    (destination / (name + '.caption.md')).write_text(caption + '\n')
    return dict(name=name, synthetic_QA_only=synthetic, caption=caption, records=len(rows))


def matched(model, destination, synthetic=False):
    if not synthetic:
        assert all(e['undefined_draws']==660 and e['valid_draws']==1340 for e in model['paired_CURE_minus_task_only'].values())
    by_key = {(r['arm'], r['condition']):r for r in model['endpoints']}
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.2), gridspec_kw={'width_ratios':[1.4, 1]}, sharey=True)
    fig.subplots_adjust(left=.175, right=.98, top=.79, bottom=.18, wspace=.22)
    left, right = axes
    offsets = [-.21, -.07, .07, .21]
    for ai, arm in enumerate(ARMS):
        for ci, condition in enumerate(CONDITIONS):
            value = by_key[arm, condition]['AP_percent']
            y = 3 - ci + offsets[ai]
            if value is None:
                left.text(99, y, 'unavailable', ha='right', va='center', fontsize=5.8, color=COLORS[ai])
            else:
                left.scatter(value, y, marker=MARKERS[ai], c=COLORS[ai], s=27, edgecolors='white', linewidth=.5, zorder=3)
        left.scatter([], [], marker=MARKERS[ai], c=COLORS[ai], s=27, label=LABELS[ai])
    left.set(yticks=[3, 2, 1, 0], yticklabels=CONDITION_LABELS, xlim=(-1, 101), ylim=(-.48, 3.48), xlabel='Four-class COCO AP (%)')
    left.set_title('(a) All declared endpoints', loc='left', pad=10)
    limits = [1.]
    for ci, condition in enumerate(CONDITIONS):
        entry = model['paired_CURE_minus_task_only'][condition]
        if entry['status'] == 'unavailable_original_metric_gate':
            right.text(.98, 3-ci, 'unavailable: geometry', transform=right.get_yaxis_transform(), ha='right', va='center', fontsize=6.2)
            continue
        delta = entry['paired_CURE_minus_task_only_pp']; limits.append(abs(delta))
        ci95 = entry['pointwise95_interval_pp']
        if ci95 is not None:
            limits.extend(abs(v) for v in ci95)
            # Explicit segment avoids assuming a percentile interval contains its point.
            right.plot(ci95, [3-ci]*2, color=COLORS[2], linewidth=1.35, zorder=2)
            right.plot(ci95, [3-ci]*2, '|', color=COLORS[2], markersize=5)
        else:
            right.annotate('CI withheld', (delta, 3-ci), xytext=(0, -10), textcoords='offset points', ha='center', fontsize=6.2)
        right.scatter(delta, 3-ci, c=COLORS[2], s=29, edgecolors='white', linewidth=.5, zorder=3)
    extent = max(2, math.ceil(max(limits)*1.15))
    right.set(xlim=(-extent, extent), xlabel='CURE − task-only (percentage points)')
    right.set_title('(b) Paired endpoint differences', loc='left', pad=10)
    right.axvline(0, color='#606060', linewidth=.75, linestyle='--', zorder=1)
    for axis in axes:
        axis.grid(axis='x', color='#E4E4E4', linewidth=.55, zorder=0)
        axis.tick_params(axis='y', length=0)
    fig.legend(*left.get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.55, .945), ncol=4, frameon=False, columnspacing=1.2, handletextpad=.4)
    fig.text(.175, .035, ('One fitted seed · 992 frames / 8 reused development sequences · 95% CIs withheld: 660/2000 unsupported draws' if all(e['pointwise95_interval_pp'] is None for e in model['paired_CURE_minus_task_only'].values()) else 'One fitted seed · 992 frames / 8 reused development sequences · pointwise descriptive intervals'), fontsize=7, color='#444444')
    rows = []
    for r in model['endpoints']:
        entry=model['paired_CURE_minus_task_only'][r['condition']]
        rows.append(dict(arm=r['arm'], condition=r['condition'], AP_percent=r['AP_percent'],
            delta_pp=entry.get('paired_CURE_minus_task_only_pp'),
            CI_low_pp=entry['pointwise95_interval_pp'][0] if entry['pointwise95_interval_pp'] else None,
            CI_high_pp=entry['pointwise95_interval_pp'][1] if entry['pointwise95_interval_pp'] else None,
            unavailable_reason=r['unavailable_reason'], synthetic_QA_only=synthetic))
    caption=('Four-class development detection and the added CURE contribution. (a) All four methods and all four declared conditions on the same 992 frames. '
        '(b) CURE-minus-task-only AP differences with descriptive pointwise 95% whole-sequence bootstrap intervals, conditional on one fitted seed and eight reused development sequences. '
        'Intervals are withheld if any resample lacks a task class; invalid selected geometry leaves AP unavailable. These intervals do not measure seed variability or establish adjusted significance. '
        'The unadapted task extension uses the adapted 82-channel ranking; it is not the earlier 80-channel, two-class camera reference.')
    if all(e['pointwise95_interval_pp'] is None for e in model['paired_CURE_minus_task_only'].values()):
        caption=caption.replace('with descriptive pointwise 95% whole-sequence bootstrap intervals', 'with all whole-sequence bootstrap intervals withheld').replace('Intervals are withheld if any resample lacks a task class;', 'At least one declared resample lacks a task class; the frozen policy withholds the complete interval;')
    return export(fig, destination, 'fig2_dfine_matched_v1', rows, caption, synthetic)


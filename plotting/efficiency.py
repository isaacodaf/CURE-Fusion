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


def efficiency(online, destination, synthetic=False):
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.2))
    fig.subplots_adjust(left=.25, right=.97, top=.83, bottom=.27, wspace=.32)
    labels=['Native camera\n(80 classes)', 'Task-only\n(82 classes)', 'CURE\n(82 classes)', 'Dropout\n(82 classes)']
    rows=[]
    for i, arm in enumerate(('native_camera', 'task_only', 'cure', 'modality_dropout')):
        r=online['results'][arm]; y=3-i
        axes[0].plot([r['median_ms'], r['p95_ms']], [y, y], color=COLORS[i], linewidth=1)
        axes[0].scatter(r['median_ms'], y, c=COLORS[i], marker='o', s=26, label='Median' if i==0 else None)
        axes[0].scatter(r['p95_ms'], y, facecolors='white', edgecolors=COLORS[i], marker='s', s=26, label='p95' if i==0 else None)
        allocated=r['peak_allocated_bytes']/2**30; reserved=r['peak_reserved_bytes']/2**30
        axes[1].plot([allocated, reserved], [y, y], color=COLORS[i], linewidth=1)
        axes[1].scatter(allocated, y, c=COLORS[i], marker='o', s=26, label='Allocated' if i==0 else None)
        axes[1].scatter(reserved, y, facecolors='white', edgecolors=COLORS[i], marker='s', s=26, label='Reserved' if i==0 else None)
        rows.append(dict(arm=arm, median_ms=r['median_ms'], p95_ms=r['p95_ms'], peak_allocated_GiB=allocated, peak_reserved_GiB=reserved,
            calls=r['measured_calls'], synthetic_QA_only=synthetic))
    for axis in axes:
        axis.set(ylim=(-.5,3.5)); axis.set_xlim(0,axis.get_xlim()[1]*1.04); axis.grid(axis='x', color='#E4E4E4', linewidth=.55)
        axis.legend(frameon=False, fontsize=7, loc='upper center', bbox_to_anchor=(.5,1.18), ncol=2)
    axes[0].set(yticks=[3,2,1,0], yticklabels=labels, xlabel='Tensor-to-output latency (ms)')
    axes[1].set(yticks=[3,2,1,0], yticklabels=['']*4, xlabel='Peak CUDA allocator memory (GiB)')
    fig.text(.10,.04,'CUDA FP32 · batch 1 · 32 fitting frames × 3 repeats · input transfer and image decode excluded',fontsize=7,color='#444444')
    caption=('Full-detector tensor-to-output execution cost on the declared A100 runtime. Each arm uses 16 warmups and 96 sequential measured calls over 32 fitting frames; points show descriptive median/p95 duration and peak allocated/reserved CUDA memory. '
        'The timed path includes D-FINE, feature capture, the adapter when present and native-style output selection, while excluding input transfer, image decode/resize and file I/O. '
        'Memory includes warmup/output-validation scratch and resident model state; it is not process memory or training memory. Native camera uses 80 classes, whereas the three adapted arms use 82; no cross-interface accuracy–latency point is asserted.')
    return export(fig,destination,'fig7_dfine_online_cost_v1',rows,caption,synthetic)


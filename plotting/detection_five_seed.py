"""Plot every verified adapter seed alongside the original single-seed endpoints."""
from pathlib import Path
import argparse
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--verified', type=Path, required=True)
p.add_argument('--original', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
v, original = json.loads(a.verified.read_text()), json.loads(a.original.read_text())
assert v['status'] == 'passed_independent_saved_artifact_replay'
arms = ['task_only', 'cure', 'modality_dropout']
names = ['Task-only', 'CURE', 'Dropout']
colors = ['#376A9F', '#B04A3D', '#48806B']
markers = ['s', 'o', '^']
conditions = ['clean', 'camera_removed', 'radar_removed', 'radar_thinned_fresh']
labels = ['Clean (primary)', 'Camera removed', 'Radar removed', 'Radar thinned']
old = {(r['arm'], r['condition']): r['AP_percent'] for r in original['endpoints']}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'pdf.fonttype': 42,
    'ps.fonttype': 42, 'svg.fonttype': 'none', 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.linewidth': .65, 'svg.hashsalt': 'cure-v15-five-seed'})
fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.05))
fig.subplots_adjust(left=.095, right=.97, bottom=.16, top=.84, wspace=.27, hspace=.65)
rows = []
for panel, (ax, condition, label) in enumerate(zip(axes.flat, conditions, labels)):
    values_all = []
    for j, arm in enumerate(arms):
        row = v['summary'][condition][arm]
        values = np.asarray(row['AP_percent'])
        x = j + np.linspace(-.10, .10, 5)
        ax.scatter(x, values, marker=markers[j], s=21, facecolors='white', edgecolors=colors[j], linewidths=.8, zorder=3)
        ax.errorbar(j + .19, row['mean_AP_percent'], yerr=row['sample_SD_AP_points'], fmt=markers[j],
            color=colors[j], markersize=5, capsize=2.5, linewidth=1.0, zorder=4)
        ax.scatter(j-.21, old[arm, condition], marker='x', s=24, color='#444444', linewidths=.9, zorder=3)
        values_all.extend(values.tolist() + [old[arm, condition]])
        for seed, ap in zip([11,23,37,53,71], values):
            rows.append(dict(study='five_adapter_seed_refit', seed=seed, arm=arm, condition=condition, AP_percent=float(ap)))
        rows.append(dict(study='original_single_seed', seed=11, arm=arm, condition=condition, AP_percent=old[arm,condition]))
    lo, hi = min(values_all), max(values_all)
    gap = max((hi-lo)*.22, .04 if condition=='camera_removed' else .18)
    ax.set_ylim(max(0,lo-gap), hi+gap)
    ax.set_xticks([0,1,2], names)
    ax.set_xlim(-.45, 2.42)
    ax.set_ylabel('COCO box AP (%)')
    ax.set_title('('+chr(97+panel)+') '+label, loc='left', fontsize=9, pad=9)
    ax.grid(axis='y', color='#E4E4E4', linewidth=.55)
    ax.tick_params(axis='x', length=0, pad=5)
handles = [plt.Line2D([],[],marker='o',color='#555555',markerfacecolor='white',linestyle='none',label='Every refit seed'),
    plt.Line2D([],[],marker='o',color='#555555',linestyle='none',label='Refit mean ± sample SD'),
    plt.Line2D([],[],marker='x',color='#444444',linestyle='none',label='Original single seed')]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.975),ncol=3,frameon=False,fontsize=7.5)
fig.text(.095,.07,'992 frames · 8 reused recordings · one frozen visual detector and archived teacher',fontsize=7,color='#444444')
fig.text(.095,.03,'Panel scales differ. SD measures adapter-seed variation; it is not a population confidence interval.',fontsize=7,color='#444444')
a.output.mkdir(parents=True,exist_ok=True)
for ext in ['pdf','svg','png']:
    metadata={'Creator':'CURE-Fusion scientific figure script'}
    if ext=='pdf': metadata.update(CreationDate=None,ModDate=None)
    if ext=='svg': metadata.update(Date=None)
    fig.savefig(a.output/('fig2_five_seed_v15.'+ext),dpi=600,metadata=metadata)
with (a.output/'fig2_five_seed_v15.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
(a.output/'fig2_five_seed_v15.json').write_text(json.dumps({'verified_source':a.verified.name,'rows':rows},indent=2)+'\n')

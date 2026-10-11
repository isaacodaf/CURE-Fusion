"""Class-level AP for every independently verified adapter-training seed."""
from pathlib import Path
import argparse, csv, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser()
p.add_argument('--verified',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();v=json.loads(a.verified.read_text())
assert v['status']=='passed_independent_saved_artifact_replay'
seeds=[11,23,37,53,71];arms=['task_only','cure','modality_dropout']
labels=['Task-only','CURE','Dropout'];classes=['person','bicycle','slidecar','doll']
colors=['#376A9F','#B04A3D','#48806B'];markers=['s','o','^']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42,
    'svg.fonttype':'none','svg.hashsalt':'cure-v15-classes','axes.spines.top':False,'axes.spines.right':False})
fig,(left,right)=plt.subplots(1,2,figsize=(7.16,3.15),gridspec_kw={'width_ratios':[1.55,1]})
fig.subplots_adjust(left=.10,right=.97,top=.78,bottom=.25,wspace=.33)
rows=[]
for i,arm in enumerate(arms):
    values=np.array([[v['endpoints'][str(s)][arm]['clean']['per_class'][c] for c in classes] for s in seeds])
    pos=np.arange(4)+(i-1)*.22
    left.errorbar(pos,values.mean(0),yerr=values.std(0,ddof=1),fmt=markers[i],color=colors[i],
        markersize=5,capsize=2.5,linewidth=.85,label=labels[i],zorder=4)
    for j in range(5):left.scatter(pos+.025*(j-2),values[j],s=12,marker=markers[i],facecolors='white',edgecolors=colors[i],linewidths=.55,zorder=3)
    for s,row in zip(seeds,values):
        for c,value in zip(classes,row):rows.append(dict(seed=s,arm=arm,condition='clean',class_name=c,AP_percent=float(value)))
delta=np.array([[v['endpoints'][str(s)]['cure']['clean']['per_class'][c]-v['endpoints'][str(s)]['task_only']['clean']['per_class'][c] for c in classes] for s in seeds])
right.errorbar(delta.mean(0),np.arange(4),xerr=delta.std(0,ddof=1),fmt='o',color=colors[1],markersize=5,capsize=2.5,linewidth=.9,zorder=4)
for j in range(5):right.scatter(delta[j],np.arange(4)+.045*(j-2),facecolors='white',edgecolors=colors[1],s=14,linewidths=.65,zorder=3)
right.axvline(0,color='#555555',ls='--',lw=.7)
left.set(xticks=range(4),xticklabels=['Person','Bicycle','Slidecar','Doll'],ylabel='COCO box AP (%)',ylim=(0,100))
right.set(yticks=range(4),yticklabels=['Person','Bicycle','Slidecar','Doll'],xlabel='CURE − task-only (AP points)')
right.invert_yaxis();left.set_title('(a) All four classes',loc='left',fontsize=10,pad=10)
right.set_title('(b) Paired seed differences',loc='left',fontsize=10,pad=10)
left.grid(axis='y',color='#E4E4E4',lw=.55);right.grid(axis='x',color='#E4E4E4',lw=.55)
fig.legend(*left.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.50,.98),ncol=3,frameon=False)
fig.text(.10,.075,'Filled: five-seed mean ± sample SD. Hollow: every adapter seed.',fontsize=8,color='#444444')
fig.text(.10,.028,'Clean condition · 992 reused frames · 2,521 objects · SD is not a population confidence interval.',fontsize=7.5,color='#444444')
a.output.mkdir(parents=True,exist_ok=True)
for ext in ['pdf','svg','png']:
    meta={'Creator':'CURE-Fusion scientific figure script'}
    if ext=='pdf':meta.update(CreationDate=None,ModDate=None)
    if ext=='svg':meta.update(Date=None)
    fig.savefig(a.output/('fig3_five_seed_classes_v15.'+ext),dpi=600,metadata=meta)
with (a.output/'fig3_five_seed_classes_v15.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(a.output/'fig3_five_seed_classes_v15.json').write_text(json.dumps({'rows':rows},indent=2)+'\n')

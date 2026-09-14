from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from . import detection as old
plt.rcParams['svg.hashsalt']='CURE_Fusion_V7_20260913'

def classes(model,destination,synthetic=False):
    assert model['class_order']==['person','bicycle','slidecar','doll']
    assert model['development_images']==992 and model['development_sequences']==8
    by={r['arm']:r for r in model['endpoints'] if r['condition']=='clean'}
    arms=old.ARMS[1:];fig,axes=plt.subplots(1,2,figsize=(7.16,3.1),gridspec_kw={'width_ratios':[1.6,1]},sharey=True)
    fig.subplots_adjust(left=.12,right=.965,top=.78,bottom=.21,wspace=.30)
    rows=[];y=np.arange(3,-1,-1)
    for k,arm in enumerate(arms,1):
        vals=np.array(by[arm]['per_class_percent'])
        assert vals.shape==(4,) and np.isfinite(vals).all() and ((vals>=0)&(vals<=100)).all()
        axes[0].scatter(vals,y+(k-2)*.18,s=31,marker=old.MARKERS[k],color=old.COLORS[k],edgecolors='white',linewidth=.45,label=old.LABELS[k],zorder=3)
        for i,name in enumerate(model['class_order']):
            rows.append(dict(arm=arm,condition='clean',class_name=name,AP_percent=float(vals[i]),CURE_minus_task_only_pp=float(by['cure']['per_class_percent'][i]-by['task_only']['per_class_percent'][i]),images=992,sequences=8,objects=2521,confidence_interval=None))
    delta=np.array(by['cure']['per_class_percent'])-np.array(by['task_only']['per_class_percent'])
    axes[1].hlines(y,0,delta,color=old.COLORS[2],linewidth=1.3)
    axes[1].scatter(delta,y,color=old.COLORS[2],marker='o',s=32,edgecolors='white',linewidth=.45,zorder=3)
    for i,d in enumerate(delta):
        axes[1].annotate(f'{d:+.2f}',(d,y[i]),xytext=(0,7),textcoords='offset points',ha='center',fontsize=7.5)
    axes[0].set(xlim=(0,100),xticks=[0,20,40,60,80,100],yticks=y,yticklabels=['Person','Bicycle','Slidecar','Doll'],ylim=(-.45,3.45),xlabel='COCO box AP (%)')
    axes[1].set(xlim=(-5.4,2),xticks=[-4,-2,0,2],xlabel='CURE − task-only (pp)')
    axes[1].axvline(0,color='#626262',lw=.7,ls='--')
    axes[0].set_title('(a) All four task classes',loc='left',pad=10)
    axes[1].set_title('(b) Matched AP differences',loc='left',pad=10)
    for a in axes:a.grid(axis='x',color='#E4E4E4',lw=.55,zorder=0);a.tick_params(axis='y',length=0)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.53,.965),ncol=3,frameon=False)
    fig.text(.12,.055,'Clean endpoint · one seed · 992 images / 8 reused sequences / 2,521 objects',fontsize=7,color='#444444')
    fig.text(.12,.014,'All classes retained; confidence intervals unavailable under the declared resampling policy.',fontsize=7,color='#444444')
    cap=('Current four-class detection localizes the control gap. (a) Clean AP for all task classes and all three identically initialized trained arms. (b) CURE-minus-task-only differences, without confidence intervals. Person and bicycle share pretrained class channels; slidecar and doll use the two task-specific outputs. The full evaluation comprises 992 images, eight reused development sequences and 2,521 objects. These are descriptive, single-seed endpoints; no class was selected by its outcome. All four sensor conditions and the unadapted reference remain in Table II and Appendix A.')
    return old.export(fig,destination,'fig3_current_classes_v1',rows,cap,synthetic)


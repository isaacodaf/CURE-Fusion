"""Camera and radar figures from recorded sensor observations."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image

BLUE = '#2166AC'
GREEN = '#14836F'
GRAY = '#596570'
INK = '#17212B'
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':8,
 'axes.labelsize':8, 'axes.titlesize':9, 'axes.titleweight':'bold',
 'axes.spines.top':False, 'axes.spines.right':False,
 'text.color':INK, 'axes.labelcolor':INK, 'pdf.fonttype':42,
 'ps.fonttype':42, 'svg.hashsalt':'cure-v11-sensor-pairs',
 'savefig.facecolor':'white'})

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(fig, out, stem):
    fig.canvas.draw()
    # Every free-standing label must remain inside the exported canvas.
    renderer = fig.canvas.get_renderer()
    texts=list(fig.texts)
    for ax in fig.axes:
        texts.extend(ax.texts)
        texts.extend([ax.title,ax._left_title,ax._right_title])
        if ax.axison:
            texts.extend([ax.xaxis.label,ax.yaxis.label])
            texts.extend(ax.get_xticklabels()+ax.get_yticklabels())
    for obj in texts:
        if not obj.get_visible() or not obj.get_text():
            continue
        box = obj.get_window_extent(renderer)
        if box.width and box.height:
            assert box.x0 >= -1 and box.y0 >= -1, obj.get_text()
            assert box.x1 <= fig.bbox.width + 1 and box.y1 <= fig.bbox.height + 1, obj.get_text()
    for ext in ('pdf', 'svg', 'png'):
        meta = {'CreationDate':None,'ModDate':None} if ext == 'pdf' else {'Date':None} if ext == 'svg' else {}
        fig.savefig(out/(stem+'.'+ext), dpi=600, metadata=meta)
    plt.close(fig)

def camera(ax, row, source, overlays=True):
    im = Image.open(source/row['image_file'])
    w,h = im.size
    assert sha(source/row['image_file']) == row['image_sha256']
    ax.imshow(im)
    ax.set(xlim=(0,w), ylim=(h,0))
    ax.set_axis_off()
    if overlays:
        for under in (True, False):
            for key,col,ls in [('gt_cxcywh',GREEN,'-'),('query_box_cxcywh',BLUE,'--')]:
                x,y,bw,bh = np.array(row[key])*[w,h,w,h]
                ax.add_patch(Rectangle((x-bw/2,y-bh/2),bw,bh,fill=False,
                    edgecolor='white' if under else col,
                    linewidth=2.9 if under else 1.4,linestyle=ls))
    return w,h

def figure5(rows, source, out):
    fig=plt.figure(figsize=(7.2,3.8))
    fig.text(.025,.97,'Current CURE outputs across the four task classes',size=11,weight='bold',va='top')
    fig.text(.025,.89,'Class-median target selection',size=8,color=GRAY,va='top')
    fig.legend(handles=[Line2D([0],[0],color=GREEN,lw=1.8,label='Selected annotation'),
        Line2D([0],[0],color=BLUE,lw=1.6,ls='--',label='Matched CURE query')],
        loc='upper right',bbox_to_anchor=(.985,.925),frameon=False,ncol=2,fontsize=8)
    grid=fig.add_gridspec(1,4,left=.025,right=.975,top=.755,bottom=.06,wspace=.15)
    for n,(row,cell) in enumerate(zip(rows,grid)):
        inner=cell.subgridspec(2,1,height_ratios=[1.3,1.3],hspace=.18)
        ax=fig.add_subplot(inner[0]);camera(ax,row,source)
        ax.set_title(f"{'ABCD'[n]}  {row['class_name']}\n{row['identity']}",loc='left',pad=6,fontsize=8.5)
        tx=fig.add_subplot(inner[1]);tx.set_axis_off()
        tx.text(0,.98,f"Query IoU {row['selected_query_iou']:.2f}",va='top',fontsize=8)
        tx.text(0,.79,f"Class score {row['class_score']:.2f}",va='top',fontsize=8)
        tx.text(0,.57,'Utility (loss units)',va='top',fontsize=8,color=GRAY)
        tx.text(0,.38,'Measured → predicted',va='top',fontsize=8,color=GRAY)
        tx.text(0,.20,f"Camera {row['camera_target']:.2f} → {row['camera_prediction']:.2f}",va='top',fontsize=8)
        def small(v):
            return f"{v:+.2g}".replace('e-0','e−').replace('e+0','e+')
        tx.text(0,.02,f"Radar {small(row['radar_target'])} → {small(row['radar_prediction'])}",va='top',fontsize=8)
    save(fig,out,'fig5_current_sensor_cases_v3')

def figure6(rows, source, out):
    audit=json.loads((source/'PAIR_AUDIT_V1.json').read_text())
    assert audit['all_returns_preserved'] and audit['total_returns']==64
    data=np.load(source/'paired_radar_inputs_v1.npz',allow_pickle=False)
    assert data['identities'].tolist()==[r['identity'] for r in rows]
    fig=plt.figure(figsize=(7.2,5.8))
    fig.text(.04,.983,'Paired camera and radar observations',size=11,weight='bold',va='top')
    fig.text(.04,.947,'The same four selected objects as Figure 5  |  Stored CURE outputs',size=8,color=GRAY,va='top')
    grid=fig.add_gridspec(2,2,left=.04,right=.975,top=.80,bottom=.19,wspace=.20,hspace=.64)
    cmap=plt.get_cmap('coolwarm')
    norm=matplotlib.colors.Normalize(vmin=-1.1,vmax=1.1)
    handles=[Line2D([0],[0],color=GREEN,lw=1.5,label='Annotation'),
             Line2D([0],[0],color=BLUE,lw=1.5,ls='--',label='CURE query'),
             Line2D([0],[0],marker='o',color='none',markerfacecolor='#C8CED4',
                        markeredgecolor=INK,markersize=4,label='Radar return')]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.027,.925),ncol=3,frameon=False,fontsize=8)
    for i,(r,cell) in enumerate(zip(rows,grid)):
        box=cell.get_position(fig)
        fig.text(box.x0,box.y1+.035,f"{'ABCD'[i]}  {r['class_name']}  |  {r['identity']}",size=9,weight='bold',va='bottom')
        inner=cell.subgridspec(1,2,width_ratios=[1.2,1],wspace=.29)
        ax=fig.add_subplot(inner[0]);w,h=camera(ax,r,source)
        ax.set_title('Camera + projected radar',size=7.5,weight='normal',pad=7)
        keep=data['radar_geometry_mask'][i]
        radar=data['radar'][i]
        xy=data['radar_xy'][i][keep]*[w,h]
        ax.scatter(xy[:,0],xy[:,1],c=radar[keep,4],cmap=cmap,norm=norm,
                   s=15,linewidths=.5,edgecolors=INK,zorder=5)
        bx=fig.add_subplot(inner[1])
        raw=data[f'raw_points_{i}']
        assert len(raw)==audit['cases'][i]['raw_return_count']
        bx.scatter(raw[:,1],raw[:,0],c=raw[:,4],cmap=cmap,norm=norm,
                   s=17,linewidths=.45,edgecolors=INK,zorder=3)
        bx.set(xlim=(-5.5,3.0),ylim=(-.4,8.5),xticks=[-4,0,2],yticks=[0,4,8])
        bx.set_aspect('equal',adjustable='box')
        bx.set_xlabel('Radar y (m)',fontsize=7.5,labelpad=2)
        bx.set_ylabel('Radar x (m)',fontsize=7.5,labelpad=1)
        bx.tick_params(labelsize=7,length=2,pad=2)
        bx.grid(alpha=.2,lw=.5)
        bx.set_title('Radar: all returns',size=7.5,weight='normal',pad=7)
        fig.text(box.x0,box.y0-.052,
           f"{len(raw)} returns retained  |  {int(keep.sum())} projected into the image",
           fontsize=7.7,color=GRAY,va='top')
    cbax=fig.add_axes([.23,.058,.54,.020])
    cb=fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cbax,
                   orientation='horizontal',ticks=[-1,0,1])
    cb.ax.tick_params(labelsize=7.5,length=2,pad=2)
    cb.set_label('Reported radial velocity (m/s) — one scale for all views',size=8,labelpad=3)
    save(fig,out,'fig6_paired_camera_radar_v1')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(exist_ok=False)
    rows=json.loads((args.source/'selected_objects.json').read_text())
    assert [x['identity'] for x in rows]==['train/020340','train/003936','train/020008','train/020145']
    figure5(rows,args.source,args.output)
    figure6(rows,args.source,args.output)
    (args.output/'FIGURE_BINDINGS.json').write_text(json.dumps({
       'result_changes':False,
       'inputs':{p.name:sha(p) for p in sorted(args.source.iterdir()) if p.is_file()},
       'exports':{p.name:sha(p) for p in sorted(args.output.iterdir()) if p.is_file()}
    },indent=2)+'\n')

if __name__=='__main__':
    main()

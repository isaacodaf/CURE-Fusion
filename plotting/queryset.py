"""Render reviewed richer matched controls, preserving all36 summary/116 result rows.

No model, metric evaluator or checkpoint loading. Source-bound saved AP only.
"""
import argparse, csv, hashlib, json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
METHODS=['task_only','cure','modality_dropout','confidence','positive_mean','fitting_RMS_margin','selector_utility','selector_zero','selector_permuted']
LABELS=['Task-only','CURE','Modality dropout','Confidence','Positive mean','Fitting RMS margin','Utility-input selector','Zero-input selector','Permuted-input selector']
CONDITIONS=['clean','camera_removed','radar_removed','radar_thinned_fresh']
NAMES=['Clean','Camera removed','Radar removed','Fresh radar thinning']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def need(v,m):
 if not v:raise ValueError(m)
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 summary=read(a.data/'summary.json');results=read(a.data/'results.json');need(len(summary)==36 and len(results)==116,'Full original rows')
 paths={'rich_review':a.data/'source.json'}
 index={(r['method'],r['condition']):r for r in summary}
 need(set(index)=={(m,c) for m in METHODS for c in CONDITIONS},'Nine rules/four conditions')
 for (m,c),r in index.items():
  rows=[z for z in results if z['method']==m and z['condition']==c]
  need(len(rows)==r['controller_runs'],'Run count')
  if m.startswith('selector_') or m in ['positive_mean','fitting_RMS_margin']:need([z['seed'] for z in rows]==[11,23,37,53,71],'All five seeds retained in order')
  for metric in ['AP_percent','AP50_percent','AP75_percent']:
   vals=[z[metric] for z in rows];d=r[metric];need(vals==d['all_values'],'Saved values')
   need(float(np.mean(vals))==d['mean'],'Summary mean');need((None if len(vals)==1 else float(np.std(vals,ddof=1)))==d['SD'],'Summary sample SD')
 a.output.mkdir(exist_ok=False)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'CURE-V10-figure6'})
 fig,axs=plt.subplots(1,4,figsize=(12.4,2.9),sharey=True,gridspec_kw={'wspace':.32})
 chosen=['selector_utility','selector_zero','selector_permuted'];colors=['#b34938','#526970','#71889b']
 for ax,c,title in zip(axs,CONDITIONS,NAMES):
  ranges=[]
  for i,(m,color) in enumerate(zip(chosen,colors)):
   d=index[m,c]['AP_percent'];vals=np.array(d['all_values']);ranges.extend(vals.tolist()+[d['mean']-d['SD'],d['mean']+d['SD']])
   ax.errorbar(d['mean'],i,xerr=d['SD'],fmt='o',color=color,ms=5,capsize=3,lw=1.4,zorder=3)
   ax.scatter(vals,np.full(5,i)+np.array([-.15,-.075,0,.075,.15]),s=15,facecolor='white',edgecolor=color,alpha=.8,zorder=4)
  ref=index['task_only',c]['AP_percent']['mean'];ranges.append(ref);ax.axvline(ref,color='#858585',ls='--',lw=1)
  span=max(ranges)-min(ranges);pad=max(span*.17,.001 if c=='camera_removed' else .025)
  ax.set_xlim(min(ranges)-pad,max(ranges)+pad);ax.set_title(title,fontsize=11,pad=10)
  ax.set_yticks(range(3));ax.set_yticklabels(['Utility input','Zero input','Permuted input']);ax.set_ylim(2.45,-.45);ax.tick_params(axis='y',length=0)
  ax.set_xlabel('COCO box AP (%)',fontsize=10);ax.grid(axis='x',color='#e5e5e5',lw=.7);ax.set_axisbelow(True);ax.locator_params(axis='x',nbins=4)
 fig.subplots_adjust(left=.13,right=.99,bottom=.24,top=.82)
 for ext in ['png','pdf','svg']:
  metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else {'Date':None} if ext=='svg' else None
  fig.savefig(a.output/('fig6_queryset_matched_v2.'+ext),dpi=220,metadata=metadata)
 plt.close(fig)
 dump(a.output/'fig6_queryset_matched_v2.json',{'all_36_summary_rows':summary,'all_116_result_rows':results,'main_panel_methods':chosen,'fixed_reference':'task_only','review_sha256':sha(paths['rich_review']),'source_sha256':sha(Path(__file__)),'AP_recomputed':False})
 with (a.output/'fig6_queryset_matched_v2.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['condition','method','controller_runs','AP_mean','AP_SD','all_AP_values'])
  for m in METHODS:
   for c in CONDITIONS:
    r=index[m,c];d=r['AP_percent'];w.writerow([c,m,r['controller_runs'],d['mean'],d['SD'],json.dumps(d['all_values'])])
 table=['| Rule | Clean | Camera removed | Radar removed | Fresh thinning |','|---|---:|---:|---:|---:|']
 for m,label in zip(METHODS,LABELS):
  cells=[]
  for c in CONDITIONS:
   d=index[m,c]['AP_percent'];cells.append(f"{d['mean']:.3f}"+(f" ± {d['SD']:.3f}" if d['SD'] is not None else ''))
  table.append('| '+label+' | '+' | '.join(cells)+' |')
 (a.output/'COMPLETE_QUERYSET_AP_TABLE_V1.md').write_text('\n'.join(table)+'\n')
 dump(a.output/'RENDER_RECEIPT_V1.json',{'status':'rendered_reviewed_saved_AP_only_pending_visual_review','all_summary_rows':36,'all_result_rows':116,'main_methods':3,'conditions':4,'review_sha256':sha(paths['rich_review']),'source_sha256':sha(Path(__file__))})
 print('Rendered all reviewed rows with three matched controls in the main panel')
if __name__=='__main__':main()

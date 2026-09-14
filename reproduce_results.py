"""Recompute saved utility statistics and regenerate scientific tables/figures.

No dataset extraction, learned forward, training, new AP evaluation or resampling
design selection. The recorded 2,000 paired draws are reproduced exactly.
"""
from pathlib import Path
import argparse,csv,hashlib,json,os,subprocess,sys,shutil
import numpy as np
from analysis import recording_math_v1 as trained
from analysis import constant_recording_math as constant
from analysis.utility_metrics import descriptive_metrics_v1
from export_tables import export_tables

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n')
def close(a,b):
 a,b=np.asarray(a,float),np.asarray(b,float)
 assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
 assert np.allclose(a,b,rtol=0,atol=1e-12),float(np.max(np.abs(a-b)))
def arrays(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def table(out,name,rows):
 write(out/(name+'.json'),rows)
 if not rows:return
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with (out/(name+'.csv')).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(list,dict)) else v for k,v in r.items()})

def utility(out):
 base=ROOT/'results/utility';registry=read(base/'numeric_registry.json')
 a=arrays(base/'head_assessment_arrays.npz');mask=a['object_mask'];truth=a['mean'][mask];u2=a['mean_sampling_variance'][mask]
 assert mask.dtype==bool and mask.shape[0]==992 and int(mask.sum())==2521
 keymap={'fresh_head_on_Task_features':'control_raw','fresh_head_on_CURE_features':'trained_CURE_feature_raw',
         'original_saved_CURE_head':'original_saved_CURE_raw','Task_head_without_utility_supervision':'frozen_task_raw'}
 reference=read(base/'head_assessment.json')['metrics'];rows=[]
 for name,key in keymap.items():
  m=descriptive_metrics_v1(a[key][mask],truth,u2)
  for metric in ['RMSE','MAE','bias','Gaussian_mean_NLL_without_additive_constant','central90_estimated_target_mean_coverage']:
   close(m[metric],reference[name][metric]);close(m[metric],registry['readout_metrics'][name][metric])
   for i,modality in enumerate(m['modality_order']):rows.append(dict(readout=name,modality=modality,metric=metric,value=m[metric][i],objects=2521))
 table(out,'utility_readouts',rows)
 a=arrays(base/'trained_object_inputs.npz');assert a['predictions'].shape==(2521,3,2)
 errors=a['predictions']-a['target'][:,None]
 contrasts,omissions,raw=trained.analyze(errors,a['group_index'])
 ref={r['contrast']:r for r in registry['all_eight_trained_readout_contrasts']}
 for r in contrasts:
  e=ref[r['contrast']]
  for k in ['CURE_error','baseline_error','CURE_minus_baseline','percent_error_reduction']:close(r[k],e[k])
  for k in ['difference_interval','relative_interval']:
   close(r[k]['percentile95'],e[k]['percentile95']);assert r[k]['undefined_draws']==e[k]['undefined_draws']==0
  selected=[x for x in omissions if x['contrast']==r['contrast']]
  assert sum(x['CURE_minus_baseline']<0 for x in selected)==e['omission_counts']['lower_CURE_error']
 table(out,'trained_readout_contrasts',contrasts);table(out,'trained_readout_recording_omissions',omissions)
 a=arrays(base/'constant_object_errors.npz');counts,sums=constant.recording_statistics(a['errors'],a['recording_index'],8)
 indices,weights=constant.paired_recording_draws(8,2000,9132026)
 assert np.array_equal(indices,raw['draw_indices']) and np.array_equal(weights,raw['multiplicities'])
 _,point=constant.pooled_metrics(np.ones((1,8),np.int64),counts,sums)
 _,boot=constant.pooled_metrics(weights,counts,sums)
 _,leave=constant.pooled_metrics(np.ones((8,8),np.int64)-np.eye(8,dtype=np.int64),counts,sums)
 pd,pr=constant.contrasts(point);bd,br=constant.contrasts(boot);ld,lr=constant.contrasts(leave)
 rows=[]
 for r in registry['all_twelve_constant_contrasts']:
  j=['fit_global_mean','fit_global_median','zero'].index(r['baseline']);m=['camera_removal','radar_thinning'].index(r['modality']);k=['RMSE','MAE'].index(r['metric'])
  close(point[0,0,m,k],r['CURE']);close(point[0,j+1,m,k],r['baseline_error'])
  close(pr[0,j,m,k],r['relative_error_reduction_percent'])
  close(constant.pointwise_interval(br[:,j,m,k])['percentile_95'],r['relative_reduction_bootstrap']['percentile_95'])
  close(constant.pointwise_interval(bd[:,j,m,k])['percentile_95'],r['difference_bootstrap']['percentile_95'])
  assert int((ld[:,j,m,k]<0).sum())==r['leave_one_out_lower_error_count']
  rows.append(r)
 table(out,'constant_readout_contrasts',rows)
 table(out,'constant_recording_omissions',read(base/'constant_omissions.json'))
 return {'readout_metric_cells':40,'trained_contrasts':8,'constant_contrasts':12,'recording_draws':2000,'objects':2521,'absolute_replay_tolerance':1e-12}

def saved_results(out):
 d=read(ROOT/'results/detection/endpoints.json');assert len(d['endpoints'])==16
 table(out,'detector_endpoints',d['endpoints'])
 rows=[]
 for e in d['endpoints']:
  close(e['AP_percent'],np.mean(e['per_class_percent']))
  for name,value in zip(d['class_order'],e['per_class_percent']):rows.append(dict(arm=e['arm'],condition=e['condition'],class_name=name,AP_percent=value))
 table(out,'detector_all_class_endpoints',rows)
 by={(x['arm'],x['condition']):x for x in d['endpoints']}
 for condition,r in d['paired_CURE_minus_task_only'].items():
  close(r['paired_CURE_minus_task_only_pp'],by['cure',condition]['AP_percent']-by['task_only',condition]['AP_percent'])
  assert r['undefined_draws']==660
 table(out,'detector_paired_differences',[dict(condition=k,**v) for k,v in d['paired_CURE_minus_task_only'].items()])
 for name in ['action','queryset']:
  summary=read(ROOT/f'results/{name}/summary.json');results=read(ROOT/f'results/{name}/results.json')
  assert len(summary)==36 and len(results)==116
  for r in summary:
   cells=[z for z in results if (z['method'],z['condition'])==(r['method'],r['condition'])]
   assert len(cells)==r['controller_runs']
   for metric in ['AP_percent','AP50_percent','AP75_percent']:
    values=[z[metric] for z in cells];assert values==r[metric]['all_values']
    close(np.mean(values),r[metric]['mean'])
    if len(values)>1:close(np.std(values,ddof=1),r[metric]['SD'])
  table(out,name+'_all_runs',results);table(out,name+'_summary',summary)
 table(out,'native_fitting_all_classes',read(ROOT/'results/native_fitting/class_results.json'))
 history=read(ROOT/'results/historical/paired_statistics.json');table(out,'historical_all_paired_statistics',history)
 selected=[r for r in history if r['condition']=='factual' and r['group'] in ['all','natural_adverse']]
 historical={}
 for r in selected:
  historical[r['candidate'],r['group']]=(r['candidate_mean_AP'],r['candidate_seed_SD'])
  historical[r['control'],r['group']]=(r['control_mean_AP'],r['control_seed_SD'])
 assert len(historical)==22
 table(out,'historical_five_seed_methods',[dict(method=m,population=g,AP_mean_percent=100*v[0],AP_sample_SD_percent=100*v[1],training_seeds=5) for (m,g),v in sorted(historical.items())])
 efficiency=read(ROOT/'results/efficiency/report.json');table(out,'online_efficiency',[dict(method=k,**v) for k,v in efficiency['results'].items()])
 sensitivity=read(ROOT/'results/sensitivity/results.json');table(out,'fixed_fitting_sensitivity',sensitivity['arms'])
 return {'detector_endpoints':16,'per_class_endpoints':64,'action_runs':116,'queryset_runs':116,'historical_methods':11,'AP_recomputed':False}

def figures(out):
 reference=read(ROOT/'results/figure_reference_hashes.json');out.mkdir()
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(out/'matplotlib_config'))
 commands={
 1:[sys.executable,'-m','plotting.architecture'],
 2:[sys.executable,'-c',"from pathlib import Path;import json;from plotting.detection import matched;matched(json.load(open('results/detection/endpoints.json')),Path(__import__('sys').argv[1]))"],
 3:[sys.executable,'-c',"from pathlib import Path;import json;from plotting.classes import classes;classes(json.load(open('results/detection/endpoints.json')),Path(__import__('sys').argv[1]))"],
 4:[sys.executable,'-m','plotting.utility','--registry','results/utility/numeric_registry.json','--registry-sha256',sha(ROOT/'results/utility/numeric_registry.json')],
 5:[sys.executable,'-m','plotting.sensors','--source','results/sensor_pairs'],
 7:[sys.executable,'-m','plotting.queryset','--data','results/queryset'],
 8:[sys.executable,'-c',"from pathlib import Path;import json;from plotting.efficiency import efficiency;efficiency(json.load(open('results/efficiency/report.json')),Path(__import__('sys').argv[1]))"]}
 for n,cmd in commands.items():
  dest=out/f'figure{n}'
  if n==1:dest.mkdir();env['CURE_FIGURE_OUTPUT']=str(dest)
  elif n in (2,3,8):cmd=cmd+[str(dest)]
  else:cmd=cmd+['--output',str(dest)]
  p=subprocess.run(cmd,cwd=ROOT,env=env,text=True,capture_output=True)
  (out/f'figure{n}.log').write_text(p.stdout+p.stderr)
  if p.returncode:raise RuntimeError('Figure '+str(n)+' failed; see output log')
 rows=[]
 for n,r in reference.items():
  dest=out/f'figure{5 if n=="6" else n}'
  for name,h in r.items():
   got=sha(dest/name);assert got==h,(name,got,h)
   rows.append(dict(figure=int(n),file=name,sha256=got))
 aliases=read(ROOT/'figures/index.json');published=out/'published';published.mkdir()
 for figure in aliases:
  dest=out/f"figure{5 if figure['number']==6 else figure['number']}"
  for item in figure['files'].values():
   source=dest/item['generator_filename'];assert sha(source)==sha(ROOT/item['path'])==item['sha256']
   shutil.copyfile(source,published/Path(item['path']).name)
 return rows

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--skip-figures',action='store_true');a=p.parse_args()
 manifest=read(ROOT/'MANIFEST_SHA256.json')
 for name,v in manifest.items():assert sha(ROOT/name)==v['sha256'] and (ROOT/name).stat().st_size==v['bytes'],name
 assert not a.output.exists(),'Choose a new output directory';a.output.mkdir(parents=True);tables=a.output/'tables';tables.mkdir()
 result={'utility_recomputation':utility(tables),'saved_result_checks':saved_results(tables),'figures':[]}
 published=export_tables(tables/'publication')
 for table in published:
  for name in table['files'].values():assert sha(tables/'publication'/name)==sha(ROOT/'tables'/name),name
 assert sha(tables/'publication/index.json')==sha(ROOT/'tables/index.json')
 raw=read(ROOT/'tables/raw/index.json')['files']
 for name,item in raw.items():assert sha(tables/Path(name).name)==sha(ROOT/name)==item['sha256'],name
 result['published_tables']={'tables':13,'rows':sum(t['rows'] for t in published),'all_finished_CSV_and_Markdown_views_exact':True,'full_precision_table_files_exact':len(raw)}
 if not a.skip_figures:result['figures']=figures(a.output.resolve()/'figures')
 result['status']='passed_saved_research_reproduction';result['training_executed']=False
 write(a.output/'REPRODUCTION.json',result);print(json.dumps(result))

if __name__=='__main__':main()

"""Check the completed matched study's saved utility and summary operands."""
from pathlib import Path
import json,statistics
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
SEEDS=[11,23,37,53,71]
ARMS=['task_only','cure','modality_dropout']
CONDITIONS=['clean','camera_removed','radar_removed','radar_thinned_fresh']
NAMES={'task_only':'Task-only','cure':'CURE','modality_dropout':'Modality dropout'}
def read(p):return json.loads(Path(p).read_text())
def close(a,b):
 a,b=np.asarray(a,float),np.asarray(b,float)
 assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
 assert np.allclose(a,b,rtol=0,atol=1e-12),(a,b)
def ms(x,d=2):return f'{statistics.mean(x):.{d}f} ± {statistics.stdev(x):.{d}f}'
def verified(root=ROOT):
 r=read(root/'results/matched_v3/VERIFIED_RESULTS.json')
 assert r['status']=='passed_independent_saved_artifact_replay'
 assert r['epochs_verified']==600 and r['endpoints_verified']==60
 return r

def verify(root=ROOT):
 r=verified(root)
 for condition in CONDITIONS:
  for arm in ARMS:
   x=[r['endpoints'][str(s)][arm][condition]['AP_percent'] for s in SEEDS]
   saved=r['summary'][condition][arm]
   close(x,saved['AP_percent']);close(statistics.mean(x),saved['mean_AP_percent'])
   close(statistics.stdev(x),saved['sample_SD_AP_points'])
  x=[r['endpoints'][str(s)]['cure'][condition]['AP_percent']-r['endpoints'][str(s)]['task_only'][condition]['AP_percent'] for s in SEEDS]
  saved=r['summary'][condition]['cure_minus_task_only']
  close(x,saved['paired_AP_points']);close(statistics.mean(x),saved['mean_AP_points']);close(statistics.stdev(x),saved['sample_SD_AP_points'])
 for seed in SEEDS:
  rows=read(root/f'results/matched_v3/utility_seed_{seed}.json');assert len(rows)==2521
  pred=np.asarray([x['mean'] for x in rows]);truth=np.asarray([x['reference_mean'] for x in rows]);u2=np.asarray([x['reference_mean_sampling_variance'] for x in rows])
  lv=np.clip(np.asarray([x['logvar'] for x in rows]),-8,5);variance=np.exp(lv)+u2;error=pred-truth
  values={'RMSE':np.sqrt((error**2).mean(0)), 'MAE':np.abs(error).mean(0),'bias':error.mean(0),
          'Gaussian_mean_NLL':(.5*(error**2/variance+np.log(variance))).mean(0),
          'central90_coverage':(np.abs(error)<=1.6448536269514722*np.sqrt(variance)).mean(0)}
  for k,v in values.items():close(v,r['utility'][str(seed)]['cure'][k])
 return dict(status='passed_saved_utility_and_summary_checks',seeds=SEEDS,AP_endpoints=60,objects_per_CURE_seed=2521,
             raw_COCO_AP_recomputed_in_this_command=False,
             prior_independent_verification=r['verification_scope'],independent_population_validation=False)

def table_rows(root,original):
 root=Path(root);r=verified(root);by={(x['arm'],x['condition']):x for x in read(root/'results/detection/endpoints.json')['endpoints']}
 rows2=[[NAMES[a]]+[ms([r['endpoints'][str(s)][a][c]['AP_percent'] for s in SEEDS],3 if c=='camera_removed' else 2) for c in CONDITIONS] for a in ARMS]
 rows2.append(['CURE − task-only (AP points)']+[ms(r['summary'][c]['cure_minus_task_only']['paired_AP_points'],3 if c=='camera_removed' else 2) for c in CONDITIONS])
 result=dict(original);result[1]=[list(x) for x in result[1]];result[1][-1]=['Fitted model','Five adapter seeds, 40 epochs per arm','Fixed endpoints, no selection'];result[2]=rows2
 result[14]=[[label]+[f"{by[a,c]['AP_percent']:.3f}" for c in CONDITIONS] for label,a in [('Unadapted extension','frozen_unadapted_pretrained_reference'),*[(NAMES[a],a) for a in ARMS]]]
 result[15]=[[str(s),NAMES[a]]+[f"{r['endpoints'][str(s)][a][c]['AP_percent']:.3f}" for c in CONDITIONS] for s in SEEDS for a in ARMS]
 labels={'clean':'Clean','camera_removed':'Camera removed','radar_removed':'Radar removed','radar_thinned_fresh':'Radar thinned'}
 result[16]=[[NAMES[a],labels[c]]+[ms([r['endpoints'][str(s)][a][c]['per_class'][cl] for s in SEEDS],3 if c=='camera_removed' else 2) for cl in ['person','bicycle','slidecar','doll']] for a in ARMS for c in CONDITIONS]
 result[17]=[[label]+[ms([factor*r['utility'][str(s)]['cure'][metric][m] for s in SEEDS],d) for m in range(2)] for metric,label,d,factor in [('RMSE','RMSE',4,1),('MAE','MAE',4,1),('bias','Prediction − target bias',5,1),('Gaussian_mean_NLL','Gaussian mean NLL',4,1),('central90_coverage','Central 90% coverage (%)',2,100)]]
 diags=read(root/'results/matched_v3/fitting_diagnostics.json')
 result[18]=[[str(s),'0 / 258',*[f"{diags[str(s)]['matched_foreground']['median_relative_tanh_slope'][m]:.6f}" for m in range(2)]] for s in SEEDS]
 return result

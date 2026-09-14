"""Generate publication table values from the included scientific operands."""
from pathlib import Path
import json

def read(p):return json.loads(Path(p).read_text())

def table_rows(root,tables):
    root=Path(root)
    d=read(root/"results/native_fitting/paired_2d_gates.json")
    data=read(root/'results/detection/endpoints.json');by={(x['arm'],x['condition']):x for x in data['endpoints']}
    arms=['frozen_unadapted_pretrained_reference','task_only','cure','modality_dropout'];conditions=['clean','camera_removed','radar_removed','radar_thinned_fresh']
    derived={}
    # Derive each numeric table from included full-precision sources and demand
    # equality to every independently authored cell, including display rounding.
    derived[2]=[[label,*[f"{by[arm,c]['AP_percent']:.2f}" for c in conditions]] for label,arm in zip(['Unadapted extension','Task-only','CURE','Modality dropout'],arms)]
    ef=read(root/'results/efficiency/report.json')['results']
    derived[3]=[[label,f"{ef[arm]['median_ms']:.2f} / {ef[arm]['p95_ms']:.2f}",f"{ef[arm]['peak_allocated_bytes']/2**30:.2f} / {ef[arm]['peak_reserved_bytes']/2**30:.2f}"] for label,arm in zip(['Native camera (80 classes)','Task-only (82 classes)','CURE (82 classes)','Dropout (82 classes)'],['native_camera','task_only','cure','modality_dropout'])]
    f=lambda x:f'{x:.2e}' if 0<x<.0001 else f'{x:.2f}'
    derived[4]=[[label+', '+c.replace('_fresh','').replace('_',' '),*[f(x) for x in by[arm,c]['per_class_percent']]] for label,arm in zip(['Reference','Task-only','CURE','Dropout'],arms) for c in conditions]
    reg=read(root/'results/utility/numeric_registry.json');constants={(x['baseline'],x['modality'],x['metric']):x for x in reg['all_twelve_constant_contrasts']}
    modalities=['camera_removal','radar_thinning']
    pair=lambda rm,ma,m:f'{rm:.4f} / {ma:.4f}' if m==0 else f'{rm:.5f} / {ma:.5f}'
    derived[5]=[]
    for label,baseline in [('CURE',None),('Global mean','fit_global_mean'),('Global median','fit_global_median'),('Zero','zero')]:
     vals=[]
     for m,mod in enumerate(modalities):
      rows=[constants[baseline or 'fit_global_mean',mod,metric] for metric in ['RMSE','MAE']]
      vals.append(pair(*[x['baseline_error' if baseline else 'CURE'] for x in rows],m))
     derived[5].append([label,*vals,'Unfitted' if baseline=='zero' else 'Optimization only'])
    derived[6]=[]
    for metric in ['RMSE','MAE']:
     for label,baseline in [('fit mean','fit_global_mean'),('fit median','fit_global_median'),('zero','zero')]:
      cells=[]
      for mod in modalities:
       x=constants[baseline,mod,metric];lo,hi=x['relative_reduction_bootstrap']['percentile_95'];cells.append(f"{x['relative_error_reduction_percent']:.2f} [{lo:.2f}, {hi:.2f}]")
      derived[6].append([metric+' / '+label,*cells])
    derived[7]=[]
    for label,key in [('Original CURE head','original_saved_CURE_head'),('Fresh CURE-feature head','fresh_head_on_CURE_features'),('Fresh Task-feature head','fresh_head_on_Task_features')]:
     x=reg['readout_metrics'][key];cov=x['central90_estimated_target_mean_coverage'];derived[7].append([label,*[pair(x['RMSE'][m],x['MAE'][m],m) for m in range(2)],f'{100*cov[0]:.2f} / {100*cov[1]:.2f}'])
    tr={(x['baseline'],x['modality'],x['metric']):x for x in reg['all_eight_trained_readout_contrasts']};derived[8]=[]
    for label,mod in [('Camera','camera_removal'),('Radar','radar_thinning')]:
     for metric in ['RMSE','MAE']:
      cells=[]
      for baseline in ['trained_on_CURE_features','trained_on_Task_features']:
       x=tr[baseline,mod,metric];lo,hi=x['relative_interval']['percentile95'];cells.append(f"{x['percent_error_reduction']:.2f} [{lo:.2f}, {hi:.2f}]; {x['omission_counts']['lower_CURE_error']}/8")
      derived[8].append([label+' '+metric,*cells])
    derived[9]=[[str(x['size']),str(x['updates']),f"{x['GT']} / {x['supported_classes']}",x['camera_present_gate'],x['camera_absent_gate']] for x in d['rows']]
    history=read(root/'results/historical/paired_statistics.json');values={}
    for x in history:
     if x['condition']=='factual' and x['group'] in ['all','natural_adverse']:
      for a in ['candidate','control']:values[x[a],x['group']]=(100*x[a+'_mean_AP'],100*x[a+'_seed_SD'])
    methods=['task_only_residual_controller','camera_only','radar_only','concat_fusion','cross_attention','matched_corruption_fusion','modality_dropout','confidence_gate','cure_without_uncertainty','cure_without_signed_negative','cure_without_temporal']
    derived[10]=[[row[0],*[f'{values[m,g][0]:.2f} ± {values[m,g][1]:.2f}' for g in ['all','natural_adverse']]] for row,m in zip(tables[9]['rows'],methods)]
    sensitivity={x['arm']:x for x in read(root/'results/sensitivity/results.json')['arms']};derived[11]=[]
    for row,arm in zip(tables[10]['rows'],['frozen_prediction','training_constant','paired_shuffled_prediction','observed_same_estimand_target']):
     cells=[]
     for prefix in ['original','camera_corrected']:
      x=sensitivity[prefix+'__'+arm];cells.append(f"{x['AP']['AP_percent']:.4f} / {x['task_risk']['mean_frame_risk']:.6f}")
     derived[11].append([row[0],*cells])
    actionmethods=['task_only','cure','modality_dropout','confidence','positive_mean','fitting_RMS_margin','selector_utility','selector_zero','selector_permuted']
    for number,name in [(12,'action'),(13,'queryset')]:
     summary={(x['method'],x['condition']):x['AP_percent'] for x in read(root/f'results/{name}/summary.json')};derived[number]=[]
     for row,m in zip(tables[number-1]['rows'],actionmethods):
      cells=[]
      for c in ['clean','camera_removed','radar_removed','radar_thinned_fresh']:
       x=summary[m,c];cells.append(f"{x['mean']:.3f}"+(f" ± {x['SD']:.3f}" if x['SD'] is not None else ''))
      derived[number].append([row[0],*cells])
    split=read(root/'configs/sew_splits.json')
    for n,role in [(1,'fit'),(2,'inner_development')]:
     assert tables[0]['rows'][0][n]==f"{split[role]['images']:,} / {len(split[role]['sequences'])}"
     assert tables[0]['rows'][1][n]==f"{sum(split[role]['class_counts']):,}"
    for n,rows in derived.items():assert rows==tables[n-1]['rows'],(n,rows,tables[n-1]['rows'])
    derived[1]=[list(row) for row in tables[0]["rows"]]
    for column,role in [(1,"fit"),(2,"inner_development")]:
        derived[1][0][column]=f"{split[role]['images']:,} / {len(split[role]['sequences'])}"
        derived[1][1][column]=f"{sum(split[role]['class_counts']):,}"
    return derived

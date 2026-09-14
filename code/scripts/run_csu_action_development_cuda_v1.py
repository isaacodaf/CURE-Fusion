"""Frozen five-controller-seed study on reused SEW data; one detector seed.

Capture, fit every declared model, and only then calculate development AP.
Outputs preserve all arms, conditions and seeds; no checkpoint selection.
"""
import argparse
import contextlib
import io
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
import numpy as np
import torch
from cure_fusion.dfine_cure_v1 import DfineCureConfigV1,require
from cure_fusion.dfine_cure_v2 import DfineNativeCureV2,TASK_CHANNELS_V2,task_top300_v2,task_label_map_v2
from cure_fusion.dfine_cure_development_v2 import Cache,sha,write,state_digest,thinning_mask,epoch_order
from cure_fusion.csu_action_v1 import action_gain_targets_v1,action_features_v1,FrameActionUtilityV1
from cure_fusion.csu_action_rules_v1 import (condition_average,fitting_normalization,rule_inputs,
    FrameActionRuleV1,expected_action_risk,sequence_risk_interval)

ROLES=('controller_fit','inner_calibration','reused_development')
ARMS=('task_only','cure','modality_dropout')
CONDITIONS=('clean','camera_removed','radar_removed','radar_thinned_fresh')


def cpu(value):
    return value.detach().cpu().numpy()


def predict_conditions(model, features):
    require(features.ndim==3 and features.shape[1]==4,'Four conditions per source frame required')
    return model(features.flatten(0,1)).reshape(features.shape[:2])


def capture(cache,models,roles,out,p):
    """Save real raw action outputs and complete target carriers, without AP."""
    summaries={}
    for role in ROLES:
        ids=roles[role]['sample_ids']; destination=out/role; destination.mkdir()
        chunks=[]; values={k:[] for k in ('features','gain','base_risk','candidate_risk')}
        invalid={arm:0 for arm in ARMS}
        for start in range(0,len(ids),16):
            batch=ids[start:start+16]; inputs,targets=cache.batch(batch)
            raw=dict(identities=np.asarray(batch),sequences=np.asarray([cache.records[s]['sequence'] for s in batch]))
            width=max(1,max(len(t['labels']) for t in targets))
            raw['target_labels']=np.full((len(batch),width),-1,np.int64)
            raw['target_boxes']=np.zeros((len(batch),width,4),np.float32)
            raw['target_mask']=np.zeros((len(batch),width),bool)
            for j,t in enumerate(targets):
                n=len(t['labels']);raw['target_labels'][j,:n]=cpu(t['labels']);raw['target_boxes'][j,:n]=cpu(t['boxes']);raw['target_mask'][j,:n]=True
            condition_values={k:[] for k in values}
            for condition in CONDITIONS:
                x=dict(inputs)
                if condition=='camera_removed':x['camera_present']=torch.zeros_like(x['camera_present'])
                if condition=='radar_removed':x['radar_mask']=torch.zeros_like(x['radar_mask'])
                if condition=='radar_thinned_fresh':
                    mask=thinning_mask(cache.raw_mask(batch),p['capture_seed'],'csu_action_development_v1_fresh',batch,0)
                    x['radar_mask']=torch.from_numpy(mask).cuda()
                raw[condition+'_radar_mask']=cpu(x['radar_mask'])
                outputs={}
                with torch.no_grad():
                    for arm,model in models.items():
                        if arm=='modality_dropout' and role!='reused_development':continue
                        output=model(**x)
                        require(output['pred_logits'].is_cuda,'A100 neural forward required')
                        labels,boxes,scores=task_top300_v2(output,torch.tensor([[640,512]]*len(batch),device='cuda'))
                        labels=task_label_map_v2(labels)
                        prefix=condition+'_'+arm+'_'
                        # Full82 logits preserve exact globaltop300 replay; no filtering.
                        for key in ('pred_logits','pred_boxes'):raw[prefix+key]=cpu(output[key])
                        raw[prefix+'labels']=cpu(labels);raw[prefix+'boxes']=cpu(boxes);raw[prefix+'scores']=cpu(scores)
                        selected=(labels>=0)&(labels<4)
                        invalid[arm]+=int((~((boxes[...,2:]-boxes[...,:2])>0).all(-1)&selected).sum())
                        require(bool(torch.isfinite(boxes).all()) and bool(torch.isfinite(scores).all()),'Nonfinite native output')
                        outputs[arm]=dict(pred_logits=output['pred_logits'][:,:,TASK_CHANNELS_V2],pred_boxes=output['pred_boxes'])
                    action=action_gain_targets_v1(outputs['task_only'],outputs['cure'],targets)
                    features=action_features_v1(outputs['task_only'],outputs['cure'])
                raw[condition+'_features']=cpu(features)
                condition_values['features'].append(cpu(features))
                for k in ('gain','base_risk','candidate_risk'):
                    raw[condition+'_'+k]=cpu(action[k]);condition_values[k].append(cpu(action[k]))
                for name in ('base_assignment','candidate_assignment'):
                    for j,a in enumerate(action[name]):
                        for key in ('queries','objects','focal','L1','GIoU'):
                            raw[f'{condition}_{name}_{j}_{key}']=cpu(a[key])
            file=destination/f'{start:06d}.npz';np.savez_compressed(file,**raw)
            chunks.append(dict(path=str(file.relative_to(out)),sha256=sha(file),identities=batch))
            for k in values:values[k].append(np.stack(condition_values[k],1))
            if start%256==0:print(json.dumps(dict(stage='capture',role=role,frames=start+len(batch),total=len(ids))),flush=True)
        compact={k:np.concatenate(v) for k,v in values.items()}
        compact.update(identities=np.asarray(ids),sequences=np.asarray([cache.records[s]['sequence'] for s in ids]))
        np.savez_compressed(out/(role+'_operands.npz'),**compact)
        summaries[role]=dict(frames=len(ids),chunks=chunks,invalid_selected_boxes=invalid)
        write(out/'CAPTURE.json',summaries)
        require(all(v==0 for v in invalid.values()),'Invalid selected boxes retained; this candidate cannot proceed to AP')
    return summaries


def fit_model(model,inputs,gain,base,candidate,seed,kind,p,destination,gain_scale):
    optimizer=torch.optim.AdamW(model.parameters(),lr=p['optimizer']['lr'],weight_decay=p['optimizer']['weight_decay'])
    logs=[];steps=0;model.train()
    initial=state_digest(model)
    for epoch in range(1,p['optimizer']['epochs']+1):
        order=epoch_order(len(inputs),seed,epoch);seen=[];total=0.
        for start in range(0,len(order),64):
            idx=order[start:start+64];seen.extend(idx.tolist())
            optimizer.zero_grad(set_to_none=True)
            prediction=predict_conditions(model,inputs[idx])
            if kind=='predictor':loss=condition_average((prediction-gain[idx]).square())/gain_scale.square()
            else:loss=expected_action_risk(prediction,base[idx],candidate[idx])
            require(bool(torch.isfinite(loss)),'Nonfinite training loss')
            loss.backward()
            require(all(v.grad is not None and bool(torch.isfinite(v.grad).all()) for v in model.parameters()),'Missing/nonfinite model gradients')
            optimizer.step();steps+=1;total+=float(loss.detach())*len(idx)
        require(len(seen)==len(set(seen))==len(inputs),'Full unique epoch coverage required')
        require(all(bool(torch.isfinite(v).all()) for v in model.parameters()),'Nonfinite trained parameters')
        logs.append(dict(epoch=epoch,frames=len(seen),steps=steps,mean_loss=total/len(inputs)))
    destination.mkdir();torch.save(model.state_dict(),destination/'FINAL.pt')
    torch.save(optimizer.state_dict(),destination/'OPTIMIZER.pt')
    write(destination/'TRAINING.json',dict(seed=seed,kind=kind,initial_state_sha256=initial,final_state_sha256=state_digest(model),epochs=logs,steps=steps,
        parameter_count=sum(v.numel() for v in model.parameters())))
    model.eval().requires_grad_(False)
    return model


def train_all(out,p):
    operands={}
    for role in ROLES:
        with np.load(out/'capture'/(role+'_operands.npz')) as z:
            operands[role]={k:torch.from_numpy(z[k].copy()).cuda() for k in ('features','gain','base_risk','candidate_risk')}
    fit=operands['controller_fit'];center,scale,gain_scale=fitting_normalization(fit['features'],fit['gain'])
    np.savez_compressed(out/'NORMALIZATION.npz',center=cpu(center),scale=cpu(scale),gain_scale=cpu(gain_scale))
    for seed in p['controller_seeds']:
        folder=out/f'seed_{seed}';folder.mkdir()
        torch.manual_seed(seed)
        predictor=FrameActionUtilityV1(center,scale).cuda()
        predictor=fit_model(predictor,fit['features'],fit['gain'],fit['base_risk'],fit['candidate_risk'],seed,'predictor',p,folder/'predictor',gain_scale)
        predictor_hash=state_digest(predictor)
        with torch.no_grad():means={r:predict_conditions(predictor,v['features']) for r,v in operands.items()}
        rms=condition_average((means['controller_fit']-fit['gain']).square()).sqrt()
        choices={r:dict(mean=cpu(m),positive_mean=cpu(m>0),fitting_RMS_margin=cpu(m-rms>0)) for r,m in means.items()}
        cal=operands['inner_calibration'];residual=(means['inner_calibration']-cal['gain']).abs()
        with np.load(out/'capture/inner_calibration_operands.npz') as z:cal_sequences=z['sequences']
        sequence_max=np.asarray([float(residual[np.where(cal_sequences==s)[0]].max()) for s in sorted(set(cal_sequences.tolist()))])
        # Nine reused calibration recordings: rank ceil((9+1)*.9)=9.
        # This is an empirical action-gain band, not latent removal CSU coverage.
        radius=float(sequence_max.max())
        np.savez_compressed(folder/'EMPIRICAL_CALIBRATION.npz',sequence_max=sequence_max,radius=np.asarray(radius),fitting_rms=cpu(rms))
        initial_rule=None
        for mode in ('utility','zero','permuted'):
            torch.manual_seed(seed+1000)
            selector=FrameActionRuleV1().cuda()
            digest=state_digest(selector)
            if initial_rule is None:initial_rule=digest
            require(digest==initial_rule,'Matched selector initialization differs')
            xx={r:rule_inputs(v['features'],means[r],center,scale,gain_scale,mode,seed,r) for r,v in operands.items()}
            selector=fit_model(selector,xx['controller_fit'],fit['gain'],fit['base_risk'],fit['candidate_risk'],seed,'selector_'+mode,p,folder/('selector_'+mode),gain_scale)
            with torch.no_grad():
                for r,x in xx.items():choices[r]['selector_'+mode]=cpu(predict_conditions(selector,x)>0)
            require(state_digest(predictor)==predictor_hash,'Fixed predictor changed during rule training')
        for r,v in operands.items():
            choices[r]['gain']=cpu(v['gain']);choices[r]['empirical_band_covers']=cpu((means[r]-v['gain']).abs()<=radius)
            np.savez_compressed(folder/(r+'_CHOICES.npz'),**choices[r])
        print(json.dumps(dict(stage='all_models_for_seed_finished',seed=seed)),flush=True)
    write(out/'ALL_TRAINING_COMPLETE.json',dict(seeds=p['controller_seeds'],AP_calculated=False,detector_seeds=1,
        source_frames=5050,epochs=40,steps_per_model=3160,learned_models_per_seed=4))


def build_gt(records,ids):
    annotations=[];images=[]
    for i,s in enumerate(ids,1):
        images.append(dict(id=i,width=640,height=512,file_name=s))
        target=records[s]['target']
        for label,box in zip(target['labels'],target['boxes']):
            cx,cy,w,h=np.asarray(box,dtype=float)*[640,512,640,512]
            annotations.append(dict(id=len(annotations)+1,image_id=i,category_id=label+1,bbox=[cx-w/2,cy-h/2,w,h],area=w*h,iscrowd=0))
    return dict(info={},images=images,annotations=annotations,categories=[dict(id=i+1,name=n) for i,n in enumerate(('person','bicycle','slidecar','doll'))])


def ap_result(gt,selected):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    predictions=[]
    for i,(labels,boxes,scores) in enumerate(selected,1):
        for label,box,score in zip(labels,boxes,scores):
            if int(label) not in (0,1,2,3):continue
            x1,y1,x2,y2=box.astype(float)
            require(x2>x1 and y2>y1,'Invalid selected geometry; no evaluator filtering')
            x1,x2=np.clip([x1,x2],0,640);y1,y2=np.clip([y1,y2],0,512)
            predictions.append(dict(image_id=i,category_id=int(label)+1,bbox=[x1,y1,x2-x1,y2-y1],score=float(score)))
    with contextlib.redirect_stdout(io.StringIO()):
        G=COCO();G.dataset=gt;G.createIndex()
        if predictions:D=G.loadRes(predictions)
        else:D=COCO();D.dataset=dict(images=gt['images'],categories=gt['categories'],annotations=[]);D.createIndex()
        e=COCOeval(G,D,'bbox');e.params.imgIds=[i['id'] for i in gt['images']];e.params.catIds=[1,2,3,4]
        e.evaluate();e.accumulate();e.summarize()
    precision=e.eval['precision']
    return dict(AP_percent=float(e.stats[0]*100),AP50_percent=float(e.stats[1]*100),AP75_percent=float(e.stats[2]*100),
        per_class_AP_percent=[float(precision[:,:,j,0,-1][precision[:,:,j,0,-1]>-1].mean()*100) for j in range(4)])


def evaluate_all(out,p,roles,records,captured):
    require((out/'ALL_TRAINING_COMPLETE.json').exists(),'All five seeds must finish before AP')
    ids=roles['reused_development']['sample_ids'];seq=[records[s]['sequence'] for s in ids]
    gt=build_gt(records,ids);write(out/'GROUND_TRUTH_COCO.json',gt)
    with np.load(out/'capture/reused_development_operands.npz') as z:
        features=z['features'].copy();gain=z['gain'].copy();base=z['base_risk'].copy();candidate=z['candidate_risk'].copy()
    selected={c:{arm:[] for arm in ARMS} for c in CONDITIONS}
    for chunk in captured['reused_development']['chunks']:
        file=out/'capture'/chunk['path'];require(sha(file)==chunk['sha256'],'Captured output changed')
        with np.load(file) as z:
            for c in CONDITIONS:
                for arm in ARMS:
                    prefix=c+'_'+arm+'_'
                    selected[c][arm].extend(zip(z[prefix+'labels'].copy(),z[prefix+'boxes'].copy(),z[prefix+'scores'].copy()))
    results=[];cache={}
    for ci,c in enumerate(CONDITIONS):
        for arm in ARMS:
            result=ap_result(gt,selected[c][arm]);results.append(dict(condition=c,method=arm,seed=None,**result))
        for seed in p['controller_seeds']:
            with np.load(out/f'seed_{seed}/reused_development_CHOICES.npz') as z:
                choices={k:z[k][:,ci].copy() for k in ('positive_mean','fitting_RMS_margin','selector_utility','selector_zero','selector_permuted')}
                mean=z['mean'][:,ci].copy();coverage=z['empirical_band_covers'][:,ci].copy()
            # Features each action: fourclass maxima/mean/std then boxmean/std.
            # Compute declared confidence directly from full canonical logits.
            confidence=[]
            for chunk in captured['reused_development']['chunks']:
                with np.load(out/'capture'/chunk['path']) as z:
                    scores=[]
                    for arm in ('task_only','cure'):
                        logits=z[c+'_'+arm+'_pred_logits'][:,:,TASK_CHANNELS_V2].astype(np.float64)
                        probability=np.exp(-np.logaddexp(0,-logits))
                        scores.append(probability.max(2).mean(1))
                    confidence.extend(scores[1]>scores[0])
            choices['confidence']=np.asarray(confidence)
            for method,choice in choices.items():
                if method=='confidence' and seed!=p['controller_seeds'][0]:continue
                key=(c,choice.tobytes())
                if key not in cache:cache[key]=ap_result(gt,[selected[c]['cure' if use else 'task_only'][i] for i,use in enumerate(choice)])
                actual_risk=np.where(choice,candidate[:,ci],base[:,ci])
                result=dict(condition=c,method=method,seed=None if method=='confidence' else seed,**cache[key],selection_fraction=float(choice.mean()),
                    harmful_selection_fraction=float((choice&(gain[:,ci]<0)).mean()),mean_frame_risk=float(actual_risk.mean()),
                    paired_sequence_risk_delta=sequence_risk_interval(actual_risk-base[:,ci],seq))
                if method in ('selector_utility','positive_mean','fitting_RMS_margin'):
                    result['utility']=dict(RMSE=float(np.sqrt(np.mean((mean-gain[:,ci])**2))),MAE=float(np.mean(np.abs(mean-gain[:,ci]))),
                        sign_agreement=float(np.mean((mean>0)==(gain[:,ci]>0))),empirical_band_coverage=float(coverage.mean()))
                results.append(result)
            write(out/'RESULTS.json',results)
            print(json.dumps(dict(stage='AP_evaluation',condition=c,seed=seed)),flush=True)
    summary=[]
    for c in CONDITIONS:
        for method in sorted(set(r['method'] for r in results)):
            rows=[r for r in results if r['condition']==c and r['method']==method]
            summary.append(dict(condition=c,method=method,controller_runs=len(rows),
                **{k:dict(mean=float(np.mean([r[k] for r in rows])),SD=float(np.std([r[k] for r in rows],ddof=1)) if len(rows)>1 else None,
                          all_values=[r[k] for r in rows]) for k in ('AP_percent','AP50_percent','AP75_percent')}))
    write(out/'SUMMARY.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('protocol','cache','checkpoints','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--protocol-sha256',required=True);args=parser.parse_args()
    require(sha(args.protocol)==args.protocol_sha256,'Frozen protocol hash differs')
    p=json.loads(args.protocol.read_text())
    require(p['launch_authorized'] is True,'Execution protocol must be frozen')
    for name,digest in p['files'].items():require(sha(ROOT/name)==digest,'Bound file changed: '+name)
    require(torch.cuda.is_available() and 'A100' in torch.cuda.get_device_name(0),'A100 required; no neural CPU/MPS fallback')
    require(not args.output.exists(),'Never overwrite a previous study');args.output.mkdir()
    started=time.time();report=dict(status='started',scope='reused_SEW_controller_development',CUDA_executed=True,
        detector_seeds=1,controller_seeds=p['controller_seeds'],fresh_confirmation=False,native_3D=False,
        AP_evaluated=False,protocol_sha256=sha(args.protocol),gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,
        versions={name:importlib.metadata.version(name) for name in ('numpy','scipy','pycocotools')},
        utility_information='Learned deterministic representation of the same40 inference features, not additional sensor information',
        efficiency_scope='Execution duration and allocatedCUDA memory recorded; no full-camera runtime or production latency claim')
    write(args.output/'REPORT.json',report)
    try:
        torch.set_num_threads(2);torch.manual_seed(p['capture_seed'])
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        old=json.loads((ROOT/p['original_protocol']).read_text())
        splits=json.loads((ROOT/old['splits_path']).read_text());roles=json.loads((ROOT/p['role_manifest']).read_text())
        cache=Cache(args.cache,old['cache_protocol_sha256'],splits,old['optimization_records_canonical_sha256'],p['capture_cache_report_sha256'])
        all_ids=[s for r in ROLES for s in roles[r]['sample_ids']]
        require(len(all_ids)==len(set(all_ids))==7172 and set(all_ids)==set(cache.records),'Complete disjoint controller roles required')
        models={}
        for arm in ARMS:
            file=args.checkpoints/arm/'epoch_040/checkpoint.pt'
            require(sha(file)==p['checkpoints'][arm]['checkpoint_sha256'],'Fixed endpoint checkpoint differs')
            model=DfineNativeCureV2(DfineCureConfigV1()).cuda()
            model.load_state_dict(torch.load(file,map_location='cuda',weights_only=False)['model'],strict=True)
            model.eval().requires_grad_(False)
            require(state_digest(model)==p['checkpoints'][arm]['state_sha256'],'Fixed endpoint state differs')
            models[arm]=model
        (args.output/'capture').mkdir()
        captured=capture(cache,models,roles,args.output/'capture',p)
        train_all(args.output,p)
        summary=evaluate_all(args.output,p,roles,cache.records,captured)
        for arm,model in models.items():
            require(state_digest(model)==p['checkpoints'][arm]['state_sha256'],'Frozen detector changed')
            require(sha(args.checkpoints/arm/'epoch_040/checkpoint.pt')==p['checkpoints'][arm]['checkpoint_sha256'],'Frozen checkpoint file changed')
        cache.verify_unchanged()
        report.update(status='completed_reused_SEW_action_study',AP_evaluated=True,summary_rows=len(summary),all_declared_models_retained=True,
            AP_interval='Withheld: original development slidecar occurs in only one recording; point AP retained',
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as exc:report.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
    try:
        for name,digest in p['files'].items():require(sha(ROOT/name)==digest,'Bound source changed after execution')
        require(sha(args.protocol)==args.protocol_sha256,'Protocol changed after execution')
        report['source_integrity']='passed'
    except BaseException as exc:report.update(status='failed_integrity',integrity_error=repr(exc))
    report['seconds']=time.time()-started;write(args.output/'REPORT.json',report)
    write(args.output/'MANIFEST.json',{str(f.relative_to(args.output)):dict(sha256=sha(f),bytes=f.stat().st_size) for f in sorted(args.output.rglob('*')) if f.is_file() and f.name!='MANIFEST.json'})
    print(json.dumps(report),flush=True)
    return 0 if report['status']=='completed_reused_SEW_action_study' else 1


if __name__=='__main__':raise SystemExit(main())

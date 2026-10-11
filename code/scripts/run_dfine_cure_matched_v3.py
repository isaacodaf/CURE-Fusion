"""Declared five-seed matched retraining with fitting-only routing temperatures."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cure_fusion.dfine_cure_v1 import DfineCureConfigV1, require
from cure_fusion.dfine_cure_v2 import DfineNativeCureV2
from cure_fusion.dfine_cure_v3 import DfineNativeCureV3
from cure_fusion.dfine_cure_development_v2 import Cache, sha, write, state_digest
from cure_fusion.dfine_cure_losses_v2 import four_class_targets_v2, four_class_detection_loss_v2
from cure_fusion.routing_temperature_v3 import fit_routing_temperatures, routing_diagnostics, scaled_routing_bias
from train_dfine_cure_development_cuda_v2 import train_arm, build_moments, evaluate_arm

ROOT = Path(__file__).resolve().parents[2]
ARMS = ('task_only', 'cure', 'modality_dropout')


def verify_source(protocol):
    for rel, expected in protocol['files'].items():
        require(sha(ROOT / rel) == expected, 'Frozen source/operand differs: ' + rel)


def inputs(protocol, cache_path):
    splits = json.loads((ROOT / protocol['splits_path']).read_text())
    cache = Cache(cache_path, protocol['cache_protocol_sha256'], splits,
                  protocol['optimization_records_canonical_sha256'], protocol['cache_report_sha256'])
    with np.load(ROOT / protocol['fit_moments_path'], allow_pickle=False) as saved:
        require(saved['identities'].tolist() == splits['fit']['sample_ids'], 'Fitting target order differs')
        temperatures, rms = fit_routing_temperatures(saved['mean'], saved['object_mask'], 0.75)
        require(np.array_equal(temperatures, np.asarray(protocol['routing_temperatures'])), 'Fitting-only temperatures differ')
        moments = dict(mean=saved['mean'].copy(), u2=saved['mean_sampling_variance'].copy(),
                       mask=saved['object_mask'].copy(),
                       positions={s:i for i,s in enumerate(saved['identities'].tolist())})
    for sid in splits['fit']['sample_ids']:
        require(int(moments['mask'][moments['positions'][sid]].sum()) == len(cache.records[sid]['target']['labels']),
                'Archived target/annotation correspondence differs')
    return splits, cache, moments, temperatures


def cuda_setup(seed):
    require(torch.cuda.is_available() and 'A100' in torch.cuda.get_device_name(0),
            'A100 CUDA required; no CPU/MPS training fallback')
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def model_config(protocol):
    config = dict(protocol['adapter_config'])
    config['shared_classes'] = tuple(config['shared_classes'])
    return DfineCureConfigV1(**config)


def fit_diagnostics(model, cache, sample_ids, temperatures):
    model.eval()
    means, biases, weights, foreground, both_available, gates = [], [], [], [], [], []
    with torch.no_grad():
        for offset in range(0, len(sample_ids), 16):
            ids = sample_ids[offset:offset+16]
            x, full = cache.batch(ids)
            out = model(**x)
            _, assignment, _ = four_class_detection_loss_v2(out, four_class_targets_v2(full))
            selected = torch.zeros(out['cev_mean'].shape[:2], dtype=torch.bool, device='cuda')
            for i, (queries, objects) in enumerate(assignment):
                selected[i, queries] = True
            available = (x['camera_present'] & x['radar_mask'].any(-1))[:, None].expand_as(selected)
            for collection, tensor in [(means,out['cev_mean']), (biases,out['routing_bias']),
                                        (weights,out['modality_weights']), (foreground,selected),
                                        (both_available,available),(gates,out['base_gate_logits'])]:
                collection.append(tensor.detach().cpu().numpy())
    mean, bias, weight, selected, available, gate = [np.concatenate(v) for v in (means,biases,weights,foreground,both_available,gates)]
    report = dict(scope='First 64 annotated fitting frames in frozen split order; no AP', frames=len(sample_ids),
                  all_queries=routing_diagnostics(mean, temperatures),
                  matched_foreground=routing_diagnostics(mean[selected], temperatures))
    if available.any():
        w=weight[available].astype(np.float64)
        gate_difference=gate[available,0]-gate[available,1]
        report['both_modalities_available']=dict(queries=int(available.sum()),
            median_weight_entropy=float(np.median(-np.sum(w*np.log(np.maximum(w,1e-30)),axis=1))),
            base_gate_camera_minus_radar_quantiles=np.quantile(gate_difference,[.05,.5,.95]).tolist())
    return report


def admission(protocol, cache, fit_ids, temperatures, seed):
    """Actual new-controller CUDA checks; historical admissions remain distinct."""
    config=model_config(protocol)
    torch.manual_seed(seed)
    original=DfineNativeCureV2(config).cuda()
    state=copy.deepcopy(original.state_dict())
    same=DfineNativeCureV3([.75,.75],config).cuda()
    same.load_state_dict({**state,'routing_temperatures':same.routing_temperatures},strict=True)
    corrected=DfineNativeCureV3(temperatures,config).cuda()
    corrected.load_state_dict({**state,'routing_temperatures':corrected.routing_temperatures},strict=True)
    admission_ids=[]
    for category in range(4):
        sid=next(s for s in fit_ids if category in cache.records[s]['target']['labels'])
        if sid not in admission_ids:admission_ids.append(sid)
    admission_ids.extend(s for s in fit_ids if s not in admission_ids)
    admission_ids=admission_ids[:16]
    x, full=cache.batch(admission_ids)
    targets=four_class_targets_v2(full)
    target_counts=[sum(int((t['labels']==k).sum()) for t in targets) for k in range(4)]
    require(all(n>0 for n in target_counts),'New supervised admission requires all four fitting classes')
    original.eval();same.eval();corrected.eval()
    with torch.no_grad():
        a,b,c=original(**x),same(**x),corrected(**x)
    require(all(torch.equal(a[k],b[k]) for k in a), 'Default-temperature V3 changes V2 arithmetic')
    require(torch.equal(a['cev_mean'],c['cev_mean']) and torch.equal(a['cev_logvar'],c['cev_logvar']),
            'Routing correction changed raw utility outputs')
    require(sum(p.numel() for p in original.parameters()) == sum(p.numel() for p in corrected.parameters()),
            'Controller capacity differs')
    probe=torch.tensor([[10.,0.]],device='cuda',requires_grad=True)
    scaled_routing_bias(probe,corrected.routing_temperatures).sum().backward()
    require(bool(torch.isfinite(probe.grad).all()) and float(probe.grad[0,0])>1e-3,
            'Camera utility-to-bias gradient remains numerically saturated at the declared probe')
    initial_diagnostics=fit_diagnostics(corrected,cache,
        [s for s in fit_ids if cache.records[s]['target']['labels']][:64],temperatures)
    corrected.train()
    optimizer=torch.optim.AdamW(corrected.parameters(),lr=.001,weight_decay=.0001)
    before_step=state_digest(corrected)
    loss,_,_=four_class_detection_loss_v2(corrected(**x),targets)
    loss.backward()
    require(bool(torch.isfinite(loss)) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in corrected.parameters()),
            'New controller has nonfinite supervised gradients')
    row_gradients=corrected.novel_class.weight.grad.square().sum(-1)
    require(bool((row_gradients>0).all()),'Both new task-class rows require nonzero gradients')
    torch.nn.utils.clip_grad_norm_(corrected.parameters(),1.)
    optimizer.step()
    require(state_digest(corrected)!=before_step and all(bool(torch.isfinite(p).all()) for p in corrected.parameters()),
            'Supervised optimizer step did not produce a finite changed state')
    result=dict(status='passed_new_controller_cuda_admission_only',CUDA_executed=True,AP_evaluated=False,
                admission_seed=seed,optimizer_steps=1,novel_row_squared_gradient_norms=row_gradients.detach().cpu().tolist(),
                default_temperature_v2_forward_identity=True,raw_utility_forward_identity=True,
                matched_parameter_count=corrected.contract()['trainable_parameters'],
                fitting_admission_ids=admission_ids,target_counts_all_four_classes=target_counts,
                probe_raw_bias_gradient=probe.grad.detach().cpu().tolist(),supervised_loss=float(loss.detach()),
                fitting_initialization_diagnostics=initial_diagnostics,
                environment=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0)))
    del original,same,corrected
    torch.cuda.empty_cache()
    return result


def train_seed(args, protocol):
    seed=args.seed
    require(seed in protocol['seeds'], 'Undeclared seed')
    cuda_setup(seed)
    splits,cache,moments,temperatures=inputs(protocol,args.cache)
    directory=args.output/('seed_%d'%seed)
    directory.mkdir(exist_ok=False)
    write(directory/'ADMISSION.json',admission(protocol,cache,splits['fit']['sample_ids'],temperatures,seed))
    torch.manual_seed(seed)
    initial=DfineNativeCureV3(temperatures,model_config(protocol)).cuda()
    state=copy.deepcopy(initial.state_dict());initial_digest=state_digest(initial)
    torch.save({k:v.cpu() for k,v in state.items()},directory/'INITIAL.pt')
    result=dict(status='training',seed=seed,initial_state_sha256=initial_digest,
                initial_file_sha256=sha(directory/'INITIAL.pt'),
                protocol_sha256=sha(args.protocol),controller_contract=initial.contract(),arms={},AP_evaluated=False)
    write(directory/'TRAINING_REPORT.json',result)
    for arm in ARMS:
        torch.manual_seed(seed)
        model=DfineNativeCureV3(temperatures,model_config(protocol)).cuda()
        model.load_state_dict(state,strict=True)
        require(state_digest(model)==initial_digest, 'Matched arm initialization differs')
        p={**protocol,'seed':seed}
        trained=train_arm(arm,model,cache,splits['fit']['sample_ids'],p,directory/arm,
                          moments if arm=='cure' else None)
        result['arms'][arm]=dict(report_sha256=sha(directory/arm/'REPORT.json'),
            final_checkpoint_sha256=sha(directory/arm/'epoch_040/checkpoint.pt'),
            final_state_sha256=state_digest(trained),
            fitting_diagnostics=fit_diagnostics(trained,cache,
                [s for s in splits['fit']['sample_ids'] if cache.records[s]['target']['labels']][:64],temperatures))
        write(directory/'TRAINING_REPORT.json',result)
        del trained,model
        torch.cuda.empty_cache()
    require(sha(args.protocol)==args.protocol_sha256,'Protocol changed during training')
    cache.verify_unchanged();verify_source(protocol)
    result.update(status='complete_all_three_training_endpoints',source_integrity='passed',cache_integrity='passed')
    write(directory/'TRAINING_REPORT.json',result)


def check_barrier(args, protocol):
    require(sha(args.protocol)==args.protocol_sha256,'Protocol changed before evaluation')
    verify_source(protocol)
    receipts={};initials=[]
    split=json.loads((ROOT/protocol['splits_path']).read_text())
    expected_fit=set(split['fit']['sample_ids'])
    for seed in protocol['seeds']:
        directory=args.output/('seed_%d'%seed)
        report=json.loads((directory/'TRAINING_REPORT.json').read_text())
        require(report['status']=='complete_all_three_training_endpoints' and report['AP_evaluated'] is False,
                'All seed/arm training must complete before evaluation')
        require(report['protocol_sha256']==sha(args.protocol), 'Training protocol differs')
        require(report.get('source_integrity')==report.get('cache_integrity')=='passed','Incomplete training integrity')
        require(sha(directory/'INITIAL.pt')==report['initial_file_sha256'],'Initial checkpoint changed')
        initial_model=DfineNativeCureV3(protocol['routing_temperatures'],model_config(protocol))
        initial_model.load_state_dict(torch.load(directory/'INITIAL.pt',map_location='cpu',weights_only=True),strict=True)
        require(state_digest(initial_model)==report['initial_state_sha256'],'Initial state receipt differs')
        initials.append(report['initial_state_sha256'])
        shared_exposure={}
        for arm in ARMS:
            bound=report['arms'][arm]
            require(sha(directory/arm/'REPORT.json')==bound['report_sha256'], 'Training report changed')
            arm_report=json.loads((directory/arm/'REPORT.json').read_text())
            require(arm_report['status']=='complete' and arm_report['arm']==arm and arm_report['epochs']==40 and len(arm_report['events'])==40, 'Incomplete training budget')
            for epoch,event in enumerate(arm_report['events'],1):
                ep=directory/arm/('epoch_%03d'%epoch)
                require(event['images']==6180 and event['updates']==97 and event['epoch']==epoch,
                        'Incomplete epoch coverage')
                require(sha(ep/'checkpoint.pt')==event['checkpoint_sha256'] and sha(ep/'updates.jsonl')==event['log_sha256'],
                        'Epoch output binding differs')
                require(json.loads((ep/'MANIFEST.json').read_text())==event,'Epoch manifest differs')
                previous=sha(directory/arm/('epoch_%03d'%(epoch-1))/'MANIFEST.json') if epoch>1 else None
                require(event['previous_epoch_manifest_sha256']==previous,'Epoch chain differs')
                rows=[json.loads(line) for line in (ep/'updates.jsonl').read_text().splitlines()]
                ids=[sid for row in rows for sid in row['identities']]
                require(len(rows)==97 and len(ids)==len(set(ids))==6180 and set(ids)==expected_fit,
                        'Training logs have incomplete/duplicate fitting coverage')
                if arm in ('task_only','cure'):
                    exposure=[{k:row[k] for k in ['identities','augmented_camera_absent','augmented_radar_mask_sha256']} for row in rows]
                    if arm=='task_only':shared_exposure[epoch]=exposure
                    else:require(exposure==shared_exposure[epoch],'Matched CURE/task-only exposure differs')
            require(sha(directory/arm/'epoch_040/checkpoint.pt')==bound['final_checkpoint_sha256'], 'Final checkpoint differs')
            final=torch.load(directory/arm/'epoch_040/checkpoint.pt',map_location='cpu',weights_only=False)
            require(final['epoch']==40 and final['arm']==arm,'Final checkpoint scope differs')
            checked_model=DfineNativeCureV3(protocol['routing_temperatures'],model_config(protocol))
            checked_model.load_state_dict(final['model'],strict=True)
            require(state_digest(checked_model)==bound['final_state_sha256'],'Final state receipt differs')
            require(torch.equal(checked_model.routing_temperatures,torch.tensor(protocol['routing_temperatures'],dtype=torch.float32)),
                    'Checkpoint routing temperatures changed')
        receipts[str(seed)]=sha(directory/'TRAINING_REPORT.json')
    require(len(set(initials))==len(protocol['seeds']), 'Independent seeds share initialization')
    return dict(status='all_15_training_endpoints_verified',seed_receipts=receipts,
                epochs_per_arm=40,updates_per_arm=3880,AP_evaluated_before_barrier=False)


def evaluate(args, protocol):
    cuda_setup(protocol['evaluation_seed'])
    barrier=check_barrier(args,protocol)
    write(args.output/'GLOBAL_TRAINING_BARRIER.json',barrier)
    splits,cache,moments,temperatures=inputs(protocol,args.cache)
    teacher=DfineNativeCureV2(model_config(protocol)).cuda()
    saved=torch.load(ROOT/protocol['teacher_checkpoint_path'],map_location='cpu',weights_only=False)
    teacher.load_state_dict(saved['model'],strict=True)
    require(state_digest(teacher)==protocol['teacher_state_sha256'], 'Archived teacher state differs')
    teacher.eval().requires_grad_(False)
    fresh=build_moments(teacher,cache,splits['inner_development']['sample_ids'],16,
        'fresh_development_thinning',args.output/'fresh_targets',protocol['evaluation_seed'])
    teacher.cpu();all_results={}
    for seed in protocol['seeds']:
        all_results[str(seed)]={}
        for arm in ARMS:
            model=DfineNativeCureV3(temperatures,model_config(protocol)).cuda()
            state=torch.load(args.output/('seed_%d'%seed)/arm/'epoch_040/checkpoint.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(state['model'],strict=True)
            all_results[str(seed)][arm]={}
            for condition in protocol['evaluation_conditions']:
                endpoint=evaluate_arm(arm,model,cache,splits['inner_development']['sample_ids'],fresh,
                    args.output/('seed_%d'%seed)/('evaluation_'+arm+'_'+condition),condition,protocol['evaluation_seed'])
                require(endpoint['status']=='complete', 'Declared endpoint unavailable; preserve raw outputs')
                all_results[str(seed)][arm][condition]=endpoint
            model.cpu();del model
            write(args.output/'EVALUATION_PROGRESS.json',all_results)
    require(sha(args.protocol)==args.protocol_sha256,'Protocol changed during evaluation')
    cache.verify_unchanged();verify_source(protocol)
    summary={}
    for condition in protocol['evaluation_conditions']:
        summary[condition]={}
        for arm in ARMS:
            values=[all_results[str(seed)][arm][condition]['AP_percent'] for seed in protocol['seeds']]
            summary[condition][arm]=dict(seeds=protocol['seeds'],AP_percent=values,
                mean_AP_percent=float(np.mean(values)),sample_SD_AP_points=float(np.std(values,ddof=1)))
        differences=np.asarray(summary[condition]['cure']['AP_percent'])-np.asarray(summary[condition]['task_only']['AP_percent'])
        summary[condition]['cure_minus_task_only']=dict(paired_AP_points=differences.tolist(),
            mean_AP_points=float(differences.mean()),sample_SD_AP_points=float(differences.std(ddof=1)))
    write(args.output/'FINAL_REPORT.json',dict(status='complete_five_seed_reused_development_study',
        protocol_sha256=sha(args.protocol),CUDA_executed=True,global_barrier=barrier,summary=summary,
        scope='Five adapter initialization/training seeds; one frozen detector and one archived teacher; eight reused recordings',
        independent_population_validation=False,recording_AP_CI_available=False,source_integrity='passed',cache_integrity='passed'))


def supervise(args, protocol):
    args.output.mkdir(exist_ok=False)
    write(args.output/'RUN_STATUS.json',dict(status='training',protocol_sha256=sha(args.protocol),
        seeds=protocol['seeds'],AP_evaluated=False,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    running=[];pending=list(protocol['seeds']);failed=[]
    while pending or running:
        while pending and len(running)<args.workers:
            seed=pending.pop(0)
            log=(args.output/('seed_%d_training.log'%seed)).open('x')
            cmd=[sys.executable,__file__,'--phase','train','--seed',str(seed),'--cache',str(args.cache),
                 '--output',str(args.output),'--protocol',str(args.protocol),'--protocol-sha256',args.protocol_sha256]
            process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
            running.append((seed,process,log))
        for seed,process,log in list(running):
            code=process.poll()
            if code is not None:
                log.close();running.remove((seed,process,log))
                if code:failed.append(dict(seed=seed,returncode=code))
                print(json.dumps(dict(seed=seed,training_returncode=code)),flush=True)
        time.sleep(5)
    if failed:
        write(args.output/'RUN_STATUS.json',dict(status='training_failed_evaluation_not_opened',failed=failed,AP_evaluated=False))
        raise RuntimeError('Training failed; outcomes preserved and evaluation remains unopened')
    evaluate(args,protocol)
    write(args.output/'RUN_STATUS.json',dict(status='complete',protocol_sha256=sha(args.protocol)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['admit','supervise','train','evaluate'],default='supervise')
    parser.add_argument('--seed',type=int)
    parser.add_argument('--workers',type=int,default=5)
    for name in ['cache','output','protocol']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--protocol-sha256',required=True)
    args=parser.parse_args()
    require(sha(args.protocol)==args.protocol_sha256,'Protocol hash differs')
    protocol=json.loads(args.protocol.read_text());verify_source(protocol)
    require(1<=args.workers<=5,'Use one to five declared workers')
    if args.phase=='admit':
        cuda_setup(protocol['seeds'][0])
        split,cache,moments,temperature=inputs(protocol,args.cache)
        result=admission(protocol,cache,split['fit']['sample_ids'],temperature,protocol['seeds'][0])
        require(sha(args.protocol)==args.protocol_sha256,'Protocol changed during admission')
        cache.verify_unchanged();verify_source(protocol)
        args.output.mkdir(exist_ok=False)
        write(args.output/'ADMISSION.json',result)
    elif args.phase=='supervise':supervise(args,protocol)
    elif args.phase=='train':train_seed(args,protocol)
    else:evaluate(args,protocol)


if __name__=='__main__':main()

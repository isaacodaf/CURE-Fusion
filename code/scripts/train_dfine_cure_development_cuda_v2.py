"""Frozen single-seed four-class D-FINE CURE development. CUDA only; no test use."""
import argparse,copy,hashlib,json,math,os,pathlib,shutil,sys,time,traceback
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
ROOT=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
import numpy as np
import torch
from cure_fusion.dfine_cure_v1 import DfineCureConfigV1,FIELDS,require,mask_native_inputs_v1
from cure_fusion.dfine_cure_v2 import DfineNativeCureV2,task_top300_v2,task_label_map_v2,TASK_CHANNELS_V2,append_task_logits_v2
from cure_fusion.dfine_cure_losses_v2 import four_class_targets_v2,four_class_detection_loss_v2,FixedDfineObjectRiskV2,cev_mean_nll_from_sampling_variance_v1
from cure_fusion.dfine_cure_development_v2 import Cache,sha,write,state_digest,thinning_mask,dropout_mask,epoch_order,training_augmentation


def keep_inputs(inputs,camera_absent=False,radar_mask=None):
 x=dict(inputs)
 if camera_absent:x['camera_present']=torch.zeros_like(x['camera_present'])
 if radar_mask is not None:x['radar_mask']=torch.from_numpy(radar_mask.copy()).to('cuda')
 return x

def build_moments(teacher,cache,sids,draws,tag,out,seed):
 """Camera removal once; iid radar-return thinning K times, detached frozen teacher."""
 out.mkdir(exist_ok=False);M=max(1,max(len(cache.records[s]['target']['labels']) for s in sids));N=len(sids)
 mean=np.zeros((N,M,2),np.float32);sampling=np.zeros_like(mean);valid=np.zeros((N,M),bool);summaries=[]
 before=state_digest(teacher);teacher.eval().requires_grad_(False);started=time.time()
 with torch.no_grad():
  for start in range(0,N,16):
   ids=sids[start:start+16];inputs,full=cache.batch(ids);targets=four_class_targets_v2(full);factual=teacher(**inputs);risk=FixedDfineObjectRiskV2(factual,targets);base=risk(factual)
   counts=[len(t['labels']) for t in targets];b=len(ids);cm=np.zeros((b,M),np.float32);rr=np.zeros((b,draws,M),np.float32);masks=[]
   # Save all object-gathered operands used by the fixed factual risk, not claimfullcounterfactualheadcapture.
   raw={k:np.zeros(shape,dtype=dtype) for k,shape,dtype in [('queries',(b,M),np.int64),('object_mask',(b,M),bool),('truth',(b,M,4),np.float32),('quality',(b,M,4),np.float32),('weights',(b,M,4),np.float32),('factual_logits',(b,M,4),np.float32),('factual_boxes',(b,M,4),np.float32),('factual_loss',(b,M),np.float32),('camera_logits',(b,M,4),np.float32),('camera_boxes',(b,M,4),np.float32),('camera_loss',(b,M),np.float32),('radar_logits',(b,draws,M,4),np.float32),('radar_boxes',(b,draws,M,4),np.float32),('radar_loss',(b,draws,M),np.float32)]}
   for i,row in enumerate(risk.rows):
    n=counts[i];q=row['queries'];valid[start+i,:n]=True;raw['object_mask'][i,:n]=True;raw['queries'][i,:n]=q.cpu().numpy()
    for k,v in [('truth','truth'),('quality','quality'),('weights','weight')]:raw[k][i,:n]=row[v].cpu().numpy()
    raw['factual_logits'][i,:n]=factual['pred_logits'][i,q][:,TASK_CHANNELS_V2].cpu().numpy();raw['factual_boxes'][i,:n]=factual['pred_boxes'][i,q].cpu().numpy();raw['factual_loss'][i,:n]=base[i].cpu().numpy()
   camera=teacher(**keep_inputs(inputs,camera_absent=True));camera_losses=risk(camera)
   for i,row in enumerate(risk.rows):
    n=counts[i];q=row['queries'];cm[i,:n]=(camera_losses[i]-base[i]).cpu().numpy();raw['camera_logits'][i,:n]=camera['pred_logits'][i,q][:,TASK_CHANNELS_V2].cpu().numpy();raw['camera_boxes'][i,:n]=camera['pred_boxes'][i,q].cpu().numpy();raw['camera_loss'][i,:n]=camera_losses[i].cpu().numpy()
   del camera,camera_losses
   for draw in range(draws):
    mask=thinning_mask(cache.raw_mask(ids),seed,tag,ids,draw);masks.append(mask);changed=teacher(**keep_inputs(inputs,radar_mask=mask));losses=risk(changed)
    for i,row in enumerate(risk.rows):
     n=counts[i];q=row['queries'];rr[i,draw,:n]=(losses[i]-base[i]).cpu().numpy();raw['radar_logits'][i,draw,:n]=changed['pred_logits'][i,q][:,TASK_CHANNELS_V2].cpu().numpy();raw['radar_boxes'][i,draw,:n]=changed['pred_boxes'][i,q].cpu().numpy();raw['radar_loss'][i,draw,:n]=losses[i].cpu().numpy()
   mean[start:start+b,:,0]=cm;mean[start:start+b,:,1]=rr.mean(1);sampling[start:start+b,:,1]=rr.var(1,ddof=1)/draws
   raw.update(camera_regret=cm,radar_regrets=rr,radar_draw_masks=np.stack(masks,axis=1),identities=np.asarray(ids))
   f=out/('%06d.npz'%start);np.savez_compressed(f,**raw);summaries.append(dict(path=f.name,sha256=sha(f),bytes=f.stat().st_size,identities=ids,four_class_objects=counts));print(json.dumps(dict(stage='target_generation',tag=tag,completed=start+b,total=N)),flush=True)
 require(state_digest(teacher)==before,'Frozen task-only teacher changed during CEV generation')
 np.savez_compressed(out/'moments.npz',mean=mean,mean_sampling_variance=sampling,object_mask=valid,identities=np.asarray(sids))
 write(out/'REPORT.json',dict(status='complete',tag=tag,teacher_state_sha256=before,images=N,four_class_objects=int(valid.sum()),camera_actual_evaluations=N,camera_operator='deterministic full camera removal; mean sampling variance exactly0, no iid/epistemic variance claim',radar_draws=draws,radar_operator='independent Bernoulli keep.5 on existing64-returnmask; allmissingallowed',source_pixel_or_augmentation_edits=False,raw_scope='Every object-gathered operand and drawmask; no fullrawcounterfactualhead or liveauthenticity claim from these receiptsalone',chunks=summaries,moments_sha256=sha(out/'moments.npz'),seconds=time.time()-started))
 return dict(mean=mean,u2=sampling,mask=valid,positions={s:i for i,s in enumerate(sids)})


def train_arm(name,model,cache,sids,protocol,out,moments=None):
 out.mkdir(exist_ok=False);model.train();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);events=[];started=time.time();seed=protocol['seed'];epochs=protocol['epochs']
 for epoch in range(1,epochs+1):
  directory=out/('epoch_%03d'%epoch);directory.mkdir();order=epoch_order(len(sids),seed,epoch);lr=1e-5+.5*(1e-3-1e-5)*(1+math.cos(math.pi*(epoch-1)/(epochs-1)))
  for group in opt.param_groups:group['lr']=lr
  epochstart=time.time();log=directory/'updates.jsonl';losses=[]
  with log.open('x') as handle:
   for offset in range(0,len(order),64):
    chosen=[sids[int(i)] for i in order[offset:offset+64]];total_objects=sum(len(cache.records[s]['target']['labels']) for s in chosen);denominator=max(total_objects,1);opt.zero_grad(set_to_none=True);aggregate={'clean_task':0.,'augmented_task':0.,'task':0.,'value':0.,'loss':0.}
    aug_camera,aug_radar=training_augmentation(cache.raw_mask(chosen),seed,epoch,chosen)
    flags=dropout_mask(seed,epoch,chosen) if name=='modality_dropout' else np.zeros((len(chosen),2),bool)
    aug_camera=aug_camera|flags[:,0];aug_radar=aug_radar&~flags[:,1,None]
    for local in range(0,len(chosen),16):
     ids=chosen[local:local+16];x,full=cache.batch(ids);targets=four_class_targets_v2(full)
     # Clean factual branch is identical in all arms; only this branch receives CEV labels.
     output=model(**x);clean_task,assignment,parts=four_class_detection_loss_v2(output,targets);n=sum(len(t['labels']) for t in targets);weight=max(n,1)/denominator
     changed=keep_inputs(x,radar_mask=aug_radar[local:local+len(ids)]);changed['camera_present']=x['camera_present']&~torch.from_numpy(aug_camera[local:local+len(ids)].copy()).to('cuda')
     augmented=model(**changed);augmented_task,_,_=four_class_detection_loss_v2(augmented,targets)
     weighted_clean=clean_task*weight;weighted_augmented=augmented_task*weight;weighted_task=weighted_clean+.25*weighted_augmented;value=weighted_task*0
     if name=='cure' and n:
      pm=[];pl=[];tm=[];tv=[]
      for i,(sid,(q,o)) in enumerate(zip(ids,assignment)):
       j=moments['positions'][sid];require(int(moments['mask'][j].sum())==len(targets[i]['labels']),'CUREobjectmapping mismatch');pm.append(output['cev_mean'][i,q]);pl.append(output['cev_logvar'][i,q]);tm.append(torch.from_numpy(moments['mean'][j,:len(o)].copy()).to('cuda')[o]);tv.append(torch.from_numpy(moments['u2'][j,:len(o)].copy()).to('cuda')[o])
      mu,lv,truth,variance=[torch.cat(v) for v in (pm,pl,tm,tv)];value=cev_mean_nll_from_sampling_variance_v1(mu,lv,truth,variance,torch.ones_like(mu,dtype=torch.bool))*(n/max(total_objects,1))
     loss=weighted_task+.1*value;require(bool(torch.isfinite(loss)),'Nonfinite studentloss');loss.backward()
     for key,tensor in [('clean_task',weighted_clean),('augmented_task',weighted_augmented),('task',weighted_task),('value',value),('loss',loss)]:aggregate[key]+=float(tensor.detach())
    require(all(p.grad is None or (p.grad.is_cuda and bool(torch.isfinite(p.grad).all())) for p in model.parameters()),'Nonfinite/nonCUDA gradient')
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);require(bool(torch.isfinite(norm)) and float(norm)>0,'Nonfinite/zero global gradient');opt.step()
    event=dict(epoch=epoch,offset=offset,identities=chosen,additional_dropout=flags.tolist(),augmented_camera_absent=aug_camera.tolist(),augmented_radar_mask_sha256=hashlib.sha256(aug_radar.tobytes()).hexdigest(),augmented_radar_counts=aug_radar.sum(-1).tolist(),objects=total_objects,learning_rate=lr,gradient_norm=float(norm),**aggregate);handle.write(json.dumps(event)+'\n');handle.flush();losses.append(aggregate['loss'])
  torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=opt.state_dict(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),epoch=epoch,arm=name),directory/'checkpoint.pt')
  receipt=dict(epoch=epoch,arm=name,images=len(order),updates=math.ceil(len(order)/64),order_sha256=hashlib.sha256(order.tobytes()).hexdigest(),mean_batch_loss=float(np.mean(losses)),seconds=time.time()-epochstart,checkpoint_sha256=sha(directory/'checkpoint.pt'),log_sha256=sha(log),state_sha256=state_digest(model),previous_epoch_manifest_sha256=sha(out/('epoch_%03d'%(epoch-1))/'MANIFEST.json') if epoch>1 else None)
  write(directory/'MANIFEST.json',receipt);events.append(receipt);print(json.dumps(dict(stage='training',arm=name,epoch=epoch,seconds=receipt['seconds'],loss=receipt['mean_batch_loss'])),flush=True)
 write(out/'REPORT.json',dict(status='complete',arm=name,epochs=epochs,events=events,seconds=time.time()-started,no_checkpoint_selection=True,automatic_resume_supported=False,task_objective='clean+.25*matchedaugmentation',CEV_clean_factual_only=name=='cure'))
 return model


def evaluate_arm(name,model,cache,sids,moments,out,condition,seed):
 """All models train first; this evaluates four_class native postprocess on992dev."""
 out.mkdir(exist_ok=False);predictions=[];annotations=[];images=[];rawfiles=[];observations=[];invalid_raw_task_boxes=0;annid=1
 if model is not None:model.eval()
 with torch.no_grad():
  for start in range(0,len(sids),16):
   ids=sids[start:start+16];x,full=cache.batch(ids);targets=four_class_targets_v2(full)
   if condition=='camera_removed':x=keep_inputs(x,camera_absent=True)
   elif condition=='radar_removed':x=keep_inputs(x,radar_mask=np.zeros_like(cache.raw_mask(ids)))
   elif condition=='radar_thinned_fresh':x=keep_inputs(x,radar_mask=thinning_mask(cache.raw_mask(ids),seed,'fresh_development_thinning',ids,0))
   else:require(condition=='clean','Undeclared evaluation condition')
   if model is not None:output=model(**x)
   else:
    z=mask_native_inputs_v1(**x);output=dict(pred_logits=append_task_logits_v2(z['camera_logits'],torch.full((*z['camera_logits'].shape[:2],2),-8.,dtype=z['camera_logits'].dtype,device='cuda')),pred_boxes=z['camera_boxes'])
   channel_labels,boxes,scores=task_top300_v2(output,torch.tensor([[640,512]]*len(ids),device='cuda'));labels=task_label_map_v2(channel_labels)
   arrays={k:v.detach().cpu().numpy() for k,v in output.items() if k in ('pred_logits','pred_boxes','cev_mean','cev_logvar','modality_weights')};arrays.update(adapted_channel_labels=channel_labels.cpu().numpy(),native_labels=labels.cpu().numpy(),native_boxes=boxes.cpu().numpy(),native_scores=scores.cpu().numpy(),identities=np.asarray(ids));f=out/('%06d.npz'%start);np.savez_compressed(f,**arrays);rawfiles.append(dict(path=f.name,sha256=sha(f),identities=ids))
   if model is not None and condition=='clean' and name!='frozen_unadapted_pretrained_reference':
    _,assign,_=four_class_detection_loss_v2(output,targets)
    for i,(sid,(q,o)) in enumerate(zip(ids,assign)):
     j=moments['positions'][sid]
     for qi,oi in zip(q.tolist(),o.tolist()):observations.append(dict(identity=sid,sequence=cache.records[sid]['sequence'],object_index=oi,query=qi,mean=output['cev_mean'][i,qi].cpu().tolist(),logvar=output['cev_logvar'][i,qi].cpu().tolist(),reference_mean=moments['mean'][j,oi].tolist(),reference_mean_sampling_variance=moments['u2'][j,oi].tolist()))
   for i,sid in enumerate(ids):
    imageid=start+i+1;images.append(dict(id=imageid,width=640,height=512,file_name=sid))
    for label,box in zip(targets[i]['labels'].tolist(),targets[i]['boxes'].cpu().numpy()):
     cx,cy,w,h=box.astype(float)*[640,512,640,512];annotations.append(dict(id=annid,image_id=imageid,category_id=label+1,bbox=[cx-w/2,cy-h/2,w,h],area=w*h,iscrowd=0));annid+=1
    for label,box,score in zip(arrays['native_labels'][i],arrays['native_boxes'][i],arrays['native_scores'][i]):
     if int(label) not in (0,1,2,3):continue
     x1,y1,x2,y2=box.astype(float);invalid_raw_task_boxes+=int(x2<=x1 or y2<=y1);x1,x2=np.clip([x1,x2],0,640);y1,y2=np.clip([y1,y2],0,512)
     predictions.append(dict(image_id=imageid,category_id=int(label)+1,bbox=[float(x1),float(y1),float(x2-x1),float(y2-y1)],score=float(score)))
 gt=dict(info={},images=images,annotations=annotations,categories=[dict(id=i+1,name=n) for i,n in enumerate(('person','bicycle','slidecar','doll'))]);write(out/'ground_truth_coco.json',gt);write(out/'predictions_coco.json',predictions);write(out/'value_observations.json',observations)
 report=dict(status='raw_outputs_complete',arm=name,condition=condition,images=len(sids),four_class_objects=len(annotations),raw=rawfiles,scope='Singlefixedseed,992nestedreuseddevelopment images,4taskclasses; no test/native3D/CUREefficacypromotion',score_threshold=None,nonpositive_raw_selected_task_boxes=invalid_raw_task_boxes,postprocess='Task-adapted82 sigmoid/globaltop300 thenmap0/1/80/81 tofourtaskclasses; no NMS; notunchangedofficial80postprocess',value_scope='Camera deterministicremoval; radar16freshthinningdraw means, notepistemic or allcorruption calibration')
 try:
  require(invalid_raw_task_boxes==0,'Nonpositive task-selected native decoded geometry; AP unavailable, raw boxes retained');require(all(p['bbox'][2]>=0 and p['bbox'][3]>=0 for p in predictions),'Invalid task selected box after native decode; AP unavailable without changing predictions')
  from pycocotools.coco import COCO
  from pycocotools.cocoeval import COCOeval
  G=COCO(str(out/'ground_truth_coco.json'))
  if predictions:D=G.loadRes(predictions)
  else:D=COCO();D.dataset=dict(images=images,categories=gt['categories'],annotations=[]);D.createIndex()
  e=COCOeval(G,D,'bbox');e.params.imgIds=list(range(1,len(sids)+1));e.params.catIds=[1,2,3,4];e.evaluate();e.accumulate();e.summarize();precision=e.eval['precision']
  report.update(status='complete',AP_percent=float(e.stats[0]*100),AP50_percent=float(e.stats[1]*100),AP75_percent=float(e.stats[2]*100),per_class={c:float(precision[:,:,j,0,-1][precision[:,:,j,0,-1]>-1].mean()*100) for j,c in enumerate(('person','bicycle','slidecar','doll'))})
 except BaseException as exc:report['AP_status']='unavailable';report['AP_error']=repr(exc)
 if observations:
  mu=np.array([o['mean'] for o in observations]);lv=np.array([o['logvar'] for o in observations]);reference=np.array([o['reference_mean'] for o in observations]);u2=np.array([o['reference_mean_sampling_variance'] for o in observations]);total=np.exp(np.clip(lv,-8,5))+u2;error=mu-reference
  report['value_object_pooled']=dict(objects=len(mu),RMSE=np.sqrt(np.mean(error**2,axis=0)).tolist(),bias=np.mean(error,axis=0).tolist(),Gaussian_mean_NLL=np.mean(.5*(error**2/total+np.log(total)),axis=0).tolist(),central90_coverage=np.mean(np.abs(error)<=1.6448536269514722*np.sqrt(total),axis=0).tolist(),order=['camera_removal','radar_thinning'])
 write(out/'REPORT.json',report);return report


def main():
 ap=argparse.ArgumentParser(description=__doc__)
 for k in ('cache','protocol','admission_report','output'):ap.add_argument('--'+k.replace('_','-'),type=pathlib.Path,required=True)
 ap.add_argument('--protocol-sha256',required=True);ap.add_argument('--admission-report-sha256',required=True);ap.add_argument('--cache-report-sha256',required=True);a=ap.parse_args();require(sha(a.protocol)==a.protocol_sha256,'Protocol mismatch');p=json.loads(a.protocol.read_text())
 for rel,h in p['files'].items():require(sha(ROOT/rel)==h,'Frozen study source mismatch: '+rel)
 # Exact four-class elementary admission is a prerequisite; V1 two-class success cannot substitute.
 require(sha(a.admission_report)==a.admission_report_sha256,'Externally verified actual admission report differs');admission=json.loads(a.admission_report.read_text());require(admission.get('status')=='passed_dfine_cure_four_class_admission_only','Complete actual CUDAadapteradmission required');require(admission.get('protocol_sha256')==p['adapter_admission_protocol_sha256'],'Wrong adapteradmission protocol')
 require(admission.get('CUDA_executed') is True and admission.get('optimizer_steps')==1 and admission.get('AP_evaluated') is False,'Wrong elementary CUDAadmission scope')
 expected_admission=json.loads((ROOT/p['adapter_admission_protocol_path']).read_text());require(admission['bound_files']==expected_admission['files'],'Actual admission source inventory differs');require(admission['source_before']==admission['source_after']==dict(commit=p['native_source_commit'],inventory_sha256=p['native_source_inventory_sha256']),'Actual admission source chain differs')
 require(admission['supervised_step']['all_four_ground_truth_classes_assigned'] is True and all(n>0 for n in admission['target_counts_all_four_classes']) and all(n>0 for n in admission['supervised_step']['novel_head_row_squared_norms']),'Complete actual four-class task admission required')
 require(torch.cuda.is_available() and 'A100' in torch.cuda.get_device_name(0),'A100 CUDA required; noCPU/MPS fallback');torch.cuda.set_device(0);torch.manual_seed(p['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 require(shutil.disk_usage(a.output.parent).free>=15*1024**3,'At least15GiBfree required for complete outputs');a.output.mkdir(parents=True,exist_ok=False);report=dict(status='started',scope='fixedseeddevelopment_only',CUDA_executed=True,model='D-FINEfrozen+four_classradarCURE',protocol_sha256=sha(a.protocol),admission_report_sha256=sha(a.admission_report),stages={},environment=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0)),AP_inspected_before_all_training_completed=False);started=time.time();write(a.output/'REPORT.json',report)
 try:
  import scipy,pycocotools.coco,pycocotools.cocoeval
  report['dependency_versions']=dict(numpy=np.__version__,scipy=scipy.__version__);report['automatic_resume_supported']=False
  splits=json.loads((ROOT/p['splits_path']).read_text());cache=Cache(a.cache,p['cache_protocol_sha256'],splits,p['optimization_records_canonical_sha256'],a.cache_report_sha256);fit=splits['fit']['sample_ids'];dev=splits['inner_development']['sample_ids'];require(len(fit)==6180 and len(dev)==992,'Frozen nested populations');report['cache_report_sha256']=cache.report_sha256;report['cache_metadata_sha256']={path.name:h for path,h in cache.bindings.items() if path.parent==cache.root};write(a.output/'REPORT.json',report)
  initial=DfineNativeCureV2(DfineCureConfigV1()).cuda();initial_state=copy.deepcopy(initial.state_dict());initial_hash=state_digest(initial);torch.save({k:v.cpu() for k,v in initial_state.items()},a.output/'INITIAL.pt');report['initial_state_sha256']=initial_hash;report['parameter_inventory']=initial.contract();write(a.output/'INITIAL.json',dict(state_sha256=initial_hash,file_sha256=sha(a.output/'INITIAL.pt'),contract=initial.contract()))
  report['current_stage']='task_only_training';write(a.output/'REPORT.json',report);models={};task=train_arm('task_only',initial,cache,fit,p,a.output/'task_only');task.eval().requires_grad_(False);teacher_hash=state_digest(task);report['teacher_policy']='Thecompletedtask-onlyendpoint is the frozenfusionteacher; noconditionalcheckpointselection or duplicateteacherfit.'
  report['current_stage']='fit_CEV_target_generation';write(a.output/'REPORT.json',report);target=build_moments(task,cache,fit,8,'fit_thinning',a.output/'fit_targets',p['seed']);models['task_only']=task.cpu();del task
  for name in ('cure','modality_dropout'):
   report['current_stage']=name+'_training';write(a.output/'REPORT.json',report);torch.manual_seed(p['seed']);model=DfineNativeCureV2(DfineCureConfigV1()).cuda();model.load_state_dict(initial_state,strict=True);require(state_digest(model)==initial_hash,'Matchedinitialstate mismatch');models[name]=train_arm(name,model,cache,fit,p,a.output/name,target if name=='cure' else None).cpu()
  report['all_training_completed_before_development_targets_or_AP']=True;report['current_stage']='fresh_development_targets';write(a.output/'REPORT.json',report)
  teacher=models['task_only'].cuda().eval().requires_grad_(False);require(state_digest(teacher)==teacher_hash,'Teacherstate changed');fresh=build_moments(teacher,cache,dev,16,'fresh_development_thinning',a.output/'fresh_development_targets',p['seed']);teacher.cpu()
  results={}
  for name in ('frozen_unadapted_pretrained_reference','task_only','cure','modality_dropout'):
   model=DfineNativeCureV2(DfineCureConfigV1()).cuda() if name=='frozen_unadapted_pretrained_reference' else models[name].cuda();
   if name=='frozen_unadapted_pretrained_reference':model.load_state_dict(initial_state,strict=True);require(state_digest(model)==initial_hash,'Unadapted reference must equal original common initialization')
   report['current_stage']='evaluation_'+name;write(a.output/'REPORT.json',report);model.eval();results[name]={}
   for condition in p['evaluation_conditions']:
    results[name][condition]=evaluate_arm(name,model,cache,dev,fresh,a.output/('evaluation_'+name+'_'+condition),condition,p['seed'])
   if model is not None:model.cpu()
  report.update(peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(),status='complete' if all(v.get('status')=='complete' for arm in results.values() for v in arm.values()) else 'completed_with_unavailable_AP',results=results,training_epochs=p['epochs'],primary_result=False,final_test_evaluated=False,efficacy_promoted=False,seconds=time.time()-started)
 except BaseException as exc:report.update(status='failed',error=repr(exc),traceback=traceback.format_exc(),seconds=time.time()-started)
 finally:
  try:
   for rel,h in p['files'].items():require(sha(ROOT/rel)==h,'Frozen study source changedafterexecution: '+rel)
   if 'cache' in locals():cache.verify_unchanged()
   require(sha(a.admission_report)==a.admission_report_sha256 and sha(a.protocol)==a.protocol_sha256,'External admission/protocol changed')
   report['final_source_integrity']='passed'
  except BaseException as exc:report['final_source_integrity']='failed';report['integrity_error']=repr(exc);report['status']='failed_integrity'
  write(a.output/'REPORT.json',report);write(a.output/'ARTIFACT_HASHES.json',{str(f.relative_to(a.output)):dict(bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(a.output.rglob('*')) if f.is_file()})
 print(json.dumps({k:v for k,v in report.items() if k in ('status','error','seconds')}),flush=True);return 0 if report['status']=='complete' else 1
if __name__=='__main__':raise SystemExit(main())

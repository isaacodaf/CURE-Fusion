"""Deterministic data/target helpers for the frozen D-FINE development comparison."""
import hashlib,json,pathlib
import numpy as np
import torch
from .dfine_cure_v1 import FIELDS,require

def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(path,value):pathlib.Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def state_digest(model):
 h=hashlib.sha256()
 for n,t in sorted(model.state_dict().items()):
  h.update(n.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def rng_for(seed,tag,sid,draw):
 salt='dfine-cure-development-v2:%d:%s:%s:%d'%(seed,tag,sid,draw)
 return np.random.default_rng(int.from_bytes(hashlib.sha256(salt.encode()).digest()[:16],'little'))
def thinning_mask(mask,seed,tag,sids,draw):
 require(mask.dtype==np.bool_ and mask.shape==(len(sids),64),'Exact parent radar mask required')
 return mask & np.stack([rng_for(seed,tag,s,draw).random(64)<.5 for s in sids])
def dropout_mask(seed,epoch,sids):
 return np.stack([rng_for(seed,'dropout_epoch_%d'%epoch,s,0).random(2)<.1 for s in sids])
def training_augmentation(mask,seed,epoch,sids):
 camera=np.array([rng_for(seed,'train_augmentation_camera',sid,epoch).random()<.1 for sid in sids],bool)
 radar=thinning_mask(mask,seed,'train_augmentation_radar_epoch_%d'%epoch,sids,0)
 return camera,radar
def epoch_order(size,seed,epoch):return np.random.default_rng(seed*100000+epoch).permutation(size)

class Cache:
 def __init__(self,root,expected_protocol_sha,splits,expected_records_digest,expected_report_sha):
  self.root=pathlib.Path(root);require(sha(self.root/'REPORT.json')==expected_report_sha,'Externally verified full-cache report differs');report=json.loads((self.root/'REPORT.json').read_text());self.bindings={self.root/'REPORT.json':expected_report_sha}
  require(report['status']=='passed_feature_cache_only' and report['mode']=='all7172' and report['images_completed']==7172 and report['final_integrity']=='passed','Complete original all7172 CUDA cache required')
  require(report['protocol_sha256']==expected_protocol_sha and report['CUDA_executed'] and report['deployed_state_sha256_before']==report['deployed_state_sha256_after']==report['final_state_sha256'],'Cache source/device/state binding')
  require(sha(self.root/'protocol.json')==expected_protocol_sha,'Actual cache protocol copy differs')
  protocol=json.loads((self.root/'protocol.json').read_text());expected_source=dict(commit=protocol['source_commit'],inventory_sha256=protocol['source_inventory_sha256'])
  require(report['source_before']==report['source_after']==expected_source,'Actual cache source inventory differs')
  require(report.get('training') is False and report.get('AP') is False and 'A100' in report['environment']['gpu'],'Cache scope/device differs')
  for name in ('protocol.json','selected_records.json','ARTIFACT_HASHES.json'):self.bindings[self.root/name]=sha(self.root/name)
  manifest=json.loads((self.root/'ARTIFACT_HASHES.json').read_text())
  for name in ('protocol.json','selected_records.json','REPORT.json'):
   require(manifest[name]['sha256']==self.bindings[self.root/name] and manifest[name]['bytes']==(self.root/name).stat().st_size,'Cache metadata manifest mismatch')
  self.report_sha256=sha(self.root/'REPORT.json');records=json.loads((self.root/'selected_records.json').read_text())
  rawrecords=[{k:v for k,v in r.items() if k!='development_role'} for r in records]
  require(hashlib.sha256(json.dumps(rawrecords,sort_keys=True,separators=(',',':')).encode()).hexdigest()==expected_records_digest,'Original optimization metadata/labelcarrier differs')
  self.records={r['identity']:r for r in records}
  require(len(self.records)==7172 and set(self.records)==set(splits['fit']['sample_ids'])|set(splits['inner_development']['sample_ids']),'Complete nested split coverage')
  arrays={k:[] for k in FIELDS};ids=[]
  for chunk in report['chunks']:
   require(manifest[chunk['path']]['sha256']==chunk['sha256'] and manifest[chunk['path']]['bytes']==chunk['bytes'],'Cache chunk manifest binding differs');path=(self.root/chunk['path']).resolve();path.relative_to(self.root.resolve());require(sha(path)==chunk['sha256'],'Cache chunk changed');self.bindings[path]=chunk['sha256']
   with np.load(path,allow_pickle=False) as z:
    for k,a in chunk['arrays'].items():
     v=z[k];require(list(v.shape)==a['shape'] and str(v.dtype)==a['dtype'] and hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()==a['sha256'],'Cache array changed')
    require(z['identities'].tolist()==chunk['identities'],'Cache identity mismatch')
    for local,sid in enumerate(chunk['identities']):
     target=self.records[sid]['target'];mask=z['target_mask'][local]
     require(np.array_equal(z['target_labels'][local,mask],target['labels']) and np.array_equal(z['target_boxes'][local,mask],np.asarray(target['boxes'],np.float32).reshape(-1,4)),'Cached allfour targetcarrier differs')
    ids.extend(chunk['identities'])
    for k in arrays:arrays[k].append(z[k].copy())
  require(len(ids)==len(set(ids))==7172 and set(ids)==set(self.records),'Duplicate/missing cached records')
  self.ids=ids;self.positions={s:i for i,s in enumerate(ids)};self.arrays={k:np.concatenate(v) for k,v in arrays.items()}
  # Immutable cache records preserve full4class carrier. V2 loss retains all four labels.
 def batch(self,sids):
  positions=[self.positions[s] for s in sids]
  inputs={k:torch.from_numpy(v[positions].copy()).to('cuda') for k,v in self.arrays.items()}
  targets=[dict(labels=torch.tensor(self.records[s]['target']['labels'],dtype=torch.long,device='cuda'),boxes=torch.tensor(self.records[s]['target']['boxes'],dtype=torch.float32,device='cuda').reshape(-1,4)) for s in sids]
  return inputs,targets
 def raw_mask(self,sids):return self.arrays['radar_mask'][[self.positions[s] for s in sids]]

 def verify_unchanged(self):
  for path,h in self.bindings.items():require(sha(path)==h,'Previously loaded cache artifact changed: '+str(path))

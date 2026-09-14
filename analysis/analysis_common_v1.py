"""CPU-only post-run algebra; no model construction/forward, training or tuning."""
import contextlib,hashlib,io,json,pathlib
import numpy as np
from scipy.special import expit
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
TASK_CHANNELS=(0,1,80,81)
CLASSES=('person','bicycle','slidecar','doll')
EPS=np.finfo(np.float32).eps

def need(value,message):
 if not value:raise ValueError(message)
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(path,value):pathlib.Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def safe(root,name):
 p=(pathlib.Path(root)/name).resolve();p.relative_to(pathlib.Path(root).resolve());return p

def close(observed,reference,*,atol,rtol,name):
 a,b=np.asarray(observed),np.asarray(reference)
 need(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),name+' nonfinite/shape')
 maximum=float(np.max(np.abs(a.astype(float)-b.astype(float)),initial=0))
 need(np.allclose(a,b,atol=atol,rtol=rtol,equal_nan=False),name+' numerical replay mismatch: '+str(maximum));return maximum

def giou(a,b):
 a,b=np.asarray(a,float),np.asarray(b,float);aw,bw=np.maximum(a[...,2:],0),np.maximum(b[...,2:],0)
 al,ah=a[...,:2]-aw/2,a[...,:2]+aw/2;bl,bh=b[...,:2]-bw/2,b[...,:2]+bw/2
 intersection=np.maximum(np.minimum(ah,bh)-np.maximum(al,bl),0).prod(-1)
 union=(ah-al).prod(-1)+(bh-bl).prod(-1)-intersection
 enclosure=np.maximum(np.maximum(ah,bh)-np.minimum(al,bl),0).prod(-1)
 iou=intersection/np.maximum(union,1e-8)
 return iou,iou-(enclosure-union)/np.maximum(enclosure,1e-8)

def fixed_risk(logits,boxes,truth,quality,weight):
 x=np.asarray(logits,float);y=np.asarray(quality,float);w=np.asarray(weight,float)
 return ((np.logaddexp(0,x)-x*y)*w).sum(-1)+5*np.abs(np.asarray(boxes,float)-np.asarray(truth,float)).sum(-1)+2*(1-giou(boxes,truth)[1])

def replay_selected(logits,boxes,channels,selected_boxes,scores):
 """Verify saved GPU top300 as a set of distinct query/channel pairs.

CPU cannot claim identical tie ordering. Check source membership, legal ties,
all300 count, score order and threshold completeness using an explicit FP32
score-replay tolerance. COCO uses saved GPU scores/order, never replacement.
"""
 logits=np.asarray(logits);boxes=np.asarray(boxes);channels=np.asarray(channels);selected_boxes=np.asarray(selected_boxes);scores=np.asarray(scores)
 need(logits.shape==(300,82) and boxes.shape==(300,4) and channels.shape==scores.shape==(300,) and selected_boxes.shape==(300,4),'Raw82/top300 shape')
 need(all(np.isfinite(a).all() for a in (logits,boxes,channels,selected_boxes,scores)),'Nonfinite saved output')
 need(channels.dtype.kind in 'iu' and ((channels>=0)&(channels<82)).all(),'Invalid selected channels')
 need(((scores>=0)&(scores<=1)).all() and (scores[:-1]>=scores[1:]).all(),'Invalid score order/range')
 low=(boxes[:,:2]-boxes[:,2:]/np.float32(2)).astype(np.float32);high=(boxes[:,:2]+boxes[:,2:]/np.float32(2)).astype(np.float32)
 expected=np.concatenate((low,high),-1)*np.array([640,512,640,512],np.float32)
 probability=expit(logits.astype(float));score_tolerance=4*EPS*np.abs(probability)+np.finfo(np.float32).tiny
 lookup={}
 for q,row in enumerate(expected):lookup.setdefault(row.tobytes(),[]).append(q)
 rows=[];cols=[];max_box=0.;max_score=0.;memo={}
 for j,(c,box,score) in enumerate(zip(channels,selected_boxes,scores)):
  key=(int(c),box.tobytes(),np.float32(score).tobytes())
  if key in memo:candidates=memo[key]
  else:
   candidate=np.array(lookup.get(box.tobytes(),[]),dtype=np.int64)
   if not len(candidate):
    # Three FP32 corner/scale operations. This is a CPU replay bound, not a detector gate.
    bound=8*EPS*np.maximum(1,np.maximum(np.abs(expected),np.abs(box)))
    candidate=np.flatnonzero((np.abs(expected.astype(float)-box.astype(float))<=bound).all(-1))
   candidates=candidate[np.abs(probability[candidate,int(c)]-float(score))<=score_tolerance[candidate,int(c)]]
   memo[key]=candidates
  need(len(candidates)>0,'Selected box/score has no raw source query')
  rows.extend([j]*len(candidates));cols.extend((candidates*82+int(c)).tolist())
 graph=csr_matrix((np.ones(len(rows),np.int8),(rows,cols)),shape=(300,300*82))
 matched=maximum_bipartite_matching(graph,perm_type='column');need((matched>=0).all() and len(set(matched.tolist()))==300,'Selected pairs cannot be joined one-to-one')
 q=matched//82;c=matched%82
 max_box=float(np.max(np.abs(expected[q].astype(float)-selected_boxes.astype(float)),initial=0));max_score=float(np.max(np.abs(probability[q,c]-scores.astype(float)),initial=0))
 selected=np.zeros((300,82),bool);selected[q,c]=True
 cutoff=float(scores[-1]);need((probability[~selected]<=cutoff+4*EPS*np.abs(probability[~selected])+np.finfo(np.float32).tiny).all(),'Higher-scoring raw pair omitted outside FP32/tie allowance')
 boundary_ambiguous=int((np.abs(probability-cutoff)<=4*EPS*np.abs(probability)+np.finfo(np.float32).tiny).sum())
 return dict(query_indices=q,channel_indices=c,max_box_error=max_box,max_score_error=max_score,boundary_ambiguous_pairs=boundary_ambiguous,cpu_tie_order_claim=False)

def coco_truth(records):
 images=[];annotations=[]
 for image_id,r in enumerate(records,1):
  images.append(dict(id=image_id,width=640,height=512,file_name=r['identity']))
  for c,b in zip(r['target']['labels'],np.asarray(r['target']['boxes'],np.float32).reshape(-1,4)):
   cx,cy,w,h=b.astype(float)*[640,512,640,512]
   annotations.append(dict(id=len(annotations)+1,image_id=image_id,category_id=int(c)+1,bbox=[cx-w/2,cy-h/2,w,h],area=w*h,iscrowd=0))
 return dict(info={},images=images,annotations=annotations,categories=[dict(id=i+1,name=n) for i,n in enumerate(CLASSES)])

def coco_predictions(channels,boxes,scores,image_id):
 predictions=[];invalid=0
 for c,box,score in zip(channels,boxes,scores):
  if int(c) not in TASK_CHANNELS:continue
  c=TASK_CHANNELS.index(int(c));x1,y1,x2,y2=box.astype(float);invalid+=int(x2<=x1 or y2<=y1)
  x1,x2=np.clip([x1,x2],0,640);y1,y2=np.clip([y1,y2],0,512)
  predictions.append(dict(image_id=image_id,category_id=c+1,bbox=[float(x1),float(y1),float(x2-x1),float(y2-y1)],score=float(score)))
 return predictions,invalid

def official_coco(gt,predictions,compact=False):
 with contextlib.redirect_stdout(io.StringIO()):
  G=COCO();G.dataset=gt;G.createIndex()
  if predictions:D=G.loadRes(predictions)
  else:D=COCO();D.dataset=dict(images=gt['images'],categories=gt['categories'],annotations=[]);D.createIndex()
  e=COCOeval(G,D,'bbox');e.params.imgIds=[r['id'] for r in gt['images']];e.params.catIds=[1,2,3,4]
  if compact:e.params.areaRng=[[0,1e10]];e.params.maxDets=[100]
  e.evaluate();e.accumulate()
 precision=e.eval['precision'][:,:,:,0,-1];per_class=[]
 for i in range(4):
  values=precision[:,:,i];per_class.append(float(values[values>=0].mean()) if (values>=0).any() else None)
 values=precision[precision>=0]
 summary=dict(AP=float(values.mean()) if len(values) else None,AP50=float(precision[0][precision[0]>=0].mean()) if (precision[0]>=0).any() else None,AP75=float(precision[5][precision[5]>=0].mean()) if (precision[5]>=0).any() else None,per_class=per_class)
 return summary,e

def matched_pack(evaluation,sequences):
 """Reuse established replication accumulator; consume exact saved COCO boxes.

Do not use its legacy normalized-box constructor, old softmax decoder, or
available-class bootstrap summary. Full4 support is enforced outside ap().
"""
 from cure_fusion.coco_cluster_bootstrap import ClusterCOCOAP
 pack=object.__new__(ClusterCOCOAP);pack.images=len(sequences);pack.sequences=list(sequences);pack.classes=[]
 need(evaluation.params.areaRng[0]==[0,1e10] and max(evaluation.params.maxDets)==100,'Official all-area/max100 axes')
 for k in range(4):
  stride=len(evaluation.params.areaRng)*pack.images;values=evaluation.evalImgs[k*stride:k*stride+pack.images]
  gt=np.array([int((~v['gtIgnore'].astype(bool)).sum()) if v is not None else 0 for v in values]);scores=[];images=[];tp=[];fp=[]
  for i,v in enumerate(values):
   if v is None:continue
   scores.extend(v['dtScores']);images.extend([i]*len(v['dtScores']));tp.append((v['dtMatches']>0)&~v['dtIgnore']);fp.append((v['dtMatches']==0)&~v['dtIgnore'])
  scores=np.asarray(scores);images=np.asarray(images,np.int64);order=np.argsort(-scores,kind='stable');scores=scores[order];images=images[order]
  tp=np.concatenate(tp,axis=1)[:,order] if tp else np.zeros((10,0),bool);fp=np.concatenate(fp,axis=1)[:,order] if fp else np.zeros((10,0),bool)
  starts=np.flatnonzero(np.r_[True,(scores[1:]!=scores[:-1])|(images[1:]!=images[:-1])]) if len(scores) else np.empty(0,np.int64)
  lengths=np.diff(np.r_[starts,len(scores)]) if len(scores) else np.empty(0,np.int64)
  pack.classes.append(dict(gt=gt,tp=tp,fp=fp,starts=starts,lengths=lengths,images=images[starts]))
 return pack

def fixed_four_ap(pack,weights,fast=True):
 if any(int(c['gt']@weights)==0 for c in pack.classes):return None
 if fast:
  from cure_fusion.coco_ap_kernel import accumulated_ap
  return accumulated_ap(pack,weights)
 return pack.ap(weights)

def physical_replica(gt,predictions,weights):
 bygt={i['id']:[] for i in gt['images']};bypred={i['id']:[] for i in gt['images']}
 for a in gt['annotations']:bygt[a['image_id']].append(a)
 for a in predictions:bypred[a['image_id']].append(a)
 images=[];annotations=[];detections=[]
 for image,count in zip(gt['images'],weights):
  for _ in range(int(count)):
   new_id=len(images)+1;images.append({**image,'id':new_id})
   for a in bygt[image['id']]:annotations.append({**a,'id':len(annotations)+1,'image_id':new_id})
   for a in bypred[image['id']]:detections.append({**a,'image_id':new_id})
 return dict(info={},images=images,annotations=annotations,categories=gt['categories']),detections

"""Shared-label sigmoid losses and frozen-estimand CEV for the new D-FINE host.

Source principles: focal Hungarian cost2/5/2; VFL(alpha=.75,gamma=2)+5L1+2GIoU.
Only person/bicycle labels are exhaustive here. The other78 logits are neither
background targets nor optimized. No FGL/DDF/denoising loss is asserted without
native corner distributions and corresponding teacher/training targets.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from .dfine_cure_v1 import require


def validate_outputs(output):
    logits,boxes=output['pred_logits'],output['pred_boxes']
    require(logits.ndim==3 and logits.shape[1:]==(300,80) and boxes.shape==(*logits.shape[:2],4), 'Native80/300 output required')
    require(bool(torch.isfinite(logits).all()) and bool(torch.isfinite(boxes).all()), 'Invalid native outputs')
    return logits,boxes


def shared_targets_v1(targets):
    """Filter a preserved SEW four-class carrier; novel labels2/3 have no COCO map."""
    out=[]
    for t in targets:
        labels,boxes=t['labels'],t['boxes']
        require(labels.dtype==torch.long and boxes.shape==(len(labels),4), 'Wrong target shape/type')
        require(bool(((labels>=0)&(labels<4)).all()) and bool(torch.isfinite(boxes).all()), 'Invalid SEW carrier')
        require(bool((boxes[:,2:]>0).all()), 'Nonpositive target dimensions')
        keep=labels<2
        out.append(dict(labels=labels[keep],boxes=boxes[keep]))
    return out


def overlap_v1(a,b):
    # Loss-only geometry: negative dimensions collapse to zero area. Raw boxes
    # and raw L1 remain unchanged; this never alters native exported predictions.
    aw,bw=a[...,2:].clamp_min(0),b[...,2:].clamp_min(0)
    alo,ahi=a[...,:2]-aw/2,a[...,:2]+aw/2
    blo,bhi=b[...,:2]-bw/2,b[...,:2]+bw/2
    inter=(torch.minimum(ahi,bhi)-torch.maximum(alo,blo)).clamp_min(0).prod(-1)
    union=(ahi-alo).prod(-1)+(bhi-blo).prod(-1)-inter
    enclosure=(torch.maximum(ahi,bhi)-torch.minimum(alo,blo)).clamp_min(0).prod(-1)
    iou=inter/union.clamp_min(1e-8)
    return iou,iou-(enclosure-union)/enclosure.clamp_min(1e-8)


def _targets(targets,batch,device):
    require(len(targets)==batch,'Target batch mismatch')
    for t in targets:
        require(t['labels'].dtype==torch.long and t['boxes'].shape==(len(t['labels']),4),'Target axes differ')
        require(len(t['labels'])<=300 and bool(((t['labels']>=0)&(t['labels']<2)).all()),'Shared person/bicycle targets only')
        require(bool(torch.isfinite(t['boxes']).all()) and bool((t['boxes'][:,2:]>0).all()),'Invalid target boxes')
    return [{k:v.to(device) for k,v in t.items() if k in ('labels','boxes')} for t in targets]


@torch.no_grad()
def match_shared_v1(output,targets):
    logits,boxes=validate_outputs(output);targets=_targets(targets,len(logits),logits.device)
    indices=[]
    for x,b,t in zip(logits,boxes,targets):
        if len(t['labels'])==0:
            empty=torch.empty(0,dtype=torch.long,device=logits.device);indices.append((empty,empty));continue
        valid=torch.nonzero((b[:,2:]>0).all(-1),as_tuple=False).flatten()
        require(len(valid)>=len(t['labels']), 'Too few positive-size native proposals for complete assignment')
        p=x[valid].sigmoid()[:,t['labels']]
        positive=.25*(1-p).pow(2)*(-(p+1e-8).log())
        negative=.75*p.pow(2)*(-(1-p+1e-8).log())
        cost=2*(positive-negative)+5*torch.cdist(b[valid],t['boxes'],p=1)-2*overlap_v1(b[valid,None],t['boxes'][None])[1]
        require(bool(torch.isfinite(cost).all()),'Non-finite assignment cost; no silent repair')
        q,o=linear_sum_assignment(cost.detach().cpu().numpy())
        indices.append((valid[torch.as_tensor(q,dtype=torch.long,device=x.device)],torch.as_tensor(o,dtype=torch.long,device=x.device)))
    return indices


def _assignment_check(indices,targets):
    require(len(indices)==len(targets),'Assignment batch differs')
    for (q,o),t in zip(indices,targets):
        n=len(t['labels'])
        require(q.dtype==o.dtype==torch.long and q.shape==o.shape==(n,), 'Complete one-to-one factual assignment required')
        require(len(q.unique())==n and len(o.unique())==n and bool(((q>=0)&(q<300)).all()),'Duplicate/out-of-range queries')
        require(torch.equal(o.sort().values,torch.arange(n,device=o.device)), 'Incomplete ground-truth identity mapping')


def shared_detection_loss_v1(output,targets,indices=None):
    logits,boxes=validate_outputs(output);targets=_targets(targets,len(logits),logits.device)
    indices=match_shared_v1(output,targets) if indices is None else indices
    _assignment_check(indices,targets)
    selected=logits[...,:2];target=torch.zeros_like(selected);quality=torch.zeros_like(selected)
    l1=boxes.sum()*0;giou=boxes.sum()*0
    for i,((q,o),t) in enumerate(zip(indices,targets)):
        q,o=q.to(logits.device),o.to(logits.device)
        truth=t['boxes'][o];pred=boxes[i,q]
        require(bool((pred[:,2:]>0).all()),'Task assignment selected invalid native geometry')
        target[i,q,t['labels'][o]]=1
        iou,g=overlap_v1(pred,truth)
        quality[i,q,t['labels'][o]]=iou.detach()
        l1=l1+(pred-truth).abs().sum();giou=giou+(1-g).sum()
    weight=.75*selected.sigmoid().detach().pow(2)*(1-target)+quality
    vfl=(F.binary_cross_entropy_with_logits(selected,quality,reduction='none')*weight).sum()
    denominator=max(sum(len(t['labels']) for t in targets),1)
    losses=dict(loss_vfl=vfl/denominator,loss_bbox=l1/denominator,loss_giou=giou/denominator)
    return losses['loss_vfl']+5*losses['loss_bbox']+2*losses['loss_giou'],indices,losses


class FixedDfineObjectRiskV1:
    """One frozen teacher, factual correspondence AND factual VFL targets/weights.

    Return object-ordered risk for the same query slots on every intervention.
    Targets are never rematched or recomputed from intervened outputs. Query
    reordering from a re-encoded camera remains a limitation, not a causal fix.
    """
    def __init__(self,factual,targets,indices=None):
        logits,boxes=validate_outputs(factual);targets=_targets(targets,len(logits),logits.device)
        indices=match_shared_v1(factual,targets) if indices is None else indices
        _assignment_check(indices,targets)
        self.rows=[]
        for i,((q,o),t) in enumerate(zip(indices,targets)):
            order=o.argsort();q=q[order].to(logits.device)
            require(bool((boxes[i,q,2:]>0).all()),'Factual utility assignment selected invalid native geometry')
            truth=t['boxes'].detach().clone();labels=t['labels']
            p=logits[i,q,:2].detach().sigmoid()
            hot=F.one_hot(labels,num_classes=2).to(logits.dtype)
            quality=overlap_v1(boxes[i,q].detach(),truth)[0].detach()[:,None]*hot
            weight=.75*p.pow(2)*(1-hot)+quality
            self.rows.append(dict(queries=q.detach().clone(),truth=truth,quality=quality.detach().clone(),weight=weight.detach().clone()))

    def __call__(self,output):
        logits,boxes=validate_outputs(output);require(len(logits)==len(self.rows),'Frozen risk batch differs')
        losses=[]
        for i,row in enumerate(self.rows):
            q=row['queries'];b=boxes[i,q]
            cls=(F.binary_cross_entropy_with_logits(logits[i,q,:2],row['quality'],reduction='none')*row['weight']).sum(-1)
            losses.append(cls+5*(b-row['truth']).abs().sum(-1)+2*(1-overlap_v1(b,row['truth'])[1]))
        return losses


def cev_mean_nll_from_sampling_variance_v1(mean,logvar,target_mean,mean_sampling_variance,mask):
    """Expected-value Gaussian working likelihood with explicit uncertainty of mean.

    A deterministic camera-removal value is computed once and has sampling
    variance0. For K independent radar draws, use unbiased draw variance/K.
    Never describe repeated deterministic camera removals as independent draws.
    """
    require(mean.shape==logvar.shape==target_mean.shape==mean_sampling_variance.shape==mask.shape and mask.dtype==torch.bool,'CEV moment axes/mask mismatch')
    require(bool(mask.any()),'No admitted CEV observations')
    mu,lv,t,v=(x[mask] for x in (mean,logvar,target_mean,mean_sampling_variance))
    require(all(bool(torch.isfinite(x).all()) for x in (mu,lv,t,v)) and bool((v>=0).all()),'Invalid admitted CEV moments')
    total=lv.clamp(-8.,5.).exp()+v.detach()
    return .5*((mu-t.detach()).square()/total+total.log()).mean()


def cev_mean_nll_v1(mean,logvar,target_mean,unbiased_draw_variance,draw_count,mask):
    """Convenience form for K>=2 independent draws, sampling variance s²/K.

    exp(logvar) is residual working variance, not automatically epistemic
    uncertainty. Use the explicit sampling-variance API for deterministic laws.
    """
    require(isinstance(draw_count,int) and draw_count>=2,'At least two declared independent draws required')
    return cev_mean_nll_from_sampling_variance_v1(mean,logvar,target_mean,unbiased_draw_variance/draw_count,mask)

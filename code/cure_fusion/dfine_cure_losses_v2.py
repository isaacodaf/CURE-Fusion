"""Versioned full four-class D-FINE/CURE loss and fixed-object CEV risk.

Every SEW label0..3 is supervised through adapted channels0/1/80/81. The other
78 native classification channels remain outside the label/background loss.
Source-style VFL and complete matching retain the reviewed V1 geometry rules.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from .dfine_cure_v1 import require
from .dfine_cure_v2 import TASK_CHANNELS_V2
from .dfine_cure_losses_v1 import (overlap_v1, _assignment_check,
    cev_mean_nll_from_sampling_variance_v1, cev_mean_nll_v1)


def validate_outputs(output):
    logits,boxes=output['pred_logits'],output['pred_boxes']
    require(logits.ndim==3 and logits.shape[1:]==(300,82) and boxes.shape==(*logits.shape[:2],4), 'Adapted82/300 output required')
    require(bool(torch.isfinite(logits).all()) and bool(torch.isfinite(boxes).all()), 'Invalid native outputs')
    return logits,boxes


def four_class_targets_v2(targets):
    """Retain every SEW label0..3 without a COCO alias or class exclusion."""
    out=[]
    for t in targets:
        labels,boxes=t['labels'],t['boxes']
        require(labels.dtype==torch.long and boxes.shape==(len(labels),4),'Wrong target shape/type')
        require(bool(((labels>=0)&(labels<4)).all()) and bool(torch.isfinite(boxes).all()),'Invalid four-class SEW target')
        require(bool((boxes[:,2:]>0).all()),'Nonpositive target dimensions')
        out.append(dict(labels=labels,boxes=boxes))
    return out


def _targets(targets,batch,device):
    require(len(targets)==batch,'Target batch mismatch')
    for t in targets:
        require(t['labels'].dtype==torch.long and t['boxes'].shape==(len(t['labels']),4),'Target axes differ')
        require(len(t['labels'])<=300 and bool(((t['labels']>=0)&(t['labels']<4)).all()),'All four SEW classes required')
        require(bool(torch.isfinite(t['boxes']).all()) and bool((t['boxes'][:,2:]>0).all()),'Invalid target boxes')
    return [{k:v.to(device) for k,v in t.items() if k in ('labels','boxes')} for t in targets]


@torch.no_grad()
def match_four_class_v2(output,targets):
    logits,boxes=validate_outputs(output);targets=_targets(targets,len(logits),logits.device)
    indices=[]
    for x,b,t in zip(logits,boxes,targets):
        if len(t['labels'])==0:
            empty=torch.empty(0,dtype=torch.long,device=logits.device);indices.append((empty,empty));continue
        valid=torch.nonzero((b[:,2:]>0).all(-1),as_tuple=False).flatten()
        require(len(valid)>=len(t['labels']), 'Too few positive-size native proposals for complete assignment')
        p=x[valid][:,TASK_CHANNELS_V2].sigmoid()[:,t['labels']]
        positive=.25*(1-p).pow(2)*(-(p+1e-8).log())
        negative=.75*p.pow(2)*(-(1-p+1e-8).log())
        cost=2*(positive-negative)+5*torch.cdist(b[valid],t['boxes'],p=1)-2*overlap_v1(b[valid,None],t['boxes'][None])[1]
        require(bool(torch.isfinite(cost).all()),'Non-finite assignment cost; no silent repair')
        q,o=linear_sum_assignment(cost.detach().cpu().numpy())
        indices.append((valid[torch.as_tensor(q,dtype=torch.long,device=x.device)],torch.as_tensor(o,dtype=torch.long,device=x.device)))
    return indices


def four_class_detection_loss_v2(output,targets,indices=None):
    logits,boxes=validate_outputs(output);targets=_targets(targets,len(logits),logits.device)
    indices=match_four_class_v2(output,targets) if indices is None else indices
    _assignment_check(indices,targets)
    selected=logits[...,TASK_CHANNELS_V2];target=torch.zeros_like(selected);quality=torch.zeros_like(selected)
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


class FixedDfineObjectRiskV2:
    """One frozen teacher, factual correspondence AND factual VFL targets/weights.

    Return object-ordered risk for the same query slots on every intervention.
    Targets are never rematched or recomputed from intervened outputs. Query
    reordering from a re-encoded camera remains a limitation, not a causal fix.
    """
    def __init__(self,factual,targets,indices=None):
        logits,boxes=validate_outputs(factual);targets=_targets(targets,len(logits),logits.device)
        indices=match_four_class_v2(factual,targets) if indices is None else indices
        _assignment_check(indices,targets)
        self.rows=[]
        for i,((q,o),t) in enumerate(zip(indices,targets)):
            order=o.argsort();q=q[order].to(logits.device)
            require(bool((boxes[i,q,2:]>0).all()),'Factual utility assignment selected invalid native geometry')
            truth=t['boxes'].detach().clone();labels=t['labels']
            p=logits[i,q][:,TASK_CHANNELS_V2].detach().sigmoid()
            hot=F.one_hot(labels,num_classes=4).to(logits.dtype)
            quality=overlap_v1(boxes[i,q].detach(),truth)[0].detach()[:,None]*hot
            weight=.75*p.pow(2)*(1-hot)+quality
            self.rows.append(dict(queries=q.detach().clone(),truth=truth,quality=quality.detach().clone(),weight=weight.detach().clone()))

    def __call__(self,output):
        logits,boxes=validate_outputs(output);require(len(logits)==len(self.rows),'Frozen risk batch differs')
        losses=[]
        for i,row in enumerate(self.rows):
            q=row['queries'];b=boxes[i,q]
            cls=(F.binary_cross_entropy_with_logits(logits[i,q][:,TASK_CHANNELS_V2],row['quality'],reduction='none')*row['weight']).sum(-1)
            losses.append(cls+5*(b-row['truth']).abs().sum(-1)+2*(1-overlap_v1(b,row['truth'])[1]))
        return losses



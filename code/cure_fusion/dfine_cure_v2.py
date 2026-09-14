"""Four-class SEW adaptation of the frozen D-FINE-native CURE V1 bridge.

Original 80-class projection has exact camera-present zero-residual identity.
Channels80/81 are separately learned slidecar/doll heads, not COCO aliases.
All82 sigmoid channels compete in global top300 before task-class filtering.
"""
import torch
from torch import nn
from .dfine_cure_v1 import (DfineNativeCureV1, DfineCureConfigV1, FIELDS,
    SOURCE_COMMIT, CHECKPOINT_SHA256, require, mask_native_inputs_v1,
    apply_native_residual_v1)

TASK_CHANNELS_V2=(0,1,80,81)


def append_task_logits_v2(native_logits,novel_logits):
    require(native_logits.shape[1:]==(300,80) and novel_logits.shape==(*native_logits.shape[:2],2),'Native80 plus novel2 axes required')
    return torch.cat((native_logits,novel_logits),-1)


def task_top300_v2(output,image_sizes_wh):
    """Native-style global selection on adapted82-class outputs, no prefilter.

Returns original channel labels; only afterwards map0/1/80/81 to task0/1/2/3.
This82-class selection is not the unchanged official80-class postprocessor.
"""
    logits,boxes=output['pred_logits'],output['pred_boxes']
    require(logits.shape[1:]==(300,82) and boxes.shape==(*logits.shape[:2],4),'Adapted82/300 output required')
    require(image_sizes_wh.shape==(len(logits),2),'Image sizes must be [width,height]')
    xyxy=torch.cat((boxes[...,:2]-boxes[...,2:]/2,boxes[...,:2]+boxes[...,2:]/2),-1)
    xyxy=xyxy*image_sizes_wh.repeat(1,2).unsqueeze(1)
    scores,index=torch.topk(logits.sigmoid().flatten(1),300,dim=-1)
    labels=index-index//82*82
    selected=xyxy.gather(1,(index//82).unsqueeze(-1).repeat(1,1,4))
    return labels,selected,scores


def task_label_map_v2(channel_labels):
    """Map native/adapted channels after global selection; unrelated labels=-1."""
    mapped=torch.full_like(channel_labels,-1)
    for task,channel in enumerate(TASK_CHANNELS_V2):mapped=torch.where(channel_labels==channel,task,mapped)
    return mapped


class DfineNativeCureV2(DfineNativeCureV1):
    """Same four-class architecture for task-only, CEV and dropout arms.

The two novel heads have unrestricted learned logits; bounding them around−8
would prevent meaningful detection. Native shared-class residual/routing/box
bounds remain unchanged. Camera removal retains V1's limited centered anchors.
"""
    def __init__(self,config=None):
        super().__init__(DfineCureConfigV1() if config is None else config)
        self.novel_class=nn.Linear(self.config.hidden,2)
        nn.init.zeros_(self.novel_class.weight);nn.init.constant_(self.novel_class.bias,-8.)

    def contract(self):
        result=super().contract()
        result.update(version=2,output_classes=82,task_channels=list(TASK_CHANNELS_V2),
            task_classes=['person','bicycle','slidecar','doll'],
            class_semantics='82 independent sigmoid logits; original80 retained; novel80/81 are learned task classes, not COCO remaps',
            novel_head='Unbounded linear logits; zero initial weight and bias−8',
            identity_scope='Original80 projection and all boxes at zero residual with camera present; adapted82 top300 may differ')
        return result

    def forward(self,**inputs):
        require(set(inputs)==set(FIELDS),'Explicit complete input contract required')
        require(inputs['camera'].is_cuda,'Neural adapter forward is CUDA-only; no CPU/MPS fallback')
        z=mask_native_inputs_v1(**inputs)
        b=len(z['camera']);h=self.config.hidden;present=z['camera_present'][:,None,None]
        cam=self.camera_norm(self.camera(z['camera'])+self.camera_logits(z['camera_logits'])+self.camera_boxes(z['camera_boxes']))
        cam=torch.where(present,cam,0.)
        q=self.query[None].expand(b,-1,-1);conditioned=q+cam
        point=z['radar'];distance=torch.linalg.vector_norm(point[...,:3],dim=-1,keepdim=True)
        rad=self.radar(torch.cat((point,distance),-1)/self.radar_scale)
        rad=rad+torch.where(z['radar_geometry_mask'][...,None],self.position(z['radar_xy']),0.)
        empty=~z['radar_mask'].any(-1,keepdim=True)
        rad=torch.cat((rad,self.null_radar.expand(b,-1,-1)),1)
        valid=torch.cat((z['radar_mask'],empty),1)
        attention=conditioned@rad.transpose(-1,-2)/h**.5
        geometry=(z['camera_boxes'][...,:2,None].transpose(-1,-2)-z['radar_xy'][:,None])
        geo_bias=-geometry.square().sum(-1)/self.config.geometry_temperature**2
        geo_bias=torch.where(z['radar_geometry_mask'][:,None],geo_bias,0.)
        attention=attention+torch.cat((geo_bias,torch.zeros_like(geo_bias[...,:1])),dim=-1) if point.shape[1] else attention
        attention=attention.masked_fill(~valid[:,None],float('-inf')).softmax(-1);local=attention@rad
        raw=self.cev(torch.cat((q,cam,local,(cam-local).abs()),-1))
        mean,logvar=raw[...,:2],raw[...,2:].clamp(-8.,5.)
        risk=mean-self.config.kappa*torch.exp(.5*logvar)
        bias=self.config.routing_bound*torch.tanh(risk/self.config.routing_temperature)
        available=torch.stack((z['camera_present'],z['radar_mask'].any(-1)),dim=-1)[:,None]
        weights=(self.gate(torch.cat((conditioned,local),-1))+bias).masked_fill(~available,-1e4).softmax(-1)
        weights=weights*available;weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)
        refined=self.refine(torch.cat((q,weights[...,:1]*cam,weights[...,1:]*local),-1))
        logits,boxes=apply_native_residual_v1(z['camera_logits'],z['camera_boxes'],self.class_delta(refined),self.box_delta(refined),self.config)
        logits=append_task_logits_v2(logits,self.novel_class(refined))
        return dict(pred_logits=logits,pred_boxes=boxes,cev_mean=mean,cev_logvar=logvar,
                    modality_weights=weights,radar_attention=attention,routing_bias=bias)

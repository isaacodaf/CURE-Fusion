"""V3 retains V2 detector arithmetic and changes only the routing denominator."""
import torch
from .dfine_cure_v2 import DfineNativeCureV2, append_task_logits_v2
from .dfine_cure_v1 import FIELDS, require, mask_native_inputs_v1, apply_native_residual_v1
from .routing_temperature_v3 import scaled_routing_bias


class DfineNativeCureV3(DfineNativeCureV2):
    def __init__(self, temperatures, config=None):
        super().__init__(config)
        values=torch.as_tensor(temperatures,dtype=torch.float32)
        require(values.shape==(2,) and bool(torch.isfinite(values).all()) and bool((values>0).all()),
                'Two finite positive routing temperatures required')
        self.register_buffer('routing_temperatures',values.clone())

    def contract(self):
        result=super().contract()
        result.update(version=3,routing_temperatures=self.routing_temperatures.detach().cpu().tolist(),
                      routing_policy='raw risk divided by fixed fitting-only modality temperatures',
                      utility_outputs_and_targets='Original raw task-loss units; original NLL unchanged')
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
        bias=scaled_routing_bias(risk,self.routing_temperatures,self.config.routing_bound)
        available=torch.stack((z['camera_present'],z['radar_mask'].any(-1)),dim=-1)[:,None]
        gate_logits=self.gate(torch.cat((conditioned,local),-1))
        weights=(gate_logits+bias).masked_fill(~available,-1e4).softmax(-1)
        weights=weights*available;weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)
        refined=self.refine(torch.cat((q,weights[...,:1]*cam,weights[...,1:]*local),-1))
        logits,boxes=apply_native_residual_v1(z['camera_logits'],z['camera_boxes'],self.class_delta(refined),self.box_delta(refined),self.config)
        logits=append_task_logits_v2(logits,self.novel_class(refined))
        return dict(pred_logits=logits,pred_boxes=boxes,cev_mean=mean,cev_logvar=logvar,
                    modality_weights=weights,radar_attention=attention,routing_bias=bias,
                    base_gate_logits=gate_logits)

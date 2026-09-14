"""Versioned D-FINE-native shared-class CURE candidate; no efficacy claim.

Frozen final decoder features and native LQE/FDR outputs are camera evidence.
Only COCO contiguous classes 0/1 receive logit residuals. All 80 independent
sigmoid logits and all 300 queries remain; there is no softmax background class.
Neural forward is CUDA-only. Pure masking/residual algebra is CPU-testable.
"""
from dataclasses import dataclass, asdict
import torch
from torch import nn

SOURCE_COMMIT = '956d1709314c2c6a4df6f34de232054578a7449f'
CHECKPOINT_SHA256 = '39361855c21f8c4757a9bb777a7ff134ae8aa110d0db67a374f081e44c2ce04b'
FIELDS = ('camera','camera_logits','camera_boxes','radar','radar_mask','radar_xy','radar_geometry_mask','camera_present')


def require(ok, message):
    if not ok:
        raise ValueError(message)


@dataclass(frozen=True)
class DfineCureConfigV1:
    hidden: int = 128
    query_count: int = 300
    camera_channels: int = 256
    native_classes: int = 80
    shared_classes: tuple = (0, 1)
    class_bound: float = 2.0
    center_bound: float = 0.1
    log_size_bound: float = 0.2
    routing_bound: float = 0.5
    routing_temperature: float = 0.75
    geometry_temperature: float = 0.25
    kappa: float = 0.0

    def __post_init__(self):
        require((self.query_count,self.camera_channels,self.native_classes,self.shared_classes)==(300,256,80,(0,1)), 'Native D-FINE-X contract required')
        require(self.hidden>0 and self.routing_temperature>0 and self.geometry_temperature>0, 'Positive dimensions/temperatures required')
        require(min(self.class_bound,self.center_bound,self.log_size_bound,self.routing_bound,self.kappa)>=0, 'Negative bounds/risk coefficient')


def mask_native_inputs_v1(camera,camera_logits,camera_boxes,radar,radar_mask,
                          radar_xy,radar_geometry_mask,camera_present):
    """Mask every unavailable camera channel before arithmetic, even NaN sentinels.

    Radar order is XYZ [metres], intensity, Doppler [m/s]. XY is the normalized
    exported-image coordinate; resize640x512->640x640 preserves normalized XY.
    Invalid geometry never becomes a position embedding at the image origin.
    Camera removal also removes all camera-frame geometry. Raw radar survives.
    """
    b = camera.shape[0]
    require(camera.shape==(b,300,256) and camera_logits.shape==(b,300,80) and camera_boxes.shape==(b,300,4), 'Native camera query axes differ')
    require(radar.ndim==3 and radar.shape[0]==b and radar.shape[2]==5, 'Radar XYZ/intensity/Doppler required')
    require(radar_mask.shape==radar.shape[:2] and radar_geometry_mask.shape==radar_mask.shape and radar_xy.shape==(*radar_mask.shape,2), 'Radar/mask/geometry axes differ')
    require(camera_present.shape==(b,) and all(x.dtype==torch.bool for x in (camera_present,radar_mask,radar_geometry_mask)), 'Explicit boolean availability masks required')
    floats=(camera,camera_logits,camera_boxes,radar,radar_xy)
    require(all(x.dtype==torch.float32 and x.device==camera.device for x in floats), 'Shared-device FP32 inputs required')
    require(all(x.device==camera.device for x in (camera_present,radar_mask,radar_geometry_mask)), 'Mask device mismatch')
    present=camera_present[:,None,None]
    c=torch.where(present,camera,0.)
    logits=torch.where(present,camera_logits,-8.)
    boxes=torch.where(present,camera_boxes,.5)
    points=torch.where(radar_mask[...,None],radar,0.)
    geometry=radar_mask & radar_geometry_mask & camera_present[:,None]
    xy=torch.where(geometry[...,None],radar_xy,0.)
    require(all(bool(torch.isfinite(x).all()) for x in (c,logits,boxes,points,xy)), 'Non-finite active sensor data')
    require(bool(((xy>=0)&(xy<=1)).all()), 'Valid projected XY must be normalized to [0,1]')
    return dict(camera=c,camera_logits=logits,camera_boxes=boxes,radar=points,
                radar_mask=radar_mask,radar_xy=xy,radar_geometry_mask=geometry,camera_present=camera_present)


def apply_native_residual_v1(logits,boxes,class_delta,box_delta,config):
    """Bit-exact zero residual identity, including boxes outside image bounds.

    Other78 logits are exact copies. Geometry remains one box per query, so a
    changed box can affect a nonshared class at that query; no geometry claim
    for the other78 classes is made. Never clip/remap before native top300.
    """
    require(class_delta.shape==(*logits.shape[:2],2) and box_delta.shape==boxes.shape, 'Residual shape mismatch')
    shared=logits[...,:2]+config.class_bound*torch.tanh(class_delta)
    pred_logits=torch.cat((shared,logits[...,2:]),dim=-1)
    xy=boxes[...,:2]+config.center_bound*torch.tanh(box_delta[...,:2])
    wh=boxes[...,2:]*torch.exp(config.log_size_bound*torch.tanh(box_delta[...,2:]))
    return pred_logits,torch.cat((xy,wh),dim=-1)


def native_top300_v1(output,image_sizes_wh):
    """Algebraic native sigmoid/global query-class top300; retain duplicate queries.

    CUDA admission compares this with the unchanged official postprocessor.
    Shared-class filtering or image-bound clipping happens only downstream.
    """
    logits,boxes=output['pred_logits'],output['pred_boxes']
    require(logits.shape[1:]==(300,80) and boxes.shape==(*logits.shape[:2],4), 'Native output shape required')
    require(image_sizes_wh.shape==(len(logits),2), 'Image sizes must be [width,height]')
    xyxy=torch.cat((boxes[...,:2]-boxes[...,2:]/2,boxes[...,:2]+boxes[...,2:]/2),-1)
    xyxy=xyxy*image_sizes_wh.repeat(1,2).unsqueeze(1)
    scores,index=torch.topk(logits.sigmoid().flatten(1),300,dim=-1)
    labels=index-index//80*80
    selected=xyxy.gather(1,(index//80).unsqueeze(-1).repeat(1,1,4))
    return labels,selected,scores


class DfineQueryExtractorV1(nn.Module):
    """Read actual final256d query without modifying the trained source or head.

    Install after model.deploy(); retain outputs AFTER native LQE and FDR.
    This hook is observational and never substitutes native tensors. Detached
    no_grad tensors remain usable by a separately trainable residual adapter.
    """
    def __init__(self,native_model):
        super().__init__()
        self.native=native_model.eval().requires_grad_(False)
        decoder=self.native.decoder
        require(decoder.eval_idx==5 and decoder.hidden_dim==256 and decoder.num_queries==300, 'Wrong deployed D-FINE-X decoder')
        head=decoder.dec_score_head[decoder.eval_idx]
        require(isinstance(head,nn.Linear) and head.in_features==256 and head.out_features==80, 'Wrong final trained score head')
        self._captured=[]
        self._handle=head.register_forward_pre_hook(self._capture)

    def _capture(self,module,args):
        self._captured.append(args[0].detach().clone())

    def train(self,mode=True):
        super().train(mode)
        self.native.eval()
        return self

    def close(self):
        self._handle.remove()

    def forward(self,images):
        require(images.is_cuda and images.dtype==torch.float32 and images.shape[1:]==(3,640,640), 'Native extraction requires CUDA FP32 RGB640 input')
        require(not self.native.training and not any(p.requires_grad for p in self.native.parameters()), 'Native model must remain frozen/eval')
        self._captured.clear()
        with torch.no_grad():
            output=self.native(images)
        require(len(self._captured)==1 and self._captured[0].shape==(len(images),300,256), 'Final query hook did not fire exactly once')
        require(output['pred_logits'].shape==(len(images),300,80) and output['pred_boxes'].shape==(len(images),300,4), 'Native trained outputs differ')
        return dict(camera=self._captured.pop(),camera_logits=output['pred_logits'].detach().clone(),
                    camera_boxes=output['pred_boxes'].detach().clone(),camera_present=torch.ones(len(images),dtype=torch.bool,device=images.device))


class DfineNativeCureV1(nn.Module):
    """One exact architecture for task-only, CEV supervision and dropout arms.

    There is intentionally no method-dependent architecture switch. Arms differ
    solely in their predeclared objective/exposure and start from one state.
    The pretrained detector is external, frozen, and never optimizer-owned here.
    """
    def __init__(self,config=None):
        super().__init__()
        self.config=DfineCureConfigV1() if config is None else config
        h=self.config.hidden
        self.query=nn.Parameter(torch.randn(300,h)*.02)
        self.camera=nn.Linear(256,h)
        self.camera_logits=nn.Linear(80,h)
        self.camera_boxes=nn.Linear(4,h)
        self.camera_norm=nn.LayerNorm(h)
        self.radar=nn.Sequential(nn.Linear(6,h),nn.GELU(),nn.Linear(h,h))
        self.register_buffer('radar_scale',torch.tensor([10.,10.,10.,50.,5.,10.]))
        self.position=nn.Linear(2,h)
        self.null_radar=nn.Parameter(torch.zeros(1,1,h))
        self.cev=nn.Sequential(nn.Linear(4*h,h),nn.LayerNorm(h),nn.GELU(),nn.Linear(h,4))
        self.gate=nn.Linear(2*h,2)
        self.refine=nn.Sequential(nn.Linear(3*h,h),nn.GELU(),nn.LayerNorm(h))
        self.class_delta=nn.Linear(h,2)
        self.box_delta=nn.Linear(h,4)
        for head in (self.class_delta,self.box_delta):
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)

    def contract(self):
        return dict(config=asdict(self.config),trainable_parameters=sum(p.numel() for p in self.parameters()),
                    radar_columns=['x','y','z','intensity','Doppler'],camera_feature_layer='decoder.dec_score_head[5] input',
                    class_semantics='80 independent sigmoid logits; shared residual indices0/1; no background channel',
                    native_detector_optimizer_owned=False,temporal_evidence_used=False)

    def forward(self,**inputs):
        require(set(inputs)==set(FIELDS), 'Explicit complete input contract required')
        require(inputs['camera'].is_cuda,'Neural adapter forward is CUDA-only; no CPU/MPS fallback')
        z=mask_native_inputs_v1(**inputs)
        b=len(z['camera']);h=self.config.hidden;present=z['camera_present'][:,None,None]
        cam=self.camera_norm(self.camera(z['camera'])+self.camera_logits(z['camera_logits'])+self.camera_boxes(z['camera_boxes']))
        cam=torch.where(present,cam,0.)
        q=self.query[None].expand(b,-1,-1)
        conditioned=q+cam
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
        attention=attention.masked_fill(~valid[:,None],float('-inf')).softmax(-1)
        local=attention@rad
        raw=self.cev(torch.cat((q,cam,local,(cam-local).abs()),-1))
        mean,logvar=raw[...,:2],raw[...,2:].clamp(-8.,5.)
        risk=mean-self.config.kappa*torch.exp(.5*logvar)
        bias=self.config.routing_bound*torch.tanh(risk/self.config.routing_temperature)
        available=torch.stack((z['camera_present'],z['radar_mask'].any(-1)),dim=-1)[:,None]
        weights=(self.gate(torch.cat((conditioned,local),-1))+bias).masked_fill(~available,-1e4).softmax(-1)
        weights=weights*available
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)
        refined=self.refine(torch.cat((q,weights[...,:1]*cam,weights[...,1:]*local),-1))
        logits,boxes=apply_native_residual_v1(z['camera_logits'],z['camera_boxes'],self.class_delta(refined),self.box_delta(refined),self.config)
        return dict(pred_logits=logits,pred_boxes=boxes,cev_mean=mean,cev_logvar=logvar,
                    modality_weights=weights,radar_attention=attention,routing_bias=bias)

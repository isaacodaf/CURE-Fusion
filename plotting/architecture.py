"""Source-bound scientific diagram of the frozen four-class D-FINE/CURE V2.

No model imports, learned execution, sensor imagery or result values are used.
Run with a Python environment containing matplotlib, numpy and Pillow.
"""
from pathlib import Path
import ast,hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Circle
from matplotlib.path import Path as MPath

import os
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
OUT=Path(os.environ['CURE_FIGURE_OUTPUT']);OUT.mkdir(exist_ok=True)
PROTOCOL='configs/protocols/DFINE_CURE_DEVELOPMENT_PROTOCOL_FROZEN_V2_20260913.json'
FILES=['code/cure_fusion/dfine_cure_v1.py','code/cure_fusion/dfine_cure_v2.py',
       'code/cure_fusion/dfine_cure_losses_v1.py','code/cure_fusion/dfine_cure_losses_v2.py',
       'code/cure_fusion/dfine_cure_development_v2.py','code/scripts/train_dfine_cure_development_cuda_v2.py']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
P=json.loads((ROOT/PROTOCOL).read_text())
for name in FILES:assert sha(ROOT/name)==P['files'][name],name

plt.rcParams.update({'font.family':'Arial','font.size':7.8,'mathtext.fontset':'dejavusans',
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'dfine-cure-frozen-v2-architecture',
 'lines.solid_capstyle':'round','savefig.facecolor':'white'})
INK='#20323D';MUTED='#52636D';EDGE='#5D6D76';FROZEN='#EFF2F5'
TEAL='#2B7378';TEAL_FILL='#EAF4F3';GOLD='#956723';GOLD_FILL='#FBF5E9';LIGHT='#D4DEE1'
fig=plt.figure(figsize=(7.16,5.95));ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,102);ax.set_ylim(0,84.8);ax.set_aspect('equal');ax.axis('off')
nodes={};edges=[];texts=[]

def txt(x,y,s,*,size=7.8,weight='normal',color=INK,ha='center',va='center',owner=None,rotation=0):
    t=ax.text(x,y,s,fontsize=size,fontweight=weight,color=color,ha=ha,va=va,linespacing=1.2,rotation=rotation)
    texts.append((t,owner));return t

def box(key,x,y,w,h,title,lines=(),*,fill='white',edge=EDGE,title_size=8.2,body_size=7.5):
    patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.0,rounding_size=0.7',lw=.8,facecolor=fill,edgecolor=edge,zorder=2)
    ax.add_patch(patch);nodes[key]=dict(x=x,y=y,width=w,height=h,title=title,lines=list(lines))
    title_y=y+h-2.65
    txt(x+w/2,title_y,title,size=title_size,weight='bold',owner=key)
    if lines:
        body_top=title_y-3.5;body_bottom=y+2.4
        locations=[(body_top+body_bottom)/2] if len(lines)==1 else [body_top-i*(body_top-body_bottom)/(len(lines)-1) for i in range(len(lines))]
        for yy,line in zip(locations,lines):txt(x+w/2,yy,line,size=body_size,owner=key)
    return patch

def arrow(key,points,*,kind='forward',color=INK,lw=.95,label=None,label_at=None):
    path=MPath(points,[MPath.MOVETO]+[MPath.LINETO]*(len(points)-1))
    patch=FancyArrowPatch(path=path,arrowstyle='-|>',mutation_scale=7.4,lw=lw,
                          linestyle=(0,(3,2.5)) if kind=='supervision' else '-',color=color,zorder=1)
    ax.add_patch(patch);edges.append(dict(id=key,points=points,kind=kind,label=label))
    if label is not None:txt(*label_at,label,size=7.0,color=color)

# Panel A: forward paths, with one native bypass and one evidence bus.
txt(3,82.0,'a',size=10,weight='bold',ha='left')
txt(7,82.0,'Inference: four-class CURE',size=10,weight='bold',ha='left')
arrow('legend_forward',[(62,82),(67,82)],lw=.9);txt(68,82,'forward',ha='left',size=7)
arrow('legend_supervision',[(82,82),(87,82)],kind='supervision',color=GOLD,lw=.9);txt(88,82,'supervision',ha='left',size=7)

box('dfine',3,63,19,12,'Frozen D-FINE-X',['RGB camera','Original trained detector'],fill=FROZEN)
box('native',26,63,22,12,'Native camera interface',['300 × 256 query features','80 logits / query; 300 boxes'],fill=FROZEN,body_size=7.4)
box('camera',26,49,22,9,'Camera features',['Projected query / logit / box'],fill=TEAL_FILL,edge=TEAL,body_size=7.0)
box('radar',3,36,19,17,'Paired radar',['XYZ, intensity, Doppler','64 slots + masks','Valid projected XY'],fill='white',body_size=7.2)
box('local',26,35,22,11,'Local radar features',['Point MLP + attention','Geometry bias; null token'],fill=TEAL_FILL,edge=TEAL,body_size=7.1)
box('cev',55,63,20,12,'Sensor utility head',[r'Mean $\mu_c,\mu_r$',r'Working variance $\sigma_c^2,\sigma_r^2$'],fill=TEAL_FILL,edge=TEAL,body_size=7.5)
box('routing',55,48,20,11,'Bounded mean routing',[r'$\beta_{im}=0.5\,\tanh(\mu_{im}/0.75)$','Learned gate + availability'],fill=TEAL_FILL,edge=TEAL,body_size=7.1)
box('refine',55,35,20,8,'Residual refinement',[r'$q_i,\ w_{ic}e_i^c,\ w_{ir}e_i^r$'],fill=TEAL_FILL,edge=TEAL,body_size=7.5)

# A shared container represents three parallel heads, not a serial cascade.
box('heads',82,48,18,27,'Adapted heads',[],fill=TEAL_FILL,edge=TEAL,title_size=8.1)
txt(91,68.6,'Native residuals',weight='bold',size=7.8,owner='heads')
txt(91,65.5,'0/1 logits: bounded',size=7.2,owner='heads')
txt(91,62.5,'Box residuals: bounded',size=7.0,owner='heads')
txt(91,59.5,'Both heads: zero init',size=7.1,owner='heads')
ax.plot([83.6,98.4],[57.4,57.4],color=TEAL,lw=.65,zorder=3)
txt(91,55.2,'Novel class logits',weight='bold',size=7.8,owner='heads')
txt(91,52.4,'Slidecar / doll: linear',size=7.2,owner='heads')
txt(91,49.9,'Unrestricted; bias −8',size=7.2,owner='heads')
box('decode',82,35,18,11,'Four task classes',['82 sigmoids → top 300','Then map 0/1/80/81'],fill='white',edge=TEAL,body_size=7.0)

arrow('image_to_native',[(22,69),(26,69)])
arrow('native_camera_embedding',[(37,63),(37,58)])
arrow('native_residual_base',[(37,75),(37,78.3),(91,78.3),(91,75)],color=MUTED,label='Original logits and boxes as the residual base',label_at=(64,79.7))
arrow('camera_to_local_attention',[(37,49),(37,46)],color=TEAL)
arrow('radar_to_local',[(22,43),(24,43),(24,40.5),(26,40.5)])
# A direction-neutral evidence trunk preserves both feature inputs at each branch.
# Arrowheads appear only at its consuming modules, avoiding a false serial flow.
ax.plot([48,51.5],[53.5,53.5],color=TEAL,lw=.95,zorder=1)
ax.plot([48,51.5],[40.5,40.5],color=TEAL,lw=.95,zorder=1)
ax.plot([51.5,51.5],[40.5,68.5],color=TEAL,lw=.95,zorder=1)
edges.append(dict(id='shared_evidence_bus',kind='forward_connector',inputs=['camera','local'],consumers=['cev','routing','refine'],arrowhead='only at consuming modules'))
for y in [40.5,53.5]:ax.add_patch(Circle((51.5,y),.22,facecolor=TEAL,edgecolor='none',zorder=4))
arrow('evidence_to_cev',[(51.5,68.5),(55,68.5)],color=TEAL)
arrow('content_evidence_to_routing',[(51.5,53.5),(55,53.5)],color=TEAL)
arrow('mean_only_to_routing',[(65,63),(65,59)],color=TEAL,label=r'$\mu$ only; $\kappa=0$',label_at=(69.9,60.5))
arrow('evidence_to_refinement',[(51.5,40.5),(53.2,40.5),(53.2,39),(55,39)],color=TEAL)
arrow('weights_to_refinement',[(65,48),(65,43)],color=TEAL,label=r'$w_c,w_r$',label_at=(70.0,45.5))
arrow('refinement_to_parallel_heads',[(75,39),(78.7,39),(78.7,61.5),(82,61.5)],color=TEAL)
arrow('heads_to_82_decode',[(91,48),(91,46)],color=TEAL)
txt(3,32.2,'Camera features and all camera-derived channels are masked when the camera is absent.',ha='left',size=7.1,color=MUTED)

# Panel B: target generation and clean-observation student supervision.
ax.plot([3,100],[29.5,29.5],lw=.65,color=LIGHT)
txt(3,26.9,'b',size=10,weight='bold',ha='left')
txt(7,26.9,'Teacher targets and factual utility supervision',size=10,weight='bold',ha='left')
box('teacher',3,10,21,13.5,'Frozen fusion teacher',['Task-only endpoint','Camera removal × 1','Radar thinning × 8 (fit)'],fill=GOLD_FILL,edge=GOLD,body_size=7.3)
box('risk',28.5,10,22,13.5,'Fixed object risk',['Same factual assignment','Same VFL quality / weights',r'$\Delta R_m=R_m-R_0$'],fill=GOLD_FILL,edge=GOLD,body_size=7.1)
box('targets',55,10,20,13.5,'Signed utility targets',[r'$\hat v_m=\mathrm{mean}(\Delta R_m)$',r'$u_c^2=0$',r'$u_r^2=s_r^2/8$'],fill=GOLD_FILL,edge=GOLD,body_size=7.5)
box('likelihood',80,10,20,13.5,'Utility-mean likelihood',[r'$V_m=\sigma_m^2+u_m^2$',r'$\frac{1}{2}[(\mu_m-\hat v_m)^2/V_m$',r'$+\log V_m]$'],fill=GOLD_FILL,edge=GOLD,body_size=7.8,title_size=7.9)
box('student',55,1.8,20,5.2,'Factual student: μ, σ²',[],fill=TEAL_FILL,edge=TEAL,title_size=7.8)
arrow('teacher_to_fixed_risk',[(24,16.8),(28.5,16.8)],color=GOLD)
arrow('risk_to_moments',[(50.5,16.8),(55,16.8)],color=GOLD)
arrow('targets_to_likelihood',[(75,16.8),(80,16.8)],color=GOLD)
arrow('student_value_outputs',[(75,4.4),(90,4.4),(90,10)],color=TEAL)
arrow('clean_cev_supervision',[(84,10),(84,7.9),(65,7.9),(65,7)],kind='supervision',color=GOLD)
txt(3,5.9,'Assessment uses 16 fresh radar draws.',ha='left',size=7.5,color=MUTED)
txt(3,2.8,'Same target, with fresh intervention draws.',ha='left',size=7.2,color=MUTED)

fig.canvas.draw();renderer=fig.canvas.get_renderer();violations=[]
for artist,owner in texts:
    if owner is None:continue
    n=nodes[owner];bb=artist.get_window_extent(renderer).transformed(ax.transData.inverted())
    if bb.x0<n['x']+.45 or bb.x1>n['x']+n['width']-.45 or bb.y0<n['y']+.35 or bb.y1>n['y']+n['height']-.35:
        violations.append(dict(text=artist.get_text(),owner=owner,bounds=[bb.x0,bb.y0,bb.x1,bb.y1]))
assert not violations,violations
for i,(artist,owner) in enumerate(texts):
    if owner is None:continue
    a=artist.get_window_extent(renderer)
    for other,other_owner in texts[i+1:]:
        if owner!=other_owner:continue
        b=other.get_window_extent(renderer)
        assert not a.overlaps(b),(owner,artist.get_text(),other.get_text())
for ext in ('pdf','svg','png'):
    metadata={'Creator':'Source-bound CURE-Fusion figure script'}
    if ext=='pdf':metadata.update(CreationDate=None,ModDate=None)
    if ext=='svg':metadata.update(Date=None)
    fig.savefig(OUT/f'fig1_csu_architecture_v1.{ext}',dpi=600,metadata=metadata)
# Lower-resolution inspection derivative only; scientific exports above stay600dpi/vector.
fig.savefig(OUT/'inspection_v1.png',dpi=170)

def anchors(name):
    tree=ast.parse((ROOT/name).read_text());return {n.name:dict(line=n.lineno,end_line=n.end_lineno) for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
prov=dict(title='Four-class CURE: counterfactual sensor utility and teacher supervision',
 scope='Implementation schematic for the separate frozen four-class development study; distinct from historical four-draw CURE. No result values or generated sensor scenes.',
 source_protocol=dict(path=PROTOCOL,sha256=sha(ROOT/PROTOCOL)),
 source_files={f:dict(sha256=sha(ROOT/f),anchors=anchors(f)) for f in FILES},
 script_sha256=sha(__file__),nodes=nodes,edges=edges,
 notation={'q_i':'Learned300query embeddings','e_i_c':'Embedded frozen camera query, original logits and boxes','e_i_r':'Attended radar features with valid-geometry bias and null token','mu':'Signed expected object-risk increase under the declared single-modality intervention','sigma2':'exp(clamped working logvariance); not epistemic uncertainty','u2':'Intervention mean sampling variance; deterministic camera0, eight-draw radar variance/8','w':'Learned content-gate logits plus bounded mean bias, softmax with sensor availability'},
 exact_constants={'native_queries':300,'native_query_channels':256,'native_logits':80,'adapted_logits':82,'task_channels':[0,1,80,81],'radar_input_columns':['x','y','z','intensity','Doppler'],'radar_capacity_slots':64,'risk_penalty_kappa':0,'routing_bound':.5,'routing_temperature':.75,'shared_logit_bound':2.,'center_bound':.1,'log_size_bound':.2,'novel_weight_init':0,'novel_bias_init':-8,'training_radar_draws':8,'fresh_calibration_radar_draws':16,'deterministic_camera_evaluations_per_object':1},
 omitted_for_readability=['MLP/LayerNorm/GELU dimensions; retained exactly in the source','Full matched clean+.25augmented VFL+5L1+2GIoU task objective and .1CEV weight; specified in caption','Per-object teacher targets remapped to clean student queries through ground-truth object identity','Unrelated78 logits remain exact copies; shared query-box residuals can alter their geometry'],
 explicit_boundaries=['Working variance has no routing path at kappa0','Frozen task-only fusion endpoint supplies teacher targets; frozen camera-only native detector is not the radar-utility teacher','Teacher input includes factual evaluation as well as the listed interventions','Camera absence retains V2 centered-anchor limitation; no radar-autonomous or native3D claim','No temporal/weather/full-corruption efficacy claim','Original80 projection identity at zero residual; adapted82 globaltop300 is a separate postprocessor'],
 typography={'family':'Arial','figure_width_inches':7.16,'figure_height_inches':5.95,'minimum_node_text_points':7.0,'png_dpi':600},
 QA={'automatic_text_within_node_bounds':True,'automatic_node_text_nonoverlap':True,'model_constructed':False,'learned_forward_executed':False,'visual_inspection_pending':True},
 artifacts={f.name:dict(sha256=sha(f),bytes=f.stat().st_size) for f in OUT.glob('fig1_csu_architecture_v1.*')})
for name in FILES:assert sha(ROOT/name)==P['files'][name],name
(OUT/'ARCHITECTURE_PROVENANCE_V1.json').write_text(json.dumps(prov,indent=2)+'\n')
print(json.dumps(dict(status='vector_and_600dpi_exports_generated',text_containment_passed=True,artifacts=prov['artifacts']),indent=2))

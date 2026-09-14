"""Fixed eight descriptive contrasts; pure arrays, no model or AP."""
import numpy as np
PREDICTORS=('CURE','trained_on_CURE_features','trained_on_Task_features')
MODALITIES=('camera_removal','radar_thinning')
METRICS=('RMSE','MAE')

def draws():
 rng=np.random.Generator(np.random.PCG64(9132026))
 indices=rng.integers(0,8,size=(2000,8),dtype=np.int64)
 weights=np.zeros((2000,8),np.int64)
 np.add.at(weights,(np.arange(2000)[:,None],indices),1)
 assert (weights.sum(1)==8).all()
 return indices,weights

def statistics(errors,groups):
 errors=np.asarray(errors,np.float64);groups=np.asarray(groups)
 assert errors.ndim==3 and errors.shape[1:]==(3,2) and np.isfinite(errors).all()
 assert groups.shape==(len(errors),) and groups.dtype.kind in 'iu' and ((groups>=0)&(groups<8)).all()
 counts=np.bincount(groups,minlength=8).astype(np.int64)
 sums=np.zeros((8,3,2,2),np.float64)
 np.add.at(sums,groups,np.stack((errors**2,abs(errors)),-1))
 assert np.isfinite(sums).all()
 return counts,sums

def pool(counts,sums,weights):
 counts,sums,weights=np.asarray(counts),np.asarray(sums,np.float64),np.asarray(weights)
 assert counts.shape==(8,) and counts.dtype.kind in 'iu' and (counts>=0).all()
 assert sums.shape==(8,3,2,2) and np.isfinite(sums).all() and (sums>=0).all()
 assert weights.ndim==2 and weights.shape[1]==8 and weights.dtype.kind in 'iu' and (weights>=0).all()
 n=weights@counts;total=np.einsum('bg,gpmt->bpmt',weights.astype(np.float64),sums,optimize=False)
 value=np.full_like(total,np.nan);np.divide(total,n[:,None,None,None],out=value,where=n[:,None,None,None]>0);value[...,0]=np.sqrt(value[...,0])
 return n,value

def difference_and_relative(value):
 assert value.ndim==4 and value.shape[1:]==(3,2,2)
 c,b=value[:,:1],value[:,1:];relative=np.full_like(b,np.nan)
 np.divide(c,b,out=relative,where=np.isfinite(b)&(b>0))
 return c-b,100*(1-relative)

def interval(v):
 v=np.asarray(v,np.float64);assert v.ndim==1 and len(v)==2000
 finite=np.isfinite(v)
 return {'percentile95':np.quantile(v,[.025,.975],method='linear').tolist() if finite.all() else None,
 'defined_draws':int(finite.sum()),'undefined_draws':int((~finite).sum())}

def analyze(errors,groups):
 counts,sums=statistics(errors,groups);indices,weights=draws();omission=np.ones((8,8),np.int64)-np.eye(8,dtype=np.int64)
 n,point=pool(counts,sums,np.ones((1,8),np.int64));bn,boot=pool(counts,sums,weights);on,leave=pool(counts,sums,omission)
 pd,pr=difference_and_relative(point);bd,br=difference_and_relative(boot);od,orr=difference_and_relative(leave)
 rows=[];omissions=[]
 for j,baseline in enumerate(PREDICTORS[1:]):
  for m,modality in enumerate(MODALITIES):
   for k,metric in enumerate(METRICS):
    name=f'{modality}/{metric}/CURE_vs_{baseline}'
    rows.append({'contrast':name,'baseline':baseline,'modality':modality,'metric':metric,'CURE_error':float(point[0,0,m,k]),'baseline_error':float(point[0,j+1,m,k]),'CURE_minus_baseline':float(pd[0,j,m,k]),'percent_error_reduction':float(pr[0,j,m,k]),'difference_interval':interval(bd[:,j,m,k]),'relative_interval':interval(br[:,j,m,k])})
    for g in range(8):omissions.append({'contrast':name,'omitted_recording_index':g,'objects':int(on[g]),'CURE_error':float(leave[g,0,m,k]),'baseline_error':float(leave[g,j+1,m,k]),'CURE_minus_baseline':float(od[g,j,m,k]),'percent_error_reduction':float(orr[g,j,m,k])})
 assert len(rows)==8 and len(omissions)==64
 raw=dict(errors=np.asarray(errors,np.float64),group_index=np.asarray(groups,np.int64),counts=counts,sums=sums,draw_indices=indices,multiplicities=weights,draw_object_counts=bn,point=point,bootstrap_metrics=boot,bootstrap_difference=bd,bootstrap_relative=br,omission_object_counts=on,omission_metrics=leave,omission_difference=od,omission_relative=orr)
 return rows,omissions,raw

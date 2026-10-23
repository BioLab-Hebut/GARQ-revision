"""R80 B. New LOO information evaluation from immutable saved memberships."""
from pathlib import Path
import json,hashlib,time,datetime,shutil,os
import numpy as np,pandas as pd
from release_paths import output_root, env_path, load_config
R=output_root();O=R/'B';O.mkdir(exist_ok=True)
def dump(path,x): path.write_text(json.dumps(x,indent=2,allow_nan=False))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def loo(x,g):
 levels,inv,size=np.unique(g,return_inverse=True,return_counts=True)
 sums=np.zeros((len(levels),x.shape[1]),np.float64);np.add.at(sums,inv,x)
 valid=size[inv]>1;means=np.full_like(x,np.nan,dtype=np.float64)
 means[valid]=(sums[inv[valid]]-x[valid])/(size[inv[valid]]-1)[:,None]
 err=((x-means)**2).sum(1)
 return err,size[inv],size
# Tiny identity gate before real inputs; no scientific result is selected.
t=np.array([[1.,0.],[3.,0.],[0.,2.],[0.,4.]])
v,_,_=loo(t,np.array([0,0,1,1]));assert np.allclose(v,[4,4,4,4])
cfg=load_config('B_inputs_locked.py')
start=time.time()
records=[];audit=[]
for run in cfg['runs']:
 p=Path(run['path']);assert p.is_file(),str(p);assert sha(p)==run['sha256'],str(p)+' hash mismatch'
 data=pd.read_csv(p,dtype={'cell_id':str,'celltype':str,'metacell':str,'prediction_native':str})
 assert data.cell_id.is_unique and data[['cell_id','celltype','metacell','prediction_native']].notna().all().all()
 rec={**run,'data':data.set_index('cell_id')}
 records.append(rec);audit.append({k:v for k,v in run.items() if k!='data'})
assert len(records)==cfg['expected_runs']
dump(O/'INPUTS_LOCKED_BEFORE_METRICS.json',{'runs':audit,'series':cfg['series'],'all_inputs_present':True,'n_runs':len(records),'additional_D1_D12':'No complete additional paired series in the supplied named manifests; original all_rare_evaluation is aggregate only; no new full model fits.'})
quality=[];pops=[];metrics=[];binsout=[];refsout=[];cohorts=[]
bins=np.array([1,2,5,10,20,40,80,np.inf]);names=['1','2–4','5–9','10–19','20–39','40–79','80+']
for s in cfg['series']:
 runs=[r for r in records if r['series']==s['series']]
 refs={}
 if s['reference_type']=='E1B_verified_npy':
  rp=s['references'][0];assert sha(rp['cell_index_path'])==rp['cell_index_sha256']
  ids=pd.read_csv(rp['cell_index_path'],dtype={'cell_id':str}).cell_id.to_numpy()
  for r in s['references']:
   assert sha(r['path'])==r['sha256'];x=np.load(r['path']).astype(np.float64)
   assert x.shape[0]==len(ids) and np.isfinite(x).all()
   mod=r['name'].split('_')[1].upper();refs[mod]=x
   refsout.append(dict(series=s['series'],modality=mod,path=r['path'],sha256=r['sha256'],row_ids_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest(),transform='none; already standardized and row-L2-normalized archived E1B reference'))
 else:
  ids=pd.read_csv(s['cells'],dtype={'cell_id':str}).cell_id.to_numpy()
  digest=hashlib.sha256('\n'.join(ids).encode()).hexdigest()
  for r in s['references']:
   z=np.load(r['path'],allow_pickle=False);assert str(z['cell_ids_sha256'])==digest
   x=z['embedding'].astype(np.float64);norm=np.linalg.norm(x,axis=1,keepdims=True)
   x=np.divide(x,norm,out=np.zeros_like(x),where=norm>0)
   refs[r['name']]=x;refsout.append(dict(series=s['series'],modality=r['name'],path=r['path'],sha256=sha(r['path']),row_ids_sha256=digest,transform='per-row L2; same as archived geometry evaluator'))
 common=np.ones(len(ids),bool);aligned={};labels=None
 for r in runs:
  d=r['data'];assert set(d.index)==set(ids)
  d=d.loc[ids];g=d.metacell.to_numpy();y=d.celltype.to_numpy();pred=d.prediction_native.to_numpy()
  if labels is None:labels=y
  assert np.array_equal(labels,y)
  size=pd.Series(g).value_counts();nper=pd.Series(g).map(size).to_numpy()
  common&=nper>1;aligned[r['run']]=(g,nper)
  if 'expected_K'in r:assert len(size)==r['expected_K']
  pop=[]
  for lab in sorted(set(y)):
   a=y==lab;b=pred==lab;tp=int((a&b).sum());n=int(a.sum());npred=int(b.sum())
   pop.append(dict(series=s['series'],dataset=s['dataset'],run=r['run'],construction=r['construction'],seed=r['seed'],condition=r['condition'],celltype=lab,n_cells=n,recall=tp/n,precision=tp/npred if npred else 0.,f1=2*tp/(n+npred),frequency=n/len(y)))
  pops.extend(pop)
  quality.append(dict(series=s['series'],dataset=s['dataset'],run=r['run'],construction=r['construction'],seed=r['seed'],condition=r['condition'],requested_K=r['requested_K'],actual_K=len(size),n_cells=len(ids),size_min=int(size.min()),size_median=float(size.median()),size_P95=float(size.quantile(.95)),size_max=int(size.max()),singleton_cells=int((nper==1).sum()),singleton_fraction=float((nper==1).mean()),assigned_coverage=1.,balanced_accuracy=float(np.mean([p['recall']for p in pop])),macro_f1=float(np.mean([p['f1']for p in pop])) ))
 assert common.sum()>1
 cohorts.append(dict(series=s['series'],n_input=len(ids),n_common_valid=int(common.sum()),common_coverage=float(common.mean()),common_ids_sha256=hashlib.sha256('\n'.join(ids[common]).encode()).hexdigest()))
 np.savez_compressed(O/(s['series']+'_common_mask.npz'),valid=common)
 for mod,x in refs.items():
  center=x[common].mean(0);den=float(np.mean(np.sum((x[common]-center)**2,axis=1)))
  assert den>0
  runbins={r['run']:np.searchsorted(bins,aligned[r['run']][1],side='right')-1 for r in runs}
  supported=set.intersection(*[set(v[common].tolist()) for v in runbins.values()])
  for r in runs:
   g,nper=aligned[r['run']];err,_,_=loo(x,g)
   assert np.isfinite(err[common]).all()
   base=dict(series=s['series'],dataset=s['dataset'],run=r['run'],construction=r['construction'],condition=r['condition'],seed=r['seed'],modality=mod)
   metrics.append({**base,'n_common_valid':int(common.sum()),'normalizer_variance':den,'mean_LOO_squared_error':float(err[common].mean()),'normalized_LOO_error':float(err[common].mean()/den)})
   for bi,name in enumerate(names):
    keep=common&(runbins[r['run']]==bi)
    binsout.append({**base,'size_bin':name,'n_cells':int(keep.sum()),'supported_by_all_arms':bi in supported,'normalized_LOO_error':float(err[keep].mean()/den) if keep.any() else np.nan})
   np.savez_compressed(O/(r['run']+'_'+mod+'_cell_errors.npz'),loo_squared_error=err)
 for q in quality:
  if q['series']==s['series']:q['common_valid_coverage']=float(common.mean())
 print(s['series'],'evaluated',len(runs),'common',int(common.sum()),flush=True)
pd.DataFrame(quality).to_csv(O/'quality_and_sizes.tsv',sep='\t',index=False)
pd.DataFrame(pops).to_csv(O/'all_populations.tsv',sep='\t',index=False)
m=pd.DataFrame(metrics);m.to_csv(O/'modality_LOO_errors.tsv',sep='\t',index=False)
pd.DataFrame(binsout).to_csv(O/'size_conditional_LOO_errors.tsv',sep='\t',index=False)
pd.DataFrame(refsout).to_csv(O/'reference_manifest.tsv',sep='\t',index=False)
pd.DataFrame(cohorts).to_csv(O/'common_cohorts.tsv',sep='\t',index=False)
m.groupby(['series','dataset','run','construction','condition'],dropna=False).normalized_LOO_error.mean().rename('equal_modality_mean_auxiliary').reset_index().to_csv(O/'equal_modality_auxiliary.tsv',sep='\t',index=False)
# Reuse already audited geometry/quality tables unchanged, avoiding duplicate evaluation.
base=env_path('GARQ_REUSED_GEOMETRY', R/'external_geometry');reused=[]
for folder,files in [('rare_repeat_geometry',['run_modality_metrics.csv','all_population_metrics.csv','all_population_summary.csv','evaluation_manifest.json']),('focused_evaluation',['focused_modality_metrics.csv','focused_celltype_metrics.csv','focused_resolution.csv','manifest.json'])]:
 dest=O/'REUSED_GEOMETRY'/folder;dest.mkdir(parents=True,exist_ok=True)
 for f in files:
  p=base/folder/f
  if p.exists():shutil.copy2(p,dest/f);reused.append(dict(path=str(p),sha256=sha(p),status='REUSED_PREVIOUSLY_VERIFIED'))
base=env_path('GARQ_D17_DOWNWEIGHT_GEOMETRY', R/'external_D17_geometry')
dest=O/'REUSED_GEOMETRY/D17_downweight';dest.mkdir(parents=True,exist_ok=True)
for f in ['modality_metrics.csv','run_metrics.csv','run_mean_sd.csv','evaluation_manifest.json']:
 if (base/f).exists():shutil.copy2(base/f,dest/f);reused.append(dict(path=str(base/f),sha256=sha(base/f),status='REUSED_PREVIOUSLY_VERIFIED'))
pd.DataFrame(reused).to_csv(O/'reused_geometry_manifest.tsv',sep='\t',index=False)
dump(O/'STATUS.json',{'status':'COMPLETE','n_runs':len(records),'series':5,'new_training':0,'new_references':0,'n_modality_results':len(m),'seconds':time.time()-start,'scope':'descriptive same-reference shared-membership evaluation; not independent held-out feature generalization'})

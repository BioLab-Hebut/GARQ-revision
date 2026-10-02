#!/usr/bin/env python
"""Frozen embedding/anchor experiment isolates the implemented batch-local quantizer."""
import os,sys,json,time,hashlib
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score,mutual_info_score,balanced_accuracy_score
from scipy.stats import entropy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'source_server'))
from model import GARQuantizer
from analysis_metrics import load_annotations,majority_statistics
run=ROOT/'runs'/sys.argv[1];out=run/'frozen_stability';out.mkdir(exist_ok=True)
if (out/'completed.json').exists():
 print('ALREADY COMPLETE',run.name);raise SystemExit(0)
cfg=json.loads((run/'effective_config.json').read_text())
checkpoint=torch.load(run/'checkpoint.pt',map_location='cpu')
E=np.load(run/'embeddings.npy');reference=np.load(run/'assignments.npy')
cells=pd.read_csv(run/'cell_index.csv',dtype={'cell_id':str,'cell_type':str})
assert len(cells)==len(E) and cells.cell_id.is_unique
labels=cells.cell_type.to_numpy();annotations=load_annotations(run,cfg['label_key'])
assert np.array_equal(annotations.astype(str).to_numpy(),labels)
q=GARQuantizer(cfg['K'],E.shape[1],k_knn=5).cuda().eval()
state={k[len('quantizer.'):]:v for k,v in checkpoint['model_state'].items() if k.startswith('quantizer.') and k!='quantizer.alpha'}
q.load_state_dict(state,strict=True)
before=q.anchors.weight.detach().cpu().numpy().copy()
e=torch.from_numpy(E).cuda()
Bs=[512,1024,2048,4096];order_seeds=[None]+list(range(1701,1711))
if cfg['dataset'].startswith('synthetic'):Bs=[256,512,1024];order_seeds=[None,1701,1702]
def assignments(order,batch):
 ans=np.empty(len(E),np.int64)
 with torch.no_grad():
  for start in range(0,len(E),batch):
   ix=order[start:start+batch]
   ids,_,_=q(e[torch.from_numpy(ix).cuda()],return_assignment=True)
   ans[ix]=ids.cpu().numpy()
 return ans
baseline=assignments(np.arange(len(E)),cfg['eval_batch_size'])
mismatch=float(np.mean(baseline!=reference))
(out/'baseline_reproduction.json').write_text(json.dumps({'changed_fraction_vs_saved_inference':mismatch,'ARI_vs_saved_inference':adjusted_rand_score(reference,baseline),'encoder_check':json.loads((run/'encoder_independence.json').read_text()),'method':'Reuses fixed encoded cells and fixed trained anchors; repeats original GARQuantizer in eval mode; excludes decoder and re-training.'},indent=2))
if mismatch>0.001:raise RuntimeError('Frozen quantizer did not reproduce saved baseline; investigate before stability comparison')
rows=[];bytype=[]
def stats(ids):
 quality,pred,_,_=majority_statistics(ids,annotations)
 _,a=np.unique(reference,return_counts=True);_,b=np.unique(ids,return_counts=True)
 return {'ARI_assignment':adjusted_rand_score(reference,ids),'NMI_assignment':normalized_mutual_info_score(reference,ids),'VI_nats':max(0.0,entropy(a)+entropy(b)-2*mutual_info_score(reference,ids)),'changed_fraction':float(np.mean(reference!=ids)),**quality},pred

for B in Bs:
 for seed in order_seeds:
  order=np.arange(len(E)) if seed is None else np.random.default_rng(seed).permutation(len(E))
  torch.cuda.synchronize();t=time.perf_counter();ids=assignments(order,B);torch.cuda.synchronize();quantizer_seconds=time.perf_counter()-t
  metric,pred=stats(ids);tag='canonical' if seed is None else str(seed)
  np.save(out/f'assignments_B{B}_{tag}.npy',ids)
  row={'dataset':cfg['dataset'],'run_id':cfg['run_id'],'batch_size':B,'order_seed':tag,'quantizer_wall_seconds':quantizer_seconds,**metric}
  rows.append(row);pd.DataFrame(rows).to_csv(out/'stability.csv',index=False)
  for label in np.unique(labels):
   mask=labels==label
   bytype.append({'batch_size':B,'order_seed':tag,'cell_type':label,'n_cells':int(mask.sum()),'majority_recall':float(np.mean(pred[mask]==label)),'changed_fraction':float(np.mean(ids[mask]!=reference[mask]))})
  pd.DataFrame(bytype).to_csv(out/'cell_type_stability.csv',index=False)
  print(B,tag,'ARI',round(metric['ARI_assignment'],4),'balanced purity',round(metric['balanced_purity'],4),flush=True)
assert np.array_equal(before,q.anchors.weight.detach().cpu().numpy()),'Frozen anchors changed'
(out/'completed.json').write_text(json.dumps({'status':'complete','tests':len(rows),'batch_sizes':Bs,'permutation_seeds':order_seeds,'anchors_unchanged':True,'baseline_mismatch':mismatch},indent=2))

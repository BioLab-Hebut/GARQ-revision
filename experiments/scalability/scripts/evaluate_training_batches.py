#!/usr/bin/env python
import sys,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score,mutual_info_score,balanced_accuracy_score
from scipy.stats import entropy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'source_server'))
from model import GARQuantizer
from analysis_metrics import load_annotations,majority_statistics,refresh_native_quality
names=json.loads((ROOT/'configs/training_sensitivity_queue.json').read_text())
out=ROOT/'deliverables';out.mkdir(exist_ok=True)
rows=[];assign={};comparisons=[]
for name in names:
 cfg=json.loads((ROOT/'configs'/name).read_text());p=ROOT/'runs'/cfg['run_id']
 assert (p/'completed.json').exists(),str(p)+' is incomplete'
 checkpoint=torch.load(p/'checkpoint.pt',map_location='cpu')
 E=np.load(p/'embeddings.npy');cells=pd.read_csv(p/'cell_index.csv',dtype={'cell_id':str,'cell_type':str});labels=cells.cell_type.to_numpy()
 q=GARQuantizer(cfg['K'],E.shape[1],5).cuda().eval()
 q.load_state_dict({k[len('quantizer.'):]:v for k,v in checkpoint['model_state'].items() if k.startswith('quantizer.') and k!='quantizer.alpha'},strict=True)
 ids=np.empty(len(E),np.int64)
 with torch.no_grad():
  for start in range(0,len(E),2048):
   z=torch.from_numpy(E[start:start+2048]).cuda()
   ids[start:start+2048]=q(z,True)[0].cpu().numpy()
 np.save(p/'assignments_fixed_eval_B2048.npy',ids)
 annotations=load_annotations(p,cfg['label_key'])
 assert np.array_equal(annotations.astype(str).to_numpy(),labels)
 quality,pred,_,_=majority_statistics(ids,annotations)
 rows.append({'dataset':cfg['dataset'],'train_batch_size':cfg['batch_size'],'eval_batch_size':2048,'seed':cfg['seed'],'K_requested':cfg['K'],**quality})
 refresh_native_quality(p)
 assign[(cfg['dataset'],cfg['seed'],cfg['batch_size'])]=(ids,cells.cell_id.to_numpy())
 del q,checkpoint
for (ds,seed,B),(ids,cells) in assign.items():
 if B==512:continue
 ref,refcells=assign[(ds,seed,512)];assert np.array_equal(cells,refcells)
 _,a=np.unique(ref,return_counts=True);_,b=np.unique(ids,return_counts=True)
 comparisons.append({'dataset':ds,'seed':seed,'train_batch_size':B,'reference_train_batch_size':512,'eval_batch_size':2048,'ARI_assignment':adjusted_rand_score(ref,ids),'NMI_assignment':normalized_mutual_info_score(ref,ids),'VI_nats':max(0.0,entropy(a)+entropy(b)-2*mutual_info_score(ref,ids))})
pd.DataFrame(rows).to_csv(out/'Training_Batch_Quality_Fixed_Evaluation.csv',index=False)
pd.DataFrame(comparisons).to_csv(out/'Training_Batch_Assignment_Stability.csv',index=False)
print('FIXED_EVALUATION_COMPLETE',len(rows))

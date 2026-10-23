from release_paths import PACKAGE_ROOT, env_path, load_json, model_source
"""Controlled frozen-representation benchmark. Labels are used only after fitting."""
from pathlib import Path
import os,sys,json,time,argparse,hashlib,random,copy
os.environ.setdefault('OMP_NUM_THREADS','8');os.environ.setdefault('OPENBLAS_NUM_THREADS','8')
os.environ.setdefault('MPLBACKEND','Agg')
import numpy as np
ROOT=env_path('GARQ_R2C2_ROOT',PACKAGE_ROOT/'results')
BASE=model_source()

def save_json(path,x):
 path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(path)

def l2(x):
 x=np.asarray(x,np.float32);v=np.linalg.norm(x,axis=1,keepdims=True)
 if not np.isfinite(x).all() or (v<1e-10).any():raise ValueError('Invalid embedding')
 return np.ascontiguousarray(x/v)

def load_representation(ds,rep):
 if rep=='learned':
  path=ROOT/'learned_training'/f'{ds}_seed1'/'embedding.npz'
 else:path=ROOT/'inputs'/ds/f'{rep}.npz'
 d=np.load(path,allow_pickle=False)
 e=l2(d['embedding']);ids=d['cell_ids'];labels=d['labels']
 ref=np.load(ROOT/'inputs'/ds/'standard.npz',allow_pickle=False)
 if not np.array_equal(ids,ref['cell_ids']) or not np.array_equal(labels,ref['labels']):raise ValueError('Cell identity or label mismatch')
 if e.shape[1]!=64:raise ValueError('All controlled representations must have 64 dimensions')
 return e,ids,labels,path

def vectorized_relation(self,e):
 import torch
 import torch.nn.functional as F
 raw=F.normalize(e,dim=1).detach() @ F.normalize(self.anchors.weight,dim=1).T
 sim=raw+0.2*(self.build_cell_knn_graph(e) @ raw)
 idx=torch.topk(sim,2,dim=1).indices
 relations=torch.zeros_like(sim).scatter_(1,idx,1.0)
 return sim,relations

def check_quantizer_equivalence():
 import torch,types
 sys.path.insert(0,str(BASE));from model import GARQuantizer
 torch.manual_seed(11);x=torch.randn(73,16)
 original=GARQuantizer(9,16);fast=copy.deepcopy(original)
 fast.encode_relation=types.MethodType(vectorized_relation,fast)
 for training in [False,True]:
  original.train(training);fast.train(training)
  torch.manual_seed(812);a,la=original(x,False)
  torch.manual_seed(812);b,lb=fast(x,False)
  assert torch.allclose(a,b,atol=1e-6) and torch.allclose(la,lb,atol=1e-6)
  assert torch.allclose(original.anchors.weight,fast.anchors.weight,atol=1e-6)
  assert torch.allclose(original.anchor_usage,fast.anchor_usage,atol=1e-6)
 original.eval();fast.eval()
 assert torch.equal(original(x,True)[0],fast(x,True)[0])
 return {'forward_and_anchor_updates':'equal within 1e-6','assignments':'identical','cases':['evaluation','training with dynamic adjustment'],'note':'Only unused top-two relation construction is vectorized. Graph coefficient is fixed at the source implementation effective value 0.2.'}

def quantizer(x,k,seed,run):
 import torch,types,faiss
 sys.path.insert(0,str(BASE));from model import GARQuantizer
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 torch.set_num_threads(8)
 device='cuda' if torch.cuda.is_available() else 'cpu'
 q=GARQuantizer(k,x.shape[1],k_knn=5).to(device)
 q.encode_relation=types.MethodType(vectorized_relation,q)
 faiss.omp_set_num_threads(8)
 km=faiss.Kmeans(x.shape[1],k,niter=20,nredo=1,spherical=True,seed=seed,gpu=False)
 km.train(x);_,initial=km.index.search(x,1);initial=initial.ravel()
 centers=np.asarray(km.centroids,dtype=np.float32).copy()
 for j in range(k):
  if np.any(initial==j):centers[j]=x[initial==j].mean(axis=0)
 q.anchors.weight.data.copy_(torch.from_numpy(centers).to(device))
 opt=torch.optim.AdamW(q.parameters(),lr=1e-3,weight_decay=1e-2)
 tensor=torch.from_numpy(x).to(device);history=[]
 generator=np.random.default_rng(seed)
 for epoch in range(250):
  q.train();order=generator.permutation(len(x));loss_sum=0.;nb=0
  for start in range(0,len(x)-511,512):
   batch=tensor[torch.from_numpy(order[start:start+512]).to(device)]
   opt.zero_grad(set_to_none=True);_,loss=q(batch,False)
   if not torch.isfinite(loss):raise FloatingPointError('Non-finite quantizer loss')
   loss.backward();opt.step();loss_sum+=float(loss.detach());nb+=1
  history.append(loss_sum/nb)
  if epoch%50==0:print('Q',seed,k,'epoch',epoch,'loss',history[-1],flush=True)
 q.eval();labels=[]
 with torch.no_grad():
  for start in range(0,len(x),2048):labels.append(q(tensor[start:start+2048],True)[0].cpu().numpy())
 y=np.concatenate(labels)
 np.savez_compressed(run/'quantizer_state.npz',anchors=q.anchors.weight.detach().cpu().numpy(),usage=q.anchor_usage.cpu().numpy(),alignment_loss_history=np.asarray(history))
 return y,{'epochs':250,'training_batch_size':512,'inference_batch_size':2048,'graph_k':5,'graph_scope':'batch-local as in supplied GARQ','alpha_effective':0.2,'objective':'original GARQuantizer anchor alignment MSE with dynamic anchor updates; no representation refitting or decoders','optimizer':'AdamW lr=0.001 weight_decay=0.01','initialization':'FAISS spherical KMeans on full frozen representation, 20 iterations, one restart, matching seed; nonempty-cluster arithmetic means','device':device,'final_alignment_loss':history[-1]}

def fit(x,method,k,seed,run):
 random.seed(seed);np.random.seed(seed)
 if method=='GARQ_quantizer':return quantizer(x,k,seed,run)
 if method=='KMeans':
  from sklearn.cluster import KMeans
  model=KMeans(n_clusters=k,random_state=seed,n_init=20,max_iter=300,algorithm='lloyd')
  y=model.fit_predict(x);return y,{'n_init':20,'max_iter':300,'inertia':float(model.inertia_),'iterations':int(model.n_iter_)}
 if method=='SEACells':
  import anndata as ad,SEACells
  try:
   import cupy as cp
   cp.random.seed(seed)
  except ImportError:pass
  a=ad.AnnData(x.copy());a.obsm['X_fixed']=x.copy()
  model=SEACells.core.SEACells(a,build_kernel_on='X_fixed',n_SEACells=k,n_waypoint_eigs=10,convergence_epsilon=1e-3,use_gpu=True)
  model.construct_kernel_matrix();model.initialize_archetypes();model.fit(min_iter=10,max_iter=100)
  y=a.obs['SEACell'].astype(str).to_numpy()
  _,y=np.unique(y,return_inverse=True)
  return y,{'kernel_input':'exact common 64D matrix','min_iter':10,'max_iter':100,'convergence_epsilon':1e-3,'n_waypoint_eigs':10,'n_iterations':len(model.RSS_iters),'use_gpu':True}
 raise ValueError(method)

def worker(args):
 x,cell_ids,labels,source=load_representation(args.dataset,args.representation)
 run=ROOT/'runs'/f'{args.dataset}__{args.representation}__{args.method}__K{args.target_k}__s{args.seed}'
 run.mkdir(parents=True,exist_ok=True)
 if (run/'DONE.json').exists():print('Already done',run);return
 meta={'dataset':args.dataset,'representation':args.representation,'method':args.method,'seed':args.seed,'target_K':args.target_k,'dimensions':64,'input_sha256':hashlib.sha256(x.tobytes()).hexdigest(),'source_file':str(source),'n_cells':len(x),'selection_rule':'minimize absolute realized-K deviation only; no annotation-dependent choices','representation_fit_seed':1 if args.representation=='learned' else 0,'representation_frozen':True}
 save_json(run/'parameters.json',meta);attempts=[];requested=args.target_k
 for attempt in range(1,4):
  dest=run/f'attempt{attempt}';dest.mkdir(exist_ok=True)
  start=time.time();y,settings=fit(x,args.method,requested,args.seed,dest)
  if len(y)!=len(x):raise ValueError('Incomplete assignment')
  y=np.asarray(y,dtype=np.int64);realized=len(np.unique(y))
  record={'attempt':attempt,'requested_K':requested,'realized_K':realized,'target_K':args.target_k,'seconds':time.time()-start,'settings':settings}
  np.savez_compressed(dest/'assignments.npz',metacell=y,cell_ids=cell_ids,labels=labels)
  save_json(dest/'run.json',record);attempts.append(record)
  if abs(realized-args.target_k)/args.target_k<=.05:break
  requested=max(2,min(len(x)-1,int(round(requested*args.target_k/max(1,realized)))))
 best=min(attempts,key=lambda r:abs(r['realized_K']-args.target_k))
 record=dict(meta,selected_attempt=best['attempt'],requested_K=best['requested_K'],realized_K=best['realized_K'],within_five_percent=abs(best['realized_K']-args.target_k)/args.target_k<=.05,attempts=attempts)
 save_json(run/'DONE.json',record);print(json.dumps(record),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--dataset');p.add_argument('--representation');p.add_argument('--method');p.add_argument('--target-k',type=int);p.add_argument('--seed',type=int)
 a=p.parse_args()
 if a.check:
  result=check_quantizer_equivalence();save_json(ROOT/'quantizer_equivalence.json',result);print(json.dumps(result))
 else:worker(a)

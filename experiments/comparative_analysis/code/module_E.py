"""Prespecified D5/D11 anchor-only three-arm mechanism experiment, 18 attempts max."""
from pathlib import Path
import os,time,json,hashlib,copy,types,traceback,datetime
os.environ.setdefault('OMP_NUM_THREADS','8');os.environ.setdefault('OPENBLAS_NUM_THREADS','8')
import numpy as np,pandas as pd,torch,torch.nn.functional as F,faiss
from sklearn.metrics import f1_score,recall_score
from sklearn.neighbors import NearestNeighbors
from settings_io import public_module
GARQuantizer = public_module('model.py','c629f59de63f4d6e3b3554a3a2b3c425b504ec4b92134167b60407b3dbbac1e8','garq_v2_public_model').GARQuantizer
from metrics_source import evaluate
from release_paths import output_root, env_path, load_config
R=output_root();O=R/'E'
INPUT=env_path('GARQ_CONTROL_INPUTS')
EXPECTED={'D5':'03bdd826323ef4f48f7038bc00aa6397e439adcc89e95b8ce8178bfbe384e860','D11':'cdf2c84ae5c7d3cca972a843b592b8bbbcccb74aeb6d11870a313305bb2bea31'}
ARMS=['Q_full','Q_no_manual_reposition','Q_no_graph']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now().astimezone().isoformat()
def dump(p,x):p.write_text(json.dumps(x,indent=2,default=lambda v:v.item() if hasattr(v,'item') else str(v)))
def relation(self,e):
 raw=F.normalize(e,dim=1).detach()@F.normalize(self.anchors.weight,dim=1).T
 sim=raw if self.no_graph else raw+.2*(self.build_cell_knn_graph(e)@raw)
 return sim,torch.zeros_like(sim).scatter_(1,torch.topk(sim,2,dim=1).indices,1.)
# The only forward delta is guarding manual updates separately from EMA.
import inspect
source=inspect.getsource(GARQuantizer.forward)
old='''        if self.training:
            self.anchor_usage.mul_(self.decay).add_(avg_probs, alpha=1 - self.decay)
            norm_distance'''
new='''        if self.training:
            self.anchor_usage.mul_(self.decay).add_(avg_probs, alpha=1 - self.decay)
        if self.training and self.manual_reposition:
            norm_distance'''
assert source.count(old)==1
import textwrap
ns={};exec(textwrap.dedent(source.replace(old,new)),{'torch':torch,'F':F},ns)
def make_q(k,d,arm,device):
 q=GARQuantizer(k,d,k_knn=5).to(device);q.no_graph=arm=='Q_no_graph';q.manual_reposition=arm!='Q_no_manual_reposition'
 q.encode_relation=types.MethodType(relation,q);q.forward=types.MethodType(ns['forward'],q)
 return q
def toy():
 torch.manual_seed(11);x=torch.randn(73,16);orig=GARQuantizer(9,16);q=make_q(9,16,'Q_full','cpu');q.load_state_dict(orig.state_dict())
 for training in [False,True]:
  orig.train(training);q.train(training)
  torch.manual_seed(812);a,l=orig(x,False)
  torch.manual_seed(812);b,j=q(x,False)
  assert torch.allclose(a,b,atol=1e-6) and torch.allclose(l,j,atol=1e-6)
  assert torch.allclose(orig.anchors.weight,q.anchors.weight,atol=1e-6)
  assert torch.allclose(orig.anchor_usage,q.anchor_usage,atol=1e-6)
 q=make_q(9,16,'Q_no_manual_reposition','cpu');q.train();before=q.anchors.weight.detach().clone()
 _,loss=q(x,False);assert torch.equal(before,q.anchors.weight);loss.backward();assert q.anchors.weight.grad.abs().sum()>0
 opt=torch.optim.AdamW(q.parameters(),lr=.001);opt.step();assert not torch.equal(before,q.anchors.weight)
 q=make_q(9,16,'Q_no_graph','cpu');assert torch.allclose(q.encode_relation(x)[0],F.normalize(x,dim=1)@F.normalize(q.anchors.weight,dim=1).T,atol=1e-6)
 return {'full_forward_and_updates':'equal original within1e-6','no_manual':'EMA retained, manual shifts disabled, anchor gradients nonzero and optimizer changes anchors','no_graph':'raw cosine assignment only, all other equations retained'}
def main():
 start=time.time()
 O.mkdir(exist_ok=True);torch.set_num_threads(8);faiss.omp_set_num_threads(8)
 assert torch.cuda.is_available(),'CUDA required for the recorded execution configuration';device='cuda:0'
 dump(O/'TECHNICAL_GATE.json',toy());(O/'forward_variant.py').write_text(textwrap.dedent(source.replace(old,new)))
 refs={d['dataset']:d['references'] for d in load_config('E_references.py')}
 allq=[];allp=[];allgeo=[];ledger=[];durations=[]
 for ds,k in [('D5',403),('D11',321)]:
  p=INPUT/ds/'standard.npz';assert sha(p)==EXPECTED[ds]
  z=np.load(p,allow_pickle=False);x=np.asarray(z['embedding'],np.float32);ids=z['cell_ids'].astype(str);labs=z['labels'].astype(str)
  assert x.shape==(len(ids),64) and len(set(ids))==len(ids) and np.isfinite(x).all()
  x=np.ascontiguousarray(x/np.linalg.norm(x,axis=1,keepdims=True))
  reference=[]
  for ref in refs[ds]:
   assert sha(ref['path'])==ref['sha256'];ci=pd.read_csv(ref['cell_index_path'],dtype=str)
   key='cell_id' if 'cell_id' in ci else ('cell' if 'cell' in ci else ci.columns[-1])
   ri=pd.Index(ci[key]).get_indexer(ids);assert (ri>=0).all() and len(set(ri))==len(ids)
   v=np.load(ref['path'])[ri];nn=NearestNeighbors(n_neighbors=16,metric='cosine',n_jobs=8).fit(v)
   neighbors=nn.kneighbors(v,return_distance=False);neighbors=np.stack([row[row!=i][:15] for i,row in enumerate(neighbors)])
   reference.append((ref['name'],v,neighbors))
  for seed in [0,1,2]:
   initdir=O/'shared_initialization'/f'{ds}_s{seed}';initdir.mkdir(parents=True,exist_ok=False)
   km=faiss.Kmeans(64,k,niter=20,nredo=1,spherical=True,seed=seed,gpu=False);km.train(x);_,initial=km.index.search(x,1);initial=initial.ravel()
   centers=km.centroids.copy()
   for j in range(k):
    if (initial==j).any():centers[j]=x[initial==j].mean(0)
   orders=np.stack([order for order in (np.random.default_rng(seed).permutation(len(x)),)]) # overwritten by one persistent RNG below
   rng=np.random.default_rng(seed);orders=np.stack([rng.permutation(len(x)) for _ in range(250)]).astype(np.int32)
   np.save(initdir/'anchors.npy',centers);np.save(initdir/'orders.npy',orders)
   statehash=sha(initdir/'anchors.npy');orderhash=sha(initdir/'orders.npy')
   for arm in ARMS:
    jid=f'{ds}_s{seed}_{arm}'
    assert len(ledger)<18;run=O/'runs'/jid;run.mkdir(parents=True,exist_ok=False)
    rec={'run_id':jid,'dataset':ds,'seed':seed,'arm':arm,'state':'STARTED','started':now(),'input_sha256':EXPECTED[ds],'initial_sha256':statehash,'orders_sha256':orderhash,'requested_K':k}
    ledger.append(rec);pd.DataFrame(ledger).to_csv(O/'FIT_ATTEMPTS.tsv',sep='\t',index=False)
    tick=time.time()
    try:
     torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);q=make_q(k,64,arm,device);q.anchors.weight.data.copy_(torch.from_numpy(centers).to(device))
     opt=torch.optim.AdamW(q.parameters(),lr=.001,weight_decay=.01);tensor=torch.from_numpy(x).to(device);hist=[]
     for ep in range(250):
      q.train();losses=[]
      for s in range(0,len(x)-511,512):
       b=tensor[torch.from_numpy(orders[ep,s:s+512].astype(np.int64)).to(device)];opt.zero_grad(set_to_none=True);_,loss=q(b,False)
       if not torch.isfinite(loss):raise FloatingPointError('Nonfinite alignment loss')
       loss.backward();opt.step();losses.append(float(loss.detach()))
      hist.append(np.mean(losses))
      if ep%50==0:print(jid,'epoch',ep,'loss',hist[-1],flush=True)
     q.eval()
     with torch.no_grad():y=np.concatenate([q(tensor[s:s+2048],True)[0].cpu().numpy() for s in range(0,len(x),2048)])
     np.savez_compressed(run/'assignments.npz',metacell=y,cell_ids=ids,labels=labs)
     np.savez_compressed(run/'quantizer_state.npz',anchors=q.anchors.weight.detach().cpu().numpy(),usage=q.anchor_usage.cpu().numpy(),loss=np.asarray(hist))
     summ,pop,groups=evaluate(labs,y,k);pred=dict(zip(groups.metacell,groups.majority_celltype));pred=np.array([pred[t] for t in y])
     summ['dominant_label_macro_F1']=f1_score(labs,pred,average='macro',zero_division=0)
     rare=[p for p in pop if p['abundance']<=.01];summ['rare_le1pct_mean_recall_ge50']=np.mean([p['recall'] for p in rare]) if rare else np.nan
     summ['rare_le1pct_mean_F1_ge50']=np.mean([p['F1'] for p in rare]) if rare else np.nan
     rec.update(state='SUCCEEDED',finished=now(),seconds=time.time()-tick,epochs=250)
     allq.append({**rec,**summ})
     for p0 in pop:allp.append({**{key:rec[key] for key in ['run_id','dataset','seed','arm']},**p0})
     sizes=np.bincount(y,minlength=k)
     for mod,v,neighbors in reference:
      raw=(y[neighbors]==y[:,None]).mean(1);chance=(sizes[y]-1)/(len(y)-1);adj=(raw-chance)/(1-chance)
      allgeo.append({'run_id':jid,'dataset':ds,'seed':seed,'arm':arm,'modality':mod,'raw_retention':raw.mean(),'size_adjusted_retention':adj.mean()})
     groups.to_csv(run/'sizes.tsv',sep='\t',index=False);dump(run/'DONE.json',{**rec,**summ})
     durations.append(rec['seconds']);del q,opt,tensor;torch.cuda.empty_cache()
    except Exception as e:
     rec.update(state='FAILED',finished=now(),seconds=time.time()-tick,error=str(e));(run/'ERROR.txt').write_text(traceback.format_exc());print('FAILED',jid,str(e),flush=True)
    pd.DataFrame(ledger).to_csv(O/'FIT_ATTEMPTS.tsv',sep='\t',index=False)
    pd.DataFrame(allq).to_csv(O/'all_run_metrics.tsv',sep='\t',index=False);pd.DataFrame(allp).to_csv(O/'all_population_metrics.tsv',sep='\t',index=False);pd.DataFrame(allgeo).to_csv(O/'all_modality_retention.tsv',sep='\t',index=False)
   if ds=='D5' and seed==0:
    if len(durations)!=3:dump(O/'PILOT_GATE.json',{'continue':False,'reason':'technical failure in complete3arm pilot'});dump(O/'STATUS.json',{'status':'PARTIAL_TECHNICAL','attempts':len(ledger),'successes':len(allq)});return
    dump(O/'PILOT_GATE.json',{'continue':True,'reason':'complete three-arm technical pilot passed; all remaining conditions retained','pilot_seconds':durations})
 dump(O/'STATUS.json',{'status':'COMPLETE' if len(allq)==18 else 'PARTIAL','attempts':len(ledger),'successes':len(allq),'full_model_fits':0,'seconds':time.time()-start,'scientific_scope':'standalone anchors in frozen64D standard multimodal input; no encoders/decoders; fixed250epoch endpoint'})
if __name__=='__main__':main()

#!/usr/bin/env python
"""Profile the frozen server implementation without changing numerical training code."""
import os, sys, json, time, random, argparse, traceback, resource, hashlib
from pathlib import Path
from contextlib import contextmanager
T0=time.perf_counter()
ROOT=Path(__file__).resolve().parents[1]
cfg=json.loads(Path(sys.argv[1]).read_text())
OUT=ROOT/'runs'/cfg['run_id']; OUT.mkdir(parents=True,exist_ok=True)
if (OUT/'completed.json').exists(): raise SystemExit('Completed run exists; refusing overwrite')
os.chdir(OUT)
os.environ.setdefault('MPLBACKEND','Agg')
sys.path.insert(0,str(ROOT/'source_server'))
stage_rows=[]; phase='imports'; graph_events=[]; graph_rows=[]; tensors=[]; data_cache={}; model_cache={}
def dump(name,value):
 p=OUT/name; tmp=p.with_suffix(p.suffix+'.tmp')
 tmp.write_text(json.dumps(value,indent=2,default=str));tmp.replace(p)
dump('state.json',{'phase':phase,'pid':os.getpid(),'started':time.time(),'config':cfg})
import numpy as np, torch, scanpy as sc, pandas as pd
import GARQ as G, data_utils as DU
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, balanced_accuracy_score
from sklearn.metrics import accuracy_score
dump('versions.json',{'python':sys.version,'torch':torch.__version__,'scanpy':sc.__version__,'numpy':np.__version__,'cuda':torch.version.cuda})
def sync():
 if torch.cuda.is_initialized(): torch.cuda.synchronize()
def flush_graph():
 global graph_events
 sync()
 for tag,a,b,n in graph_events:
  graph_rows.append({'phase':tag,'n':n,'gpu_ms':a.elapsed_time(b)})
 graph_events=[]
@contextmanager
def stage(name,extra=None):
 global phase
 old=phase;phase=name;sync();start=time.perf_counter()
 dump('state.json',{'phase':name,'pid':os.getpid(),'time':time.time(),**(extra or {})})
 try: yield
 finally:
  sync();elapsed=time.perf_counter()-start
  flush_graph()
  stage_rows.append({'phase':name,'seconds':elapsed,'parent_phase':old,
   'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
   'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0})
  pd.DataFrame(stage_rows).to_csv(OUT/'stages.csv',index=False)
  phase=old;dump('state.json',{'phase':old,'pid':os.getpid(),'time':time.time()})
def install_graph(net):
 original=net.quantizer.build_cell_knn_graph
 def graph(e):
  a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
  a.record();ans=original(e);b.record()
  graph_events.append((phase,a,b,int(e.shape[0])))
  return ans
 net.quantizer.build_cell_knn_graph=graph
 for i,enc in enumerate(net.encoders):
  def trace_forward(fn,idx,batch_first):
   def traced(src,*a,**kw):
    shape=list(src.shape)
    if not any(t['modality_index']==idx and t['input_shape']==shape for t in tensors):
     tensors.append({'modality_index':idx,'input_shape':shape,'batch_first':batch_first})
    return fn(src,*a,**kw)
   return traced
  enc.transformer.forward=trace_forward(enc.transformer.forward,i,bool(enc.transformer.self_attn.batch_first))
orig_load=G.load_data
orig_pre=DU.preprocess
orig_read=sc.read_h5ad
def read(*a,**k):
 with stage('read_h5ad',{'file':str(a[0]) if a else str(k)}): return orig_read(*a,**k)
sc.read_h5ad=read
def preprocess(adata,data_type):
 shape=list(adata.shape);dtype=str(adata.X.dtype)
 with stage('preprocess_'+data_type):
  result=orig_pre(adata,data_type)
 data_cache.setdefault('input_metadata',[]).append({'modality':data_type,'raw_shape':shape,'raw_dtype':dtype,'selected_shape':list(result[0].shape),'selected_dtype':str(result[0].dtype)})
 return result
DU.preprocess=preprocess
def load(args):
 with stage('load_and_preprocess'):
  result=orig_load(args)
 ad,train,ev,dims=result
 data_cache.update({'ad':ad,'train':train,'eval':ev,'dims':dims})
 for x in ad[1:]:
  if not np.array_equal(ad[0].obs_names.to_numpy(),x.obs_names.to_numpy()):raise RuntimeError('Modality cell ordering differs')
 pd.DataFrame({'row':np.arange(ad[0].n_obs),'cell_id':ad[0].obs_names.astype(str),'cell_type':ad[0].obs[args.type_key].astype(str).to_numpy()}).to_csv(OUT/'cell_index.csv',index=False)
 dump('effective_config.json',{**cfg,'train_batch_size':train.batch_size,'eval_batch_size':ev.batch_size,'train_shuffle':True,'train_drop_last':True,'eval_shuffle':False,'eval_drop_last':False,'num_workers':train.num_workers,'pin_memory':train.pin_memory,'input_metadata':data_cache['input_metadata']})
 return result
G.load_data=load
orig_init=G.init_gart_anchors
def init(*a,**kw):
 net=kw['model'];model_cache['net']=net;install_graph(net)
 with stage('anchor_initialization'):return orig_init(*a,**kw)
G.init_gart_anchors=init
for key in ['warm_one_epoch','train_one_epoch']:
 orig=getattr(G,key)
 def make_train(fn,label):
  def wrapped(*a,**kw):
   with stage(label,{'epoch':kw['epoch']+1,'max_epoch':cfg['epoch']}):
    value=fn(*a,**kw)
   if value is not None and not np.all(np.isfinite(value)):raise RuntimeError('Nonfinite epoch losses')
   if (kw['epoch']+1)%10==0:print('EPOCH',kw['epoch']+1,'TIME',round(time.perf_counter()-T0,1),flush=True)
   return value
  return wrapped
 setattr(G,key,make_train(orig,key))
orig_infer=G.inference
def infer(*a,**kw):
 net=kw['model']
 with stage('save_checkpoint'):
  torch.save({'model_state':{k:v.detach().cpu() for k,v in net.state_dict().items()},'input_dims':data_cache['dims'],'config':cfg,'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate()},OUT/'checkpoint.pt')
 with stage('inference'):
  result=orig_infer(*a,**kw)
 embeds,ids,delta,ratio,loss=result
 if not np.isfinite(embeds).all():raise RuntimeError('Nonfinite embeddings')
 with stage('save_reference_arrays'):
  np.save(OUT/'embeddings.npy',embeds);np.save(OUT/'assignments.npy',ids);np.save(OUT/'delta_confidence.npy',delta)
 # Controlled encoder-only check after baseline inference; preserve main outputs.
 with stage('encoder_batch_independence_check'):
  net.eval(); ds=data_cache['eval'].dataset;m=min(1024,len(ds))
  def enc(order,b):
   out=np.empty((m,embeds.shape[1]),np.float32)
   with torch.no_grad():
    for j in range(0,m,b):
     idx=order[j:j+b]; xs=[x[idx].to(kw['device']) for x in ds.x_list]
     h=net(xs);z=torch.cat(h,dim=1).detach().cpu().numpy();out[idx]=z
   return out
  order=np.arange(m);v1=enc(order,256);v2=enc(np.random.default_rng(20260914).permutation(m),512)
  dump('encoder_independence.json',{'n':m,'batch_sizes':[256,512],'max_abs_error':float(np.max(np.abs(v1-v2))),'allclose_atol_1e-5_rtol_1e-4':bool(np.allclose(v1,v2,atol=1e-5,rtol=1e-4))})
 return result
G.inference=infer
orig_agg=G.compute_metacell
def agg(*a,**kw):
 with stage('aggregation'):return orig_agg(*a,**kw)
G.compute_metacell=agg
orig_write=sc.AnnData.write_h5ad
def write(self,*a,**kw):
 with stage('write_h5ad'):return orig_write(self,*a,**kw)
sc.AnnData.write_h5ad=write
# These post-hoc diagnostic graphics do not contribute to learned assignments.
# Excluded consistently in every run; do not describe timing as the full plotting CLI.
excluded=['plot_metacell_umap','plot_metacell_size','plot_celltype_purity','plot_compactness_separation']
for key in excluded:setattr(G,key,lambda *a,**kw:None)
sc.pp.neighbors=lambda *a,**kw:None
sc.tl.umap=lambda *a,**kw:None
sc.pl.umap=lambda *a,**kw:None
args=argparse.Namespace(data_file=cfg['data_files'],data_type=cfg['modalities'],save_name=cfg['run_id'],
 n_GARQs=cfg['K'],type_key=cfg['label_key'],anchers_init='Kmeans',epoch=cfg['epoch'],batch_size=cfg['batch_size'],
 converge_threshold=10,seed=cfg['seed'],device='cuda',k_knn=5)
random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.random.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
try:
 with stage('core_pipeline'):
  G.main(args)
 ids=np.load(OUT/'assignments.npy');labels=data_cache['ad'][0].obs[args.type_key].astype(str).to_numpy()
 unique,counts=np.unique(ids,return_counts=True);pred=np.empty(len(ids),dtype=object);purity=[]
 for k in unique:
  ix=ids==k;values,n=np.unique(labels[ix],return_counts=True);pred[ix]=values[np.argmax(n)];purity.append(float(n.max()/n.sum()))
 quality={'K_requested':args.n_GARQs,'K_actual':len(unique),'n_cells':len(ids),'ARI_celltype':adjusted_rand_score(labels,ids),'NMI_celltype':normalized_mutual_info_score(labels,ids),'balanced_purity':balanced_accuracy_score(labels,pred),'cell_weighted_purity':accuracy_score(labels,pred),'mean_metacell_purity':float(np.mean(purity)),'size_min':int(counts.min()),'size_median':float(np.median(counts)),'size_mean':float(counts.mean()),'size_max':int(counts.max())}
 pd.DataFrame({'metacell':unique,'size':counts,'purity':purity}).to_csv(OUT/'metacell_sizes.csv',index=False)
 pd.DataFrame([{'cell_type':v,'n_cells':int(np.sum(labels==v)),'majority_recall':float(np.mean(pred[labels==v]==v))} for v in np.unique(labels)]).to_csv(OUT/'cell_type_quality.csv',index=False)
 dump('quality.json',quality);dump('tensor_shapes.json',tensors)
 pd.DataFrame(graph_rows).to_csv(OUT/'graph_timings.csv',index=False)
 dump('completed.json',{'status':'complete','wall_seconds_with_imports_and_checks':time.perf_counter()-T0,'core_pipeline_seconds':[x['seconds'] for x in stage_rows if x['phase']=='core_pipeline'][0],'main_ru_maxrss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'excluded_posthoc_visualization':excluded+['neighbors_for_visualization','umap'],'instrumentation':'CUDA events on every graph call, synchronized epoch boundaries; full metacell and assignment h5ad files written; checkpoint and encoder diagnostic separately timed','quality':quality})
 print('COMPLETED',cfg['run_id'],quality,flush=True)
except BaseException as e:
 dump('failed.json',{'error':repr(e),'traceback':traceback.format_exc(),'phase':phase,'elapsed':time.perf_counter()-T0});raise

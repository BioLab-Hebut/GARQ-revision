from release_paths import PACKAGE_ROOT, env_path, load_json, model_source
"""Export a reproducible learned representation using unmodified GARQ training."""
from pathlib import Path
import argparse,os,sys,time,json,hashlib,random
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('OMP_NUM_THREADS','8')
os.environ.setdefault('OPENBLAS_NUM_THREADS','8')
BASE=model_source()
ROOT=env_path('GARQ_R2C2_ROOT',PACKAGE_ROOT/'results')
parser=argparse.ArgumentParser();parser.add_argument('--dataset',required=True);parser.add_argument('--seed',type=int,default=1)
args=parser.parse_args()
import numpy as np,pandas as pd,torch
from types import SimpleNamespace
sys.path.insert(0,str(BASE))
pre=load_json(env_path('GARQ_R2C2_PREFLIGHT',PACKAGE_ROOT/'configs/preflight.py'))
cfg=pre['datasets'][args.dataset]['config']
run=ROOT/'learned_training'/f'{args.dataset}_seed{args.seed}'
run.mkdir(parents=True,exist_ok=True)
os.chdir(run)
if (run/'DONE.json').exists():
 print('Already completed',run);sys.exit(0)
source=(BASE/'GARQ.py').read_text()
cut='    if not os.path.exists("./save/"):'
assert source.count(cut)==1
training_source=source.split(cut)[0]+'    return net, adata_list, embeds, ids, delta_confs, rec_q_percent, loss_ancher\n'
(run/'training_through_inference.py').write_text(training_source)
namespace={'__name__':'garq_frozen_representation_export'}
exec(compile(training_source,str(run/'training_through_inference.py'),'exec'),namespace)
random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
torch.set_num_threads(8)
torch.cuda.reset_peak_memory_stats()
k={'D5':403,'D11':321}[args.dataset]
params=dict(device='cuda',data_file=cfg['paths'],data_type=cfg['mods'],n_GARQs=k,k_knn=5,epoch=300,batch_size=512,anchers_init='Kmeans',converge_threshold=10,type_key='celltype',save_name=f'{args.dataset}_seed{args.seed}')
(run/'parameters.json').write_text(json.dumps(params,indent=2))
start=time.time()
net,adata,embedding,ids,confidence,rec,loss=namespace['main'](SimpleNamespace(**params))
cell_ids=adata[0].obs_names.astype(str).to_numpy()
for a in adata[1:]:assert np.array_equal(cell_ids,a.obs_names.astype(str).to_numpy())
assert np.isfinite(embedding).all() and len(embedding)==cfg['n']
assert embedding.shape[1]==64
labels=adata[0].obs['celltype'].astype(str).to_numpy()
np.savez_compressed(run/'embedding.npz',embedding=embedding,cell_ids=cell_ids.astype(str),labels=labels.astype(str),assignments=ids.astype(int),confidence=confidence)
pd.DataFrame({'cell_id':cell_ids,'celltype':labels,'metacell':ids}).to_csv(run/'assignments.csv.gz',index=False)
torch.save(net.state_dict(),run/'checkpoint.pt')
info={'dataset':args.dataset,'seed':args.seed,'n_cells':len(ids),'embedding_dimensions':embedding.shape[1],'requested_K':k,'realized_K':len(np.unique(ids)),'seconds':time.time()-start,'quantized_reconstruction_ratio':float(rec),'anchor_loss':float(loss),'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'cell_ids_sha256':hashlib.sha256('\n'.join(cell_ids).encode()).hexdigest(),'source_model_sha256':pre['source_sha256']['model.py'],'training':'Original GARQ training and inference; output stage replaced only to save cell IDs, embeddings, checkpoint and parameters.'}
(run/'DONE.json').write_text(json.dumps(info,indent=2));print(json.dumps(info),flush=True)

from release_paths import PACKAGE_ROOT, env_path, load_json
from pathlib import Path
import argparse,os,time,json
os.environ.setdefault('OMP_NUM_THREADS','8');os.environ.setdefault('OPENBLAS_NUM_THREADS','8')
import numpy as np
from mofapy2.run.entry_point import entry_point
ROOT=env_path('GARQ_R2C2_ROOT',PACKAGE_ROOT/'results')
parser=argparse.ArgumentParser();parser.add_argument('--dataset',required=True);args=parser.parse_args()
dest=ROOT/'inputs'/args.dataset
data=np.load(dest/'mofa_inputs.npz',allow_pickle=False)
ent=entry_point()
ent.set_data_options(scale_views=True,center_groups=True,use_float32=False)
views=['RNA','ADT' if args.dataset=='D5' else 'ATAC']
ent.set_data_matrix([[data['block0'].astype(float)],[data['block1'].astype(float)]],likelihoods=['gaussian','gaussian'],views_names=views,groups_names=['paired_cells'],samples_names=[data['cell_ids'].tolist()])
ent.set_model_options(factors=64,spikeslab_factors=False,spikeslab_weights=True,ard_factors=False,ard_weights=True)
ent.set_train_options(iter=1000,convergence_mode='medium',seed=0,dropR2=None,verbose=False,quiet=True,weight_views=True)
start=time.time();ent.build();ent.run()
z=np.asarray(ent.model.getExpectations()['Z']['E'],dtype=np.float32)
assert z.shape==(len(data['cell_ids']),64) and np.isfinite(z).all()
z=z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
np.savez_compressed(dest/'MOFA.npz',embedding=z,cell_ids=data['cell_ids'],labels=data['labels'])
ent.save(str(dest/'mofa_model.hdf5'),save_data=False)
info={'dataset':args.dataset,'method':'MOFA+','input':'same two modality blocks as balanced concatenation','factors':64,'representation_seed':0,'normalization':'scale_views=True; weight_views=True; Gaussian views; posterior factors followed by per-cell L2 normalization','seconds':time.time()-start,'max_iterations':1000,'convergence_mode':'medium'}
(dest/'MOFA_metadata.json').write_text(json.dumps(info,indent=2));print(json.dumps(info),flush=True)

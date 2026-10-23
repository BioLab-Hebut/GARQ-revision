from release_paths import PACKAGE_ROOT, env_path, load_json
from pathlib import Path
import json,hashlib,os,sys
os.environ.setdefault('OMP_NUM_THREADS','8');os.environ.setdefault('OPENBLAS_NUM_THREADS','8')
import numpy as np,pandas as pd,anndata as ad
from sklearn.decomposition import PCA
ROOT=env_path('GARQ_R2C2_ROOT',PACKAGE_ROOT/'results')
pre=load_json(env_path('GARQ_R2C2_PREFLIGHT',PACKAGE_ROOT/'configs/preflight.py'))
def norm(x):
 x=np.asarray(x,dtype=np.float32);length=np.linalg.norm(x,axis=1,keepdims=True)
 if (length<1e-10).any():raise ValueError('zero vector')
 return x/length
def sha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
for ds,info in pre['datasets'].items():
 dest=ROOT/'inputs'/ds;dest.mkdir(parents=True,exist_ok=True)
 a=ad.read_h5ad(info['config']['fixed'])
 ids=a.obs_names.astype(str).to_numpy();labels=a.obs['celltype'].astype(str).to_numpy()
 assert info['fixed']['ids_match'] and info['fixed']['labels_equal_in_order']
 x=np.asarray(a.obsm['X_fixed'],np.float32)
 assert np.isfinite(x).all()
 blockkeys=['X_rna_pca','X_adt_pca' if ds=='D5' else 'X_atac_lsi']
 widths=[a.obsm[key].shape[1] for key in blockkeys]
 # Split the exact published fixed matrix to preserve its documented block balancing.
 blocks=np.split(x,np.cumsum(widths)[:-1],axis=1)
 pca=PCA(n_components=64,svd_solver='full')
 z=norm(pca.fit_transform(x))
 np.savez_compressed(dest/'standard.npz',embedding=z,cell_ids=ids.astype(str),labels=labels.astype(str))
 np.savez_compressed(dest/'mofa_inputs.npz',block0=blocks[0],block1=blocks[1],cell_ids=ids.astype(str),labels=labels.astype(str))
 np.savez_compressed(dest/'evaluation.npz',RNA=blocks[0],other=blocks[1],cell_ids=ids.astype(str),labels=labels.astype(str),modality=info['config']['mods'][1])
 pd.DataFrame({'cell_id':ids,'celltype':labels}).to_csv(dest/'cells.csv',index=False)
 details={'dataset':ds,'source_fixed_file':info['config']['fixed'],'source_fixed_metadata':info['fixed']['uns'],'source_matrix_sha256':sha(x),'shared_embedding_sha256':sha(z),'source_dimensions':x.shape[1],'dimensions':64,'pca64_retained_variance':float(pca.explained_variance_ratio_.sum()),'normalization':'PCA to common 64 dimensions, then per-cell L2 normalization; PCA is fitted without annotations.','block_widths':widths,'block_norm_ranges':[[float(np.linalg.norm(b,axis=1).min()),float(np.linalg.norm(b,axis=1).max())] for b in blocks],'n_cells':len(ids)}
 (dest/'standard_metadata.json').write_text(json.dumps(details,indent=2))
 print(json.dumps(details),flush=True)

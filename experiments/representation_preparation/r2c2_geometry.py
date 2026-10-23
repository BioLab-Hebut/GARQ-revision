from release_paths import PACKAGE_ROOT, env_path, load_json
"""Common per-modality Pearson geometry on supplied normalized/log counts."""
from pathlib import Path
import os,json,argparse,time
os.environ.setdefault('OMP_NUM_THREADS','8');os.environ.setdefault('OPENBLAS_NUM_THREADS','8')
import numpy as np,pandas as pd,anndata as ad
from scipy import sparse
from r2c2_evaluate import ROOT,jobs

def correlation(path,ds,mod,cache):
    if cache.exists():return np.load(cache,mmap_mode='r')
    start=time.time();a=ad.read_h5ad(path)
    ref=np.load(ROOT/'inputs'/ds/'standard.npz');assert np.array_equal(np.asarray(a.obs_names,str),ref['cell_ids'])
    x=sparse.csr_matrix(a.X,dtype=np.float32);n,p=x.shape
    assert np.isfinite(x.data).all() and x.data.min()>=0
    total=np.asarray(x.sum(1)).ravel();x=sparse.diags(np.divide(1e4,total,out=np.zeros_like(total),where=total>0)).dot(x).tocsr();x.data=np.log1p(x.data)
    sums=np.asarray(x.sum(1)).ravel().astype(np.float64)
    norm=np.sqrt(np.asarray(x.multiply(x).sum(1)).ravel().astype(np.float64)-sums*sums/p)
    if (norm<=0).any():raise ValueError('Zero variance count profile')
    print(ds,mod,'computing pairwise Pearson',x.shape,flush=True)
    c=np.empty((n,n),dtype=np.float32)
    if os.environ.get('GEOMETRY_USE_GPU')=='1':
        import torch
        torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
        dense=torch.from_numpy(x.toarray()).cuda();dense-=dense.mean(1,keepdim=True)
        dense/=torch.linalg.vector_norm(dense,dim=1,keepdim=True)
        with torch.no_grad():
            for startrow in range(0,n,1024):c[startrow:startrow+1024]=(dense[startrow:startrow+1024]@dense.T).cpu().numpy()
        del dense;torch.cuda.empty_cache()
    else:
        # Sparse products avoid materializing all ATAC features on the CPU.
        for startrow in range(0,n,512):
            end=min(startrow+512,n)
            dots=(x[startrow:end]@x.T).toarray().astype(np.float64)
            dots-=sums[startrow:end,None]*sums[None,:]/p
            dots/=norm[startrow:end,None]*norm[None,:]
            c[startrow:end]=np.clip(dots,-1,1)
    # Independent double-precision spot checks of the actual count profiles.
    ix=np.array([0,n//3,n//2,n-1]);expected=np.corrcoef(x[ix].toarray().astype(np.float64))
    max_error=float(np.max(np.abs(c[np.ix_(ix,ix)]-expected)))
    assert max_error<3e-5,max_error
    np.fill_diagonal(c,1);np.save(cache,c)
    cache.with_suffix('.json').write_text(json.dumps({'dataset':ds,'modality':mod,'source':path,'n_cells':n,'n_features':p,'normalization':'each cell total to 10000, log1p, Pearson correlation over all features in the provided input file','matrix_dtype':'float32','gpu':os.environ.get('GEOMETRY_USE_GPU')=='1','double_precision_spotcheck_max_error':max_error,'seconds':time.time()-start},indent=2))
    return c

def geom(c,y):
    n=len(y);clusters=np.unique(y);comp=0.;sep=0.;comp_ex=0.;ex_weight=0;rows=[]
    for cluster in clusters:
        ix=np.flatnonzero(y==cluster);size=len(ix)
        inside=np.asarray(c[np.ix_(ix,ix)]);avg=float(inside.mean(dtype=np.float64))
        block=np.asarray(c[ix,:]).copy();block[:,ix]=-np.inf
        d=float(np.mean(1-block.max(1),dtype=np.float64))
        comp+=avg*size/n;sep+=d*size/n
        excl=(float(inside.sum(dtype=np.float64))-size)/(size*(size-1)) if size>1 else np.nan
        if size>1:comp_ex+=excl*size;ex_weight+=size
        rows.append({'metacell':int(cluster),'size':size,'compactness_per_metacell':avg,'separation_per_metacell':d,'paper_compactness_scaled':avg*size/n*len(clusters),'paper_separation_scaled':d*size/n*len(clusters)})
    return {'compactness_cell_weighted':comp,'separation_cell_weighted':sep,'compactness_excluding_diagonal_cell_weighted':comp_ex/ex_weight,'compactness_paper_scaled_median':float(np.median([r['paper_compactness_scaled'] for r in rows])),'separation_paper_scaled_median':float(np.median([r['paper_separation_scaled'] for r in rows]))},rows

def main(ds,prepare_only):
    out=ROOT/'evaluation';out.mkdir(exist_ok=True);cache=ROOT/'geometry_cache';cache.mkdir(exist_ok=True)
    pre=load_json(env_path('GARQ_R2C2_PREFLIGHT',PACKAGE_ROOT/'configs/preflight.py'))['datasets'][ds]['config']
    c={mod:correlation(path,ds,mod,cache/f'{ds}_{mod}.npy') for mod,path in zip(pre['mods'],pre['paths'])}
    if prepare_only:return
    summary=[];mc=[]
    for name,meta,y,ids,labels in jobs():
        if meta['dataset']!=ds:continue
        for mod,corr in c.items():
            metrics,details=geom(corr,y);summary.append(dict(run=name,dataset=ds,modality=mod,**metrics))
            for row in details:mc.append(dict(run=name,dataset=ds,modality=mod,**row))
        print('EVALUATED',name,flush=True)
    pd.DataFrame(summary).to_csv(out/f'geometry_{ds}.csv',index=False)
    pd.DataFrame(mc).to_csv(out/f'geometry_metacells_{ds}.csv',index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--prepare-only',action='store_true');args=p.parse_args();main(args.dataset,args.prepare_only)

"""Uniform MOFA+ reanalysis of all construction controls using fixed features."""
import sys,json,time,contextlib,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'pythonlibs'))
import numpy as np,pandas as pd,anndata as ad,scanpy as sc
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score,adjusted_mutual_info_score,confusion_matrix
from mofapy2.run.entry_point import entry_point
from run_bimodal_downstream import canon
RUN=ROOT/'analysis';FEATURES=RUN/'MOFA_common_features.json'
rawroot=Path('/workspace/garq/data/RNA_ATAC_ADT/GSE158013')

def select_features():
    features={}
    for m,n in [('RNA',2500),('ATAC',3500),('ADT',None)]:
        a=ad.read_h5ad(rawroot/f'GSE158013_{m.lower()}.h5ad')
        if n:
            sc.pp.normalize_total(a,target_sum=10000);sc.pp.log1p(a)
            sc.pp.highly_variable_genes(a,n_top_genes=n,flavor='seurat')
            features[m]=a.var_names[a.var.highly_variable].tolist()
        else:features[m]=a.var_names.tolist()
    payload={'features':features,'selection':'Once on original paired cells, Scanpy seurat normalized-dispersion HVG; log1p total-normalized RNA/ATAC; all ADT. Identical lists in every arm and batch.','RNA_n':2500,'ATAC_n':3500,'ADT_n':46,'normalization':'RNA/ATAC total 10000 then log1p; ADT CLR log(x+1)-row_mean(log(x+1)); feature z-score ddof=1 clipped at +/-10','MOFA':{'factors':10,'likelihoods':'Gaussian','iterations_max':1000,'convergence_mode':'medium','scale_views':False,'spikeslab_weights':True,'seed':1},'clustering':{'method':'KMeans','n_clusters':12,'n_init':20,'random_state':1},'source_workflow':'/workspace/garq/vscode/MultiomeBenchmarking-main/three/run.py; /workspace/garq/vscode/MultiomeBenchmarking-main/0901/seacell/run.py','changes_for_control':'Fixed raw-cell feature lists and explicit deterministic fitting/clustering seeds; every arm receives identical measured views and settings.'}
    FEATURES.write_text(json.dumps(payload,indent=2));print('FEATURES_READY',{m:len(f) for m,f in features.items()},flush=True)

def preprocess(path,features,mod):
    a=ad.read_h5ad(path)
    assert a.var_names.is_unique
    if mod in ('RNA','ATAC'):
        sc.pp.normalize_total(a,target_sum=10000);sc.pp.log1p(a)
        x=a[:,features].X.toarray() if sparse.issparse(a.X) else np.asarray(a[:,features].X)
    else:
        x=a.X.toarray() if sparse.issparse(a.X) else np.asarray(a.X)
        x=np.log1p(x);x-=x.mean(axis=1,keepdims=True)
    x=np.asarray(x,dtype=np.float64);sd=x.std(axis=0,ddof=1);sd[sd==0]=1
    x=(x-x.mean(0))/sd;x=np.clip(x,-10,10)
    return np.asarray(x,dtype=np.float32),canon(a.obs_names),a.obs.celltype.astype(str).to_numpy()

def run_batch(tag):
    spec=json.loads(FEATURES.read_text());features=spec['features'];bd=RUN/tag
    arms=json.loads((bd/'MOFA_input_manifest.json').read_text())
    od=bd/'MOFA';od.mkdir(exist_ok=True);rows=[]
    for arm,paths in arms.items():
        start=time.time();folder=od/arm;folder.mkdir(exist_ok=True)
        if (folder/'metrics.json').exists():
            rows.append(json.loads((folder/'metrics.json').read_text()));continue
        print('MOFA_START',tag,arm,flush=True)
        matrices=[];refids=None
        for m in ['RNA','ATAC','ADT']:
            x,ids,labels=preprocess(paths[m],features[m],m)
            if refids is None:refids=ids;ref_labels=labels
            assert ids==refids and np.array_equal(labels,ref_labels)
            matrices.append([x])
        with (folder/'training.log').open('w') as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
            model=entry_point();model.set_data_options(scale_views=False,scale_groups=False,center_groups=True,use_float32=True)
            model.set_data_matrix(matrices,likelihoods=['gaussian']*3,views_names=['RNA','ATAC','ADT'],groups_names=['paired_cells'],samples_names=[refids],features_names=[[m+'::'+f for f in features[m]] for m in ['RNA','ATAC','ADT']])
            model.set_model_options(factors=10,spikeslab_weights=True,ard_weights=True)
            model.set_train_options(iter=1000,convergence_mode='medium',seed=1,gpu_mode=False,verbose=False,quiet=False,outfile=str(folder/'model.hdf5'))
            model.build();model.run();model.save(str(folder/'model.hdf5'),save_data=False)
            latent=model.model.nodes['Z'].getExpectation()
        assert latent.shape[0]==len(refids) and np.isfinite(latent).all()
        pred=KMeans(n_clusters=12,n_init=20,random_state=1).fit_predict(latent)
        y=pd.Categorical(ref_labels).codes
        mat=confusion_matrix(y,pred);ii,jj=linear_sum_assignment(-mat)
        row=dict(batch=tag,arm=arm,K=len(y),ARI=float(adjusted_rand_score(y,pred)),NMI=float(normalized_mutual_info_score(y,pred)),AMI=float(adjusted_mutual_info_score(y,pred)),ACC=float(mat[ii,jj].sum()/mat.sum()),seconds=time.time()-start)
        pd.DataFrame(latent,index=refids,columns=[f'Factor_{i+1}' for i in range(latent.shape[1])]).to_csv(folder/'latent.csv')
        pd.DataFrame({'metacell_id':refids,'celltype':ref_labels,'cluster':pred}).to_csv(folder/'clustering.csv',index=False)
        (folder/'metrics.json').write_text(json.dumps(row,indent=2));rows.append(row)
        pd.DataFrame(rows).to_csv(od/'MOFA_metrics.csv',index=False)
        print('MOFA_DONE',row,flush=True)
    (od/'DONE').write_text('All 10 arms completed')

if __name__=='__main__':
    if sys.argv[1]=='select':select_features()
    else:run_batch(sys.argv[1])

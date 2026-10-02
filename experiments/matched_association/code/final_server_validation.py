from pathlib import Path
import sys,json,hashlib,tarfile,importlib.metadata,datetime
import numpy as np,pandas as pd,h5py
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score,adjusted_mutual_info_score,confusion_matrix
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parent.parent;RUN=ROOT/'analysis';E=ROOT/'final_evidence';E.mkdir(exist_ok=True)
rows=[];models=[];status={}
for tag in ['full','matched_seed1','matched_seed2','matched_seed3','single_cell']:
    bd=RUN/tag
    status[tag]={'association_and_TF_done':(bd/'SUPPLIED_WORKFLOWS_DONE').exists(),'MOFA_done':(bd/'MOFA/DONE').exists()}
    assert all(status[tag].values()),status[tag]
    for p in sorted((bd/'MOFA').glob('*/metrics.json')):
        r=json.loads(p.read_text());d=pd.read_csv(p.parent/'clustering.csv');latent=pd.read_csv(p.parent/'latent.csv',index_col=0)
        assert len(d)==r['K'] and latent.shape==(r['K'],10) and np.isfinite(latent.to_numpy()).all()
        assert latent.index.tolist()==d.metacell_id.tolist()
        y=pd.Categorical(d.celltype).codes;pred=d.cluster.to_numpy();cm=confusion_matrix(y,pred);ii,jj=linear_sum_assignment(-cm)
        actual={'ARI':adjusted_rand_score(y,pred),'NMI':normalized_mutual_info_score(y,pred),'AMI':adjusted_mutual_info_score(y,pred),'ACC':cm[ii,jj].sum()/cm.sum()}
        assert all(abs(actual[k]-r[k])<1e-12 for k in actual)
        with h5py.File(p.parent/'model.hdf5') as h:
            elbo=np.asarray(h['training_stats']['elbo']);ev=elbo[np.isfinite(elbo)]
            assert len(ev)>0
        tf=pd.read_csv(bd/'TF_gene'/r['arm']/'fixed_metacell_order_lasso_coefficients.csv')
        assert tf.shape[0]==390 and tf.coefficient.notna().all() and tf['lambda'].notna().all()
        models.append({'batch':tag,'arm':r['arm'],'K':r['K'],'MOFA_latent_shape':list(latent.shape),'clustering_metrics_independently_recomputed':True,'TF_all_39_models_have_fitted_lambda':True,'last_finite_ELBO':float(ev[-1]),'finite_ELBO_evaluations':len(ev),'model_hdf5_sha256':hashlib.sha256((p.parent/'model.hdf5').read_bytes()).hexdigest()})
        rows.append(r)
        destination=E/tag/r['arm'];destination.mkdir(parents=True,exist_ok=True)
        for name in ['training.log','latent.csv','clustering.csv','metrics.json']:
            (destination/name).write_bytes((p.parent/name).read_bytes())
assert len(rows)==41
pd.DataFrame(rows).to_csv(E/'all_MOFA_metrics.csv',index=False)
(E/'execution_status.json').write_text(json.dumps(status,indent=2))
(E/'model_validation.json').write_text(json.dumps(models,indent=2))
versions={'python':sys.version,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
for pkg in ['anndata','scanpy','numpy','scipy','pandas','scikit-learn','matplotlib','h5py']:
    versions[pkg]=importlib.metadata.version(pkg)
versions['mofapy2']='0.7.2 (task-local installation from server source checkout)'
(E/'python_versions.json').write_text(json.dumps(versions,indent=2))
codefiles={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'code').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
(E/'executed_code_hashes.json').write_text(json.dumps(codefiles,indent=2))
with tarfile.open(ROOT/'final_evidence.tar.gz','w:gz') as tar:tar.add(E,arcname='final_evidence')
print(json.dumps({'complete_models':len(rows),'all_batches_finished':status,'output':str(E)},indent=2))

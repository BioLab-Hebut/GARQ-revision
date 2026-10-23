"""Reuse saved MetaQ memberships for previously specified D18 downstream controls.
No model training; read-only original inputs; all outputs isolated in revision13.
"""
from pathlib import Path
import os, sys, json, hashlib, shutil, traceback, time
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse

OUT = Path(os.environ.get('GARQ_METAQ_OUTPUT', '/path/to/new/metaq_matched'))
OLD = Path(os.environ.get('GARQ_ASSOCIATION_INPUT', '/path/to/matched_association_inputs'))
ASSIGN = Path(os.environ.get('GARQ_METAQ_ASSIGNMENTS', '/path/to/metaq/cell_assignments.csv'))
RAW = Path(os.environ.get('GARQ_D18_RAW', '/path/to/D18_raw'))
CODE=Path(os.environ.get('GARQ_ASSOCIATION_CODE', str(Path(__file__).resolve().parents[1]/'matched_association/code')))
sys.path.insert(0, str(CODE))
import run_bimodal_downstream as workflow
import mofa_controls as mofa

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    analysis=OUT/'analysis'; analysis.mkdir(exist_ok=True)
    features=analysis/'MOFA_common_features.json'
    if not features.exists(): shutil.copy2(OLD/'analysis/MOFA_common_features.json',features)
    assert sha(features)==sha(OLD/'analysis/MOFA_common_features.json')
    mofa.RUN=analysis; mofa.FEATURES=features
    raw={v:ad.read_h5ad(RAW/f'GSE158013_{v.lower()}.h5ad',backed='r') for v in ['RNA','ATAC','ADT']}
    ids=workflow.canon(raw['RNA'].obs_names); workflow.check_unique(ids,'raw RNA')
    labels=raw['RNA'].obs.celltype.astype(str).to_numpy()
    df=pd.read_csv(ASSIGN); a_ids=workflow.canon(df.cell_id)
    workflow.check_unique(a_ids,'MetaQ membership'); assert set(a_ids)==set(ids)
    by_id=dict(zip(a_ids,df.metacell_id.astype(str)))
    groups,_=workflow.normalize_groups([by_id[x] for x in ids])
    by_label=dict(zip(a_ids,df.cell_type.astype(str)))
    assert all(by_label[i]==y for i,y in zip(ids,labels))
    eligible=int(sum(np.sum(groups==g)>=20 for g in np.unique(groups)))
    assert len(ids)==25517 and groups.max()+1==505 and eligible==418
    metadata={'assignment_sha256':sha(ASSIGN),'native_assignment_seed':1,'requested_K':537,'occupied_K':505,'eligible_K_ge20':eligible,'matched_K':369,'cells_per_metacell':20,'sampling_seeds':[1,2,3],'cell_id_sha256':workflow.digest(ids),'common_features_sha256':sha(features),'source_hashes':{p.name:sha(p) for p in [CODE/'run_bimodal_downstream.py',CODE/'mofa_controls.py',CODE/'source/atac.ipynb',CODE/'source/adt.ipynb',CODE/'source/compare.R']}}
    (OUT/'provenance.json').write_text(json.dumps(metadata,indent=2))
    coverage=[]
    for seed in [1,2,3]:
        tag=f'matched_seed{seed}'; bd=analysis/tag; bd.mkdir(exist_ok=True)
        folder=bd/'MetaQ';folder.mkdir(exist_ok=True)
        g=workflow.sample_equal(groups,20,369,seed)
        assert np.sum(g>=0)==7380 and np.all(np.bincount(g[g>=0])==20)
        pd.DataFrame({'cell_id':ids,'metacell_id':g,'celltype':labels}).to_csv(folder/'memberships.csv.gz',index=False)
        paths={}
        for mod, a in raw.items():
            path=folder/f'D18_MetaQ_{mod}_matched_seed{seed}.h5ad'; paths[mod]=str(path)
            if not path.exists(): workflow.aggregate_view(a,g,ids,labels,path,'MetaQ','saved_native_membership')
        ag=ad.read_h5ad(paths['RNA']); predicted=ag.obs.celltype.astype(str).to_numpy()[g[g>=0]]
        retained=labels[g>=0]
        counts={ct:int(np.sum(retained==ct)) for ct in np.unique(labels)}
        recalls={ct:float(np.mean(predicted[retained==ct]==ct)) for ct in np.unique(retained)}
        coverage.append({'batch':tag,'arm':'MetaQ','seed':seed,'K':369,'assigned_cells':7380,'celltype_coverage':counts,'balanced_majority_accuracy':float(np.mean(list(recalls.values()))),'class_recalls':recalls})
        arms={'MetaQ':paths};(bd/'MOFA_input_manifest.json').write_text(json.dumps(arms,indent=2))
        print('ASSOCIATIONS_START',tag,flush=True)
        workflow.run_associations(bd,arms)
        print('TF_START',tag,flush=True)
        workflow.run_tf(bd,arms,'/usr/bin/Rscript',seed)
        print('MOFA_START',tag,flush=True)
        mofa.run_batch(tag)
        print('SEED_DONE',tag,flush=True)
        (OUT/'size_and_coverage.json').write_text(json.dumps(coverage,indent=2))
    for a in raw.values(): a.file.close()
    (OUT/'COMPLETE.json').write_text(json.dumps({'status':'PASS','seeds':3,'analyses':['MOFA','peak_gene','RNA_protein','TF_gene']},indent=2))

if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        (OUT/'FAILED.txt').write_text(traceback.format_exc());raise

"""Audited GSE158013 construction controls; all measured views share membership."""
from pathlib import Path
import sys,json,hashlib,datetime
import numpy as np,pandas as pd,anndata as ad
from scipy import sparse
from run_bimodal_downstream import canon,digest,check_unique,normalize_groups,sample_equal,map_groups,COMBOS

ROOT=Path(__file__).resolve().parent.parent
rawroot=Path('/workspace/garq/data/RNA_ATAC_ADT/GSE158013')
save=Path('/workspace/garq/vscode/GARQ20260905/save')
run=ROOT/'analysis'
run.mkdir(exist_ok=False)
raw={m:ad.read_h5ad(rawroot/f'GSE158013_{m.lower()}.h5ad') for m in ['RNA','ATAC','ADT']}
ids=canon(raw['RNA'].obs_names);labels=raw['RNA'].obs.celltype.astype(str).to_numpy()
assert len(ids)==25517
for m,a in raw.items():
    mid=canon(a.obs_names);check_unique(mid,m)
    assert set(mid)==set(ids)
    ix={x:i for i,x in enumerate(mid)};order=[ix[x] for x in ids]
    if order!=list(range(len(ids))):raw[m]=a[order].copy()
    raw[m].obs_names=ids
    assert np.array_equal(raw[m].obs.celltype.astype(str),labels)
    raw[m].X=sparse.csr_matrix(raw[m].X,dtype=np.float32)

mapping={f'GARQ_{c}':dict(method='GARQ',construction=c,path=str(save/f'GSE158013637_{c}_637metacell_k5_ids.h5ad'),column='metacell') for c in COMBOS}
for method,folder,fn,col in [('SEACells','3_seacell','GSE158013_ad.h5ad','SEACell'),('MetaCellV2','4_mc2','GSE158013_metacells2_ad.h5ad','metacell'),('SuperCell','5_supercell','GSE158013_supercell.h5ad','metacell')]:
    mapping[method]=dict(method=method,construction='RNA_ATAC_ADT',path=f'/workspace/garq/results/{folder}/three/GSE158013/{fn}',column=col)
groups={}
for arm,r in mapping.items():
    a=ad.read_h5ad(r['path'],backed='r')
    if arm=='SuperCell':
        xx=raw['RNA'].X.astype(float);xx=sparse.diags(1e4/np.asarray(xx.sum(1)).ravel())@xx;xx.data=np.log1p(xx.data)
        err=xx-sparse.csr_matrix(a.X[:,:])
        assert np.max(np.abs(err.data),initial=0)<1e-5 and a.var_names.tolist()==raw['RNA'].var_names.tolist()
        r['positional_validation']='entire X equals log1p(normalize_total(raw_RNA,10000)) within 1e-5; identical gene and celltype order'
    obs=a.obs.drop(columns=[c for c in ['metacell','membership','SEACell'] if c!=r['column'] and c in a.obs])
    vals,alignment,col=map_groups(obs,ids,labels,r['method']=='GARQ' or arm=='SuperCell')
    g,glabels=normalize_groups(vals);groups[arm]=g
    r.update(alignment=alignment,assignment_hash=digest(vals),file_sha256=hashlib.sha256(Path(r['path']).read_bytes()).hexdigest(),realized_K=len(glabels),unassigned_cells=int((g<0).sum()))
    a.file.close()

M=min(int((np.bincount(g[g>=0])>=20).sum()) for g in groups.values())
plan={'dataset':'GSE158013','raw_paths':{m:str(rawroot/f'GSE158013_{m.lower()}.h5ad') for m in raw},'raw_cell_id_hash':digest(ids),'raw_dimensions':{m:list(a.shape) for m,a in raw.items()},'raw_celltype_hash':digest(labels),'arms':mapping,'construction_seed':1,'construction_requested_K':637,'evaluation_views':list(raw),'aggregation':'arithmetic mean of raw count X; identical membership for all evaluation views','matched_K':M,'matched_size':20,'sampling_seeds':[1,2,3],'unimodal_downstream':'included','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(run/'run_plan.json').write_text(json.dumps(plan,indent=2))
print('VALIDATED',len(groups),'arms; exact matched K',M,'size 20',flush=True)
types=sorted(set(labels));ct_codes=np.array([types.index(x) for x in labels]);all_inputs={};summaries=[]
batches=[('full',groups,1)]+[(f'matched_seed{s}',{n:sample_equal(g,20,M,s) for n,g in groups.items()},s) for s in [1,2,3]]
for tag,batch,seed in batches:
    bd=run/tag;bd.mkdir();arm_inputs={}
    for arm,g in batch.items():
        folder=bd/arm;folder.mkdir();valid=g>=0;k=int(g.max())+1;counts=np.bincount(g[valid],minlength=k)
        membership=sparse.csr_matrix((np.ones(valid.sum(),np.float32)/counts[g[valid]],(g[valid],np.flatnonzero(valid))),shape=(k,len(ids)))
        confusion=sparse.csr_matrix((np.ones(valid.sum()),(g[valid],ct_codes[valid])),shape=(k,len(types))).toarray()
        maj=np.array(types)[confusion.argmax(1)];purity=confusion.max(1)/counts
        obs=pd.DataFrame({'n_cells':counts,'celltype':maj,'purity':purity,'metacell_id':[f'{arm}__{j}' for j in range(k)]})
        obs.index=obs.metacell_id
        pd.DataFrame({'cell_id':ids,'group':g,'celltype':labels}).to_csv(folder/'memberships.csv.gz',index=False)
        obs.to_csv(folder/'metacell_metadata.csv',index=False)
        arm_inputs[arm]={}
        for m,a in raw.items():
            x=membership@a.X
            out=ad.AnnData(x.astype(np.float32),obs=obs.copy(),var=a.var.copy())
            out.uns['provenance']={'dataset':'GSE158013','construction_modalities':mapping[arm]['construction'],'evaluation_modality':m,'source_assignment':mapping[arm]['path'],'aggregation':'mean_raw_counts','batch':tag,'seed':seed}
            p=folder/f'GSE158013__construct-{mapping[arm]["construction"]}__view-{m}__{tag}.h5ad'
            out.write_h5ad(p,compression='lzf');arm_inputs[arm][m]=str(p)
            print(tag,arm,m,list(out.shape),flush=True)
            del x,out
        correct=confusion[np.arange(k),confusion.argmax(1)]
        majority_for_cells=maj[g[valid]]
        bal=np.mean([np.mean(majority_for_cells[labels[valid]==ct]==ct) for ct in types if np.any(labels[valid]==ct)])
        summaries.append(dict(batch=tag,arm=arm,seed=seed,K=k,assigned_cells=int(valid.sum()),size_min=int(counts.min()),size_median=float(np.median(counts)),size_max=int(counts.max()),size_mean=float(counts.mean()),mean_metacell_purity=float(purity.mean()),balanced_majority_accuracy=float(bal),celltype_coverage={t:int(np.sum(valid&(labels==t))) for t in types}))
    (bd/'MOFA_input_manifest.json').write_text(json.dumps(arm_inputs,indent=2))
    all_inputs[tag]=arm_inputs
    (run/'all_input_manifests.json').write_text(json.dumps(all_inputs,indent=2))
    (run/'size_and_coverage.json').write_text(json.dumps(summaries,indent=2))
    print('BATCH_COMPLETE',tag,flush=True)
(run/'AGGREGATION_DONE').write_text(datetime.datetime.now().isoformat())

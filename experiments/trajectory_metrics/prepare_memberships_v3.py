import json, sys, hashlib, time
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

out=Path(sys.argv[1]); cfg=json.loads((out/'analysis_plan.json').read_text())
def canon(x): return pd.Series(x).astype(str).replace({'Tumor (Neuronal-like)':'Tumor Cells (Dedifferentiated)'}).to_numpy()
def dense(x): return x.toarray() if sparse.issparse(x) else np.asarray(x)
raw=ad.read_h5ad(cfg['single_rna'],backed='r')
rng=np.random.default_rng(20261006)
genes=sorted(set(cfg['gene_panel']) | set(np.asarray(raw.var_names)[rng.choice(raw.n_vars,65,replace=False)]))
cols=raw.var_names.get_indexer(genes); cols_order=np.argsort(cols)
blocks=[]
for start in range(0,raw.n_obs,512):
    blocks.append(sparse.csr_matrix(raw.X[start:start+512,:]))
full=sparse.vstack(blocks,format='csr');del blocks
X=full[:,cols].toarray().astype(np.float64)
totals=np.asarray(full.sum(axis=1)).ravel()
if np.any(totals<=0):raise ValueError('Zero RNA library')
L=np.log1p(X*(10000/totals[:,None]))
types=canon(raw.obs['celltype']); barcodes=raw.obs_names.astype(str).to_numpy()
assert len(set(barcodes))==len(barcodes)
members=pd.DataFrame({'cell':barcodes,'celltype':types})
report={'fixed_validation_genes':genes,'n_original':raw.n_obs,'methods':{}}
for name,m in cfg['methods'].items():
    print('VALIDATE',name,flush=True)
    a=ad.read_h5ad(m['members'],backed='r'); b=ad.read_h5ad(m['rna'],backed='r')
    if a.n_obs!=raw.n_obs: raise ValueError('membership cell count mismatch '+name)
    idx=a.obs_names.astype(str).to_numpy()
    numeric=np.array_equal(idx,np.arange(raw.n_obs).astype(str))
    if numeric:
        position=np.arange(raw.n_obs); mode='numeric original order candidate; group expression validation required'
    else:
        position=a.obs_names.get_indexer(barcodes)
        if np.any(position<0): raise ValueError('missing original barcode '+name)
        mode='exact original barcode join'
    if not np.array_equal(canon(a.obs.iloc[position]['celltype']),types): raise ValueError('original state ordering mismatch '+name)
    lab=a.obs.iloc[position][m['column']].astype(str).to_numpy()
    if np.any(pd.isna(lab)): raise ValueError('membership missing '+name)
    unique=sorted(set(lab)); label_pos={k:i for i,k in enumerate(unique)}
    count=np.zeros(len(unique)); sums=np.zeros((len(unique),len(genes)))
    z=np.array([label_pos[k] for k in lab]); np.add.at(count,z,1);np.add.at(sums,z,X)
    means=sums/count[:,None]
    log_sums=np.zeros_like(sums);np.add.at(log_sums,z,L)
    log_means=log_sums/count[:,None]
    bc=b.var_names.get_indexer(genes)
    if np.any(bc<0): raise ValueError('gene not found '+name)
    so=np.argsort(bc); Y=dense(b.X[:,bc[so]])[:,np.argsort(so)].astype(np.float64)
    if len(unique)!=b.n_obs: raise ValueError('occupied K mismatch '+name)
    tested=[]; chosen=None
    for aggregation,expected in [('mean_raw_counts',means),('sum_raw_counts',sums),('mean_log1p_norm10k',log_means),('sum_log1p_norm10k',log_sums)]:
        scale=np.maximum(np.max(np.abs(expected),axis=0),1.)
        dist=cdist(expected/scale,Y/scale)
        rr,cc=linear_sum_assignment(dist)
        dif=np.abs(expected[rr]-Y[cc]); tol=2e-5*np.maximum(1,np.abs(expected[rr]))
        ok=bool(np.all(dif<=tol))
        tested.append({'aggregation':aggregation,'all_within_float_tolerance':ok,'max_abs_error':float(dif.max()),'max_scaled_error':float((dif/np.maximum(1,np.abs(expected[rr]))).max())})
        if ok:
            if chosen is not None: raise ValueError('ambiguous aggregation '+name)
            # Each expected group must have exactly one output within tolerance.
            candidates=np.array([np.sum(np.all(np.abs(Y-e)<=2e-5*np.maximum(1,np.abs(e)),axis=1)) for e in expected])
            if not np.all(candidates==1): raise ValueError('ambiguous group identity '+name)
            chosen=(aggregation,rr,cc)
    if chosen is None:
        report['methods'][name]={'passed':False,'tested':tested,'index_mode':mode}
        (out/'membership_validation_v3.json').write_text(json.dumps(report,indent=2))
        raise ValueError('group expression does not reproduce RNA output '+name)
    aggregation,rr,cc=chosen
    remap={unique[r]:str(b.obs_names[c]) for r,c in zip(rr,cc)}
    mapped=np.array([remap[k] for k in lab]);members[name]=mapped
    exports=pd.DataFrame({'original_membership':unique,'rna_metacell':[remap[k] for k in unique],'n_original_cells':count.astype(int)})
    exports.to_csv(out/f'membership_export_mapping_{name}.csv',index=False)
    for col in ['n_cells','supercell_size']:
        if col in b.obs:
            expected_count={remap[k]:int(count[label_pos[k]]) for k in unique}
            if not all(int(b.obs.loc[k,col])==expected_count[str(k)] for k in b.obs_names): raise ValueError('output group size mismatch '+name)
    extra={}
    if name=='SuperCell':
        extra['original_input_file']=str(a.uns.get('input_file',''))
        extra['input_file']=str(m.get('input_file',extra['original_input_file']))
        source=ad.read_h5ad(extra['input_file'],backed='r')
        if not np.array_equal(source.obs_names.astype(str),barcodes):raise ValueError('SuperCell source barcodes differ')
        sc=source.var_names.get_indexer(genes); so=np.argsort(sc)
        sx=dense(source.X[:,sc[so]])[:,np.argsort(so)]
        if np.allclose(sx,X,rtol=2e-5,atol=2e-5):extra['input_value_convention']='raw counts'
        elif np.allclose(sx,L,rtol=2e-5,atol=2e-5):extra['input_value_convention']='per-cell log1p norm10k'
        else:raise ValueError('SuperCell input values do not match either verified original convention')
        ac=a.var_names.get_indexer(genes); so=np.argsort(ac)
        ax=dense(a.X[:,ac[so]])[:,np.argsort(so)]
        if not np.allclose(ax,sx,rtol=2e-5,atol=2e-5): raise ValueError('SuperCell membership row expression differs')
        source.file.close()
    indicator=sparse.csr_matrix((1/count[z],(z,np.arange(raw.n_obs))),shape=(len(unique),raw.n_obs))
    grouped=indicator@full
    by_export=np.empty(len(cc),dtype=int);by_export[cc]=rr
    derived=ad.AnnData(X=grouped[by_export,:].astype(np.float32),obs=b.obs.copy(),var=raw.var.copy())
    derived.uns['provenance']={'original_raw_rna':cfg['single_rna'],'original_membership':m['members'],'aggregation':'mean raw RNA counts, exact validated exported row IDs'}
    derived.write_h5ad(out/f'common_raw_{name}.h5ad',compression='gzip')
    report['methods'][name]={'passed':True,'index_mode':mode,'aggregation':aggregation,'n_metacells':len(unique),'tested':tested,**extra}
    (out/'membership_validation_v3.json').write_text(json.dumps(report,indent=2))
    a.file.close();b.file.close()
members.to_csv(out/'original_cell_memberships.csv',index=False)
report['passed']=True
(out/'membership_validation_v3.json').write_text(json.dumps(report,indent=2))
target=np.flatnonzero(np.isin(types,cfg['states']))
print('PREPARE_SINGLE',len(target),flush=True)
if not (out/'single_three_states.h5ad').exists():
    sub=ad.AnnData(X=full[target,:],obs=raw.obs.iloc[target].copy(),var=raw.var.copy());sub.obs['celltype']=types[target]
    sub.write_h5ad(out/'single_three_states.h5ad',compression='gzip')
raw.file.close()
print('PREPARE_SUCCESS',flush=True)

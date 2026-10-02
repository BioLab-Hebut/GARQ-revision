"""Export compact numerical evidence; independently verify aggregation and correlations."""
from pathlib import Path
import json,shutil,tarfile,hashlib,sys
import numpy as np,pandas as pd,anndata as ad
from scipy.stats import pearsonr
ROOT=Path(__file__).resolve().parent.parent;RUN=ROOT/'analysis';OUT=ROOT/'export'
OUT.mkdir(exist_ok=True)
for sub in ['data','source_panels','evidence','code']:(OUT/sub).mkdir(exist_ok=True)
for p in [RUN/'run_plan.json',RUN/'size_and_coverage.json',RUN/'MOFA_common_features.json']:
    if p.exists():shutil.copy2(p,OUT/'data'/p.name)
for p in (ROOT/'code').glob('*.py'):shutil.copy2(p,OUT/'code'/p.name)
shutil.copytree(ROOT/'code/source',OUT/'code/source',dirs_exist_ok=True)
for name,path in [('original_MOFA_GARQ.py','/workspace/garq/vscode/MultiomeBenchmarking-main/three/run.py'),('original_MOFA_SEACells.py','/workspace/garq/vscode/MultiomeBenchmarking-main/0901/seacell/run.py'),('original_TF_lasso.R','/workspace/garq/Rworkspace/MetaQ/three/compare/16_lasso.R')]:
    shutil.copy2(path,OUT/'code/source'/name)
rawroot=Path('/workspace/garq/data/RNA_ATAC_ADT/GSE158013')
feature_spec={'RNA':['GZMB','CD8A','NCAM1','TCF7','TBX21'],'ATAC':['chr17-1120285-1120785','chr2-86786482-86786982'],'ADT':['CD8a','CD56']}
raw_vectors={}
for m,fs in feature_spec.items():
    a=ad.read_h5ad(rawroot/f'GSE158013_{m.lower()}.h5ad',backed='r')
    raw_vectors[m]=np.asarray(a[:,fs].X.toarray() if hasattr(a[:,fs].X,'toarray') else a[:,fs].X)
    a.file.close()
profiles=[];associations=[];tfrows=[];mofarows=[];validations=[]
for tag in ['full','matched_seed1','matched_seed2','matched_seed3','single_cell']:
    bd=RUN/tag
    if not (bd/'MOFA_input_manifest.json').exists():continue
    arms=json.loads((bd/'MOFA_input_manifest.json').read_text())
    for arm,paths in arms.items():
        table=None;errmax=0.
        for m,fs in feature_spec.items():
            a=ad.read_h5ad(paths[m],backed='r')
            if table is None:
                table=pd.DataFrame({'batch':tag,'arm':arm,'metacell_id':a.obs_names,'celltype':a.obs.celltype.astype(str).to_numpy(),'n_cells':a.obs.n_cells.to_numpy() if 'n_cells' in a.obs else 1})
            x=a[:,fs].X;x=x.toarray() if hasattr(x,'toarray') else np.asarray(x)
            for j,f in enumerate(fs):table[m+'::'+f]=x[:,j]
            if tag!='single_cell':
                member=pd.read_csv(bd/arm/'memberships.csv.gz');g=member.group.to_numpy();valid=g>=0;k=len(table);counts=np.bincount(g[valid],minlength=k)
                assert np.array_equal(counts,table.n_cells)
                for j in range(len(fs)):
                    expected=np.bincount(g[valid],weights=raw_vectors[m][valid,j],minlength=k)/counts
                    err=float(np.max(np.abs(expected-x[:,j])));errmax=max(errmax,err)
                    assert np.allclose(expected,x[:,j],rtol=1e-6,atol=1e-6),(tag,arm,m,fs[j],err)
            a.file.close()
        if tag.startswith('matched'):
            assert len(table)==369 and np.all(table.n_cells==20)
        profiles.append(table)
        validations.append({'batch':tag,'arm':arm,'independent_mean_max_absolute_error':errmax,'n_profiles':len(table),'exact_match_passed':tag.startswith('matched')})
        p=bd/'TF_gene'/arm/'fixed_metacell_order_lasso_coefficients.csv'
        if p.exists():
            d=pd.read_csv(p);assert len(d)==390 and np.isfinite(d.coefficient).all();d.insert(0,'arm',arm);d.insert(0,'batch',tag);tfrows.append(d)
        p=bd/'MOFA'/arm/'metrics.json'
        if p.exists():mofarows.append(json.loads(p.read_text()))
    for mod,name in [('ATAC','metacell_rna_atac_gene_peak_correlations_with_pvalue.csv'),('ADT','metacell_rna_adt_marker_correlations_with_pvalue.csv')]:
        p=bd/('RNA_'+mod)/name
        if p.exists():
            df=pd.read_csv(p);df.insert(0,'batch',tag);df.insert(1,'analysis','peak_gene' if mod=='ATAC' else 'RNA_protein')
            for _,r in df.iterrows():
                tab=next(t for t in profiles if t.batch.iloc[0]==tag and t.arm.iloc[0]==r.algorithm)
                x=tab['RNA::'+r.gene].to_numpy(dtype=float);y=tab[mod+'::'+(r.associated_peaks if mod=='ATAC' else r.protein)].to_numpy(dtype=float)
                if mod=='ADT':keep=(x>.025)&(y>.025);x=x[keep];y=y[keep]
                x=np.log1p(np.maximum(x,0));y=np.log1p(np.maximum(y,0))
                if mod=='ADT':x=np.clip(x,*np.quantile(x,[.01,.99]));y=np.clip(y,*np.quantile(y,[.01,.99]))
                rr,pp=pearsonr(x,y);assert abs(rr-r.correlation_r)<1e-10,(tag,r.algorithm,r.gene,rr,r.correlation_r)
                assert len(x)==r.n_metacells_after_filter
            associations.append(df)
    for p in bd.rglob('*.pdf'):
        if 'MOFA' in p.parts:continue
        dest=OUT/'source_panels'/tag/p.relative_to(bd);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    for p in bd.glob('*execution.log'):
        dest=OUT/'evidence'/tag/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    for fn in ['ordered_genes_from_original_metacell.csv','ordered_tfs_from_original_metacell.csv']:
        p=bd/'TF_gene'/fn
        if p.exists():shutil.copy2(p,OUT/'data'/fn)
pd.concat(profiles).to_csv(OUT/'data'/'selected_profiles.csv.gz',index=False)
pd.concat(associations).to_csv(OUT/'data'/'all_association_statistics.csv',index=False)
pd.concat(tfrows).to_csv(OUT/'data'/'all_TF_gene_coefficients.csv',index=False)
pd.DataFrame(mofarows).to_csv(OUT/'data'/'all_MOFA_metrics.csv',index=False)
(OUT/'evidence'/'independent_validation.json').write_text(json.dumps(validations,indent=2))
status={tag:{'association_and_TF_done':(RUN/tag/'SUPPLIED_WORKFLOWS_DONE').exists(),'MOFA_done':(RUN/tag/'MOFA/DONE').exists()} for tag in ['full','matched_seed1','matched_seed2','matched_seed3','single_cell']}
(OUT/'evidence'/'execution_status.json').write_text(json.dumps(status,indent=2))
with tarfile.open(ROOT/'results_export.tar.gz','w:gz') as tar:tar.add(OUT,arcname='results')
print(json.dumps({'status':status,'MOFA_results':len(mofarows),'association_rows':sum(len(d) for d in associations),'TF_rows':sum(len(d) for d in tfrows),'profiles':sum(len(d) for d in profiles),'independent_checks':len(validations)},indent=2))

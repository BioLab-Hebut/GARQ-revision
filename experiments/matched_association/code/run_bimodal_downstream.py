"""GSE158013 R2 comment 4: reuse saved assignments without filename ambiguity.

Runs all seven unimodal, bimodal and trimodal GARQ constructions.
All constructions are evaluated using all three
measured views. Original source notebooks and compare.R are preserved in source/.
MOFA+ inputs are exported; its original fitting/clustering script is not invented.
"""
from pathlib import Path
import argparse, ast, csv, datetime, hashlib, json, re, subprocess, sys

COMBOS=('RNA','ATAC','ADT','RNA_ATAC','RNA_ADT','ATAC_ADT','RNA_ATAC_ADT')
DATASET='GSE158013'
MODS=('RNA','ATAC','ADT')
ROOT=Path(__file__).resolve().parent

def canon(values):
    return [re.sub(r'\.(\d+)$',r'-\1',str(v)) for v in values]

def digest(values):
    return hashlib.sha256('\n'.join(map(str,values)).encode()).hexdigest()

def check_unique(values,what):
    if len(values)!=len(set(values)):raise ValueError(f'{what}: duplicate identifiers')

def map_groups(obs,raw_ids,raw_labels,allow_positional):
    import numpy as np
    keys=[x for x in ['metacell','SEACell','membership'] if x in obs.columns]
    if len(keys)!=1:raise ValueError(f'Expected one assignment column, found {keys}')
    ids=canon(obs.index);check_unique(ids,'assignment cells')
    if set(ids)==set(raw_ids):
        vals=obs[keys[0]].astype(str).to_numpy();lookup=dict(zip(ids,vals))
        return np.array([lookup[x] for x in raw_ids]),'barcode',keys[0]
    positional=ids==[str(i) for i in range(len(raw_ids))]
    labels_ok='celltype' in obs and np.array_equal(obs['celltype'].astype(str).to_numpy(),raw_labels)
    if allow_positional and positional and labels_ok:
        return obs[keys[0]].astype(str).to_numpy(),'verified_GARQ_positional_order',keys[0]
    raise ValueError('Cannot align assignments. For the previously audited GARQ positional files only, use --allow-verified-positional-garq; celltype order is also checked.')

def normalize_groups(values):
    import numpy as np
    invalid={'-1','-1.0','nan','None','outliers','__missing__','<NA>'}
    labels=sorted(set(str(x) for x in values)-invalid)
    lookup={v:i for i,v in enumerate(labels)}
    return np.array([lookup.get(str(x),-1) for x in values],dtype=int),labels

def sample_equal(groups,size,m,seed):
    import numpy as np
    rng=np.random.default_rng(seed)
    eligible=[k for k in np.unique(groups) if k>=0 and np.sum(groups==k)>=size]
    if m>len(eligible):raise ValueError('Too few eligible groups')
    chosen=sorted(rng.choice(eligible,m,replace=False).tolist())
    result=np.full(len(groups),-1,dtype=int)
    for new,old in enumerate(chosen):
        selected=rng.choice(np.flatnonzero(groups==old),size,replace=False)
        result[selected]=new
    return result

def aggregate_view(adata,groups,raw_ids,raw_labels,out_path,arm,combo,chunk_rows=1024):
    import numpy as np
    import pandas as pd
    import anndata as ad
    from scipy import sparse
    ids=canon(adata.obs_names);check_unique(ids,'raw view cells')
    if set(ids)!=set(raw_ids):raise ValueError('Raw modalities do not have the same paired cells')
    lookup={v:i for i,v in enumerate(raw_ids)}
    order=np.array([lookup[x] for x in ids]);local=groups[order]
    valid=groups>=0;k=int(groups.max())+1
    counts=np.bincount(groups[valid],minlength=k)
    if (counts==0).any():raise ValueError('Empty groups must be removed before aggregation')
    result=sparse.csr_matrix((k,adata.n_vars),dtype=np.float64)
    for start in range(0,adata.n_obs,chunk_rows):
        end=min(start+chunk_rows,adata.n_obs);g=local[start:end];ok=g>=0
        membership=sparse.csr_matrix((1/counts[g[ok]],(g[ok],np.flatnonzero(ok))),shape=(k,end-start))
        block=adata.X[start:end,:]
        result=result+membership@sparse.csr_matrix(block)
    majority=[]
    for i in range(k):
        labels,cc=np.unique(raw_labels[groups==i],return_counts=True)
        majority.append(labels[np.argmax(cc)])
    obs=pd.DataFrame({'metacell_id':[f'{arm}__{i}' for i in range(k)],'n_cells':counts,'celltype':majority})
    obs.index=obs.metacell_id
    out=ad.AnnData(result.astype(np.float32),obs=obs,var=adata.var.copy())
    out.uns['reviewer2_comment4']={'dataset':DATASET,'construction_modalities':combo,'evaluation_views':'RNA_ATAC_ADT','aggregation':'arithmetic_mean_of_input_X','input_path':str(adata.filename)}
    out.write_h5ad(out_path,compression='gzip')
    return {'K':k,'assigned_cells':int(valid.sum()),'min_size':int(counts.min()),'median_size':float(np.median(counts)),'q1':float(np.quantile(counts,.25)),'q3':float(np.quantile(counts,.75)),'max_size':int(counts.max()),'mean_size':float(counts.mean())}

def patch_notebook(path,outdir,file_map):
    """Change only the input map/output directory; retain supplied analysis settings."""
    nb=json.loads(path.read_text(encoding='utf8'))
    cell=next(c for c in nb['cells'] if c['cell_type']=='code')
    tree=ast.parse(''.join(cell['source']))
    seen=set()
    for node in tree.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            if name=='algo_file_map':node.value=ast.parse(repr(file_map),mode='eval').body;seen.add(name)
            if name=='OUTDIR':node.value=ast.parse(f'Path({str(outdir)!r})',mode='eval').body;seen.add(name)
    if seen!={'algo_file_map','OUTDIR'}:raise ValueError('Notebook structure differs from supplied source')
    script=outdir.parent/(path.stem+'_adapted.py')
    adapted=ast.unparse(ast.fix_missing_locations(tree)).replace('_3algorithms_','_all_arms_')
    script.write_text('import matplotlib\nmatplotlib.use("Agg")\n'+adapted+'\n',encoding='utf8')
    return script

def run_logged(command,logfile):
    with logfile.open('w',encoding='utf8') as f:
        subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True,cwd=logfile.parent)

def run_associations(batchdir,arms):
    for modality,filename in [('ATAC','atac.ipynb'),('ADT','adt.ipynb')]:
        m={name:{'rna_h5ad':d['RNA'],modality.lower()+'_h5ad':d[modality]} for name,d in arms.items()}
        out=batchdir/('RNA_'+modality);out.mkdir(exist_ok=True)
        script=patch_notebook(ROOT/'source'/filename,out,m)
        run_logged([sys.executable,str(script)],batchdir/(modality+'_execution.log'))

def run_tf(batchdir,arms,rscript,seed):
    text=(ROOT/'source/compare.R').read_text(encoding='utf8')
    quote=lambda s:json.dumps(str(s),ensure_ascii=False)
    inputs='algorithm_inputs <- list(\n'+',\n'.join(quote(name)+' = list(rna_h5ad = '+quote(d['RNA'])+')' for name,d in arms.items())+'\n)'
    text,n=re.subn(r'algorithm_inputs <- list\([\s\S]*?\n\)',lambda _:inputs,text,count=1)
    if n!=1:raise ValueError('Could not replace compare.R input list')
    text=re.sub(r'^setwd\(.*?\)',lambda _:'setwd('+quote(batchdir)+')',text,count=1,flags=re.M)
    text=re.sub(r'^base_outdir <- .*$',lambda _:'base_outdir <- '+quote(batchdir/'TF_gene'),text,count=1,flags=re.M)
    text='set.seed('+str(seed)+')\n'+text
    path=batchdir/'compare_adapted.R';path.write_text(text,encoding='utf8')
    run_logged([rscript,str(path)],batchdir/'TF_gene_execution.log')

def main():
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument('--raw-root',type=Path,default=Path('/workspace/garq/data/RNA_ATAC_ADT/GSE158013'))
    pa.add_argument('--save-root',type=Path,default=Path('/workspace/garq/vscode/GARQ20260905/save'))
    pa.add_argument('--output-root',type=Path,default=Path('/workspace/garq/vscode/GARQ20260905/reviewer2_comment4'))
    pa.add_argument('--allow-verified-positional-garq',action='store_true')
    pa.add_argument('--include-comparators',action='store_true')
    pa.add_argument('--match-size',type=int,default=20)
    pa.add_argument('--seeds',default='1,2,3')
    pa.add_argument('--run-associations',action='store_true')
    pa.add_argument('--run-tf',action='store_true')
    pa.add_argument('--rscript',default='Rscript')
    pa.add_argument('--prepare-only',action='store_true',help='Audit paths/assignments and write plan without aggregating')
    args=pa.parse_args()
    import numpy as np
    import anndata as ad
    raw={v:ad.read_h5ad(args.raw_root/f'{DATASET}_{v.lower()}.h5ad',backed='r') for v in MODS}
    ref=raw['RNA'];ids=canon(ref.obs_names);check_unique(ids,'RNA cells')
    if len(ids)!=25517:raise ValueError('Expected 25,517 GSE158013 cells; wrong dataset or processing batch')
    labels=ref.obs['celltype'].astype(str).to_numpy()
    for v,a in raw.items():
        if set(canon(a.obs_names))!=set(ids):raise ValueError(f'{v}: raw cell set differs')
    mapping={f'GARQ_{c}':{'method':'GARQ','construction':c,'path':str(args.save_root/f'{DATASET}637_{c}_637metacell_k5_ids.h5ad')} for c in COMBOS}
    if args.include_comparators:
        for method,folder,fn in [('SEACells','3_seacell','GSE158013_ad.h5ad'),('MetaCellV2','4_mc2','GSE158013_metacells2_ad.h5ad'),('SuperCell','5_supercell','GSE158013_supercell.h5ad')]:
            mapping[method]={'method':method,'construction':'RNA_ATAC_ADT','path':f'/workspace/garq/results/{folder}/three/{DATASET}/{fn}'}
    groups={}
    for name,r in mapping.items():
        a=ad.read_h5ad(r['path'],backed='r')
        vals,alignment,col=map_groups(a.obs,ids,labels,args.allow_verified_positional_garq and r['method']=='GARQ')
        g,_=normalize_groups(vals);groups[name]=g
        r.update(alignment=alignment,assignment_column=col,assignment_hash=digest(vals),realized_K=int(g.max()+1),unassigned_cells=int((g<0).sum()))
        a.file.close()
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run=args.output_root/f'{DATASET}__{digest(ids)[:12]}'/stamp
    run.mkdir(parents=True,exist_ok=False)
    seeds=[int(x) for x in args.seeds.split(',')]
    matched_m=min(sum(np.sum(g==i)>=args.match_size for i in np.unique(g) if i>=0) for g in groups.values()) if args.match_size>0 else 0
    if args.match_size>0 and matched_m<3:raise ValueError('Insufficient groups for size matching')
    plan={'dataset':DATASET,'raw_paths':{v:str(a.filename) for v,a in raw.items()},'cell_id_hash':digest(ids),'arms':mapping,'evaluation_views':MODS,'unimodal_downstream':'included: use saved membership to aggregate every paired measured view','match_size':args.match_size,'matched_K':int(matched_m),'seeds':seeds,'MOFA_status':'requires original MOFA fitting and clustering workflow; exported inputs only'}
    (run/'run_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf8')
    print(run,flush=True)
    if args.prepare_only:return
    batches=[('full',groups,1)]
    if args.match_size>0:
        batches.extend((f'matched_K{matched_m}_size{args.match_size}_seed{seed}',{n:sample_equal(g,args.match_size,matched_m,seed) for n,g in groups.items()},seed) for seed in seeds)
    metrics=[];all_inputs={}
    for tag,batch,seed in batches:
        batchdir=run/tag;batchdir.mkdir();arm_inputs={}
        for name,g in batch.items():
            folder=batchdir/name;folder.mkdir()
            with (folder/'memberships.csv').open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['cell_id','metacell_id','celltype'])
                w.writerows((cid,int(group),ct) for cid,group,ct in zip(ids,g,labels) if group>=0)
            arm_inputs[name]={}
            for modality,a in raw.items():
                path=folder/f'{DATASET}__construct-{mapping[name]["construction"]}__view-{modality}__K{g.max()+1}__{tag}.h5ad'
                stats=aggregate_view(a,g,ids,labels,path,name,mapping[name]['construction'])
                arm_inputs[name][modality]=str(path)
            coverage={ct:int(np.sum((labels==ct)&(g>=0))) for ct in np.unique(labels)}
            metrics.append(dict(batch=tag,arm=name,**stats,celltype_coverage=coverage))
            print(tag,name,stats,flush=True)
        all_inputs[tag]=arm_inputs
        (batchdir/'MOFA_input_manifest.json').write_text(json.dumps(arm_inputs,indent=2),encoding='utf8')
        if args.run_associations:run_associations(batchdir,arm_inputs)
        if args.run_tf:run_tf(batchdir,arm_inputs,args.rscript,seed)
    (run/'size_and_coverage.json').write_text(json.dumps(metrics,indent=2),encoding='utf8')
    (run/'all_input_manifests.json').write_text(json.dumps(all_inputs,indent=2),encoding='utf8')
    for a in raw.values():a.file.close()
    print('Completed aggregation and requested association/TF stages. MOFA fit has not been run.',flush=True)

if __name__=='__main__':main()

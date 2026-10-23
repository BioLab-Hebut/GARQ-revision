from release_paths import PACKAGE_ROOT, env_path, load_json
from pathlib import Path
import json,argparse,hashlib,time
import numpy as np,pandas as pd
from r2c2_metrics import evaluate,test_metrics
ROOT=env_path('GARQ_R2C2_ROOT',PACKAGE_ROOT/'results')

def jobs():
    for path in sorted((ROOT/'runs').glob('*/DONE.json')):
        meta=json.loads(path.read_text());name=path.parent.name
        data=np.load(path.parent/f"attempt{meta['selected_attempt']}"/'assignments.npz')
        yield name,meta,data['metacell'],data['cell_ids'],data['labels']

def main(posthoc=False):
    out=ROOT/'evaluation';out.mkdir(exist_ok=True)
    (out/'metric_validation.json').write_text(json.dumps(test_metrics(),indent=2))
    summaries=[];populations=[];sizes=[];rare=[];sampling_manifest=[]
    sample_root=ROOT/'posthoc_samples';sample_root.mkdir(exist_ok=True)
    samplings={}
    for ds in ['D5','D11']:
        r=np.load(ROOT/'inputs'/ds/'standard.npz');labels=r['labels'];cellids=r['cell_ids']
        samplings[ds]=[]
        for ct in np.unique(labels):
            target=np.flatnonzero(labels==ct);others=np.flatnonzero(labels!=ct);abundance=len(target)/len(labels)
            if abundance>.02:continue
            for level in [.01,.005,.002,.001]:
                keep=int(np.floor(level*len(others)/(1-level)))
                if keep<1 or keep>len(target):continue
                for seed in [0,1,2]:
                    key=f'{ds}|{ct}|{level}|{seed}';digest=hashlib.sha256(key.encode()).hexdigest()
                    rng=np.random.default_rng(int(digest[:16],16));sample=np.sort(np.r_[others,rng.choice(target,keep,replace=False)])
                    name=digest[:20]+'.npz';path=sample_root/name
                    if not path.exists():np.savez_compressed(path,positions=sample,cell_ids=cellids[sample])
                    info={'dataset':ds,'celltype':ct,'sampling_seed':seed,'target_abundance':level,'original_abundance':abundance,'original_target_cells':len(target),'retained_target_cells':keep,'n_cells_after_sampling':len(sample),'realized_abundance':keep/len(sample),'selection_rule':'all annotated populations with original abundance <= 2%; every feasible level; not selected by recovery performance','sample_file':name,'sampled_cell_ids_sha256':hashlib.sha256('\n'.join(cellids[sample]).encode()).hexdigest()}
                    samplings[ds].append((info,sample));sampling_manifest.append(info)
    for name,meta,y,ids,labels in jobs():
        ds=meta['dataset'];ref=np.load(ROOT/'inputs'/ds/'standard.npz')
        assert np.array_equal(ids,ref['cell_ids']) and np.array_equal(labels,ref['labels'])
        fields={k:meta[k] for k in ['dataset','representation','method','seed','target_K','requested_K','realized_K','within_five_percent','input_sha256']};fields['run']=name
        summary,pop,mc=evaluate(labels,y,meta['requested_K'])
        summaries.append(dict(fields,**{k:v for k,v in summary.items() if k not in fields}))
        for row in pop:populations.append(dict(fields,**row))
        for row in mc.to_dict('records'):sizes.append(dict(fields,**row))
        if posthoc:
            for samplemeta,indices in samplings[ds]:
                sy=y[indices];sl=labels[indices]
                # Target-only counts avoid computing irrelevant population summaries thousands of times.
                cluster,inv=np.unique(sy,return_inverse=True);sz=np.bincount(inv)
                hits=np.bincount(inv,weights=(sl==samplemeta['celltype']).astype(int),minlength=len(sz))
                dominated=hits/sz>=.5;tp=hits[dominated].sum();positive=sz[dominated].sum();n=samplemeta['retained_target_cells']
                precision=tp/positive if positive else 0.;recall=tp/n
                rare.append(dict(fields,**samplemeta,realized_K_after_sampling=len(sz),compression_after_sampling=len(sy)/len(sz),precision=precision,recall=recall,F1=2*precision*recall/(precision+recall) if precision+recall else 0.,pooled_purity=precision if positive else np.nan,mean_metacell_purity=float((hits[dominated]/sz[dominated]).mean()) if dominated.any() else np.nan,target_dominated_metacells=int(dominated.sum()),median_metacell_size=float(np.median(sz[dominated])) if dominated.any() else np.nan,min_metacell_size=int(sz[dominated].min()) if dominated.any() else np.nan,analysis='post_hoc_fixed_memberships_no_refitting'))
    for filename,rows in [('controlled_summary.csv',summaries),('controlled_all_population_metrics.csv',populations),('controlled_metacell_sizes.csv',sizes)]:pd.DataFrame(rows).to_csv(out/filename,index=False)
    if posthoc:
        pd.DataFrame(rare).to_csv(out/'controlled_posthoc_all_eligible_populations.csv',index=False)
        pd.DataFrame(sampling_manifest).to_csv(out/'posthoc_sampling_manifest.csv',index=False)
    print(json.dumps({'runs':len(summaries),'population_rows':len(populations),'posthoc_rows':len(rare),'sampling_cases':len(sampling_manifest)}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--posthoc',action='store_true');a=p.parse_args();main(a.posthoc)

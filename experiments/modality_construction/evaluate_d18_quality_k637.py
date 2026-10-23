"""Independently evaluate the new D18 K637 seed1 paired quality quartet.

Requires evaluate_d18_k637.py beside this script (its independent metric code).
All quality deltas use the NEW clean baseline, never the historical baseline.
Separately reports new-clean/historical-clean agreement. No training or mutation
of old counts, references, assignments, or evaluation products is performed.
"""
import argparse
import datetime as dt
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import evaluate_d18_k637 as independent
from evaluate_d18_k637 import (align, annotation_metrics, canonical, finite_mean,
    ids_hash, neighborhood, read_json, require, sha_file, within_cosine)

from release_paths import PACKAGE_ROOT, env_path
NEW_ROOT = env_path('GARQ_MODALITY_OUTPUT',PACKAGE_ROOT/'results')
OLD_ROOT = env_path('GARQ_REFERENCE_ROOT',PACKAGE_ROOT/'external_references')
MODS = ('RNA','ATAC','ADT')
CONDITIONS = ('clean','RNA_thin50','ATAC_thin50','ADT_thin50')
N, K, SEED = 25517,637,1


def load_assignment(path, cells):
    sha_file(path)
    a = pd.read_csv(path,dtype={'cell_id':str,'celltype':str})
    a = a.iloc[align(cells.cell_id,a.cell_id)].reset_index(drop=True)
    require(np.array_equal(a.celltype.to_numpy(),cells.celltype.to_numpy()),'Assignment annotation mismatch: '+str(path))
    ids = pd.to_numeric(a.metacell,errors='raise').to_numpy()
    require(np.isfinite(ids).all() and np.all(ids==np.rint(ids)) and ids.min()>=0 and ids.max()<K,'Invalid anchor assignments')
    return ids.astype(np.int64)


def evaluate_one(ids,cells,refs,populations,meta):
    pt,anchors = annotation_metrics(ids,cells.celltype)
    sizes = np.unique(ids,return_counts=True)[1]
    resolution = {**meta,'n_cells':len(ids),'requested_k':K,'realized_k':len(sizes),
        'empty_anchors':K-len(sizes),'cells_per_nonempty_anchor':N/len(sizes),
        'size_min':int(sizes.min()),'size_median':float(np.median(sizes)),
        'size_p95':float(np.quantile(sizes,.95)),'size_max':int(sizes.max()),
        'balanced_accuracy':float(pt.recall.mean()),'macro_f1':float(pt.f1.mean()),
        'rare_macro_f1':float(pt.loc[pt.rare_le_1pct,'f1'].mean()),
        'majority_tie_anchors':int(anchors.majority_tie.sum())}
    views,metrics = [],{}
    for mod,(z,knn) in refs.items():
        retention,chance,adjusted = neighborhood(ids,knn)
        cosine = within_cosine(z,ids)
        for population,mask in populations:
            if not mask.any():
                continue
            row={**meta,'evaluation_modality':mod,'population':population,'n_evaluated':int(mask.sum()),
                'requested_k':K,'realized_k':len(sizes),'neighbor_retention':float(retention[mask].mean()),
                'chance_retention':float(chance[mask].mean()),'size_adjusted_retention':finite_mean(adjusted[mask]),
                'within_anchor_cosine':finite_mean(cosine[mask]),'n_homogeneity_defined':int(np.isfinite(cosine[mask]).sum())}
            views.append(row)
            if population=='all_cells':
                metrics[mod]=row
        bytype = pd.DataFrame({'celltype':cells.celltype,mod+'_neighbor_retention':retention,
            mod+'_size_adjusted_retention':adjusted,mod+'_within_anchor_cosine':cosine}).groupby('celltype').mean()
        pt = pt.merge(bytype,on='celltype',validate='one_to_one')
    for key,value in meta.items():
        pt[key]=value
    return resolution,views,pt,metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=NEW_ROOT/'D18_K637_quality')
    parser.add_argument('--reference-root',type=Path,default=OLD_ROOT)
    parser.add_argument('--historical-evaluation-dir',type=Path,default=NEW_ROOT/'D18_K637_evaluation')
    args = parser.parse_args()
    root,old,historical = args.root.resolve(),args.reference_root.resolve(),args.historical_evaluation_dir.resolve()
    require(root!=old and old not in root.parents,'Quality outputs cannot be inside the historical directory')
    prepared = read_json(root/'prepared/manifest.json')
    cells_path = Path(prepared['cells_csv'])
    require(sha_file(cells_path)==prepared['cells_csv_sha256'],'Fixed cell table changed')
    cells = pd.read_csv(cells_path,dtype=str)
    require(len(cells)==N and cells.cell_id.is_unique and not cells.celltype.isna().any(),'Invalid reference population')
    require(ids_hash(cells.cell_id)==prepared['cell_ids_sha256'],'Quality reference cell ID hash changed')
    old_manifest = read_json(old/'prepared/D18/manifest.json')
    settings = read_json(old/'references/D18/settings.json')
    require(settings['prepared_manifest_sha256']==sha_file(old/'prepared/D18/manifest.json'),'Old reference/manifest mismatch')
    require(int(settings['evaluation_k'])==15,'Expected fixed 15-neighbor references')
    require(old_manifest['cell_ids_sha256']==prepared['cell_ids_sha256'],'New and historical reference cells differ')
    require(prepared['old_manifest_sha256']==sha_file(old/'prepared/D18/manifest.json'),'Prepared manifest changed after training')
    require(prepared['thinning_probability']==.5 and prepared['thinning_random_seed']==20260918,'Unexpected quality perturbation')
    for mod,record in prepared['modalities'].items():
        for prefix in ('clean','thin'):
            require(sha_file(Path(record[prefix+'_path']))==record[prefix+'_sha256'],'Prepared count file changed: '+mod+'/'+prefix)
    refs = {}
    for mod in MODS:
        path = old/'references/D18'/(mod+'.npz')
        sha_file(path)
        with np.load(path,allow_pickle=False) as saved:
            require(str(saved['cell_ids_sha256'])==ids_hash(cells.cell_id),'Reference row hash mismatch')
            z,knn=saved['embedding'],saved['neighbors']
        require(z.shape[0]==N and np.isfinite(z).all(),'Invalid reference embedding')
        require(knn.shape==(N,15) and np.issubdtype(knn.dtype,np.integer),'Invalid reference neighbors')
        require(knn.min()>=0 and knn.max()<N and not (knn==np.arange(N)[:,None]).any(),'Invalid neighbor indices')
        refs[mod]=(z,knn)
    overlap = np.asarray([len(set(r)&set(a)) for r,a in zip(refs['RNA'][1],refs['ATAC'][1])])
    jaccard = overlap/(30-overlap)
    cutoff = float(np.quantile(jaccard,.25))
    populations = [('all_cells',np.ones(N,dtype=bool)),('low_RNA_ATAC_concordance',jaccard<=cutoff),
        ('rare_annotations_le_1pct',cells.celltype.map(cells.celltype.value_counts(normalize=True)).to_numpy()<=.01)]
    resolution,views,types,all_metrics,assignments,runs = [],[],[],{},{},[]
    for condition in CONDITIONS:
        name = 'D18_RNA_ATAC_ADT_'+condition+'_seed1_K637'
        path = root/'runs'/name
        job,result = read_json(path/'job.json'),read_json(path/'result.json')
        require(job['dataset']=='D18' and job['seed']==SEED and job['requested_k']==K and job['target_k']==K,'Unexpected job configuration')
        require(job['condition']==condition and tuple(job['modalities'])==MODS,'Unexpected perturbation condition')
        require(job['epoch']==300 and job['batch_size']==512 and job['k_knn']==5 and job['converge_threshold']==10,'Unexpected training settings')
        require(result['job_sha256']==sha_file(path/'job.json'),'Result/job provenance mismatch')
        require(job['prepared_manifest_sha256']==sha_file(root/'prepared/manifest.json'),'Job/count manifest mismatch')
        require(job['source_manifest_sha256']==sha_file(Path(job['source_dir'])/'source_manifest.json'),'Source manifest changed')
        expected_paths = [prepared['data_paths'][m+('_thin50' if condition==m+'_thin50' else '_clean')] for m in MODS]
        require(job['data_files']==expected_paths,'Job did not use the specified single perturbed layer')
        ids = load_assignment(path/'assignments.csv.gz',cells)
        require(result['n_cells']==N and result['realized_k']==len(np.unique(ids)),'Result resolution mismatch')
        require(result['cell_ids_sha256']==ids_hash(cells.cell_id),'Result cell hash mismatch')
        meta = {'dataset':'D18','construction':'RNA+ATAC+ADT','condition':condition,'seed':SEED,
            'run':name,'series':'new_K637_seed1_paired_quality','n_paired_seeds':1}
        row,v,pt,metrics = evaluate_one(ids,cells,refs,populations,meta)
        resolution.append(row); views.extend(v); types.append(pt)
        all_metrics[condition]=metrics; assignments[condition]=ids
        runs.append({'condition':condition,'job_path':str(path/'job.json'),'job_sha256':sha_file(path/'job.json'),
            'assignment_sha256':sha_file(path/'assignments.csv.gz'),'source_manifest_sha256':job['source_manifest_sha256'],
            'realized_k':row['realized_k']})
    require(len({x['source_manifest_sha256'] for x in runs})==1,'Quality conditions used different source snapshots')
    by_condition = {x['condition']:x for x in resolution}
    clean = by_condition['clean']
    deltas,summaries = [],[]
    for mod in MODS:
        condition=mod+'_thin50'
        current=by_condition[condition]
        ari=float(adjusted_rand_score(assignments['clean'],assignments[condition]))
        summary={'dataset':'D18','seed':SEED,'n_paired_seeds':1,'perturbed_modality':mod,
            'requested_k':K,'realized_k_clean':clean['realized_k'],'realized_k_thinned':current['realized_k'],
            'assignment_ARI_vs_clean':ari,'balanced_accuracy_clean':clean['balanced_accuracy'],
            'balanced_accuracy_thinned':current['balanced_accuracy'],
            'balanced_accuracy_delta':current['balanced_accuracy']-clean['balanced_accuracy'],
            'macro_f1_delta':current['macro_f1']-clean['macro_f1']}
        base_summary = summary.copy()
        for view in MODS:
            baseline,value=all_metrics['clean'][view],all_metrics[condition][view]
            delta={**base_summary,'evaluation_modality':view,'unaltered_input_modality':view!=mod,
                'neighbor_retention_clean':baseline['neighbor_retention'],'neighbor_retention_thinned':value['neighbor_retention'],
                'retention_delta':value['neighbor_retention']-baseline['neighbor_retention'],
                'size_adjusted_retention_clean':baseline['size_adjusted_retention'],
                'size_adjusted_retention_thinned':value['size_adjusted_retention'],
                'size_adjusted_retention_delta':value['size_adjusted_retention']-baseline['size_adjusted_retention'],
                'within_anchor_cosine_clean':baseline['within_anchor_cosine'],
                'within_anchor_cosine_thinned':value['within_anchor_cosine'],
                'within_anchor_cosine_delta':value['within_anchor_cosine']-baseline['within_anchor_cosine']}
            deltas.append(delta)
            summary[view+'_retention_delta']=delta['retention_delta']
            summary[view+'_size_adjusted_retention_delta']=delta['size_adjusted_retention_delta']
            summary[view+'_within_anchor_cosine_delta']=delta['within_anchor_cosine_delta']
        summaries.append(summary)
    # Validate the independently exported historical baseline, including the original positional mapping.
    historic_manifest = read_json(historical/'D18_K637_evaluation_manifest.json')
    require(historic_manifest['requested_K']==K and historic_manifest['n_cells']==N,'Wrong historical evaluation series')
    historic_run = next(x for x in historic_manifest['runs'] if x['construction']=='RNA+ATAC+ADT')
    historic_source = Path(historic_run['source_file'])
    require(sha_file(historic_source)==historic_run['source_sha256'],'Historical source changed')
    old_ids = load_assignment(historical/'assignments/D18_RNA_ATAC_ADT_K637.csv.gz',cells)
    raw_rna = ad.read_h5ad(old_manifest['inputs']['RNA']['path'],backed='r')
    historical_a = ad.read_h5ad(historic_source,backed='r')
    try:
        raw_order = canonical(raw_rna.obs_names)
        require(raw_rna.n_obs==N and historical_a.n_obs==N,'Historical/raw size mismatch')
        require(np.array_equal(historical_a.obs_names.to_numpy(),np.arange(N).astype(str)),'Unexpected historical positional IDs')
        require(np.array_equal(historical_a.obs.celltype.astype(str).to_numpy(),raw_rna.obs.celltype.astype(str).to_numpy()),'Historical labels do not match raw row order')
        mapped = pd.to_numeric(historical_a.obs.metacell).to_numpy()[align(cells.cell_id,raw_order)]
        require(np.array_equal(old_ids,mapped),'Historical export differs from actual saved assignments')
    finally:
        raw_rna.file.close(); historical_a.file.close()
    old_row,_,_,old_metrics = evaluate_one(old_ids,cells,refs,populations,{'dataset':'D18','condition':'historical_clean','seed':SEED})
    reproduction = {'dataset':'D18','requested_k':K,'seed':SEED,
        'new_realized_k':clean['realized_k'],'historical_realized_k':old_row['realized_k'],
        'ARI':float(adjusted_rand_score(old_ids,assignments['clean'])),
        'NMI':float(normalized_mutual_info_score(old_ids,assignments['clean'])),
        'exact_anchor_id_array_equal':bool(np.array_equal(old_ids,assignments['clean'])),
        'new_balanced_accuracy':clean['balanced_accuracy'],'historical_balanced_accuracy':old_row['balanced_accuracy'],
        'balanced_accuracy_delta':clean['balanced_accuracy']-old_row['balanced_accuracy']}
    for mod in MODS:
        reproduction[mod+'_retention_delta']=all_metrics['clean'][mod]['neighbor_retention']-old_metrics[mod]['neighbor_retention']
        reproduction[mod+'_adjusted_retention_delta']=all_metrics['clean'][mod]['size_adjusted_retention']-old_metrics[mod]['size_adjusted_retention']
    out = root/'summary'
    out.mkdir(parents=True,exist_ok=True)
    tables = {'D18_quality_resolution_and_accuracy.csv':pd.DataFrame(resolution),
        'D18_quality_fixed_reference_metrics.csv':pd.DataFrame(views),
        'D18_quality_celltype_metrics.csv':pd.concat(types,ignore_index=True),
        'D18_quality_perturbation_deltas.csv':pd.DataFrame(deltas),
        'D18_quality_stability_summary.csv':pd.DataFrame(summaries),
        'D18_clean_reproduction_vs_historical.csv':pd.DataFrame([reproduction])}
    for filename,table in tables.items():
        table.to_csv(out/filename,index=False)
    report = {'evaluated_at':dt.datetime.now(dt.timezone.utc).isoformat(),
        'dataset':'D18','n_cells':N,'requested_K':K,'training_seed':SEED,'n_paired_seeds':1,
        'perturbation':'One modality at a time binomial count thinning, p=0.5; other modalities unchanged',
        'baseline':'New clean run under exactly the same source/environment/settings as new thinning runs',
        'reference_root':str(old),'reference_settings':settings,'quality_root':str(root),
        'uncertainty':'One paired computational seed; point estimates only, no between-seed SD or stability range',
        'realized_K':'Fixed requested K637; occupied K reported as observed without resolution-matching retries',
        'comparison_to_history':reproduction,'runs':runs,'input_files':independent.INPUT_RECORDS,
        'script_sha256':sha_file(Path(__file__)),'metric_implementation_sha256':sha_file(Path(independent.__file__)),
        'output_sha256':{f:sha_file(out/f) for f in tables},
        'interpretation':'Thinning sensitivity and clean-run reproducibility are separate comparisons; do not attribute new versus historical clean differences to reduced data quality.'}
    (out/'D18_quality_evaluation_manifest.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)
    print('NEW CLEAN VS HISTORICAL',json.dumps(reproduction),flush=True)
    print('DONE: four-condition single-seed paired quality evaluation',flush=True)


if __name__=='__main__':
    main()

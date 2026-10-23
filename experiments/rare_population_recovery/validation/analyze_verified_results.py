"""Prespecified aggregate analysis, gated on all 144 independently verified fits."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from settings_io import read_settings
import math
from pathlib import Path
from statistics import mean, stdev

METHODS=['GARQ_quantizer','SEACells','KMeans']
FRACTIONS=[1.,.5,.25,.1]
COMPRESSION=[30,50]
TOLERANCE=1e-12
METRICS=['rare_macro_F1','population_macro_F1','rare_macro_precision','rare_macro_recall',
         'balanced_accuracy_plurality','coverage','mean_metacell_purity','size_median','size_p95',
         'size_max','cells_in_metacells_smaller_than_5_fraction']

def read(path):return read_settings(path)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(condition,message):
    if not condition:raise ValueError(message)
def save(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csv_save(path,rows):
    require(bool(rows),'Refusing an empty analysis table')
    with Path(path).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
def check_close(actual,expected,message):
    require(math.isclose(actual,expected,abs_tol=1e-10,rel_tol=1e-9),message)
def avg_sd(values):
    vals=[float(x) for x in values if x is not None]
    return (mean(vals) if vals else None,stdev(vals) if len(vals)>1 else None,len(vals))

def load_validated(validation_dir,protocol_path):
    report=read(validation_dir/'INDEPENDENT_RESULT_VALIDATION.json')
    aggregate=read(validation_dir/'INDEPENDENT_AGGREGATE_RESULTS.json')
    protocol=read(protocol_path)
    require(report['status']=='PASS' and report['failed_checks']==0,'Complete independent PASS is required')
    require(report['all_conditions_complete'] and report['verified_fits']==144 and report['verified_cohorts']==24,
            'All 144 fits and 24 preparations must be verified')
    require(report['plan_sha256']==aggregate['plan_sha256'],'Validation/aggregate plan SHA mismatch')
    runs=aggregate['runs'];require(len(runs)==144,'Exactly 144 aggregate runs are required')
    grid={}
    for item in runs:
        j=item['job'];s=item['summary'];d=protocol['datasets'][j['dataset']]
        key=(j['dataset'],float(j['fraction']),int(j['compression']),j['method'],int(j['seed']))
        require(key not in grid,'Duplicate experiment run')
        grid[key]=item
        rare={p['celltype'] for p in d['populations'] if p['prespecified_rare']}
        all_classes={p['celltype'] for p in d['populations']}
        pops={p['celltype']:p for p in item['populations']}
        require(len(pops)==len(item['populations']) and set(pops)==all_classes,'Every original population must remain represented')
        require(len(rare)==d['n_rare_populations'],'Fixed rare-population set mismatch')
        check_close(s['rare_macro_F1'],mean(pops[c]['F1'] for c in rare),'Rare macro-F1 population denominator mismatch')
        check_close(s['population_macro_F1'],mean(p['F1'] for p in pops.values()),'All macro-F1 population denominator mismatch')
        require(s['rare_population_count']==len(rare),'Rare population count mismatch')
        expected_n=sum(math.floor(j['fraction']*p['original_count']) if p['prespecified_rare'] else p['original_count']
                       for p in d['populations'])
        require(j['n_retained']==expected_n,'Cohort size does not match the locked rare-population thinning rule')
        require(s['n_cells']==j['n_retained'] and s['requested_K']==round(j['n_retained']/j['compression']),
                'Retained N and requested K are inconsistent')
        for p in pops.values():
            require(all(math.isfinite(p[m]) and 0<=p[m]<=1 for m in ['F1','precision','recall']),
                    'Invalid recovery metric; missing groups must have zero recovery, not missing values')
        require(all(math.isfinite(s[m]) for m in METRICS),'Non-finite condition metric')
    expected={(d,f,c,m,s) for d in ['D5','D11'] for f in FRACTIONS for c in COMPRESSION for m in METHODS for s in [0,1,2]}
    require(set(grid)==expected,'The 144-run prespecified grid is incomplete or different')
    return report,protocol,grid

def analyze(protocol,grid):
    conditions=[];pairs=[];population_rows=[]
    for dataset in ['D5','D11']:
        d=protocol['datasets'][dataset]
        rare={p['celltype'] for p in d['populations'] if p['prespecified_rare']}
        for fraction in FRACTIONS:
            for compression in COMPRESSION:
                for method in METHODS:
                    items=[grid[(dataset,fraction,compression,method,seed)] for seed in [0,1,2]]
                    summaries=[r['summary'] for r in items]
                    n={r['n_cells'] for r in summaries};k={r['requested_K'] for r in summaries}
                    require(len(n)==len(k)==1,'Same condition has unequal retained N or requested K')
                    row=dict(dataset=dataset,retained_fraction=fraction,compression_setting=compression,method=method,
                             n_runs=3,n_cells=n.pop(),requested_K=k.pop(),rare_population_count=len(rare),
                             all_population_count=d['n_populations'],realized_K_min=min(r['realized_K'] for r in summaries),
                             realized_K_max=max(r['realized_K'] for r in summaries))
                    for metric in METRICS:
                        row[metric+'_mean'],row[metric+'_sd'],_=avg_sd(r[metric] for r in summaries)
                    conditions.append(row)
                    if fraction==.1:
                        for label in sorted(p['celltype'] for p in d['populations']):
                            selected=[next(p for p in item['populations'] if p['celltype']==label) for item in items]
                            p_row=dict(dataset=dataset,retained_fraction=fraction,compression_setting=compression,
                                       method=method,celltype=label,original_rare_class=label in rare,n_runs=3)
                            for metric in ['n_celltype','abundance','F1','precision','recall','pooled_purity',
                                           'mean_metacell_purity','target_dominated_metacells',
                                           'median_metacell_size','mean_metacell_size']:
                                p_row[metric+'_mean'],p_row[metric+'_sd'],p_row[metric+'_valid_n']=avg_sd(p[metric] for p in selected)
                            population_rows.append(p_row)
                for comparator in METHODS[1:]:
                    pair=dict(dataset=dataset,retained_fraction=fraction,compression_setting=compression,comparator=comparator,n_paired_runs=3)
                    for metric in ['rare_macro_F1','population_macro_F1']:
                        diffs=[grid[(dataset,fraction,compression,'GARQ_quantizer',s)]['summary'][metric]-
                               grid[(dataset,fraction,compression,comparator,s)]['summary'][metric] for s in [0,1,2]]
                        pair[metric+'_delta_mean']=mean(diffs);pair[metric+'_delta_sd']=stdev(diffs)
                        pair[metric+'_direction']='higher' if mean(diffs)>TOLERANCE else 'lower' if mean(diffs)<-TOLERANCE else 'tied'
                        for seed,value in enumerate(diffs):pair[metric+'_delta_seed_'+str(seed)]=value
                    pairs.append(pair)
    descriptive={}
    for comparator in METHODS[1:]:
        subset=[p for p in pairs if p['comparator']==comparator]
        descriptive[comparator]={}
        for metric in ['rare_macro_F1','population_macro_F1']:
            descriptive[comparator][metric]={
                'n_conditions':16,**{name:sum(p[metric+'_direction']==name for p in subset) for name in ['higher','tied','lower']},
                'descriptive_mean_delta_across_conditions':mean(p[metric+'_delta_mean'] for p in subset),
                'note':'Conditions share data; this is descriptive, not a significance test'}
    return conditions,pairs,population_rows,descriptive

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--validation-dir',type=Path,required=True)
    p.add_argument('--protocol-counts',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    report,protocol,grid=load_validated(args.validation_dir,args.protocol_counts)
    conditions,pairs,populations,descriptive=analyze(protocol,grid)
    require(len(conditions)==48 and len(pairs)==32 and len(populations)==276,'Unexpected aggregate dimensions')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    files={'condition_summary_48.csv':conditions,'paired_comparisons_32.csv':pairs,
           'severe_retention_all_population_276.csv':populations}
    for name,rows in files.items():csv_save(args.output_dir/name,rows)
    save(args.output_dir/'descriptive_comparison_summary.json',descriptive)
    inputs=[args.validation_dir/'INDEPENDENT_RESULT_VALIDATION.json',args.validation_dir/'INDEPENDENT_AGGREGATE_RESULTS.json',args.protocol_counts]
    metadata={'status':'COMPLETE_VALIDATED','plan_sha256':report['plan_sha256'],'fits':144,'method_conditions':48,
              'comparison_conditions':16,'paired_comparisons':32,'primary_metric':'rare_macro_F1',
              'secondary_metric':'population_macro_F1','fixed_numerical_tie_tolerance':TOLERANCE,
              'p_values_computed':False,'manuscript_edited':False,'cell_level_data_used':False,
              'input_sha256':{str(path.resolve()):sha(path) for path in inputs},
              'output_sha256':{name:sha(args.output_dir/name) for name in [*files,'descriptive_comparison_summary.json']},
              'script_sha256':sha(__file__),
              'scope':'Shared standard representations recomputed after thinning; controlled aggregators, not full GARQ encoder retraining'}
    save(args.output_dir/'analysis_metadata.json',metadata)
    print(json.dumps({'status':metadata['status'],'method_conditions':48,'paired_comparisons':32,'output_dir':str(args.output_dir)},indent=2))

if __name__=='__main__':main()

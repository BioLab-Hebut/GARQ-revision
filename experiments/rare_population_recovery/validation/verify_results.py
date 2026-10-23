"""Server-only independent verification. Output contains aggregates, never cell IDs."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from settings_io import read_settings
import math
from pathlib import Path
from statistics import mean, stdev
import sys
import numpy as np
from independent_metrics import evaluate, test_independent_metrics

def read(path):
    return read_settings(path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):
            h.update(block)
    return h.hexdigest()

def id_digest(ids):
    return hashlib.sha256('\n'.join(map(str,ids)).encode()).hexdigest()

def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return None
    if hasattr(value,'item'):return clean(value.item())
    return value

def save(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

class Checks:
    def __init__(self):self.n=0;self.failures=[]
    def check(self,name,condition,context=None):
        self.n+=1
        if not bool(condition):self.failures.append({'check':name,'context':context})
    def equal(self,name,actual,expected,context=None):
        if expected is None:
            passed=actual is None or actual=='' or isinstance(actual,float) and math.isnan(actual)
        elif isinstance(expected,(float,int)) and not isinstance(expected,bool):
            try:passed=math.isclose(float(actual),float(expected),rel_tol=1e-9,abs_tol=1e-10)
            except (TypeError,ValueError):passed=False
        else:passed=actual==expected
        self.check(name,passed,dict(context or {},actual=clean(actual),expected=clean(expected)))

def validated_done(root,job,plan_hash,checks):
    path=root/'jobs'/job['job_id']/'DONE.json'
    if not path.exists():return None
    result=read(path)
    context={'job':job['job_id']}
    checks.equal('completed status',result.get('status'),'SUCCEEDED',context)
    checks.equal('completed plan',result.get('plan_sha256'),plan_hash,context)
    checks.equal('completed job metadata',result.get('job'),job,context)
    attempt=Path(result['attempt_dir'])
    checks.check('completed attempt belongs to job',attempt.parent.resolve()==path.parent.resolve(),context)
    for name,digest in result['output_sha256'].items():
        artifact=attempt/name
        checks.check('completed artifact exists',artifact.is_file(),dict(context,artifact=name))
        if artifact.is_file():checks.equal('artifact SHA256',sha(artifact),digest,dict(context,artifact=name))
    return result

def expected_grid(registry):
    output=[]
    for dataset in ['D5','D11']:
        d=registry['datasets'][dataset]
        rare=set(d['rare_classes'])
        for fraction in [1.,.5,.25,.1]:
            retained=sum(math.floor(fraction*n) if label in rare else n for label,n in d['class_counts'].items())
            for seed in [0,1,2]:
                cohort=f'{dataset}__f{round(fraction*100):03d}__s{seed}'
                output.append({'job_id':cohort+'__prepare','kind':'prepare','cohort':cohort,
                               'dataset':dataset,'fraction':fraction,'seed':seed,'n_retained':retained})
                for compression in [30,50]:
                    for method in ['GARQ_quantizer','SEACells','KMeans']:
                        output.append({'job_id':f'{cohort}__r{compression}__{method}','kind':'fit','cohort':cohort,
                                       'dataset':dataset,'fraction':fraction,'seed':seed,'n_retained':retained,
                                       'compression':compression,'requested_K':round(retained/compression),'method':method})
    return output

def check_cohort(plan,job,done,checks,raw_cache):
    import anndata as ad
    d=plan['registry']['datasets'][job['dataset']]
    if job['dataset'] not in raw_cache:
        raw=ad.read_h5ad(d['rna_path'],backed='r')
        try:
            raw_ids=np.asarray(raw.obs_names.astype(str),dtype=str)
            raw_labels=raw.obs[d.get('label_key','celltype')].astype(str).to_numpy(dtype=str)
        finally:raw.file.close()
        second=ad.read_h5ad(d['other_path'],backed='r')
        try:second_ids=np.asarray(second.obs_names.astype(str),dtype=str)
        finally:second.file.close()
        checks.check('paired original IDs agree',np.array_equal(raw_ids,second_ids),{'dataset':job['dataset']})
        checks.equal('original annotations counts',dict(Counter(raw_labels)),d['class_counts'],{'dataset':job['dataset']})
        checks.equal('original cell number',len(raw_ids),d['n_expected'],{'dataset':job['dataset']})
        checks.equal('original cell ID hash',id_digest(raw_ids),d['cell_ids_sha256'],{'dataset':job['dataset']})
        raw_cache[job['dataset']]=(raw_ids,raw_labels)
    raw_ids,raw_labels=raw_cache[job['dataset']]
    # Independent reconstruction of the prescribed RNG and prefix rule.
    generator=np.random.default_rng(job['seed'])
    selected=set(i for i,label in enumerate(raw_labels) if label not in d['rare_classes'])
    for label in sorted(d['rare_classes']):
        indices=np.array([i for i,value in enumerate(raw_labels) if value==label],dtype=int)
        permutation=generator.permutation(indices)
        selected.update(map(int,permutation[:math.floor(job['fraction']*len(indices))]))
    expected_indices=np.asarray(sorted(selected),dtype=np.int64)
    attempt=Path(done['attempt_dir'])
    with np.load(attempt/'standard.npz',allow_pickle=False) as data:
        ids=np.asarray(data['cell_ids'],dtype=str)
        labels=np.asarray(data['labels'],dtype=str)
        embedding=np.asarray(data['embedding'])
        indices=np.asarray(data['source_row_indices'])
    context={'job':job['job_id']}
    checks.check('cohort sampling exactly follows fixed seeds/fractions',np.array_equal(indices,expected_indices),context)
    checks.check('retained original IDs/order',np.array_equal(ids,raw_ids[expected_indices]),context)
    checks.check('retained original labels',np.array_equal(labels,raw_labels[expected_indices]),context)
    checks.equal('retained N',len(ids),job['n_retained'],context)
    checks.equal('retained unique IDs',len(set(ids)),len(ids),context)
    checks.equal('representation shape',list(embedding.shape),[job['n_retained'],64],context)
    checks.check('representation finite',np.isfinite(embedding).all(),context)
    checks.check('representation unit norms',np.allclose(np.linalg.norm(embedding,axis=1),1,atol=2e-6,rtol=0),context)
    metadata=read(attempt/'cohort_metadata.json')
    checks.equal('cohort metadata job',metadata['job'],job,context)
    checks.equal('cohort metadata IDs',metadata['retained_cell_ids_sha256'],id_digest(ids),context)
    checks.equal('cohort counts from raw',metadata['retained_counts'],dict(Counter(labels)),context)
    checks.equal('fixed representation seed',metadata['representation_fit_seed'],0,context)
    checks.equal('fixed original rare list',sorted(metadata['rare_classes']),sorted(d['rare_classes']),context)
    digest=hashlib.sha256(np.ascontiguousarray(embedding).tobytes()).hexdigest()
    checks.equal('stored embedding SHA',metadata['embedding_sha256'],digest,context)
    for item in metadata['source_files']:
        source=Path(item['path']).stat()
        checks.equal('original file size stable',source.st_size,item['bytes'],{'dataset':job['dataset']})
        checks.equal('original file mtime stable',source.st_mtime_ns,item['mtime_ns'],{'dataset':job['dataset']})
    # Match the controlled runner's documented second L2 operation.
    x=np.asarray(embedding,np.float32)
    fitted_x=np.ascontiguousarray(x/np.linalg.norm(x,axis=1,keepdims=True))
    return {'ids':ids,'labels':labels,'npz_sha256':sha(attempt/'standard.npz'),
            'fit_embedding_sha256':hashlib.sha256(fitted_x.tobytes()).hexdigest()}

def check_fit(plan,job,done,cohort,checks):
    attempt=Path(done['attempt_dir']);context={'job':job['job_id']}
    with np.load(attempt/'assignments.npz',allow_pickle=False) as data:
        ids=np.asarray(data['cell_ids'],dtype=str)
        labels=np.asarray(data['labels'],dtype=str)
        membership=np.asarray(data['metacell'])
    checks.check('assignment IDs exactly match cohort',np.array_equal(ids,cohort['ids']),context)
    checks.check('assignment labels exactly match cohort',np.array_equal(labels,cohort['labels']),context)
    checks.equal('assignment shape',list(membership.shape),[job['n_retained']],context)
    run=read(attempt/'run.json')
    checks.equal('run metadata job',run['job'],job,context)
    checks.equal('fixed requested K',run['requested_K'],round(job['n_retained']/job['compression']),context)
    checks.equal('no K calibration',run['K_calibration'],False,context)
    checks.equal('no attempt outcome selection',run['fit_attempt_selection'],False,context)
    checks.equal('fit input NPZ hash',run['input_npz_sha256'],cohort['npz_sha256'],context)
    checks.equal('fit input IDs hash',run['input_cell_ids_sha256'],id_digest(ids),context)
    checks.equal('fit exact common matrix hash',run['input_embedding_sha256'],cohort['fit_embedding_sha256'],context)
    expected_settings={
        'GARQ_quantizer':{'epochs':250,'training_batch_size':512,'inference_batch_size':2048,'graph_k':5,'alpha_effective':.2,
                         'optimizer':'AdamW lr=0.001 weight_decay=0.01','device':'cuda'},
        'SEACells':{'min_iter':10,'max_iter':100,'convergence_epsilon':.001,'n_waypoint_eigs':10,'use_gpu':True},
        'KMeans':{'n_init':20,'max_iter':300}}
    for field,value in expected_settings[job['method']].items():
        checks.equal('original fit hyperparameter '+field,run['settings'].get(field),value,context)
    for name,source in plan['registry']['sources'].items():
        checks.equal('fit original source SHA',done['source_hashes'][name]['sha256'],source['sha256'],dict(context,source=name))
    summary,populations,sizes=evaluate(labels,membership,job['requested_K'])
    rare=plan['registry']['datasets'][job['dataset']]['rare_classes']
    for metric,key in [('F1','rare_macro_F1'),('recall','rare_macro_recall'),('precision','rare_macro_precision')]:
        summary[key]=mean(populations[label][metric] for label in rare)
    summary['rare_population_count']=len(rare)
    reported=read(attempt/'metrics.json')
    checks.equal('metrics metadata job',reported['job'],job,context)
    checks.equal('inclusive recovery threshold',reported['target_fraction_threshold'],.5,context)
    for field,value in summary.items():checks.equal('independent summary '+field,reported['summary'].get(field),value,context)
    checks.equal('realized K agrees with assignments',run['realized_K'],summary['realized_K'],context)
    reported_pop={p['celltype']:p for p in reported['all_populations']}
    csv_pop={p['celltype']:p for p in csv_rows(attempt/'all_population_metrics.csv')}
    checks.equal('complete population set in JSON',sorted(reported_pop),sorted(populations),context)
    checks.equal('complete population set in CSV',sorted(csv_pop),sorted(populations),context)
    for label,record in populations.items():
        for field,value in record.items():
            checks.equal('independent population JSON '+field,reported_pop.get(label,{}).get(field),value,dict(context,population=label))
            checks.equal('independent population CSV '+field,csv_pop.get(label,{}).get(field),value,dict(context,population=label))
    reported_sizes={r['metacell']:r for r in csv_rows(attempt/'metacell_sizes.csv')}
    checks.equal('all metacell size rows present',len(reported_sizes),len(sizes),context)
    checks.check('metacell key set matches',set(reported_sizes)==set(sizes),context)
    for group,record in sizes.items():
        for field,value in record.items():
            # Do not export metacell IDs even when a check fails.
            checks.equal('independent metacell '+field,reported_sizes.get(group,{}).get(field),value,context)
    return summary,populations

def check_summaries(root,computed,checks,plan=None,statuses=None):
    summary=root/'summary'
    if not (summary/'per_run_summary.csv').exists():return {'present':False}
    reports={r['job_id']:r for r in csv_rows(summary/'per_run_summary.csv')}
    checks.equal('summary includes every successful fit',sorted(reports),sorted(computed))
    if plan is not None and (summary/'completion.json').exists():
        completion=read(summary/'completion.json')
        checks.equal('summary completion plan hash',completion.get('plan_sha256'),plan['plan_sha256'])
        checks.equal('summary planned fits',completion.get('planned_fits'),144)
        checks.equal('summary successful fits',completion.get('successful_fits'),len(computed))
        checks.equal('summary completion status',completion.get('status'),'COMPLETE' if len(computed)==144 else 'PARTIAL')
    if plan is not None and statuses is not None and (summary/'all_planned_runs.csv').exists():
        rows=csv_rows(summary/'all_planned_runs.csv')
        planned={j['job_id']:j for j in plan['jobs'] if j['kind']=='fit'}
        observed={r['job_id']:r for r in rows}
        states={s['job']:s for s in statuses}
        checks.equal('all planned run rows exactly once',len(rows),144)
        checks.equal('all planned fit IDs reported',sorted(observed),sorted(planned))
        for job_id,job in planned.items():
            row=observed.get(job_id,{})
            context={'job':job_id}
            for field,value in job.items():checks.equal('planned run '+field,row.get(field),value,context)
            state=states[job_id]
            checks.equal('recorded failed attempts',row.get('failed_attempts'),state['failed_attempt_count'],context)
            checks.equal('recorded total attempts',row.get('n_attempts'),state['attempt_count'],context)
            checks.equal('reported fit success',row.get('status')=='SUCCEEDED',state['status']=='SUCCEEDED',context)
    for job_id,item in computed.items():
        for field,value in item['summary'].items():
            checks.equal('independent per-run summary CSV '+field,reports.get(job_id,{}).get(field),value,{'job':job_id})
    if (summary/'condition_mean_sd.csv').exists():
        groups=defaultdict(list)
        for item in computed.values():
            j=item['job'];groups[(j['dataset'],j['fraction'],j['compression'],j['method'])].append(item['summary'])
        metrics=['rare_macro_F1','rare_macro_recall','rare_macro_precision','population_macro_F1',
                 'balanced_accuracy_plurality','coverage','realized_K','size_median','size_p95','size_max',
                 'cells_in_metacells_smaller_than_5_fraction']
        for row in csv_rows(summary/'condition_mean_sd.csv'):
            key=(row['dataset'],float(row['fraction']),int(row['compression']),row['method'])
            rows=groups[key];context={'condition':list(key)}
            checks.equal('condition successful count',row['n_successful'],len(rows),context)
            checks.equal('condition planned seeds',row['n_planned'],3,context)
            for metric in metrics:
                values=[r[metric] for r in rows]
                checks.equal('condition mean '+metric,row[metric+'_mean'],mean(values) if values else None,context)
                checks.equal('condition SD '+metric,row[metric+'_sd'],stdev(values) if len(values)>1 else None,context)
            checks.equal('condition min realized K',row['realized_K_min'],min((r['realized_K'] for r in rows),default=None),context)
            checks.equal('condition max realized K',row['realized_K_max'],max((r['realized_K'] for r in rows),default=None),context)
    if (summary/'all_population_per_run.csv').exists():
        reported={(r['job_id'],r['celltype']):r for r in csv_rows(summary/'all_population_per_run.csv')}
        expected={(job_id,pop['celltype']):(item,pop) for job_id,item in computed.items() for pop in item['populations']}
        checks.equal('all per-population rows retained',sorted(reported),sorted(expected))
        for key,(item,pop) in expected.items():
            context={'job':key[0],'population':key[1]}
            if plan is not None:
                rare=plan['registry']['datasets'][item['job']['dataset']]['rare_classes']
                checks.equal('per-population fixed rare flag',reported.get(key,{}).get('original_rare_class'),str(pop['celltype'] in rare),context)
            for field,value in pop.items():
                checks.equal('per-population CSV '+field,reported.get(key,{}).get(field),value,context)
    if (summary/'all_population_mean_sd.csv').exists():
        groups=defaultdict(list)
        for item in computed.values():
            j=item['job']
            for pop in item['populations']:
                groups[(j['dataset'],j['fraction'],j['compression'],j['method'],pop['celltype'])].append(pop)
        rows=csv_rows(summary/'all_population_mean_sd.csv')
        checks.equal('population condition row count',len(rows),len(groups))
        fields=['F1','precision','recall','pooled_purity','mean_metacell_purity','target_dominated_metacells',
                'median_metacell_size','mean_metacell_size','n_celltype','abundance']
        for row in rows:
            key=(row['dataset'],float(row['fraction']),int(row['compression']),row['method'],row['celltype'])
            context={'condition':list(key)}
            if plan is not None:
                rare=plan['registry']['datasets'][row['dataset']]['rare_classes']
                checks.equal('population mean fixed rare flag',row.get('original_rare_class'),str(row['celltype'] in rare),context)
            for field in fields:
                values=[p[field] for p in groups[key] if p[field] is not None]
                checks.equal('population aggregate mean '+field,row[field+'_mean'],mean(values) if values else None,context)
                checks.equal('population aggregate SD '+field,row[field+'_std'],stdev(values) if len(values)>1 else None,context)
                checks.equal('population aggregate valid count '+field,row[field+'_count'],len(values),context)
    return {'present':True,'checked_per_run':len(computed)}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args()
    test_result=test_independent_metrics()
    checks=Checks();plan=read(args.plan);registry=plan['registry'];root=Path(registry['output_root'])
    calculated=hashlib.sha256(json.dumps({k:v for k,v in plan.items() if k!='plan_sha256'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    checks.equal('immutable plan hash',plan['plan_sha256'],calculated)
    checks.equal('approved protocol',registry['protocol_approved'],True)
    checks.equal('input/source preflight',registry['preflight_status'],'PASS')
    expected=expected_grid(registry)
    checks.equal('planned job count',len(plan['jobs']),168)
    checks.equal('planned unique job IDs',len({j['job_id'] for j in plan['jobs']}),168)
    checks.equal('prespecified complete job grid',{j['job_id']:j for j in plan['jobs']},{j['job_id']:j for j in expected})
    for name,item in registry['sources'].items():checks.equal('current scientific source SHA',sha(item['path']),item['sha256'],{'source':name})
    for ds,item in registry['datasets'].items():
        expected=sorted(c for c,n in item['class_counts'].items() if n/item['n_expected']<=.02)
        checks.equal('fixed original <=2% rare class set',sorted(item['rare_classes']),expected,{'dataset':ds})
    raw_cache,cohorts,computed={},{},{}
    statuses=[]
    for job in plan['jobs']:
        done=validated_done(root,job,plan['plan_sha256'],checks)
        attempts=list((root/'jobs'/job['job_id']).glob('attempt_*/result.json'))
        failed_attempts=sum(read(p).get('status')=='FAILED' for p in attempts)
        status={'job':job['job_id'],'kind':job['kind'],'status':'SUCCEEDED' if done else 'FAILED' if failed_attempts else 'PENDING',
                'attempt_count':len(attempts),'failed_attempt_count':failed_attempts}
        statuses.append(status)
        if not done:continue
        try:
            if job['kind']=='prepare':
                cohorts[job['cohort']]=check_cohort(plan,job,done,checks,raw_cache)
            else:
                if job['cohort'] not in cohorts:
                    raise ValueError('Successful fit has no verified cohort')
                values,pop=check_fit(plan,job,done,cohorts[job['cohort']],checks)
                computed[job['job_id']]={'job':job,'summary':values,'populations':list(pop.values())}
        except Exception as error:
            checks.check('result can be independently verified',False,{'job':job['job_id'],'error_type':type(error).__name__,
                         'error':'inspection failed; server-only traceback required for debugging; no cell IDs exported'})
    summary_check=check_summaries(root,computed,checks,plan,statuses)
    complete=len(computed)==144 and len(cohorts)==24
    if args.require_complete:checks.check('all 144 fits and 24 preparations complete',complete)
    status='FAIL' if checks.failures else 'PASS' if complete else 'PARTIAL'
    result={'status':status,'generated_utc':datetime.now(timezone.utc).isoformat(),
            'plan_sha256':plan['plan_sha256'],'validator_sha256':sha(__file__),
            'independent_metrics_sha256':sha(Path(__file__).with_name('independent_metrics.py')),
            'checks':checks.n,'failed_checks':len(checks.failures),'failures':checks.failures,
            'planned_fits':144,'verified_fits':len(computed),'planned_cohorts':24,'verified_cohorts':len(cohorts),
            'all_conditions_complete':complete,'attempt_statuses':statuses,'summary_check':summary_check,
            'toy_checks':test_result,'cell_level_data_exported':False,
            'independence':'Counter-based evaluation and manual quantiles; experiment evaluator is never imported or called',
            'scope':'validates saved assignments, shared cohort/matrix provenance, fixed K/conditions and numerical summaries; does not retrain or independently rerun all model fits'}
    save(args.output_dir/'INDEPENDENT_RESULT_VALIDATION.json',result)
    save(args.output_dir/'INDEPENDENT_AGGREGATE_RESULTS.json',{'plan_sha256':plan['plan_sha256'],
         'scope':'aggregate population statistics only; no cell IDs, metacell IDs or embeddings','runs':list(computed.values())})
    print(json.dumps({k:result[k] for k in ['status','checks','failed_checks','verified_fits','verified_cohorts','all_conditions_complete','cell_level_data_exported']},indent=2))
    return 1 if status=='FAIL' else 0

if __name__=='__main__':
    raise SystemExit(main())

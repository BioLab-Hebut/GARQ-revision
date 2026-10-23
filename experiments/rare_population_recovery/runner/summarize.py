"""Summarize every planned condition, including missing and failed runs."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from common import load_plan, latest_success, read_json, write_json, now

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    args = p.parse_args()
    plan = load_plan(args.plan)
    root = Path(plan['registry']['output_root'])
    output = root/'summary'
    output.mkdir(parents=True,exist_ok=True)
    status_rows, per_run, population = [], [], []
    for job in plan['jobs']:
        if job['kind']!='fit':
            continue
        directory = root/'jobs'/job['job_id']
        attempts = [read_json(p) for p in sorted(directory.glob('attempt_*/result.json'))]
        done = latest_success(directory,plan['plan_sha256'])
        prepared = latest_success(root/'jobs'/(job['cohort']+'__prepare'),plan['plan_sha256'])
        row = dict(job, status='SUCCEEDED' if done else 'FAILED' if attempts else 'BLOCKED_PREPARATION' if not prepared else 'NOT_COMPLETED',
                   n_attempts=len(attempts),failed_attempts=sum(a['status']=='FAILED' for a in attempts))
        status_rows.append(row)
        if done is None:
            continue
        metrics = read_json(Path(done['attempt_dir'])/'metrics.json')
        per_run.append(dict(job, **metrics['summary']))
        rare = set(plan['registry']['datasets'][job['dataset']]['rare_classes'])
        for record in metrics['all_populations']:
            population.append(dict(job, **record, original_rare_class=record['celltype'] in rare))
    pd.DataFrame(status_rows).to_csv(output/'all_planned_runs.csv',index=False)
    pd.DataFrame(per_run).to_csv(output/'per_run_summary.csv',index=False)
    pd.DataFrame(population).to_csv(output/'all_population_per_run.csv',index=False)
    metrics = ['rare_macro_F1','rare_macro_recall','rare_macro_precision','population_macro_F1',
               'balanced_accuracy_plurality','coverage','realized_K','size_median','size_p95','size_max',
               'cells_in_metacells_smaller_than_5_fraction']
    grouped = defaultdict(list)
    for row in per_run:
        grouped[(row['dataset'],row['fraction'],row['compression'],row['method'])].append(row)
    summaries = []
    conditions = sorted({(r['dataset'],r['fraction'],r['compression'],r['method']) for r in status_rows})
    for key in conditions:
        rows = grouped[key]
        summary = dict(zip(['dataset','fraction','compression','method'],key))
        summary.update(n_successful=len(rows),n_planned=3,complete_three_seeds=len(rows)==3,
                       successful_seeds=','.join(str(r['seed']) for r in sorted(rows,key=lambda x:x['seed'])))
        for metric in metrics:
            values = np.asarray([r[metric] for r in rows],dtype=float)
            summary[metric+'_mean'] = float(values.mean()) if len(values) else None
            summary[metric+'_sd'] = float(values.std(ddof=1)) if len(values)>=2 else None
        summary['realized_K_min'] = min((r['realized_K'] for r in rows),default=None)
        summary['realized_K_max'] = max((r['realized_K'] for r in rows),default=None)
        summaries.append(summary)
    pd.DataFrame(summaries).to_csv(output/'condition_mean_sd.csv',index=False)
    if population:
        frame = pd.DataFrame(population)
        group_cols = ['dataset','fraction','compression','method','celltype','original_rare_class']
        group = frame.groupby(group_cols,dropna=False)
        means = group[['F1','precision','recall','pooled_purity','mean_metacell_purity',
                       'target_dominated_metacells','median_metacell_size','mean_metacell_size',
                       'n_celltype','abundance']].agg(['mean','std','count'])
        means.columns = ['_'.join(col) for col in means.columns]
        means.reset_index().to_csv(output/'all_population_mean_sd.csv',index=False)
    write_json(output/'completion.json',{'generated_utc':now(),'plan_sha256':plan['plan_sha256'],
                                        'planned_fits':144,'successful_fits':len(per_run),
                                        'status':'COMPLETE' if len(per_run)==144 else 'PARTIAL',
                                        'scientific_scope':'reconstruct controlled aggregators after thinning and refitting standard representations; not end-to-end GARQ retraining',
                                        'selection':'all planned runs; failed attempts retained; no outcome-dependent selection',
                                        'reporting':'partial means are labeled n_successful; absence of target-dominated groups remains F1=0; undefined purity remains missing'})
    print({'planned_fits':144,'successful_fits':len(per_run),'output':str(output)},flush=True)

if __name__=='__main__':
    main()

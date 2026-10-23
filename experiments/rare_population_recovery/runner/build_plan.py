"""Validate a root-provided registry and freeze the 24-cohort/144-fit plan."""
import argparse
from pathlib import Path
from common import read_json, write_json, validate_registry, build_jobs, object_sha, now

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--registry', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--validate-only', action='store_true')
    args = p.parse_args()
    registry = validate_registry(read_json(args.registry))
    jobs = build_jobs(registry)
    plan = {'schema_version': 1, 'created_utc': now(), 'registry': registry, 'jobs': jobs,
            'protocol': {'retention_fractions':[1,.5,.25,.1], 'seeds':[0,1,2], 'compression_inputs':[30,50],
                         'sampling':'all original <=2% classes; nested within seed/class; floor(f*n); all other cells retained',
                         'representation':'recomputed from retained raw assays, original standard functions, seed 0; shared 64D matrix',
                         'fit':'one fit per method/condition/seed; fixed requested K=round(N_retained/compression); no K calibration',
                         'report':'all target populations, all seeds, all methods; failed and unassigned outcomes retained'}}
    plan['plan_sha256'] = object_sha(plan)
    if not args.validate_only:
        if args.output.exists():
            raise FileExistsError('Refusing to overwrite a frozen plan')
        write_json(args.output, plan)
    print({'status':'PASS', 'prepare_jobs':sum(x['kind']=='prepare' for x in jobs), 'fits':sum(x['kind']=='fit' for x in jobs), 'plan_sha256':plan['plan_sha256'], 'written':not args.validate_only})

if __name__ == '__main__':
    main()

"""R69 orchestration utilities. No scientific model implementation lives here."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from settings_io import read_settings
import math
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone

METHODS = ('GARQ_quantizer', 'SEACells', 'KMeans')
FRACTIONS = (1.0, 0.5, 0.25, 0.1)
SEEDS = (0, 1, 2)
COMPRESSIONS = (30, 50)

def now():
    return datetime.now(timezone.utc).isoformat()

def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def ids_sha(ids):
    # Match the historical controlled-input registry: NO terminal newline.
    return hashlib.sha256('\n'.join(map(str, ids)).encode()).hexdigest()

def clean_json(x):
    if isinstance(x, dict):
        return {str(k): clean_json(v) for k,v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean_json(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if hasattr(x, 'item'):
        return clean_json(x.item())
    return x

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(clean_json(value), f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write('\n')
        os.replace(temp, path)
    finally:
        if Path(temp).exists():
            Path(temp).unlink()

def read_json(path):
    return read_settings(path)

def import_exact(name, item):
    path = Path(item['path'])
    if file_sha(path) != item['sha256']:
        raise ValueError(f'Source changed: {path}')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def validate_registry(registry, check_files=True):
    if registry.get('protocol_approved') is not True:
        raise ValueError('Protocol must be approved before plan generation')
    if registry.get('preflight_status') != 'PASS':
        raise ValueError('Registry requires completed input/source preflight PASS')
    if set(registry['datasets']) != {'D5', 'D11'}:
        raise ValueError('Only the approved two datasets are permitted')
    if registry.get('threads', 8) != 8:
        raise ValueError('Original quantizer sets 8 threads; preserve 8 here')
    for name in ['benchmark', 'representation', 'metrics', 'model']:
        item = registry['sources'][name]
        if check_files and file_sha(item['path']) != item['sha256']:
            raise ValueError(f'Source mismatch: {name}')
    if check_files:
        for name in ['prepare', *METHODS]:
            if not Path(registry['python'][name]).is_file():
                raise FileNotFoundError(registry['python'][name])
    for ds, item in registry['datasets'].items():
        if sum(item['class_counts'].values()) != item['n_expected']:
            raise ValueError(f'Annotation counts do not sum to N: {ds}')
        rare = sorted(k for k,v in item['class_counts'].items() if v/item['n_expected'] <= .02)
        if rare != sorted(item['rare_classes']):
            raise ValueError(f'Rare class list does not match <=2%: {ds}')
        if item['other_modality'] != ('ADT' if ds == 'D5' else 'ATAC'):
            raise ValueError(f'Wrong modality: {ds}')
        for key in ['rna_path', 'other_path']:
            if check_files and not Path(item[key]).is_file():
                raise FileNotFoundError(item[key])
        if any(math.floor(.1*item['class_counts'][c]) < 1 for c in rare):
            raise ValueError('Approved smallest fraction must retain every target class')
    return registry

def cohort_name(dataset, fraction, seed):
    return f'{dataset}__f{int(round(fraction*100)):03d}__s{seed}'

def build_jobs(registry):
    jobs = []
    for ds, item in registry['datasets'].items():
        rare = set(item['rare_classes'])
        for fraction in FRACTIONS:
            n = sum(math.floor(fraction*v) if k in rare else v for k,v in item['class_counts'].items())
            for seed in SEEDS:
                cohort = cohort_name(ds, fraction, seed)
                jobs.append({'job_id': cohort+'__prepare', 'kind': 'prepare', 'cohort': cohort,
                             'dataset': ds, 'fraction': fraction, 'seed': seed, 'n_retained': n})
                for compression in COMPRESSIONS:
                    k = round(n/compression)
                    for method in METHODS:
                        jobs.append({'job_id': f'{cohort}__r{compression}__{method}', 'kind': 'fit',
                                     'cohort': cohort, 'dataset': ds, 'fraction': fraction, 'seed': seed,
                                     'n_retained': n, 'compression': compression, 'requested_K': k,
                                     'method': method})
    return jobs

def load_plan(path):
    plan = read_json(path)
    expected = object_sha({k:v for k,v in plan.items() if k != 'plan_sha256'})
    if plan.get('plan_sha256') != expected:
        raise ValueError('Plan checksum mismatch')
    validate_registry(plan['registry'])
    if plan['jobs'] != build_jobs(plan['registry']):
        raise ValueError('Job grid changed from prespecified protocol')
    return plan

def source_record(registry):
    return {k: {'path': v['path'], 'sha256': file_sha(v['path'])} for k,v in registry['sources'].items()}

def runtime_versions():
    import importlib.metadata as md
    versions = {}
    for package in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'scanpy', 'anndata', 'torch', 'einops', 'SEACells', 'cupy-cuda12x', 'cupy-cuda11x', 'faiss-cpu', 'faiss-gpu']:
        try:
            versions[package] = md.version(package)
        except md.PackageNotFoundError:
            pass
    return versions

def latest_success(job_dir, plan_hash):
    job_dir = Path(job_dir)
    done = job_dir/'DONE.json'
    if not done.exists():
        return None
    record = read_json(done)
    if record.get('status') != 'SUCCEEDED' or record.get('plan_sha256') != plan_hash:
        raise ValueError(f'Incompatible existing result: {done}')
    attempt = Path(record['attempt_dir'])
    for artifact, digest in record['output_sha256'].items():
        if file_sha(attempt/artifact) != digest:
            raise ValueError(f'Completed artifact changed: {attempt/artifact}')
    return record

"""One isolated preparation or fit, importing locked original scientific code."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
try:
    import resource
except ImportError:  # Allows local read-only protocol tests on Windows.
    resource = None
import sys
import time
import traceback

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[name] = '8'
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
from common import (load_plan, import_exact, latest_success, write_json, source_record,
                    runtime_versions, file_sha, ids_sha, now, read_json)

def selected_rows(labels, counts, rare_classes, fraction, seed):
    """One permutation per fixed rare class; prefixes make fractions nested."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=str)
    keep = np.ones(len(labels), dtype=bool)
    for label in sorted(rare_classes):
        rows = np.flatnonzero(labels == label)
        if len(rows) != counts[label]:
            raise ValueError('Rare-class source count mismatch: '+label)
        order = rng.permutation(rows)
        number = int(np.floor(fraction*len(rows)))
        keep[rows] = False
        keep[order[:number]] = True
    return np.flatnonzero(keep)

def prepare(plan, job, dest):
    import anndata as ad
    from scipy import sparse
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize
    registry = plan['registry']
    data = registry['datasets'][job['dataset']]
    module = import_exact('r69_original_standard', registry['sources']['representation'])
    if module.RANDOM_STATE != 0 or module.N_HVG != 3000 or module.N_PCA_RNA != 50 or module.N_LSI_ATAC != 50 or module.N_PCA_ADT != 20:
        raise ValueError('Unexpected standard-representation constants')
    raw = []
    file_meta = []
    try:
        for key in ('rna_path','other_path'):
            path = Path(data[key])
            stat = path.stat()
            file_meta.append({'path':str(path),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns})
            raw.append(ad.read_h5ad(path, backed='r'))
        ids = np.asarray(raw[0].obs_names.astype(str), dtype=str)
        if len(np.unique(ids)) != len(ids) or len(ids) != data['n_expected']:
            raise ValueError('Wrong original cell count or duplicate cell IDs')
        if not np.array_equal(ids, np.asarray(raw[1].obs_names.astype(str), dtype=str)):
            raise ValueError('Raw paired assays must have the same complete cell order')
        if data.get('cell_ids_sha256') and ids_sha(ids) != data['cell_ids_sha256']:
            raise ValueError('Raw cell-ID hash differs from verified registry')
        labels = raw[0].obs[data.get('label_key','celltype')].astype(str).to_numpy(dtype=str)
        counts = dict(Counter(labels))
        if counts != data['class_counts']:
            raise ValueError('Current raw annotations differ from approved population counts')
        fingerprint = {'source_files':file_meta,'cell_ids_sha256':ids_sha(ids),'class_counts':counts}
        lock_path = Path(registry['output_root'])/'input_fingerprints'/(job['dataset']+'.json')
        if lock_path.exists():
            if read_json(lock_path) != fingerprint:
                raise ValueError('Raw dataset changed since the first cohort preparation')
        else:
            write_json(lock_path, fingerprint)
        rows = selected_rows(labels, counts, data['rare_classes'], job['fraction'], job['seed'])
        if len(rows) != job['n_retained']:
            raise ValueError('Retained cohort count differs from frozen plan')
        assays = [a[rows,:].to_memory() for a in raw]
    finally:
        for a in raw:
            a.file.close()
    kept_ids, kept_labels = ids[rows], labels[rows]
    for a in assays:
        values = a.X.data if sparse.issparse(a.X) else np.asarray(a.X)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('Invalid raw nonnegative assay input')
        if not np.array_equal(a.obs_names.astype(str), kept_ids):
            raise ValueError('Retained assay cell order changed')
    blocks = [module.compute_rna_pca(assays[0])]
    other = module.compute_adt_pca(assays[1]) if data['other_modality']=='ADT' else module.compute_atac_lsi(assays[1])
    blocks.append(other)
    del assays
    balanced = [module.standardize_and_l2(b) for b in blocks]
    joined = normalize(np.hstack(balanced).astype(np.float32), norm='l2', axis=1).astype(np.float32)
    pca = PCA(n_components=64, svd_solver='full')
    embedding = np.asarray(pca.fit_transform(joined), dtype=np.float32)
    lengths = np.linalg.norm(embedding, axis=1, keepdims=True)
    if not np.isfinite(embedding).all() or (lengths < 1e-10).any():
        raise ValueError('Invalid standard representation after PCA64')
    embedding = np.ascontiguousarray(embedding / lengths)
    np.savez_compressed(dest/'standard.npz', embedding=embedding, cell_ids=kept_ids,
                        labels=kept_labels, source_row_indices=rows)
    population_counts = {c:int(np.sum(kept_labels==c)) for c in sorted(counts)}
    if any(n==0 for n in population_counts.values()):
        raise ValueError('Approved design unexpectedly lost an entire annotation class')
    metadata = {'job':job,'source_files':file_meta,'source_cell_ids_sha256':ids_sha(ids),
                'retained_cell_ids_sha256':ids_sha(kept_ids),'retained_counts':population_counts,
                'rare_classes':data['rare_classes'],'representation_fit_seed':0,
                'blocks': [50,20 if data['other_modality']=='ADT' else 50], 'dimensions':64,
                'pca64_retained_variance':float(pca.explained_variance_ratio_.sum()),
                'source_values':'raw assay X; normalize/HVG/PCA/LSI fitted only on the retained cohort',
                'embedding_sha256':__import__('hashlib').sha256(embedding.tobytes()).hexdigest()}
    write_json(dest/'cohort_metadata.json', metadata)
    return ['standard.npz','cohort_metadata.json']

def fit(plan, job, dest):
    registry = plan['registry']
    root = Path(registry['output_root'])
    prepared = latest_success(root/'jobs'/(job['cohort']+'__prepare'), plan['plan_sha256'])
    if prepared is None:
        raise RuntimeError('Cohort preparation is incomplete')
    source = Path(prepared['attempt_dir'])/'standard.npz'
    with np.load(source, allow_pickle=False) as data:
        embedding = np.asarray(data['embedding'], dtype=np.float32)
        ids = np.asarray(data['cell_ids'], dtype=str)
        labels = np.asarray(data['labels'], dtype=str)
    if embedding.shape != (job['n_retained'],64) or len(ids)!=len(labels):
        raise ValueError('Prepared input shape differs from planned cohort')
    if len(np.unique(ids)) != len(ids):
        raise ValueError('Duplicate retained IDs')
    backend = import_exact('r69_original_benchmark', registry['sources']['benchmark'])
    backend.BASE = Path(registry['sources']['model']['path']).parent
    if (backend.BASE/'model.py').resolve() != Path(registry['sources']['model']['path']).resolve():
        raise ValueError('Locked quantizer source must be named model.py')
    x = backend.l2(embedding)  # Preserve original benchmark input normalization.
    method = job['method']
    if method == 'GARQ_quantizer':
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; no silent CPU fallback permitted')
        torch.cuda.reset_peak_memory_stats()
    elif method == 'SEACells':
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError('CuPy CUDA unavailable; no alternative implementation permitted')
    started = time.perf_counter()
    assignments, settings = backend.fit(x, method, job['requested_K'], job['seed'], dest)
    fit_seconds = time.perf_counter()-started
    assignments = np.asarray(assignments)
    if assignments.ndim != 1 or len(assignments)!=len(ids):
        raise ValueError('Incomplete or malformed assignment output')
    metric = import_exact('r69_original_metrics', registry['sources']['metrics'])
    metric.test_metrics()
    summary, population, sizes = metric.evaluate(labels, assignments, requested_k=job['requested_K'])
    rare = registry['datasets'][job['dataset']]['rare_classes']
    per_type = {p['celltype']:p for p in population}
    if set(per_type) != set(registry['datasets'][job['dataset']]['class_counts']):
        raise ValueError('Population evaluation omitted a prespecified class')
    summary['rare_macro_F1'] = float(np.mean([per_type[c]['F1'] for c in rare]))
    summary['rare_macro_recall'] = float(np.mean([per_type[c]['recall'] for c in rare]))
    summary['rare_macro_precision'] = float(np.mean([per_type[c]['precision'] for c in rare]))
    summary['rare_population_count'] = len(rare)
    if method == 'GARQ_quantizer':
        import torch
        summary['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
        summary['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved()
    np.savez_compressed(dest/'assignments.npz', metacell=assignments, cell_ids=ids, labels=labels)
    import pandas as pd
    pd.DataFrame(population).to_csv(dest/'all_population_metrics.csv', index=False)
    sizes.to_csv(dest/'metacell_sizes.csv', index=False)
    write_json(dest/'metrics.json', {'job':job,'summary':summary,'all_populations':population,
                                    'evaluation':'original r2c2_metrics; fixed original rare class set; all classes including zero recovery',
                                    'target_fraction_threshold':0.5})
    write_json(dest/'run.json', {'job':job,'fit_seconds':fit_seconds,'settings':settings,
                               'requested_K':job['requested_K'],'realized_K':summary['realized_K'],
                               'input_npz_sha256':file_sha(source),'input_cell_ids_sha256':ids_sha(ids),
                               'input_embedding_sha256':__import__('hashlib').sha256(x.tobytes()).hexdigest(),
                               'K_calibration':False,'fit_attempt_selection':False})
    outputs = ['assignments.npz','all_population_metrics.csv','metacell_sizes.csv','metrics.json','run.json']
    if method == 'GARQ_quantizer':
        outputs.append('quantizer_state.npz')
    return outputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--attempt-dir', type=Path, required=True)
    args = parser.parse_args()
    plan = load_plan(args.plan)
    job = next(j for j in plan['jobs'] if j['job_id']==args.job_id)
    dest = args.attempt_dir.resolve()
    expected_parent = Path(plan['registry']['output_root']).resolve()/'jobs'/job['job_id']
    if dest.parent != expected_parent:
        raise ValueError('Attempt directory is outside the planned job')
    if (dest/'result.json').exists():
        raise FileExistsError('A completed attempt cannot be overwritten')
    dest.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    record = {'status':'RUNNING','started_utc':now(),'job':job,'plan_sha256':plan['plan_sha256'],
              'pid':os.getpid(),'source_hashes':source_record(plan['registry']),
              'python':sys.executable,'versions':runtime_versions(),
              'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'), 'threads':8}
    write_json(dest/'RUNNING.json', record)
    try:
        outputs = prepare(plan,job,dest) if job['kind']=='prepare' else fit(plan,job,dest)
        record.update(status='SUCCEEDED', attempt_dir=str(dest),
                      output_sha256={name:file_sha(dest/name) for name in outputs})
        exit_code = 0
    except BaseException as error:
        record.update(status='FAILED',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        print(record['traceback'],flush=True)
        exit_code = 1
    record.update(finished_utc=now(),wall_seconds=time.perf_counter()-started,
                  peak_cpu_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource else None,
                  final_versions=runtime_versions())
    write_json(dest/'result.json',record)
    return exit_code

if __name__=='__main__':
    raise SystemExit(main())

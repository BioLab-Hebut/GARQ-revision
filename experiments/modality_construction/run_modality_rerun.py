#!/usr/bin/env python3
"""Rerun the 16 documented arms, with durable state and native GARQ outputs."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime

from release_paths import PACKAGE_ROOT, env_path, load_json
WORK = env_path('GARQ_HISTORICAL_WORK',PACKAGE_ROOT/'results/historical_generation_work')
BATCH = WORK / 'reports/modality_rerun_20260910'
PYTHON = str(env_path('GARQ_TRAIN_PYTHON',sys.executable))
DATA = load_json(env_path('GARQ_HISTORICAL_CONFIG',PACKAGE_ROOT/'configs/historical_16arm_inputs.py'))
GUARD = threading.RLock()


def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def jobs():
    out = []
    for dataset, (k, prefix, paths) in DATA.items():
        for n in range(1, len(paths) + 1):
            for mods in itertools.combinations(paths, n):
                name = prefix + '_' + '_'.join(mods)
                outputs = [str(WORK / 'save' / f'{name}_{m}_{k}metacell_k5.h5ad') for m in mods]
                outputs.append(str(WORK / 'save' / f'{name}_{k}metacell_k5_ids.h5ad'))
                out.append(dict(dataset=dataset, k=k, modalities=list(mods),
                                input_paths=[paths[m] for m in mods], save_name=name,
                                status='pending', seed=1, epoch=300,
                                log_path=str(WORK / 'log' / f'{name}_log.log'), outputs=outputs))
    return out


def save(state):
    with GUARD:
        state['updated_at'] = now()
        tmp = BATCH / 'state.json.tmp'
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp.replace(BATCH / 'state.json')



def prepare():
    """Read metadata and retain small pre-rerun evidence before any overwrite."""
    import anndata as ad
    BATCH.mkdir(parents=True, exist_ok=True)
    (WORK/'log').mkdir(exist_ok=True)
    (WORK/'save').mkdir(exist_ok=True)
    state_path = BATCH / 'state.json'
    if state_path.exists():
        raise RuntimeError('Batch already prepared; refusing to replace its state/evidence')
    state = dict(status='prepared', started_at=now(), jobs=jobs(),
                 work_dir=str(WORK), python=PYTHON, controller_pid=None, input_audit=[])
    evidence = BATCH / 'evidence_before'
    (evidence / 'log').mkdir(parents=True)
    (evidence / 'code').mkdir()
    (evidence / 'save').mkdir()
    for path in list(WORK.glob('*.py')) + ([WORK / 'run_trimodel.sh'] if (WORK / 'run_trimodel.sh').exists() else []):
        shutil.copy2(path, evidence / 'code' / path.name)
    for job in state['jobs']:
        p = Path(job['log_path'])
        if p.exists():
            shutil.copy2(p, evidence / 'log' / p.name)
        for p in (WORK / 'save').glob(job['save_name'] + '_*.txt'):
            shutil.copy2(p, evidence / 'save' / p.name)
        for modality in job['modalities']:
            pattern = f"{job['save_name']}_{job['k']}metacell_k5_{modality}_*cells_correlation.npy"
            if list((WORK / 'save').glob(pattern)):
                raise RuntimeError('New full-name cache already exists; inspect before rerun: ' + pattern)
    old_master = WORK / 'log/run_trimodel_seed1_master.log'
    if old_master.exists():
        shutil.copy2(old_master, evidence / 'log' / old_master.name)
    old_report = WORK / 'reports/GARQ_modality_ablation_seed1/GARQ_modality_ablation_seed1_report.html'
    if old_report.exists():
        shutil.copy2(old_report, evidence / old_report.name)
    outputs = []
    for job in state['jobs']:
        for name in job['outputs']:
            p = Path(name)
            if p.exists():
                outputs.append(dict(path=name, bytes=p.stat().st_size,
                                    mtime=p.stat().st_mtime, sha256=sha(p)))
    (evidence / 'output_hashes.json').write_text(json.dumps(outputs, indent=2), encoding='utf-8')
    for dataset, (_, _, paths) in DATA.items():
        reference = None
        for modality, path in paths.items():
            obj = ad.read_h5ad(path, backed='r')
            original = list(map(str, obj.obs_names))
            ids = [re.sub(r'\.(\d+)$', r'-\1', x) for x in original] if dataset == 'GSE158013' else original
            if reference is None:
                reference = ids
            info = dict(dataset=dataset, modality=modality, path=path,
                        shape=list(obj.shape), obs_columns=list(obj.obs.columns),
                        same_row_order=ids == reference, unique_ids=len(set(ids)) == len(ids),
                        id_sha256=hashlib.sha256('\n'.join(original).encode()).hexdigest(),
                        identity_check='terminal .number to -number for D18 comparison only' if dataset == 'GSE158013' else 'exact')
            obj.file.close()
            state['input_audit'].append(info)
            if not info['same_row_order'] or not info['unique_ids']:
                raise RuntimeError('Input pairing mismatch: ' + path)
    state['code_before'] = {p.name: sha(p) for p in (evidence / 'code').glob('*.py')}
    save(state)
    print(json.dumps(state['input_audit'], ensure_ascii=False), flush=True)
    print('PREPARED', flush=True)


def available(gpu):
    try:
        r = subprocess.run(['nvidia-smi', '-i', str(gpu),
                            '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, check=True)
        mem, util = map(int, r.stdout.strip().split(','))
        free_kb = int(re.search(r'MemAvailable:\s+(\d+)', Path('/proc/meminfo').read_text()).group(1))
        return mem < 1024 and util < 10 and free_kb > 130 * 1024 * 1024
    except Exception as e:
        print('Resource check:', repr(e), flush=True)
        return False


def run_one(job, gpu, state):
    start = time.time()
    with GUARD:
        job.update(status='running', gpu=gpu, started_at=now(), start_epoch=start)
        save(state)
    cmd = [PYTHON, '-u', str(WORK / 'GARQ.py'), '--data_path', *job['input_paths'],
           '--data_type', *job['modalities'], '--metacell_num', str(job['k']),
           '--save_name', job['save_name']]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1', MPLBACKEND='Agg',
               OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', NUMBA_NUM_THREADS='8')
    path = Path(job['log_path'])
    with path.open('w', encoding='utf-8', buffering=1) as log:
        log.write('RERUN_BATCH=modality_rerun_20260910\n')
        log.write('START_TIME=' + job['started_at'] + '\n')
        log.write('COMMAND_JSON=' + json.dumps(cmd, ensure_ascii=False) + '\n')
        log.write('CUDA_VISIBLE_DEVICES=' + str(gpu) + '\n')
        log.write('DEFAULT_SEED=1; EPOCH=300; K_KNN=5\n')
        log.flush()
        proc = subprocess.Popen(cmd, cwd=WORK, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT)
        with GUARD:
            job['pid'] = proc.pid
            save(state)
        rc = proc.wait()
        log.write('\nPYTHON_EXIT_CODE=' + str(rc) + '\nEND_TIME=' + now() + '\n')
    missing = [p for p in job['outputs'] if not Path(p).is_file() or Path(p).stat().st_size == 0
               or Path(p).stat().st_mtime < start]
    txt = path.read_text(encoding='utf-8', errors='replace')
    native_metric_lines = re.findall(r'^\* .*$', txt, re.M)
    complete = '======= Inference Done =======' in txt and bool(native_metric_lines)
    with GUARD:
        job.update(exit_code=rc, ended_at=now(), wall_seconds=round(time.time() - start, 2),
                   missing_or_stale=missing, native_metric_lines=native_metric_lines,
                   status='success' if rc == 0 and not missing and complete else 'failed')
        if job['status'] == 'failed':
            job['error_tail'] = txt[-3000:]
        save(state)
    print(job['save_name'], job['status'], 'exit=', rc, flush=True)


def worker(gpu, state):
    while True:
        with GUARD:
            pending = next((j for j in state['jobs'] if j['status'] == 'pending'), None)
            if pending is None:
                return
        if not available(gpu):
            time.sleep(30)
            continue
        with GUARD:
            job = next((j for j in state['jobs'] if j['status'] == 'pending'), None)
            if job is None:
                return
            job['status'] = 'assigned'
        try:
            run_one(job, gpu, state)
        except Exception as e:
            with GUARD:
                job.update(status='failed', ended_at=now(), error=str(e))
                save(state)
            print('JOB ERROR', job['save_name'], repr(e), flush=True)


def run():
    with (BATCH / 'controller.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads((BATCH / 'state.json').read_text(encoding='utf-8'))
        if state['status'] != 'prepared':
            raise RuntimeError('Run requires prepared state; no implicit duplicate rerun')
        state.update(status='running', controller_pid=os.getpid(), launched_at=now(),
                     code_after={p.name: sha(p) for p in WORK.glob('*.py')})
        save(state)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for future in [pool.submit(worker, gpu, state) for gpu in (0, 1)]:
                future.result()
        state['status'] = 'complete' if all(j['status'] == 'success' for j in state['jobs']) else 'completed_with_failures'
        state['ended_at'] = now()
        save(state)
        (BATCH / 'FINISHED').write_text(state['status'] + '\n', encoding='utf-8')
        print(state['status'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.prepare == args.run:
        parser.error('Choose exactly one of --prepare / --run')
    prepare() if args.prepare else run()

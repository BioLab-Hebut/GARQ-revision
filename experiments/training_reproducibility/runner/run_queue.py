#!/usr/bin/env python3
"""Serial Linux/CUDA queue for the prespecified D1-D12 repeated-training cohort."""
import argparse
import csv
import importlib.metadata
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

from common import EXPECTED_MODEL_SHA256, MODEL_FILES, SEEDS, input_snapshot, key, load_plan, read_json, sha256, source_hashes, validate_run, write_json


def runtime_versions():
    versions = {}
    for package in ('torch', 'scanpy', 'anndata', 'numpy', 'pandas', 'scipy', 'scikit-learn',
                    'faiss-cpu', 'faiss-gpu', 'h5py', 'psutil'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def preflight(ds, gpu, required_gpu_mib):
    import psutil
    for path in ds['data_files']:
        if not Path(path).is_file():
            raise RuntimeError(f'Missing input: {path}')
    available = psutil.virtual_memory().available
    if available < float(ds['minimum_available_ram_gib']) * 1024 ** 3:
        raise RuntimeError(f'Insufficient RAM: {available / 1024**3:.2f} GiB available; '
                           f'{ds["minimum_available_ram_gib"]} GiB required by dataset preflight.')
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                      '--format=csv,noheader,nounits'], text=True, timeout=10)
    free = {int(row.split(',')[0]): int(row.split(',')[1]) for row in output.strip().splitlines()}
    if free.get(gpu, -1) < required_gpu_mib:
        raise RuntimeError(f'GPU {gpu} requires at least {required_gpu_mib} MiB free; observed {free.get(gpu)}.')
    return {'available_ram_bytes': available, 'free_gpu_mib': free[gpu],
            'required_ram_gib': ds['minimum_available_ram_gib'], 'required_gpu_mib': required_gpu_mib}


def monitor(process, out, interval):
    import psutil
    started = time.monotonic()
    peaks = {name: None for name in ('main_rss_bytes', 'tree_pss_sum_bytes', 'process_gpu_bytes')}
    errors = set()
    with (out / 'resource_samples.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['seconds', 'phase', *peaks])
        writer.writeheader()
        while process.poll() is None:
            values = {name: None for name in peaks}
            pids = []
            try:
                parent = psutil.Process(process.pid)
                children = [parent] + parent.children(recursive=True)
                pss_values = []
                for child in children:
                    try:
                        memory = child.memory_full_info()
                        pids.append(child.pid)
                        if child.pid == parent.pid:
                            values['main_rss_bytes'] = memory.rss
                        pss_values.append(getattr(memory, 'pss', None))
                    except psutil.NoSuchProcess:
                        continue
                if pss_values and all(v is not None for v in pss_values):
                    values['tree_pss_sum_bytes'] = sum(pss_values)
            except (psutil.Error, OSError) as exc:
                errors.add(f'CPU memory sampling: {type(exc).__name__}')
            try:
                output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                                  '--format=csv,noheader,nounits'], text=True, timeout=5)
                matched = []
                for line in output.strip().splitlines():
                    fields = [s.strip() for s in line.split(',')]
                    if len(fields) == 2 and fields[0].isdigit() and int(fields[0]) in pids:
                        if fields[1].isdigit():
                            matched.append(int(fields[1]) * 1024 ** 2)
                        else:
                            errors.add('GPU memory reported unavailable for job PID')
                if matched:
                    values['process_gpu_bytes'] = sum(matched)
            except (subprocess.SubprocessError, ValueError, OSError) as exc:
                errors.add(f'GPU memory sampling: {type(exc).__name__}')
            try:
                phase = read_json(out / 'state.json').get('phase', 'unknown')
            except (OSError, ValueError):
                phase = 'startup'
            writer.writerow({'seconds': round(time.monotonic() - started, 3), 'phase': phase, **values})
            handle.flush()
            for name, value in values.items():
                if value is not None:
                    peaks[name] = value if peaks[name] is None else max(peaks[name], value)
            time.sleep(interval)
    return {'returncode': process.wait(), 'process_wall_seconds': time.monotonic() - started,
            'peaks': peaks, 'sampling_errors': sorted(errors),
            'definitions': 'Sampled job PID-tree PSS and nvidia-smi process GPU memory; unavailable samples are null. '
                           'CUDA/PyTorch stage values are separately written by the unchanged profile runner.'}


def stage_source(source, work):
    hashes = source_hashes(source / 'source_server')
    if hashes != EXPECTED_MODEL_SHA256:
        raise RuntimeError('Model modules differ from the verified original R2Q5 backend; do not launch.')
    paths = [Path('source_server') / name for name in MODEL_FILES]
    paths += [Path('scripts/profile_runner.py'), Path('scripts/analysis_metrics.py')]
    for relative in paths:
        src, dst = source / relative, work / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and sha256(src) != sha256(dst):
            raise RuntimeError(f'Existing staged source differs; use a new work-root: {dst}')
        if not dst.exists():
            shutil.copy2(src, dst)
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True, help='Unchanged original scalability directory')
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--minimum-free-gpu-mib', type=int, default=16000)
    parser.add_argument('--poll-seconds', type=float, default=2)
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--validate-only', action='store_true', help='Validate plan/staging/reuse; never launch training')
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('The serial queue requires Linux (flock, original CUDA runtime).')
    if args.minimum_free_gpu_mib < 16000 or args.poll_seconds <= 0:
        parser.error('Keep GPU threshold >=16000 MiB and a positive sampling interval.')
    import fcntl
    import psutil
    datasets, entries = load_plan(args.registry, args.manifest)
    work, source = args.work_root.resolve(), args.source_root.resolve()
    if work == source or source in work.parents:
        parser.error('Use a new output root outside the original source directory.')
    work.mkdir(parents=True, exist_ok=True)
    with (work / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        hashes = stage_source(source, work)
        identity = {'registry_sha256': sha256(args.registry), 'manifest_sha256': sha256(args.manifest),
                    'source_sha256': hashes, 'profile_runner_sha256': sha256(work / 'scripts/profile_runner.py'),
                    'analysis_metrics_sha256': sha256(work / 'scripts/analysis_metrics.py')}
        identity_path = work / 'plan_identity.json'
        if identity_path.exists() and read_json(identity_path) != identity:
            raise RuntimeError('Plan or code changed in an existing output root; do not mix cohorts.')
        write_json(identity_path, identity)
        status_path = work / 'queue_status.json'
        state = read_json(status_path) if status_path.exists() else {'runs': {}, 'protocol': identity}
        state.update({'status': 'running', 'supervisor_pid': os.getpid(), 'updated_unix': time.time()})
        write_json(status_path, state)
        any_failure = False
        for name in sorted(datasets, key=lambda s: int(s[1:])):
            ds = datasets[name]
            for seed in SEEDS:
                task = key(name, seed)
                entry, previous = entries[(name, seed)], state['runs'].get(task, {})
                try:
                    if entry.get('reuse'):
                        summary, *_ = validate_run(entry['reuse']['path'], ds, seed, hashes, entry['reuse'])
                        state['runs'][task] = {'status': 'reused_verified', **summary}
                    elif previous.get('status') == 'complete':
                        summary, *_ = validate_run(previous['path'], ds, seed, hashes)
                        state['runs'][task] = {'status': 'complete', **summary}
                    elif args.validate_only:
                        state['runs'][task] = {**previous, 'status': previous.get('status', 'pending'),
                                               'dataset': name, 'seed': seed}
                    else:
                        if previous.get('status') == 'running' and psutil.pid_exists(previous.get('pid', -1)):
                            raise RuntimeError(f'Previous child PID may still be active; inspect before relaunch: {previous}')
                        if previous.get('status') in ('failed', 'preflight_failed') and not args.retry_failed:
                            any_failure = True
                            continue
                        check = preflight(ds, args.gpu, args.minimum_free_gpu_mib)
                        (work / 'runs').mkdir(exist_ok=True)
                        prefix = f'{name}_K{ds["K"]}_B512_seed{seed}__attempt'
                        attempts = [int(p.name[len(prefix):]) for p in (work / 'runs').glob(prefix + '*')
                                    if p.name[len(prefix):].isdigit()]
                        attempt = max(attempts, default=0) + 1
                        run_id = prefix + f'{attempt:03d}'
                        out = work / 'runs' / run_id
                        out.mkdir(exist_ok=False)
                        cfg = {'dataset': name, 'run_id': run_id, 'epoch': 300, 'batch_size': 512, 'seed': seed,
                               'K': ds['K'], 'data_files': ds['data_files'], 'modalities': ds['modalities'],
                               'label_key': ds['label_key'], 'gpu': args.gpu,
                               'minimum_available_ram_gib': ds['minimum_available_ram_gib'],
                               'minimum_free_gpu_mib': args.minimum_free_gpu_mib}
                        config = work / 'configs' / (run_id + '.json')
                        write_json(config, cfg)
                        write_json(out / 'r66_run_manifest.json', {'dataset_spec': ds, 'source_sha256': hashes,
                                   'config': cfg, 'preflight': check, 'started_unix': time.time(),
                                   'input_file_snapshot': input_snapshot(ds['data_files']),
                                   'python': sys.version, 'executable': sys.executable,
                                   'package_versions': runtime_versions(),
                                   'profile_runner_sha256': identity['profile_runner_sha256']})
                        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
                                           MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
                        with (out / 'console.log').open('x') as log:
                            process = subprocess.Popen([sys.executable, '-u', str(work / 'scripts/profile_runner.py'), str(config)],
                                                       cwd=out, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                                       start_new_session=True)
                            state['runs'][task] = {'status': 'running', 'dataset': name, 'seed': seed,
                                                   'pid': process.pid, 'path': str(out), 'attempt': attempt}
                            write_json(status_path, state)
                            try:
                                resource = monitor(process, out, args.poll_seconds)
                            except BaseException:
                                if process.poll() is None:
                                    os.killpg(process.pid, signal.SIGTERM)
                                    try:
                                        process.wait(timeout=30)
                                    except subprocess.TimeoutExpired:
                                        os.killpg(process.pid, signal.SIGKILL)
                                        process.wait()
                                raise
                        write_json(out / 'resource_summary.json', resource)
                        if resource['returncode']:
                            raise RuntimeError(f'Child exited {resource["returncode"]}; inspect {out / "console.log"}')
                        summary, *_ = validate_run(out, ds, seed, hashes)
                        state['runs'][task] = {'status': 'complete', 'attempt': attempt, **summary}
                except BaseException as exc:
                    any_failure = True
                    record = dict(state['runs'].get(task, {}), status='failed', dataset=name, seed=seed,
                                  error=str(exc), traceback=traceback.format_exc(), failed_unix=time.time())
                    state['runs'][task] = record
                    if record.get('path') and not entry.get('reuse'):
                        write_json(Path(record['path']) / 'queue_failed.json', record)
                    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                        state.update({'status': 'interrupted', 'updated_unix': time.time()})
                        write_json(status_path, state)
                        raise
                state['updated_unix'] = time.time()
                write_json(status_path, state)
                print(task, state['runs'][task]['status'], flush=True)
        complete = sum(r.get('status') in ('complete', 'reused_verified') for r in state['runs'].values())
        state.update({'status': 'complete' if complete == 36 else ('validated_plan' if args.validate_only else 'incomplete'),
                      'completed_runs': complete, 'expected_runs': 36, 'updated_unix': time.time()})
        write_json(status_path, state)
        return 1 if any_failure else 0


if __name__ == '__main__':
    raise SystemExit(main())

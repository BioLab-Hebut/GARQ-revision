"""Read-only provenance checks for the prespecified 30-run primary-resolution rerun."""
from pathlib import Path
import hashlib
import os
import json
from settings_io import read_settings
import numpy as np

DEFAULT_ROOT = Path(os.environ.get('GARQ_CONTROL_OUTPUT', '/path/to/new/controlled_results'))
DEFAULT_INPUT_ROOT = Path(os.environ.get('GARQ_CONTROL_INPUT', '/path/to/controlled_inputs'))
TARGETS = {'D5': (12103, 403), 'D11': (9631, 321)}
VARIANTS = [('standard', 'GARQ_quantizer'), ('standard', 'SEACells'),
            ('standard', 'KMeans'), ('learned', 'GARQ_quantizer'), ('MOFA', 'GARQ_quantizer')]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def ids_sha256(ids):
    return hashlib.sha256('\n'.join(np.asarray(ids, str)).encode()).hexdigest()


def run_name(job):
    return f"{job['dataset']}__{job['representation']}__{job['method']}__K{job['target_k']}__s{job['seed']}"


def expected_jobs():
    return [dict(dataset=ds, representation=rep, method=method, seed=seed,
                 target_k=k, target_compression=30)
            for ds, (_, k) in TARGETS.items() for seed in (0, 1, 2)
            for rep, method in VARIANTS]


def read_config(path):
    path=Path(path)
    if not path.exists() and path.suffix=='.py' and path.with_suffix('.json').exists():
        path=path.with_suffix('.json')
    config = read_settings(path)
    actual = config['jobs']
    expected = {run_name(j): j for j in expected_jobs()}
    if len(actual) != 30 or len({run_name(j) for j in actual}) != 30:
        raise ValueError('Configuration must contain exactly 30 unique prespecified runs')
    for job in actual:
        name = run_name(job)
        if name not in expected or any(job.get(k) != v for k, v in expected[name].items()):
            raise ValueError(f'Unexpected resolution, variant or seed: {job}')
    return config


def input_references(input_root):
    refs = {}
    for ds, (n, _) in TARGETS.items():
        with np.load(input_root / 'inputs' / ds / 'standard.npz', allow_pickle=False) as data:
            refs[ds] = dict(cell_ids=data['cell_ids'].copy(), labels=data['labels'].copy())
        if len(refs[ds]['cell_ids']) != n or len(np.unique(refs[ds]['cell_ids'])) != n:
            raise ValueError(f'{ds}: unexpected or duplicate cell IDs')
    return refs


def representation_hashes(input_root, refs, learned_root=None):
    learned_root = input_root if learned_root is None else learned_root
    hashes = {}
    for ds in TARGETS:
        for rep in ('standard', 'learned', 'MOFA'):
            path = (learned_root / 'learned_training' / f'{ds}_seed1' / 'embedding.npz'
                    if rep == 'learned' else input_root / 'inputs' / ds / f'{rep}.npz')
            with np.load(path, allow_pickle=False) as data:
                x = np.asarray(data['embedding'], np.float32)
                for field in ('cell_ids', 'labels'):
                    if not np.array_equal(data[field], refs[ds][field]):
                        raise ValueError(f'{ds}/{rep}: {field} order mismatch')
            norms = np.linalg.norm(x, axis=1, keepdims=True)
            if x.shape != (TARGETS[ds][0], 64) or not np.isfinite(x).all() or (norms < 1e-10).any():
                raise ValueError(f'{ds}/{rep}: invalid 64D embedding')
            # Byte-for-byte normalization used by the frozen-representation runner.
            x = np.ascontiguousarray(x / norms)
            hashes[(ds, rep)] = hashlib.sha256(x.tobytes()).hexdigest()
    return hashes


def read_run(root, job, ref, expected_embedding_hash=None):
    name = run_name(job)
    path = root / 'runs' / name
    meta = json.loads((path / 'DONE.json').read_text())
    for key in ('dataset', 'representation', 'method', 'seed'):
        if meta[key] != job[key]:
            raise ValueError(f'{name}: metadata mismatch for {key}')
    if meta['target_K'] != job['target_k'] or meta['n_cells'] != TARGETS[job['dataset']][0]:
        raise ValueError(f'{name}: target K or cell count mismatch')
    if meta.get('representation_frozen') is not True or meta.get('dimensions') != 64:
        raise ValueError(f'{name}: representation must be frozen and 64-dimensional')
    if expected_embedding_hash is not None and meta['input_sha256'] != expected_embedding_hash:
        raise ValueError(f'{name}: fitted embedding differs from the frozen source')
    attempts = meta['attempts']
    best = min(attempts, key=lambda a: abs(a['realized_K'] - job['target_k']))
    if meta['selected_attempt'] != best['attempt']:
        raise ValueError(f'{name}: selected attempt does not follow K-only selection')
    for key in ('requested_K', 'realized_K'):
        if meta[key] != best[key]:
            raise ValueError(f'{name}: selected attempt mismatch for {key}')
    assignment_path = path / f"attempt{meta['selected_attempt']}" / 'assignments.npz'
    with np.load(assignment_path, allow_pickle=False) as data:
        ids, labels, y = data['cell_ids'].copy(), data['labels'].copy(), data['metacell'].copy()
    if not np.array_equal(ids, ref['cell_ids']) or not np.array_equal(labels, ref['labels']):
        raise ValueError(f'{name}: assignment cell order or labels differ from source')
    if y.ndim != 1 or len(y) != len(ids) or not np.issubdtype(y.dtype, np.integer) or (y < 0).any():
        raise ValueError(f'{name}: incomplete or invalid memberships')
    realized = len(np.unique(y))
    within = abs(realized - job['target_k']) / job['target_k'] <= .05
    if meta['realized_K'] != realized or bool(meta['within_five_percent']) != within:
        raise ValueError(f'{name}: realized K metadata does not match assignments')
    # A realized-K deviation is reported, never used here to omit a completed run.
    fields = {k: meta[k] for k in ('dataset', 'representation', 'method', 'seed', 'target_K',
              'requested_K', 'realized_K', 'within_five_percent', 'input_sha256')}
    fields.update(run=name, target_compression=30)
    provenance = dict(run=name, assignment_file=str(assignment_path),
                      assignment_file_sha256=sha256_file(assignment_path),
                      ordered_cell_ids_sha256=ids_sha256(ids),
                      embedding_sha256=meta['input_sha256'], selected_attempt=meta['selected_attempt'])
    return fields, y, labels, provenance


def load_runs(root, config, refs, hashes):
    expected = {run_name(job) for job in config['jobs']}
    available = {p.parent.name for p in (root / 'runs').glob('*/DONE.json')}
    if available != expected:
        raise ValueError(f'Run grid incomplete or contaminated. Missing={sorted(expected-available)}; '
                         f'unexpected={sorted(available-expected)}')
    return [read_run(root, job, refs[job['dataset']], hashes[(job['dataset'], job['representation'])])
            for job in config['jobs']]

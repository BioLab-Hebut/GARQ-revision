"""R66 runner validation; does not import or change the GARQ model."""
import csv
import hashlib
import json
from settings_io import read_settings
from pathlib import Path

K_MAP = dict(zip((f'D{i}' for i in range(1, 13)),
                 (491, 298, 475, 433, 403, 396, 282, 405, 496, 346, 321, 459)))
N_MAP = dict(zip((f'D{i}' for i in range(1, 13)),
                 (24559, 7468, 26169, 8670, 12103, 13870, 9876, 8105, 34774, 6924, 9631, 9190)))
MODEL_FILES = ('GARQ.py', 'model.py', 'engine.py', 'data_utils.py', 'eval_utils.py')
EXPECTED_MODEL_SHA256 = {
    'GARQ.py': '49c97de6440cd8ed76eca677a9225214752856461cc27930a0d0a465ae3179ab',
    'model.py': 'c629f59de63f4d6e3b3554a3a2b3c425b504ec4b92134167b60407b3dbbac1e8',
    'engine.py': '25a2ba5fda0c4ed06602d65ceef3a061f8bdf9c021740aa59fbde05198498765',
    'data_utils.py': 'f86c775960003b070d5fcd71f789bdc7d53357bae16db21b1b54aedd5760e97a',
    'eval_utils.py': 'e38819dc53085912c128401c852e86382f2d52abf1cb0e028ff9a72c43bbf9d8',
}
SEEDS = (1, 2, 3)


def read_json(path):
    return read_settings(path)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(source):
    return {name: sha256(Path(source) / name) for name in MODEL_FILES}


def key(dataset, seed):
    return f'{dataset}_seed{seed}'


def load_plan(registry_path, manifest_path):
    registry, manifest = read_json(registry_path), read_json(manifest_path)
    datasets = {d['id']: d for d in registry['datasets']}
    if len(registry['datasets']) != 12 or set(datasets) != set(K_MAP):
        raise ValueError('Registry must contain exactly D1-D12, without duplicates.')
    for name, ds in datasets.items():
        if ds.get('preflight_status') != 'PASS':
            raise ValueError(f'Input identity/order preflight has not passed: {name}')
        if int(ds['K']) != K_MAP[name] or int(ds['n_cells']) != N_MAP[name]:
            raise ValueError(f'Current manuscript K/N mismatch: {name}')
        expected_modalities = ['RNA', 'ADT'] if int(name[1:]) <= 6 else ['RNA', 'ATAC']
        if ds['modalities'] != expected_modalities or len(ds['data_files']) != 2:
            raise ValueError(f'Input modalities/order mismatch: {name}')
        if not ds['label_key'] or float(ds['minimum_available_ram_gib']) <= 0:
            raise ValueError(f'Missing label key or positive RAM threshold: {name}')
        if any(not Path(p).is_absolute() for p in ds['data_files']):
            raise ValueError(f'Use resolved absolute input paths: {name}')
    entries = {(r['dataset'], int(r['seed'])): r for r in manifest['runs']}
    expected = {(d, s) for d in datasets for s in SEEDS}
    if len(manifest['runs']) != 36 or set(entries) != expected:
        raise ValueError('Manifest must contain exactly D1-D12 x seeds 1,2,3.')
    return datasets, entries


def cell_digest(cells):
    return hashlib.sha256(('\n'.join(cells) + '\n').encode('utf-8')).hexdigest()


def input_snapshot(paths):
    result = []
    for value in paths:
        path = Path(value).resolve()
        stat = path.stat()
        result.append({'path': str(path), 'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
    return result


def validate_run(path, ds, seed, hashes, reuse=None):
    """Validate actual artifacts, returning metadata and the saved partition."""
    import numpy as np
    path = Path(path).resolve()
    done = read_json(path / 'completed.json')
    resource = read_json(path / 'resource_summary.json')
    cfg = read_json(path / 'effective_config.json')
    if done.get('status') != 'complete' or int(resource['returncode']) != 0:
        raise ValueError(f'Run is not successfully completed: {path}')
    required = {'K': ds['K'], 'seed': seed, 'epoch': 300,
                'batch_size': 512, 'train_batch_size': 512, 'eval_batch_size': 2048}
    for field, value in required.items():
        if int(cfg[field]) != int(value):
            raise ValueError(f'{path.name}: {field}={cfg[field]} differs from {value}')
    if cfg['modalities'] != ds['modalities'] or cfg['label_key'] != ds['label_key']:
        raise ValueError(f'Modalities/annotation key differ: {path}')
    if not cfg['train_shuffle'] or not cfg['train_drop_last'] or cfg['eval_shuffle'] or cfg['eval_drop_last']:
        raise ValueError(f'Loader ordering/drop_last settings differ: {path}')
    if int(cfg['num_workers']) != 4 or not cfg['pin_memory']:
        raise ValueError(f'Training loader settings differ: {path}')
    if reuse:
        proof = reuse.get('verification', {})
        if proof.get('input_identity_verified') is not True or proof.get('canonical_order_verified') is not True:
            raise ValueError(f'Historical run input/order verification missing: {path}')
        if proof.get('source_sha256') != hashes or not proof.get('evidence'):
            raise ValueError(f'Historical run source verification/evidence missing: {path}')
        actual_source = path.parent.parent / 'source_server'
        if actual_source.is_dir() and source_hashes(actual_source) != hashes:
            raise ValueError(f'Historical model source files differ: {path}')
    else:
        if [str(Path(p).resolve()) for p in cfg['data_files']] != [str(Path(p).resolve()) for p in ds['data_files']]:
            raise ValueError(f'Input paths differ: {path}')
        provenance = read_json(path / 'r66_run_manifest.json')
        if provenance['source_sha256'] != hashes or provenance['dataset_spec'] != ds:
            raise ValueError(f'New-run provenance differs: {path}')
        if provenance['input_file_snapshot'] != input_snapshot(ds['data_files']):
            raise ValueError(f'Input file size/mtime changed since this run; verify before reuse: {path}')
    for artifact in ('checkpoint.pt', 'embeddings.npy', 'assignments.npy', 'cell_index.csv'):
        if not (path / artifact).is_file() or not (path / artifact).stat().st_size:
            raise ValueError(f'Missing/empty artifact: {path / artifact}')
    ids = np.load(path / 'assignments.npy', allow_pickle=False)
    if ids.ndim != 1 or len(ids) != ds['n_cells'] or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError(f'Invalid assignment vector: {path}')
    if ids.min() < 0 or ids.max() >= ds['K']:
        raise ValueError(f'Assignment outside requested anchor range: {path}')
    embeds = np.load(path / 'embeddings.npy', mmap_mode='r', allow_pickle=False)
    if embeds.ndim != 2 or embeds.shape[0] != ds['n_cells'] or not np.isfinite(embeds).all():
        raise ValueError(f'Invalid saved embeddings: {path}')
    with (path / 'cell_index.csv').open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    cells, labels = [r['cell_id'] for r in rows], [r['cell_type'] for r in rows]
    if len(cells) != len(ids) or len(set(cells)) != len(cells) or any(not c for c in cells):
        raise ValueError(f'Cell IDs are missing, duplicated or incomplete: {path}')
    if [int(r['row']) for r in rows] != list(range(len(ids))):
        raise ValueError(f'Cell index is not canonical row order: {path}')
    if ds.get('cell_id_sha256') and cell_digest(cells) != ds['cell_id_sha256']:
        raise ValueError(f'Canonical cell-ID hash differs: {path}')
    if any(not label or label.lower() in ('nan', 'none') for label in labels):
        raise ValueError(f'Missing cell annotations: {path}')
    metadata = cfg.get('input_metadata', [])
    if len(metadata) != 2 or any(int(m['raw_shape'][0]) != ds['n_cells'] for m in metadata):
        raise ValueError(f'Input metadata does not identify full cohort: {path}')
    if ds.get('raw_shapes') and [m['raw_shape'] for m in metadata] != ds['raw_shapes']:
        raise ValueError(f'Raw feature dimensions differ: {path}')
    if int(done['quality']['K_actual']) != len(np.unique(ids)) or int(done['quality']['n_cells']) != len(ids):
        raise ValueError(f'Completion metadata and assignments disagree: {path}')
    fixed_path = path / 'assignments_fixed_eval_B2048.npy'
    if fixed_path.exists() and not np.array_equal(ids, np.load(fixed_path, allow_pickle=False)):
        raise ValueError(f'Saved native B2048 and fixed B2048 assignments differ: {path}')
    summary = {'dataset': ds['id'], 'seed': seed, 'path': str(path), 'n_cells': len(ids),
               'requested_K': ds['K'], 'actual_K': len(np.unique(ids)),
               'train_batch_size': 512, 'eval_batch_size': 2048,
               'assignment_sha256': sha256(path / 'assignments.npy'),
               'cell_index_sha256': sha256(path / 'cell_index.csv'),
               'cell_id_sha256': cell_digest(cells), 'reused': bool(reuse)}
    return summary, ids, cells, labels

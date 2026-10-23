"""Recompute the locked population and post-hoc metrics from all 30 new assignments.

No fitting, sample generation, ranking-based selection, or secondary-resolution data.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from r2c2_metrics import evaluate, test_metrics
from rerun_common import (DEFAULT_ROOT, DEFAULT_INPUT_ROOT, TARGETS, ids_sha256,
                          read_config, input_references, representation_hashes, load_runs,
                          sha256_file)


def load_saved_samples(input_root, refs):
    """Verify the existing sample files against the original deterministic sampling rule."""
    manifest_path = input_root / 'evaluation' / 'posthoc_sampling_manifest.csv'
    manifest = pd.read_csv(manifest_path)
    rows = {row['sample_file']: row for row in manifest.to_dict('records')}
    if len(rows) != len(manifest):
        raise ValueError('Duplicate sample IDs in the source manifest')
    samplings = {ds: [] for ds in TARGETS}
    seen = set()
    for ds, ref in refs.items():
        labels, ids = ref['labels'], ref['cell_ids']
        for ct in np.unique(labels):
            target = np.flatnonzero(labels == ct)
            others = np.flatnonzero(labels != ct)
            abundance = len(target) / len(labels)
            if abundance > .02:
                continue
            for level in (.01, .005, .002, .001):
                keep = int(np.floor(level * len(others) / (1 - level)))
                if keep < 1 or keep > len(target):
                    continue
                for seed in (0, 1, 2):
                    digest = hashlib.sha256(f'{ds}|{ct}|{level}|{seed}'.encode()).hexdigest()
                    filename = digest[:20] + '.npz'
                    row = rows[filename]
                    expected_fields = dict(dataset=ds, celltype=ct, sampling_seed=seed,
                                           target_abundance=level, original_target_cells=len(target),
                                           retained_target_cells=keep)
                    for key, value in expected_fields.items():
                        if row[key] != value:
                            raise ValueError(f'{filename}: source manifest mismatch for {key}')
                    with np.load(input_root / 'posthoc_samples' / filename, allow_pickle=False) as data:
                        indices, sampled_ids = data['positions'].copy(), data['cell_ids'].copy()
                    rng = np.random.default_rng(int(digest[:16], 16))
                    expected_indices = np.sort(np.r_[others, rng.choice(target, keep, replace=False)])
                    if not np.array_equal(indices, expected_indices):
                        raise ValueError(f'{filename}: saved sampling positions changed')
                    if not np.array_equal(sampled_ids, ids[indices]):
                        raise ValueError(f'{filename}: sampled cell order mismatch')
                    if row['sampled_cell_ids_sha256'] != ids_sha256(sampled_ids):
                        raise ValueError(f'{filename}: sampled cell hash mismatch')
                    if row['n_cells_after_sampling'] != len(indices):
                        raise ValueError(f'{filename}: sampled cell count mismatch')
                    if not np.isclose(row['realized_abundance'], keep / len(indices), atol=1e-15, rtol=0):
                        raise ValueError(f'{filename}: realized abundance mismatch')
                    seen.add(filename)
                    samplings[ds].append((row, indices))
    if seen != set(rows):
        raise ValueError('Source manifest contains an omitted or ineligible sampling case')
    return manifest, samplings


def posthoc_row(y, labels, info, indices):
    """Same target-only calculation as the original r2c2_evaluate.py."""
    sy, sl = y[indices], labels[indices]
    cluster, inv = np.unique(sy, return_inverse=True)
    sizes = np.bincount(inv)
    hits = np.bincount(inv, weights=(sl == info['celltype']).astype(int), minlength=len(sizes))
    dominated = hits / sizes >= .5
    tp, positive, n = hits[dominated].sum(), sizes[dominated].sum(), info['retained_target_cells']
    precision, recall = tp / positive if positive else 0., tp / n
    return dict(realized_K_after_sampling=len(cluster), compression_after_sampling=len(sy) / len(cluster),
                precision=precision, recall=recall,
                F1=2 * precision * recall / (precision + recall) if precision + recall else 0.,
                pooled_purity=precision if positive else np.nan,
                mean_metacell_purity=float((hits[dominated] / sizes[dominated]).mean()) if dominated.any() else np.nan,
                target_dominated_metacells=int(dominated.sum()),
                median_metacell_size=float(np.median(sizes[dominated])) if dominated.any() else np.nan,
                min_metacell_size=int(sizes[dominated].min()) if dominated.any() else np.nan,
                analysis='post_hoc_fixed_memberships_no_refitting')


def evaluate_runs(runs, samplings):
    summaries, populations, sizes, rare = [], [], [], []
    for fields, y, labels, _ in runs:
        summary, pop, mc = evaluate(labels, y, fields['requested_K'])
        summaries.append(dict(fields, **{k: v for k, v in summary.items() if k not in fields}))
        populations.extend(dict(fields, **row) for row in pop)
        sizes.extend(dict(fields, **row) for row in mc.to_dict('records'))
        for info, indices in samplings[fields['dataset']]:
            rare.append(dict(fields, **{k: v for k, v in info.items() if k not in fields},
                             **posthoc_row(y, labels, info, indices)))
        print(json.dumps(dict(event='evaluated', run=fields['run'], realized_K=summary['realized_K'])), flush=True)
    return dict(controlled_summary=summaries, controlled_all_population_metrics=populations,
                controlled_metacell_sizes=sizes, controlled_posthoc_all_eligible_populations=rare)


def main(args):
    root, input_root = Path(args.root), Path(args.input_root)
    config_path = Path(args.config) if args.config else root / 'config.py'
    config = read_config(config_path)
    refs = input_references(input_root)
    hashes = representation_hashes(input_root, refs)
    runs = load_runs(root, config, refs, hashes)
    manifest, samplings = load_saved_samples(input_root, refs)
    results = evaluate_runs(runs, samplings)
    out = root / 'evaluation'
    out.mkdir(exist_ok=True)
    for name, rows in results.items():
        pd.DataFrame(rows).to_csv(out / f'{name}.csv', index=False)
    manifest.to_csv(out / 'posthoc_sampling_manifest.csv', index=False)
    (out / 'metric_validation.json').write_text(json.dumps(test_metrics(), indent=2))
    audit = dict(status='passed', runs=30, target_compression=30,
                 target_K={ds: k for ds, (_, k) in TARGETS.items()},
                 config_sha256=sha256_file(config_path), source_root=str(input_root),
                 population_rows=len(results['controlled_all_population_metrics']),
                 posthoc_rows=len(results['controlled_posthoc_all_eligible_populations']),
                 sampling_cases=len(manifest),
                 source_sampling_manifest_sha256=sha256_file(input_root / 'evaluation' / 'posthoc_sampling_manifest.csv'),
                 cell_identity_and_order='verified for every run and representation',
                 fixed_embedding_hashes={f'{ds}/{rep}': h for (ds, rep), h in hashes.items()},
                 posthoc_samples='existing sample positions and ordered IDs verified; no new samples generated',
                 run_selection='all 30 prespecified runs retained, no metric-based selection',
                 assignments=[run[3] for run in runs])
    (out / 'evaluation_provenance.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(dict(event='complete', runs=30, population_rows=audit['population_rows'],
                          posthoc_rows=audit['posthoc_rows'], sampling_cases=len(manifest))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(DEFAULT_ROOT))
    parser.add_argument('--input-root', default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument('--config')
    main(parser.parse_args())

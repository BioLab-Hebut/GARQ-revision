"""Completion gate for the 30-run rerun; never modifies assignments or source inputs."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from r2c2_metrics import evaluate
from rerun_common import (DEFAULT_ROOT, DEFAULT_INPUT_ROOT, read_config, input_references,
                          representation_hashes, load_runs, run_name)
from evaluate_rerun import load_saved_samples, posthoc_row


def same_number(a, b):
    return bool(np.isclose(a, b, atol=1e-12, rtol=1e-12, equal_nan=True))


def main(args):
    root, input_root = Path(args.root), Path(args.input_root)
    out = root / 'evaluation'
    config = read_config(root / 'config.py')
    refs = input_references(input_root)
    runs = load_runs(root, config, refs, representation_hashes(input_root, refs))
    manifest, samples = load_saved_samples(input_root, refs)
    summary = pd.read_csv(out / 'controlled_summary.csv')
    population = pd.read_csv(out / 'controlled_all_population_metrics.csv')
    sizes = pd.read_csv(out / 'controlled_metacell_sizes.csv')
    posthoc = pd.read_csv(out / 'controlled_posthoc_all_eligible_populations.csv')
    expected_runs = {run_name(job) for job in config['jobs']}
    assert len(summary) == 30 and set(summary['run']) == expected_runs
    assert len(population) == 690
    assert len(manifest) == 249 and len(posthoc) == 3735
    assert summary['target_compression'].eq(30).all()
    assert summary['coverage'].eq(1).all()
    assert np.allclose(summary['compression_N_over_K'], summary['n_cells'] / summary['realized_K'])
    checked_rows = 0
    for fields, y, labels, _ in runs:
        name, ds = fields['run'], fields['dataset']
        size = sizes.loc[sizes['run'].eq(name)]
        assert len(size) == fields['realized_K'] and size['size'].sum() == len(y)
        pop = population.loc[population['run'].eq(name)]
        assert set(pop['celltype']) == set(np.unique(labels))
        post = posthoc.loc[posthoc['run'].eq(name)]
        assert set(post['sample_file']) == {row['sample_file'] for row, _ in samples[ds]}
        assert len(post) == len(samples[ds])
        # Independent full confusion-matrix evaluation checks the optimized target-only path.
        for index in (0, len(samples[ds]) // 2, len(samples[ds]) - 1):
            info, indices = samples[ds][index]
            _, population_full, _ = evaluate(labels[indices], y[indices])
            target = next(row for row in population_full if row['celltype'] == info['celltype'])
            optimized = posthoc_row(y, labels, info, indices)
            exported = post.loc[post['sample_file'].eq(info['sample_file'])].iloc[0]
            for key in ('precision', 'recall', 'F1', 'pooled_purity', 'mean_metacell_purity',
                        'target_dominated_metacells', 'median_metacell_size', 'min_metacell_size'):
                assert same_number(target[key], optimized[key]), (name, info['sample_file'], key)
                assert same_number(target[key], exported[key]), (name, info['sample_file'], key)
            checked_rows += 1
    geometry_rows = 0
    for ds, mods in [('D5', {'RNA', 'ADT'}), ('D11', {'RNA', 'ATAC'})]:
        geometry = pd.read_csv(out / f'geometry_{ds}.csv')
        assert len(geometry) == 30 and set(geometry['modality']) == mods
        ds_runs = {fields['run'] for fields, _, _, _ in runs if fields['dataset'] == ds}
        for mod in mods:
            subset = geometry.loc[geometry['modality'].eq(mod)]
            assert set(subset['run']) == ds_runs and len(subset) == 15
        assert np.isfinite(geometry.select_dtypes('number').to_numpy()).all()
        geometry_rows += len(geometry)
    audit = dict(status='passed', runs=30, population_rows=len(population), sampling_cases=len(manifest),
                 posthoc_rows=len(posthoc), posthoc_independent_full_matrix_checks=checked_rows,
                 geometry_rows=geometry_rows, target_compression=30,
                 source_and_grid_validation='all passed; no secondary-resolution runs',
                 metric_bounds='precision, recall, F1 and purity checked in [0,1]')
    for frame in (population, posthoc):
        for key in ('precision', 'recall', 'F1', 'pooled_purity', 'mean_metacell_purity'):
            observed = frame[key].dropna()
            assert observed.between(0, 1).all(), key
    (out / 'FINAL_VALIDATION.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(DEFAULT_ROOT))
    parser.add_argument('--input-root', default=str(DEFAULT_INPUT_ROOT))
    main(parser.parse_args())

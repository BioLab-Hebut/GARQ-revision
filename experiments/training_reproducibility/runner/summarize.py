#!/usr/bin/env python3
"""Read saved original-model assignments; compute fixed-B repeat ARI and label quality."""
import argparse
import csv
import importlib.util
import itertools
from pathlib import Path
import statistics
import sys

import numpy as np
from sklearn.metrics import adjusted_rand_score

from common import SEEDS, key, load_plan, read_json, sha256, source_hashes, validate_run, write_json


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='New empty summary directory')
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    datasets, entries = load_plan(args.registry, args.manifest)
    work = args.work_root.resolve()
    identity = read_json(work / 'plan_identity.json')
    if identity['registry_sha256'] != sha256(args.registry) or identity['manifest_sha256'] != sha256(args.manifest):
        raise ValueError('Summary plan differs from the executed queue.')
    if (identity['source_sha256'] != source_hashes(work / 'source_server') or
        identity['analysis_metrics_sha256'] != sha256(work / 'scripts/analysis_metrics.py')):
        raise ValueError('Staged model or analysis helper changed after queue preparation.')
    state = read_json(work / 'queue_status.json')
    spec = importlib.util.spec_from_file_location('r66_original_analysis_metrics', work / 'scripts/analysis_metrics.py')
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    args.out.mkdir(parents=True, exist_ok=False)
    per_run, pairs, summaries, recall, missing, validated = [], [], [], [], [], []
    for name in sorted(datasets, key=lambda s: int(s[1:])):
        ds = datasets[name]
        partitions, reference_cells, reference_labels, reference_categories = {}, None, None, None
        qualities = []
        for seed in SEEDS:
            record = state['runs'].get(key(name, seed), {})
            if record.get('status') not in ('complete', 'reused_verified'):
                missing.append({'dataset': name, 'seed': seed, 'status': record.get('status', 'missing'),
                                'reason': record.get('error', 'No verified completed run')})
                continue
            reuse = entries[(name, seed)].get('reuse')
            info, ids, cells, labels = validate_run(record['path'], ds, seed, identity['source_sha256'], reuse)
            annotations = analysis.load_annotations(Path(record['path']), ds['label_key'])
            if annotations.astype(str).tolist() != labels:
                raise ValueError(f'Saved annotation and cell index differ: {name} seed {seed}')
            categories = list(annotations.cat.categories) if hasattr(annotations, 'cat') else None
            if reference_cells is not None and (cells != reference_cells or labels != reference_labels or categories != reference_categories):
                raise ValueError(f'Cell order/labels/category order differ across seeds: {name}')
            reference_cells, reference_labels, reference_categories = cells, labels, categories
            quality, predictions, _, ties = analysis.majority_statistics(ids, annotations)
            qualities.append(quality['balanced_purity'])
            partitions[seed] = ids
            per_run.append({**info, 'balanced_accuracy': quality['balanced_purity'],
                            'cell_weighted_purity': quality['cell_weighted_purity'],
                            'mean_metacell_purity': quality['mean_metacell_purity'], 'majority_tie_groups': ties})
            values = np.asarray(labels)
            for label in np.unique(values):
                mask = values == label
                recall.append({'dataset': name, 'seed': seed, 'cell_type': label,
                               'n_cells': int(mask.sum()), 'majority_recall': float(np.mean(predictions[mask] == label))})
            validated.append(info)
        if len(partitions) == 3:
            aris = []
            for first, second in itertools.combinations(SEEDS, 2):
                ari = float(adjusted_rand_score(partitions[first], partitions[second]))
                aris.append(ari)
                pairs.append({'dataset': name, 'train_batch_size': 512, 'eval_batch_size': 2048,
                              'seed_a': first, 'seed_b': second, 'ARI_assignment': ari})
            summaries.append({'dataset': name, 'requested_K': ds['K'], 'n_cells': ds['n_cells'],
                              'train_batch_size': 512, 'eval_batch_size': 2048, 'training_runs': 3,
                              'pairwise_comparisons': 3, 'ARI_mean': statistics.mean(aris),
                              'ARI_min': min(aris), 'ARI_max': max(aris),
                              'balanced_accuracy_mean': statistics.mean(qualities),
                              'balanced_accuracy_sample_sd': statistics.stdev(qualities)})
    write_csv(args.out / 'per_run_quality.csv', per_run)
    write_csv(args.out / 'pairwise_training_ari.csv', pairs)
    write_csv(args.out / 'dataset_summary.csv', summaries)
    write_csv(args.out / 'cell_type_recall.csv', recall)
    report = {'status': 'PASS' if len(validated) == 36 and len(summaries) == 12 else 'PARTIAL',
              'validated_runs': len(validated), 'completed_datasets': len(summaries),
              'expected_runs': 36, 'expected_datasets': 12, 'missing': missing,
              'source': identity, 'runs': validated,
              'definition': 'Native original-model assignments, training B512 and canonical inference B2048; '
                            'three pairwise ARIs per dataset share the three training runs and are not independent replicates. '
                            'Balanced accuracy and per-class recall preserve the original categorical majority-tie rule.'}
    write_json(args.out / 'validation.json', report)
    print(f'{report["status"]}: {len(validated)}/36 runs, {len(summaries)}/12 complete datasets')
    return 0 if report['status'] == 'PASS' or args.allow_partial else 2


if __name__ == '__main__':
    sys.exit(main())

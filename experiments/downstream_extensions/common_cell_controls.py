"""D18 common-cell sensitivity using six frozen native partitions.

No GARQ/comparator training or reassignment is performed. Every arm uses exactly
the intersection of valid cell IDs, with native boundaries retained. Native
metacell counts/sizes vary; this complements, rather than replaces, the 369x20
control. Original downstream workflows are reused. Molecular correlations run
once; MOFA/KMeans and TF-LASSO use three technical fitting seeds on the same data.
"""
from pathlib import Path
import argparse
import os
import hashlib
import inspect
import json
import shutil
import sys
import traceback

import anndata as ad
import numpy as np
import pandas as pd


DEFAULT_OLD = Path('/path/to/project_home/vscode/GARQ20260905/reviewer2_comment4_20260917')
DEFAULT_RAW = Path('/path/to/project_home/data/RNA_ATAC_ADT/GSE158013')
DEFAULT_METAQ = Path('/path/to/data_home/GARQ_revision/revision_results/phase2/01_size_resolution/D18/MetaQ/full_seed1_K002_realizedmatched/cell_assignments.csv')
DEFAULT_OUT = Path('/path/to/project_home/GARQ_revision14_20261002/common_cell')
ARMS = ('GARQ_RNA_ATAC_ADT', 'MetaQ', 'SEACells', 'MetaCellV2', 'SuperCell', 'GARQ_RNA')
MODALITIES = ('RNA', 'ATAC', 'ADT')


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def dump(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf8')


def aligned_memberships(path, ids, labels, workflow, metaq=False):
    df = pd.read_csv(path)
    group_col, label_col = ('metacell_id', 'cell_type') if metaq else ('group', 'celltype')
    required = {'cell_id', group_col, label_col}
    if not required.issubset(df.columns):
        raise ValueError(f'{path}: missing columns {required - set(df.columns)}')
    members = workflow.canon(df.cell_id)
    workflow.check_unique(members, f'{path}: membership IDs')
    if set(members) != set(ids):
        raise ValueError(f'{path}: membership cell universe differs from raw D18')
    lookup = {value: index for index, value in enumerate(members)}
    order = np.asarray([lookup[value] for value in ids])
    source_labels = df[label_col].astype(str).to_numpy()[order]
    if not np.array_equal(source_labels, labels):
        raise ValueError(f'{path}: cell-type metadata differs after ID alignment')
    source_groups = df[group_col].astype(str).to_numpy()[order]
    normalized, native_names = workflow.normalize_groups(source_groups)
    return normalized, source_groups, native_names


def restrict_partition(native, common):
    """Remove only non-common cells, then relabel occupied native groups."""
    if np.any(native[common] < 0):
        raise ValueError('Common pool contains an unassigned native cell')
    occupied, inverse = np.unique(native[common], return_inverse=True)
    groups = np.full(len(native), -1, dtype=int)
    groups[common] = inverse
    counts = np.bincount(groups[common], minlength=len(occupied))
    if np.any(counts <= 0) or not np.array_equal(groups >= 0, common):
        raise AssertionError('Common-cell compression changed pool membership')
    return groups, occupied, counts


def majority_quality(groups, labels, all_types):
    """Original alphabetical majority tie rule; no absent classes are imputed."""
    valid = groups >= 0
    k = int(groups.max()) + 1
    counts = np.bincount(groups[valid], minlength=k)
    if k < 1 or np.any(counts == 0):
        raise ValueError('Empty/noncontiguous occupied groups')
    code = {value: index for index, value in enumerate(all_types)}
    truth = np.asarray([code[value] for value in labels], dtype=int)
    confusion = np.zeros((k, len(all_types)), dtype=np.int64)
    np.add.at(confusion, (groups[valid], truth[valid]), 1)
    majority = confusion.argmax(axis=1)
    predicted = np.full(len(labels), -1, dtype=int)
    predicted[valid] = majority[groups[valid]]
    correct = predicted == truth
    recalls = {}
    coverage = {}
    for value, index in code.items():
        mask = truth == index
        retained = mask & valid
        denominator = int(mask.sum())
        retained_count = int(retained.sum())
        coverage[value] = retained_count
        recalls[value] = {
            'original_cells': denominator,
            'retained_cells': retained_count,
            'excluded_cells': denominator - retained_count,
            'correctly_majority_labelled_cells': int(np.sum(mask & correct)),
            'full_cohort_recall': float(np.sum(mask & correct) / denominator),
            'retained_cell_recall': float(np.mean(correct[retained])) if retained_count else None,
        }
    retained_recalls = [r['retained_cell_recall'] for r in recalls.values() if r['retained_cell_recall'] is not None]
    full_recalls = [r['full_cohort_recall'] for r in recalls.values()]
    sizes = {
        'min': int(counts.min()), 'q1': float(np.quantile(counts, .25)),
        'median': float(np.median(counts)), 'q3': float(np.quantile(counts, .75)),
        'p95': float(np.quantile(counts, .95)), 'max': int(counts.max()),
        'mean': float(counts.mean()), 'singleton_groups': int(np.sum(counts == 1)),
        'groups_lt5': int(np.sum(counts < 5)),
    }
    purity = confusion.max(axis=1) / counts
    return {
        'K': k, 'assigned_cells': int(valid.sum()), 'excluded_cells': int((~valid).sum()),
        'original_classes': len(all_types), 'retained_classes': len(retained_recalls),
        'mean_metacell_purity': float(purity.mean()),
        'cell_weighted_metacell_purity': float(np.sum(confusion.max(axis=1)) / counts.sum()),
        'balanced_majority_accuracy_retained_present_classes': float(np.mean(retained_recalls)),
        'balanced_majority_accuracy_retained_all_original_classes': float(np.mean(retained_recalls)) if len(retained_recalls) == len(all_types) else None,
        'balanced_majority_accuracy_full_cohort_excluded_unrecovered': float(np.mean(full_recalls)),
        'size_summary': sizes, 'celltype_coverage': coverage, 'class_recalls': recalls,
    }, np.asarray(all_types)[majority], purity


def parameterized_mofa(mofa, analysis, features, out):
    """Change only the two original stochastic seeds and descriptive DONE text."""
    source = inspect.getsource(mofa.run_batch)
    replacements = {
        'def run_batch(tag):': 'def run_batch_with_seed(tag, seed):',
        "seed=1,gpu_mode=False": "seed=seed,gpu_mode=False",
        'random_state=1).fit_predict(latent)': 'random_state=seed).fit_predict(latent)',
        "'All 10 arms completed'": "'All six common-cell arms completed'",
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError(f'MOFA source structure changed: {old!r}')
        source = source.replace(old, new)
    # No feature selection, preprocessing, fitting settings, or metrics are edited.
    adapted = out / 'mofa_run_batch_seed_parameter.py'
    adapted.write_text(source, encoding='utf8')
    exec(compile(source, str(adapted), 'exec'), mofa.__dict__)
    mofa.RUN = analysis
    mofa.FEATURES = features
    return mofa.run_batch_with_seed, sha(adapted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-root', type=Path, default=DEFAULT_OLD)
    parser.add_argument('--raw-root', type=Path, default=DEFAULT_RAW)
    parser.add_argument('--metaq-assignment', type=Path, default=DEFAULT_METAQ)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--rscript', default='/usr/bin/Rscript')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)
    code_root = Path(os.environ.get('GARQ_ASSOCIATION_CODE', str(Path(__file__).resolve().parents[1] / 'matched_association/code')))
    sys.path.insert(0, str(code_root))
    import run_bimodal_downstream as workflow
    import mofa_controls as mofa

    analysis = out / 'analysis'
    analysis.mkdir(exist_ok=True)
    feature_source = args.old_root / 'analysis/MOFA_common_features.json'
    features = analysis / 'MOFA_common_features.json'
    if features.exists() and sha(features) != sha(feature_source):
        raise ValueError('Output fixed-feature file differs from original')
    if not features.exists():
        shutil.copy2(feature_source, features)
    feature_spec = json.loads(features.read_text())
    if feature_spec['MOFA']['factors'] != 10:
        raise ValueError('Expected original ten-factor MOFA design')

    raw = {mod: ad.read_h5ad(args.raw_root / f'GSE158013_{mod.lower()}.h5ad', backed='r') for mod in MODALITIES}
    try:
        ids = workflow.canon(raw['RNA'].obs_names)
        workflow.check_unique(ids, 'raw RNA cells')
        labels = raw['RNA'].obs.celltype.astype(str).to_numpy()
        all_types = sorted(np.unique(labels).tolist())
        if len(ids) != 25517 or len(all_types) != 12:
            raise ValueError('Unexpected original D18 cell/class count')
        for mod, a in raw.items():
            mid = workflow.canon(a.obs_names)
            workflow.check_unique(mid, f'raw {mod} cells')
            if set(mid) != set(ids):
                raise ValueError(f'{mod}: different paired raw-cell universe')
            order = {value: index for index, value in enumerate(mid)}
            local_labels = a.obs.celltype.astype(str).to_numpy()[[order[value] for value in ids]]
            if not np.array_equal(local_labels, labels):
                raise ValueError(f'{mod}: cell-type metadata differs after barcode alignment')
            if not a.var_names.is_unique or not set(feature_spec['features'][mod]).issubset(a.var_names):
                raise ValueError(f'{mod}: missing/duplicated fixed MOFA features')

        native = {}
        source_groups = {}
        assignment_info = {}
        for arm in ARMS:
            path = args.metaq_assignment if arm == 'MetaQ' else args.old_root / 'analysis/full' / arm / 'memberships.csv.gz'
            native[arm], source_groups[arm], native_names = aligned_memberships(path, ids, labels, workflow, metaq=(arm == 'MetaQ'))
            assignment_info[arm] = {
                'path': str(path), 'sha256': sha(path),
                'aligned_source_group_sha256': workflow.digest(source_groups[arm]),
                'occupied_native_K': len(native_names),
                'native_unassigned_cells': int(np.sum(native[arm] < 0)),
                'construction_modalities': 'RNA_ATAC_ADT' if arm in ('GARQ_RNA_ATAC_ADT', 'MetaQ') else 'RNA',
            }
        common = np.logical_and.reduce([native[arm] >= 0 for arm in ARMS])
        if not common.any():
            raise ValueError('No common valid cell pool')
        metadata = {
            'design': 'frozen_native_partitions_common_valid_cell_intersection',
            'dataset': 'D18/GSE158013', 'new_model_training': False,
            'new_inference_or_reassignment': False, 'count_size_matching': False,
            'original_N': len(ids), 'common_N': int(common.sum()),
            'excluded_N': int((~common).sum()), 'common_coverage_fraction': float(common.mean()),
            'raw_cell_id_sha256': workflow.digest(ids),
            'raw_celltype_sha256': workflow.digest(labels),
            'common_cell_id_sha256': workflow.digest(np.asarray(ids)[common]),
            'excluded_cell_id_sha256': workflow.digest(np.asarray(ids)[~common]),
            'original_label_counts': {value: int(np.sum(labels == value)) for value in all_types},
            'common_label_counts': {value: int(np.sum((labels == value) & common)) for value in all_types},
            'raw_paths': {mod: str(a.filename) for mod, a in raw.items()},
            'raw_shapes': {mod: list(a.shape) for mod, a in raw.items()},
            'assignments': assignment_info,
            'aggregation': 'arithmetic_mean_of_raw_X_over_every_common_cell_in_each_native_group',
            'majority_tie_rule': 'alphabetical original cell-type label, identical to original workflow',
            'full_cohort_BA_definition': 'mean of 12 original-class recalls; cells excluded from the evaluated partition remain in each original-class denominator and count as unrecovered',
            'retained_BA_definition': 'mean recalls in common cell pool; absent original classes remain explicitly unavailable in the all-original-class result',
            'molecular_correlations': 'one deterministic original-workflow analysis on each common-cell aggregation',
            'MOFA_fitting_and_KMeans_seeds': [1, 2, 3],
            'TF_crossvalidation_seeds': [1, 2, 3],
            'replicate_scope': 'technical fit/CV sensitivity conditional on one frozen native partition per arm; same common cell pool in all fits',
            'fixed_features_sha256': sha(features),
            'source_sha256': {str(path): sha(path) for path in [
                Path(__file__), code_root / 'run_bimodal_downstream.py',
                code_root / 'mofa_controls.py', code_root / 'source/atac.ipynb',
                code_root / 'source/adt.ipynb', code_root / 'source/compare.R']},
        }
        provenance = out / 'provenance.json'
        if provenance.exists() and json.loads(provenance.read_text()) != metadata:
            raise ValueError('Existing output provenance differs; choose a fresh output root')
        dump(provenance, metadata)
        pd.DataFrame({'cell_id': ids, 'celltype': labels, 'in_common_pool': common}).to_csv(out / 'common_cell_pool.csv.gz', index=False)
        quality = []
        native_quality = []
        groups_by_arm = {}
        for arm in ARMS:
            result, _, _ = majority_quality(native[arm], labels, all_types)
            native_quality.append({'arm': arm, 'scope': 'native_full_cohort', **result})
            groups, occupied, counts = restrict_partition(native[arm], common)
            groups_by_arm[arm] = groups
            result, majority, purity = majority_quality(groups, labels, all_types)
            quality.append({'arm': arm, 'scope': 'common_cell_pool', **result})
            folder = analysis / 'common_native' / arm
            folder.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({'cell_id': ids, 'group': groups, 'celltype': labels,
                          'native_source_group': source_groups[arm]}).to_csv(folder / 'memberships.csv.gz', index=False)
            pd.DataFrame({'metacell_id': [f'{arm}__{i}' for i in range(len(counts))],
                          'normalized_native_group': occupied, 'n_cells': counts,
                          'celltype': majority, 'purity': purity}).to_csv(folder / 'metacell_metadata.csv', index=False)
        dump(out / 'native_full_cohort_quality.json', native_quality)
        dump(out / 'size_and_coverage.json', quality)
        flat_rows = []
        for row in quality + native_quality:
            flat_rows.append({key: value for key, value in row.items() if not isinstance(value, dict)})
        pd.DataFrame(flat_rows).to_csv(out / 'quality_metrics.csv', index=False)
        print('COMMON_POOL', int(common.sum()), 'EXCLUDED', int((~common).sum()), flush=True)
        print('NATIVE_K', {arm: assignment_info[arm]['occupied_native_K'] for arm in ARMS}, flush=True)
        if args.prepare_only:
            print('PREPARE_ONLY_COMPLETE', flush=True)
            return

        arm_inputs = {}
        for arm in ARMS:
            folder = analysis / 'common_native' / arm
            groups = groups_by_arm[arm]
            counts = np.bincount(groups[common])
            paths = {}
            for mod, a in raw.items():
                path = folder / f'D18_{arm}_{mod}_common_native.h5ad'
                if not path.exists():
                    workflow.aggregate_view(a, groups, ids, labels, path, arm, assignment_info[arm]['construction_modalities'])
                check = ad.read_h5ad(path, backed='r')
                try:
                    expected_ids = [f'{arm}__{i}' for i in range(len(counts))]
                    if check.obs_names.tolist() != expected_ids or check.n_obs != len(counts) or check.n_vars != a.n_vars:
                        raise ValueError(f'{path}: aggregate dimensions or profile IDs mismatch')
                    if not np.array_equal(check.obs.n_cells.to_numpy(), counts):
                        raise ValueError(f'{path}: aggregate constituent counts mismatch')
                finally:
                    check.file.close()
                paths[mod] = str(path)
            arm_inputs[arm] = paths
            print('AGGREGATED', arm, len(counts), int(counts.sum()), flush=True)
        base = analysis / 'common_native'
        dump(base / 'MOFA_input_manifest.json', arm_inputs)
        if not (base / 'ASSOCIATIONS_DONE').exists():
            workflow.run_associations(base, arm_inputs)
            (base / 'ASSOCIATIONS_DONE').write_text('One deterministic analysis per frozen common-cell partition', encoding='utf8')
        fit_mofa, adapted_sha = parameterized_mofa(mofa, analysis, features, out)
        dump(out / 'mofa_seed_adaptation.json', {
            'original_run_batch_source_sha256': hashlib.sha256(inspect.getsource(mofa.run_batch).encode()).hexdigest(),
            'adapted_source_sha256': adapted_sha,
            'changes': ['MOFA seed argument', 'KMeans random_state argument', 'function name/signature', 'DONE description'],
            'other_statistics_and_preprocessing_changes': False,
        })
        for seed in (1, 2, 3):
            tag = f'fit_seed{seed}'
            batch = analysis / tag
            batch.mkdir(exist_ok=True)
            dump(batch / 'MOFA_input_manifest.json', arm_inputs)
            if not (batch / 'TF_DONE').exists():
                print('TF_START', tag, flush=True)
                workflow.run_tf(batch, arm_inputs, args.rscript, seed)
                for arm in ARMS:
                    coefficients = pd.read_csv(batch / 'TF_gene' / arm / 'fixed_metacell_order_lasso_coefficients.csv')
                    if len(coefficients) != 390 or not np.isfinite(coefficients.coefficient).all() or not np.isfinite(coefficients['lambda']).all():
                        raise ValueError(f'{tag}/{arm}: TF output missing, nonfinite, or fallback lambda')
                (batch / 'TF_DONE').write_text('Six arms, 390 finite coefficients and tuning lambdas each', encoding='utf8')
            print('MOFA_FIT_START', tag, flush=True)
            fit_mofa(tag, seed)
            for arm in ARMS:
                metrics = json.loads((batch / 'MOFA' / arm / 'metrics.json').read_text())
                if metrics['K'] != int(groups_by_arm[arm].max()) + 1:
                    raise ValueError(f'{tag}/{arm}: MOFA profile count differs')
            print('FIT_SEED_DONE', tag, flush=True)
        dump(out / 'COMPLETE.json', {
            'status': 'PASS', 'arms': list(ARMS), 'common_cells': int(common.sum()),
            'analyses': ['MOFA', 'original_peak_gene_examples', 'original_RNA_protein_pairs', 'fixed_TF_gene_LASSO'],
            'deterministic_association_runs_per_arm': 1,
            'technical_fit_seeds': [1, 2, 3], 'new_model_training': False,
            'native_metacell_count_and_size_are_not_matched': True,
        })
    finally:
        for a in raw.values():
            a.file.close()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        # Only an isolated analysis output is written; original inputs are untouched.
        parser_out = DEFAULT_OUT
        if '--output-root' in sys.argv:
            parser_out = Path(sys.argv[sys.argv.index('--output-root') + 1])
        parser_out.mkdir(parents=True, exist_ok=True)
        (parser_out / 'FAILED.txt').write_text(traceback.format_exc(), encoding='utf8')
        raise

"""Apply the original geometry definitions to new assignments and existing correlations."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from rerun_common import (DEFAULT_ROOT, DEFAULT_INPUT_ROOT, TARGETS, read_config, input_references,
                          representation_hashes, load_runs, sha256_file)


def geom(c,y):
    # Verbatim metric implementation from work/r2c2_geometry.py.
    n=len(y);clusters=np.unique(y);comp=0.;sep=0.;comp_ex=0.;ex_weight=0;rows=[]
    for cluster in clusters:
        ix=np.flatnonzero(y==cluster);size=len(ix)
        inside=np.asarray(c[np.ix_(ix,ix)]);avg=float(inside.mean(dtype=np.float64))
        block=np.asarray(c[ix,:]).copy();block[:,ix]=-np.inf
        d=float(np.mean(1-block.max(1),dtype=np.float64))
        comp+=avg*size/n;sep+=d*size/n
        excl=(float(inside.sum(dtype=np.float64))-size)/(size*(size-1)) if size>1 else np.nan
        if size>1:comp_ex+=excl*size;ex_weight+=size
        rows.append({'metacell':int(cluster),'size':size,'compactness_per_metacell':avg,'separation_per_metacell':d,'paper_compactness_scaled':avg*size/n*len(clusters),'paper_separation_scaled':d*size/n*len(clusters)})
    return {'compactness_cell_weighted':comp,'separation_cell_weighted':sep,'compactness_excluding_diagonal_cell_weighted':comp_ex/ex_weight,'compactness_paper_scaled_median':float(np.median([r['paper_compactness_scaled'] for r in rows])),'separation_paper_scaled_median':float(np.median([r['paper_separation_scaled'] for r in rows]))},rows


def main(args):
    root, input_root = Path(args.root), Path(args.input_root)
    config = read_config(Path(args.config) if args.config else root / 'config.py')
    refs = input_references(input_root)
    runs = load_runs(root, config, refs, representation_hashes(input_root, refs))
    ds = args.dataset
    n = TARGETS[ds][0]
    mods = ('RNA', 'ADT') if ds == 'D5' else ('RNA', 'ATAC')
    cache_provenance, summary, metacells = [], [], []
    for mod in mods:
        path = input_root / 'geometry_cache' / f'{ds}_{mod}.npy'
        meta_path = path.with_suffix('.json')
        meta = json.loads(meta_path.read_text())
        if meta['dataset'] != ds or meta['modality'] != mod or meta['n_cells'] != n:
            raise ValueError(f'{ds}/{mod}: cached correlation metadata mismatch')
        c = np.load(path, mmap_mode='r')
        if c.shape != (n, n):
            raise ValueError(f'{ds}/{mod}: cached matrix dimensions mismatch')
        ix = np.array([0, n // 3, n // 2, n - 1])
        spot = np.asarray(c[np.ix_(ix, ix)])
        if not np.isfinite(spot).all() or not np.allclose(np.diag(spot), 1, atol=1e-6):
            raise ValueError(f'{ds}/{mod}: invalid correlation cache')
        cache_provenance.append(dict(dataset=ds, modality=mod, source_file=str(path),
                                     file_sha256=sha256_file(path), metadata=meta))
        for fields, y, labels, _ in runs:
            if fields['dataset'] != ds:
                continue
            metrics, details = geom(c, y)
            summary.append(dict(run=fields['run'], dataset=ds, modality=mod, **metrics))
            metacells.extend(dict(run=fields['run'], dataset=ds, modality=mod, **row) for row in details)
            print(json.dumps(dict(event='geometry_evaluated', run=fields['run'], modality=mod)), flush=True)
        del c
    out = root / 'evaluation'
    out.mkdir(exist_ok=True)
    pd.DataFrame(summary).to_csv(out / f'geometry_{ds}.csv', index=False)
    pd.DataFrame(metacells).to_csv(out / f'geometry_metacells_{ds}.csv', index=False)
    (out / f'geometry_provenance_{ds}.json').write_text(json.dumps(dict(
        status='passed', dataset=ds, runs=15, modality_rows=len(summary),
        cache_action='read-only reuse; correlations not recomputed', caches=cache_provenance,
        metric_definitions='unchanged from original r2c2_geometry.py'), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(DEFAULT_ROOT))
    parser.add_argument('--input-root', default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument('--config')
    parser.add_argument('--dataset', required=True, choices=list(TARGETS))
    main(parser.parse_args())

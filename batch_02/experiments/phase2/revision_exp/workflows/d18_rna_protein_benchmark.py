"""D18 cross-method RNA-protein associations on frozen assignments.

This full-feature analysis is descriptive/circular. Feature-excluded reruns remain
required for confirmatory claims.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse, stats

RNA = Path(os.environ.get("GARQ_DATA_ROOT", "./data")) / 'RNA_ATAC_ADT/GSE158013/GSE158013_rna.h5ad'
ADT = Path(os.environ.get("GARQ_DATA_ROOT", "./data")) / 'RNA_ATAC_ADT/GSE158013/GSE158013_adt.h5ad'
ALIASES = {"CD56": "NCAM1", "CD16": "FCGR3A", "CD25": "IL2RA", "CD127": "IL7R"}


def aggregate(x, codes: np.ndarray, k: int) -> np.ndarray:
    m = sparse.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(k, len(codes)))
    value = m @ x
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def residualize(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = values.astype(float).copy()
    for group in np.unique(groups):
        mask = groups == group
        out[mask] -= out[mask].mean()
    return out


def bootstrap_r(rng, x, y, groups, iterations=1000):
    vals = []
    for _ in range(iterations):
        idx = rng.integers(0, len(x), len(x))
        rx, ry = residualize(x[idx], groups[idx]), residualize(y[idx], groups[idx])
        vals.append(stats.pearsonr(rx, ry).statistic if np.std(rx) and np.std(ry) else np.nan)
    return np.nanpercentile(vals, [2.5, 97.5])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--assignments", type=Path, required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    assn = pd.read_csv(args.assignments)
    labels, codes = np.unique(assn.metacell_id.astype(str), return_inverse=True)
    rna, adt = ad.read_h5ad(RNA), ad.read_h5ad(ADT)
    if not np.array_equal(assn.cell_id.astype(str), rna.obs_names.astype(str)):
        raise ValueError("assignment/RNA mismatch")
    canonical_adt = adt.obs_names.astype(str).str.replace(r"\.([0-9]+)$", r"-\1", regex=True)
    if not np.array_equal(rna.obs_names.astype(str), canonical_adt):
        raise ValueError("RNA/ADT canonical ID mismatch")

    adt_names = list(map(str, adt.var_names))
    rna_names = list(map(str, rna.var_names))
    rna_lookup = {x.upper(): i for i, x in enumerate(rna_names)}
    pairs = []
    for j, protein in enumerate(adt_names):
        gene = ALIASES.get(protein, protein).upper()
        if gene in rna_lookup:
            pairs.append((gene, protein, rna_lookup[gene], j))
    required = {("NCAM1", "CD56"), ("CD8A", "CD8a")}
    if not required.issubset({(g, p) for g, p, _, _ in pairs}):
        raise ValueError("prespecified RNA-protein pair missing")

    k = len(labels)
    rna_counts = aggregate(rna.X[:, [i for _, _, i, _ in pairs]], codes, k)
    adt_counts = aggregate(adt.X[:, [j for _, _, _, j in pairs]], codes, k)
    rna_values = np.log1p(rna_counts / np.maximum(rna_counts.sum(axis=1, keepdims=True), 1) * 1e4)
    log_adt = np.log1p(adt_counts)
    adt_values = log_adt - log_adt.mean(axis=1, keepdims=True)
    source = rna.obs["celltype"].astype(str).to_numpy()
    dominant = []
    for i in range(k):
        values, counts = np.unique(source[codes == i], return_counts=True)
        dominant.append(values[np.argmax(counts)])
    dominant = np.asarray(dominant)
    rng = np.random.default_rng(args.seed)
    rows = []
    for index, (gene, protein, _, _) in enumerate(pairs):
        x, y = rna_values[:, index], adt_values[:, index]
        rx, ry = residualize(x, dominant), residualize(y, dominant)
        lo, hi = bootstrap_r(rng, x, y, dominant)
        rows.append({
            "status": "PASS", "method": args.method, "seed": args.seed, "realized_K": k,
            "gene": gene, "protein": protein, "prespecified": (gene, protein) in required,
            "pearson_r": stats.pearsonr(x, y).statistic,
            "spearman_rho": stats.spearmanr(x, y).statistic,
            "partial_pearson_r": stats.pearsonr(rx, ry).statistic,
            "partial_ci_low": lo, "partial_ci_high": hi,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"PASS method={args.method} seed={args.seed} K={k} pairs={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()

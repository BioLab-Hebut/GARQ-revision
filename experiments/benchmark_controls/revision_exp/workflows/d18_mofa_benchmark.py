"""Frozen cross-method D18 MOFA+ benchmark, one assignment per invocation."""

from __future__ import annotations

import argparse
import os
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, balanced_accuracy_score, normalized_mutual_info_score


SOURCES = {
    "RNA": Path(os.environ.get("GARQ_DATA_ROOT", "./data")) / 'RNA_ATAC_ADT/GSE158013/GSE158013_rna.h5ad',
    "ATAC": Path(os.environ.get("GARQ_DATA_ROOT", "./data")) / 'RNA_ATAC_ADT/GSE158013/GSE158013_atac.h5ad',
    "ADT": Path(os.environ.get("GARQ_DATA_ROOT", "./data")) / 'RNA_ATAC_ADT/GSE158013/GSE158013_adt.h5ad',
}
SETTINGS = {
    "rna_features": 2000,
    "atac_features": 5000,
    "adt_features": "all",
    "factors": 15,
    "mofa_seed": 0,
    "max_iterations": 1000,
    "convergence_mode": "medium",
    "scale_views": True,
    "scale_groups": False,
    "spikeslab_weights": True,
    "ard_factors": True,
    "ard_weights": True,
}


def top_dispersion(x: sparse.spmatrix, n: int) -> np.ndarray:
    x = x.tocsr().astype(np.float64)
    mean = np.asarray(x.mean(axis=0)).ravel()
    mean2 = np.asarray(x.multiply(x).mean(axis=0)).ravel()
    score = (mean2 - mean * mean) / np.maximum(mean, 1e-8)
    return np.argsort(score, kind="stable")[-min(n, x.shape[1]):]


def aggregate(x: sparse.spmatrix, codes: np.ndarray, k: int) -> np.ndarray:
    membership = sparse.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(k, len(codes)))
    result = membership @ x
    if sparse.issparse(result):
        result = result.toarray()
    return np.asarray(result, dtype=np.float64)


def zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std(axis=0)
    keep = sd > 1e-8
    x = x[:, keep]
    return (x - x.mean(axis=0)) / x.std(axis=0)


def cluster_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, np.ndarray]:
    true_levels, ti = np.unique(y_true, return_inverse=True)
    pred_levels, pi = np.unique(y_pred, return_inverse=True)
    mat = np.zeros((len(pred_levels), len(true_levels)), dtype=int)
    np.add.at(mat, (pi, ti), 1)
    rows, cols = linear_sum_assignment(-mat)
    mapping = {pred_levels[r]: true_levels[c] for r, c in zip(rows, cols)}
    mapped = np.asarray([mapping.get(v, true_levels[0]) for v in y_pred])
    return float((mapped == y_true).mean()), mapped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    assignments = pd.read_csv(args.assignments)
    labels, codes = np.unique(assignments["metacell_id"].astype(str), return_inverse=True)
    k = len(labels)
    rna = ad.read_h5ad(SOURCES["RNA"])
    atac = ad.read_h5ad(SOURCES["ATAC"])
    adt = ad.read_h5ad(SOURCES["ADT"])
    if not np.array_equal(assignments.cell_id.astype(str), rna.obs_names.astype(str)):
        raise ValueError("Assignment/RNA row order mismatch")
    if not np.array_equal(rna.obs_names.astype(str), atac.obs_names.astype(str)):
        raise ValueError("RNA/ATAC row order mismatch")

    rna_idx = top_dispersion(rna.X, SETTINGS["rna_features"])
    atac_prevalence = np.asarray((atac.X > 0).sum(axis=0)).ravel()
    atac_idx = np.argsort(atac_prevalence, kind="stable")[-SETTINGS["atac_features"]:]
    rna_counts = aggregate(rna.X[:, rna_idx], codes, k)
    atac_counts = aggregate(atac.X[:, atac_idx], codes, k)
    adt_counts = aggregate(adt.X, codes, k)

    rna_view = np.log1p(rna_counts / np.maximum(rna_counts.sum(axis=1, keepdims=True), 1) * 1e4)
    tf = atac_counts / np.maximum(atac_counts.sum(axis=1, keepdims=True), 1)
    idf = np.log1p(k / np.maximum((atac_counts > 0).sum(axis=0), 1))
    atac_view = np.log1p(tf * idf * 1e4)
    log_adt = np.log1p(adt_counts)
    adt_view = log_adt - log_adt.mean(axis=1, keepdims=True)
    views = [zscore(rna_view), zscore(atac_view), zscore(adt_view)]

    source_labels = rna.obs["celltype"].astype(str).to_numpy()
    dominant = []
    for index in range(k):
        values, counts = np.unique(source_labels[codes == index], return_counts=True)
        dominant.append(values[np.argmax(counts)])
    dominant = np.asarray(dominant)

    from mofapy2.run.entry_point import entry_point
    ep = entry_point()
    ep.set_data_options(scale_groups=SETTINGS["scale_groups"], scale_views=SETTINGS["scale_views"])
    ep.set_data_matrix([[views[0]], [views[1]], [views[2]]], views_names=["RNA", "ATAC", "ADT"], groups_names=["D18"])
    ep.set_model_options(
        factors=SETTINGS["factors"], spikeslab_weights=SETTINGS["spikeslab_weights"],
        ard_factors=SETTINGS["ard_factors"], ard_weights=SETTINGS["ard_weights"],
    )
    ep.set_train_options(
        iter=SETTINGS["max_iterations"], convergence_mode=SETTINGS["convergence_mode"],
        seed=SETTINGS["mofa_seed"], verbose=False,
    )
    ep.build(); ep.run()
    model_path = args.output_dir / "model.hdf5"
    ep.save(str(model_path), save_data=False)
    factors = np.asarray(ep.model.nodes["Z"].getExpectation())
    if factors.shape[0] != k and factors.shape[1] == k:
        factors = factors.T
    if factors.shape[0] != k:
        raise ValueError(f"Unexpected MOFA factor shape {factors.shape}; expected {k} samples")
    n_types = len(np.unique(dominant))
    predicted = KMeans(n_clusters=n_types, random_state=0, n_init=20).fit_predict(factors)
    acc, mapped = cluster_accuracy(dominant, predicted)
    metrics = {
        "status": "PASS", "dataset": "D18", "method": args.method, "seed": args.seed,
        "cells": len(assignments), "realized_K": k, "factors": SETTINGS["factors"],
        "dominant_cell_types": n_types,
        "ARI": adjusted_rand_score(dominant, predicted),
        "NMI": normalized_mutual_info_score(dominant, predicted),
        "AMI": adjusted_mutual_info_score(dominant, predicted),
        "ACC": acc, "balanced_accuracy": balanced_accuracy_score(dominant, mapped),
        "wall_time_seconds": time.perf_counter() - started,
    }
    pd.DataFrame([metrics]).to_csv(args.output_dir / "mofa_metrics.csv", index=False)
    pd.DataFrame(factors, index=labels).to_csv(args.output_dir / "factors.csv")
    (args.output_dir / "settings.json").write_text(json.dumps(SETTINGS, indent=2) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

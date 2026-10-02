"""E3 fixed-representation aggregation controls."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler, normalize

from revision_exp.metrics.metacell import evaluate_assignments
from revision_exp.utils.provenance import peak_rss_bytes, write_json
from revision_exp.workflows.legacy import _canonicalize_obs_names, _prepare_uniform_subset


def _dense(matrix: Any) -> np.ndarray:
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def _rna_pca(adata: ad.AnnData, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    work = adata.copy()
    sc.pp.normalize_total(work, target_sum=1e4)
    sc.pp.log1p(work)
    n_top = min(3000, work.n_vars)
    sc.pp.highly_variable_genes(work, n_top_genes=n_top)
    selected = work.var["highly_variable"].to_numpy()
    matrix = _dense(work[:, selected].X)
    n_components = min(50, matrix.shape[0] - 1, matrix.shape[1])
    embedding = PCA(n_components=n_components, svd_solver="randomized", random_state=seed).fit_transform(matrix)
    return embedding, {
        "preprocessing": "normalize_total_1e4+log1p+HVG+PCA",
        "selected_features": int(selected.sum()),
        "components": n_components,
    }


def _atac_lsi(adata: ad.AnnData, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = sparse.csr_matrix(adata.X, dtype=np.float64)
    if matrix.data.size and matrix.data.min() < 0:
        raise ValueError("ATAC TF-IDF requires non-negative input")
    depth = np.asarray(matrix.sum(axis=1)).ravel()
    inv_depth = np.divide(1.0, depth, out=np.zeros_like(depth), where=depth > 0)
    term_frequency = sparse.diags(inv_depth) @ matrix
    document_frequency = np.asarray((matrix > 0).sum(axis=0)).ravel()
    inverse_document_frequency = np.log1p(matrix.shape[0] / (1.0 + document_frequency))
    tfidf = term_frequency.multiply(inverse_document_frequency).tocsr()
    n_components = min(51, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
    raw_lsi = TruncatedSVD(n_components=n_components, random_state=seed).fit_transform(tfidf)
    first_depth_correlation = (
        float(np.corrcoef(raw_lsi[:, 0], depth)[0, 1])
        if np.std(depth) > 0 and np.std(raw_lsi[:, 0]) > 0
        else float("nan")
    )
    remove_first = bool(np.isfinite(first_depth_correlation) and abs(first_depth_correlation) >= 0.5)
    start = 1 if remove_first else 0
    embedding = raw_lsi[:, start : start + min(50, raw_lsi.shape[1] - start)]
    return embedding, {
        "preprocessing": "TF-IDF+TruncatedSVD_LSI",
        "input_features": int(matrix.shape[1]),
        "components": int(embedding.shape[1]),
        "first_component_depth_correlation": first_depth_correlation,
        "remove_first_component_threshold_abs_correlation": 0.5,
        "first_component_removed": remove_first,
    }


def _adt_pca(adata: ad.AnnData, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    counts = _dense(adata.X).astype(np.float64, copy=False)
    if counts.size and counts.min() < 0:
        raise ValueError("ADT CLR requires non-negative input")
    logged = np.log1p(counts)
    clr = logged - logged.mean(axis=1, keepdims=True)
    n_components = min(30, counts.shape[1] - 1, counts.shape[0] - 1)
    embedding = PCA(n_components=n_components, svd_solver="randomized", random_state=seed).fit_transform(clr)
    return embedding, {
        "preprocessing": "per-cell CLR(log1p centered)+PCA",
        "input_features": int(counts.shape[1]),
        "components": n_components,
    }


def _block(data_type: str, adata: ad.AnnData, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    if data_type == "RNA":
        return _rna_pca(adata, seed)
    if data_type == "ATAC":
        return _atac_lsi(adata, seed)
    if data_type == "ADT":
        return _adt_pca(adata, seed)
    raise ValueError(f"Unsupported modality: {data_type}")


def run_fixed_representation_benchmark(
    config: dict[str, Any], output_dir: Path, result_root: Path
) -> None:
    if config["implementation_tag"] != "diagnostic_fixed_representation":
        raise ValueError("Fixed representation controls require diagnostic_fixed_representation")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_paths = [Path(path) for path in config["data_files"]]
    paths = [str(path) for path in source_paths]
    subset_info = {"subset_applied": False}
    if config.get("cell_limit"):
        paths, subset_info = _prepare_uniform_subset(
            source_paths,
            result_root / "cache" / "subsets",
            config["dataset"],
            int(config["cell_limit"]),
            int(config.get("subset_seed", config["seed"])),
            config.get("obs_name_canonicalization"),
        )
    write_json(output_dir / "subset_provenance.json", subset_info)
    adatas = [ad.read_h5ad(path) for path in paths]
    rules = config.get("obs_name_canonicalization") or [None] * len(adatas)
    if len(rules) != len(adatas):
        raise ValueError("obs_name_canonicalization must match data_files length")
    for item, rule in zip(adatas, rules):
        item.obs_names = _canonicalize_obs_names(item.obs_names.astype(str).to_numpy(), rule)
    reference_names = adatas[0].obs_names.astype(str).to_numpy()
    if any(not np.array_equal(reference_names, item.obs_names.astype(str).to_numpy()) for item in adatas[1:]):
        raise ValueError("Paired modalities do not have identical canonical cell order")
    stage_rows = []
    blocks = []
    block_provenance = []
    for index, (data_type, adata) in enumerate(zip(config["data_types"], adatas)):
        started = time.perf_counter()
        embedding, provenance = _block(data_type, adata, int(config["seed"]) + index)
        standardized = StandardScaler().fit_transform(embedding)
        normalized = normalize(standardized, norm="l2", axis=1)
        blocks.append(normalized.astype(np.float32))
        block_provenance.append({"data_type": data_type, **provenance, "postprocessing": "feature_standardize+per-cell_L2"})
        stage_rows.append(
            {
                "stage": f"representation_{data_type}",
                "wall_time_seconds": time.perf_counter() - started,
                "peak_cpu_rss_bytes": peak_rss_bytes(),
                "output_shape": str(list(normalized.shape)),
            }
        )
    fixed = np.concatenate(blocks, axis=1)
    np.save(output_dir / "fixed_representation.npy", fixed)
    write_json(
        output_dir / "fixed_representation_provenance.json",
        {
            "dataset": config["dataset"],
            "shape": list(fixed.shape),
            "equal_weight_concatenation": True,
            "blocks": block_provenance,
        },
    )
    labels = (
        adatas[0].obs[config.get("label_key", "celltype")].astype(str).to_numpy()
        if config.get("label_key", "celltype") in adatas[0].obs
        else None
    )
    summary_tables = []
    per_type_tables = []
    for method in config.get("aggregation_methods", ["KMeans", "MiniBatchKMeans"]):
        started = time.perf_counter()
        if method == "KMeans":
            estimator = KMeans(n_clusters=int(config["requested_K"]), n_init=10, random_state=int(config["seed"]))
        elif method == "MiniBatchKMeans":
            estimator = MiniBatchKMeans(
                n_clusters=int(config["requested_K"]),
                n_init=10,
                batch_size=int(config.get("minibatch_size", 256)),
                random_state=int(config["seed"]),
            )
        else:
            raise ValueError(f"Unsupported aggregation method: {method}")
        assignments_array = estimator.fit_predict(fixed)
        stage_rows.append(
            {
                "stage": f"aggregation_{method}",
                "wall_time_seconds": time.perf_counter() - started,
                "peak_cpu_rss_bytes": peak_rss_bytes(),
                "output_shape": str(list(assignments_array.shape)),
            }
        )
        assignments = pd.DataFrame(
            {
                "cell_id": reference_names,
                "method": method,
                "dataset": config["dataset"],
                "seed": int(config["seed"]),
                "requested_K": int(config["requested_K"]),
                "realized_K": int(np.unique(assignments_array).size),
                "metacell_id": assignments_array,
            }
        )
        if labels is not None:
            assignments["cell_type"] = labels
        assignments.to_csv(output_dir / f"cell_assignments_{method}.csv", index=False)
        tables = evaluate_assignments(
            assignments,
            requested_k=int(config["requested_K"]),
            dataset=config["dataset"],
            method=method,
            seed=int(config["seed"]),
            implementation_tag=config["implementation_tag"],
            resolution=int(config["requested_K"]) / len(assignments),
        )
        summary = tables["metacell_size_summary"].copy()
        summary["representation"] = "fixed_equal_weight_PCA_LSI_CLR"
        summary_tables.append(summary)
        if "per_type_metrics_long" in tables:
            per_type = tables["per_type_metrics_long"].copy()
            per_type["representation"] = "fixed_equal_weight_PCA_LSI_CLR"
            per_type_tables.append(per_type)
    pd.concat(summary_tables, ignore_index=True).to_csv(output_dir / "fixed_representation_benchmark.csv", index=False)
    if per_type_tables:
        pd.concat(per_type_tables, ignore_index=True).to_csv(output_dir / "fixed_representation_per_type.csv", index=False)
    pd.DataFrame(stage_rows).to_csv(output_dir / "stage_profile.csv", index=False)

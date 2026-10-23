"""Evaluate every construction in the same clean, modality-specific spaces."""
import json
from release_paths import load_json as _portable_load_json
from pathlib import Path
import numpy as np
import pandas as pd

from r2c1_core import (MODALITIES, align_order, dump_json, ids_hash, neighborhood_metrics,
                       sha_file, type_metrics, within_anchor_cosine)


def load_json(path):
    return _portable_load_json(path)


def mean_finite(values):
    values = np.asarray(values)
    valid = np.isfinite(values)
    return float(np.mean(values[valid])) if valid.any() else np.nan


def exact_neighbors(z, k):
    from sklearn import config_context
    from sklearn.neighbors import NearestNeighbors
    if len(z) <= k:
        raise ValueError("Too few cells for the prespecified evaluation k")
    with config_context(working_memory=512):
        raw = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute", n_jobs=8).fit(z).kneighbors(z, return_distance=False)
    # Explicit exclusion handles tied/duplicated embeddings without assuming self is first.
    return np.asarray([row[row != i][:k] for i, row in enumerate(raw)], dtype=np.int32)


def make_reference(config, name, root):
    import anndata as ad
    import scanpy as sc
    from scipy import sparse
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.preprocessing import StandardScaler
    prepared = root / "prepared" / name
    target = root / "references" / name
    target.mkdir(parents=True, exist_ok=True)
    mods = [m for m in MODALITIES if m in config["datasets"][name]]
    settings = {"prepared_manifest_sha256": sha_file(prepared / "manifest.json"),
                "evaluation_k": config["evaluation_k"], "evaluation_code_sha256": sha_file(__file__),
                "random_state": 0, "RNA": "total=1e4; log1p; top3000 Seurat HVG; zscore clip10; centered PCA50",
                "ATAC": "TF=row fraction; IDF=log1p(n_cells/nonzero_cells); log1p(1e4*TF*IDF); SVD51, discard component1",
                "ADT": "cell CLR log1p(x/exp(mean(log1p(x)))); feature zscore; PCA20",
                "neighbors": "exact cosine kNN in clean fixed representations, exclude self",
                "labels": "fixed supplied annotations, not inputs to representation selection or training"}
    if (target / "settings.json").exists():
        if load_json(target / "settings.json") != settings:
            raise RuntimeError("Reference settings changed; create a new result directory")
    else:
        dump_json(target / "settings.json", settings)
    references = {}
    cells = pd.read_csv(prepared / "cells.csv", dtype=str)
    for mod in mods:
        cache = target / f"{mod}.npz"
        if cache.is_file():
            with np.load(cache, allow_pickle=False) as saved:
                if str(saved["cell_ids_sha256"]) != ids_hash(cells.cell_id):
                    raise RuntimeError("Cached reference cell IDs differ")
                references[mod] = (saved["embedding"], saved["neighbors"])
            continue
        print(f"EVALUATION {name}: build clean {mod} reference", flush=True)
        a = ad.read_h5ad(prepared / f"{mod}_clean.h5ad")
        if not np.array_equal(a.obs_names.to_numpy(), cells.cell_id.to_numpy()):
            raise RuntimeError("Reference input order mismatch")
        x = sparse.csr_matrix(a.X, dtype=np.float64)
        if mod == "RNA":
            sc.pp.normalize_total(a, target_sum=1e4)
            sc.pp.log1p(a)
            sc.pp.highly_variable_genes(a, n_top_genes=min(3000, a.n_vars), flavor="seurat")
            b = a[:, a.var.highly_variable].copy()
            sc.pp.scale(b, max_value=10)
            values = b.X.toarray() if sparse.issparse(b.X) else b.X
            z = PCA(n_components=min(50, min(values.shape) - 1), svd_solver="randomized", random_state=0).fit_transform(values)
        elif mod == "ATAC":
            nonzero = np.asarray((x > 0).sum(axis=0)).ravel()
            x = x[:, nonzero > 0]
            tf = sparse.diags(1 / np.asarray(x.sum(axis=1)).ravel()) @ x
            idf = np.log1p(x.shape[0] / nonzero[nonzero > 0])
            weighted = tf.multiply(idf).tocsr()
            weighted.data = np.log1p(weighted.data * 1e4)
            z = TruncatedSVD(n_components=min(51, min(weighted.shape) - 1), random_state=0).fit_transform(weighted)[:, 1:]
        else:
            values = x.toarray()
            denominator = np.exp(np.log1p(values).mean(axis=1))
            clr = np.log1p(values / denominator[:, None])
            scaled = StandardScaler().fit_transform(clr)
            z = PCA(n_components=min(20, min(scaled.shape) - 1), svd_solver="full").fit_transform(scaled)
        z = np.asarray(z, dtype=np.float32)
        if z.shape[1] < 1 or not np.isfinite(z).all():
            raise RuntimeError("Invalid evaluation representation")
        neighbors = exact_neighbors(z, config["evaluation_k"])
        np.savez_compressed(cache, embedding=z, neighbors=neighbors, cell_ids_sha256=ids_hash(cells.cell_id))
        references[mod] = (z, neighbors)
        del a, x
    rna, atac = references["RNA"][1], references["ATAC"][1]
    intersections = np.asarray([len(set(x) & set(y)) for x, y in zip(rna, atac)])
    jaccard = intersections / (2 * config["evaluation_k"] - intersections)
    cutoff = float(np.quantile(jaccard, .25))
    low = jaccard <= cutoff  # Keep all boundary ties; report the actual number, which can exceed 25%.
    context = cells.copy()
    context["RNA_ATAC_neighbor_jaccard"] = jaccard
    context["low_concordance_including_cutoff_ties"] = low
    context.to_csv(target / "cell_reference_context.csv.gz", index=False)
    context.groupby("celltype").agg(n_cells=("cell_id", "size"),
        mean_RNA_ATAC_neighbor_jaccard=("RNA_ATAC_neighbor_jaccard", "mean"),
        low_concordance_fraction=("low_concordance_including_cutoff_ties", "mean")).to_csv(target / "type_reference_context.csv")
    dump_json(target / "discordance_definition.json", {"jaccard_cutoff": cutoff,
               "cells_selected": int(low.sum()), "fraction_selected": float(low.mean()),
               "definition": "Lower quartile of clean RNA/ATAC neighbor Jaccard with all cutoff ties retained"})
    return cells, references, context


def evaluate(config, names, root):
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    root = Path(root)
    requested_names = list(names)
    # Running D18 first and D17 later must not erase the earlier dataset's summary.
    completed_names = [name for name in config["datasets"]
                       if any((root / "runs").glob(f"{name}_*/selected.json"))]
    names = list(dict.fromkeys(requested_names + completed_names))
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    view_rows, type_tables, resolution_rows, agreements, perturbations, seed_rows, norm_tables = [], [], [], [], [], [], []
    for name in names:
        selected_files = sorted((root / "runs").glob(f"{name}_*/selected.json"))
        if not selected_files:
            continue
        cells, references, context = make_reference(config, name, root)
        all_assignments = []
        for selection_path in selected_files:
            selection = load_json(selection_path)
            attempt = Path(selection["attempt_dir"])
            job = load_json(attempt / "job.json")
            result = load_json(attempt / "result.json")
            a = pd.read_csv(attempt / "assignments.csv.gz", dtype={"cell_id": str, "celltype": str})
            order = align_order(cells.cell_id, a.cell_id)
            a = a.iloc[order].reset_index(drop=True)
            if not np.array_equal(a.celltype.to_numpy(), cells.celltype.to_numpy()):
                raise RuntimeError("Assignment annotation mismatch")
            ids = a.metacell.to_numpy()
            meta = {"dataset": name, "construction": "+".join(job["modalities"]), "condition": job["condition"],
                    "seed": job["seed"], "run": job["name"], "requested_k": job["requested_k"],
                    "realized_k": result["realized_k"], "within_target_tolerance": selection["within_target_tolerance"]}
            resolution_rows.append({**meta, **result, "attempt_dir": str(attempt)})
            per_type, anchors = type_metrics(ids, cells.celltype, config["rare_frequency_threshold"],
                                             config["retention_purity_threshold"])
            cell_metrics = context.copy()
            cell_metrics["anchor"] = ids
            metrics_by_view = {}
            for mod, (z, neighbors) in references.items():
                retention, null, adjusted = neighborhood_metrics(ids, neighbors)
                homogeneity = within_anchor_cosine(z, ids)
                metrics_by_view[mod] = {"retention": float(retention.mean()),
                                       "adjusted": mean_finite(adjusted), "cosine": mean_finite(homogeneity)}
                cell_metrics[mod + "_neighbor_retention"] = retention
                cell_metrics[mod + "_chance_retention"] = null
                cell_metrics[mod + "_adjusted_retention"] = adjusted
                cell_metrics[mod + "_within_anchor_cosine"] = homogeneity
                for population, mask in [("all_cells", np.ones(len(ids), dtype=bool)),
                         ("low_RNA_ATAC_concordance", context.low_concordance_including_cutoff_ties.to_numpy()),
                         ("rare_annotations_le_1pct", cells.celltype.map(cells.celltype.value_counts(normalize=True)).to_numpy() <= config["rare_frequency_threshold"])]:
                    if not mask.any():
                        continue
                    view_rows.append({**meta, "evaluation_modality": mod, "population": population,
                                      "n_evaluated": int(mask.sum()), "neighbor_retention": float(retention[mask].mean()),
                                      "chance_retention": float(null[mask].mean()),
                                      "size_adjusted_retention": mean_finite(adjusted[mask]),
                                      "within_anchor_cosine": mean_finite(homogeneity[mask]),
                                      "n_homogeneity_defined": int(np.isfinite(homogeneity[mask]).sum())})
                by_type = pd.DataFrame({"celltype": cells.celltype, mod + "_neighbor_retention": retention,
                                        mod + "_within_anchor_cosine": homogeneity}).groupby("celltype").mean()
                per_type = per_type.merge(by_type, on="celltype", how="left")
                by_anchor = pd.DataFrame({"anchor": ids, mod + "_within_anchor_cosine": homogeneity,
                                         mod + "_neighbor_retention": retention}).groupby("anchor").mean()
                anchors = anchors.merge(by_anchor, on="anchor", how="left")
            for key, value in meta.items():
                per_type[key] = value
            type_tables.append(per_type)
            cell_metrics.to_csv(attempt / "cell_metrics.csv.gz", index=False)
            anchors.to_csv(attempt / "anchor_metrics.csv.gz", index=False)
            per_type.to_csv(attempt / "type_metrics.csv", index=False)
            block = pd.read_csv(attempt / "embedding_block_norms.csv.gz")
            for mod in job["modalities"]:
                norm_tables.append({**meta, "encoding_modality": mod,
                                    "mean_embedding_norm": float(block[mod + "_norm"].mean()),
                                    "mean_squared_norm_share": float(block[mod + "_squared_norm_share"].mean())})
            all_assignments.append((job, meta, ids, metrics_by_view))
        full = "+".join([m for m in MODALITIES if m in config["datasets"][name]])
        for i, (job_a, meta_a, ids_a, views_a) in enumerate(all_assignments):
            for job_b, meta_b, ids_b, views_b in all_assignments[i + 1:]:
                if job_a["seed"] == job_b["seed"]:
                    agreements.append({"dataset": name, "seed": job_a["seed"], "run_a": job_a["name"], "run_b": job_b["name"],
                                       "ARI": adjusted_rand_score(ids_a, ids_b),
                                       "NMI": normalized_mutual_info_score(ids_a, ids_b)})
                if meta_a["construction"] == meta_b["construction"] == full and job_a["condition"] == job_b["condition"] == "clean":
                    seed_rows.append({"dataset": name, "seed_a": job_a["seed"], "seed_b": job_b["seed"],
                                      "ARI": adjusted_rand_score(ids_a, ids_b),
                                      "requested_k_a": job_a["requested_k"], "requested_k_b": job_b["requested_k"]})
            if job_a["condition"] != "clean":
                baseline = next((x for x in all_assignments if x[0]["seed"] == job_a["seed"]
                                 and x[0]["condition"] == "clean" and x[1]["construction"] == full), None)
                if baseline is not None:
                    for mod in references:
                        perturbations.append({"dataset": name, "seed": job_a["seed"], "perturbed_modality": "ATAC",
                                              "evaluation_modality": mod, "unaltered_input_modality": mod != "ATAC",
                                              "assignment_ARI_vs_clean": adjusted_rand_score(baseline[2], ids_a),
                                              "neighbor_retention_clean": baseline[3][mod]["retention"],
                                              "neighbor_retention_thinned": views_a[mod]["retention"],
                                              "retention_delta": views_a[mod]["retention"] - baseline[3][mod]["retention"],
                                              "size_adjusted_retention_delta": views_a[mod]["adjusted"] - baseline[3][mod]["adjusted"],
                                              "within_anchor_cosine_delta": views_a[mod]["cosine"] - baseline[3][mod]["cosine"],
                                              "requested_k_fixed": job_a["requested_k"] == baseline[0]["requested_k"],
                                              "realized_k_clean": baseline[1]["realized_k"], "realized_k_thinned": meta_a["realized_k"]})
    tables = {"modality_metrics.csv": pd.DataFrame(view_rows),
              "celltype_metrics.csv": pd.concat(type_tables, ignore_index=True) if type_tables else pd.DataFrame(),
              "resolution.csv": pd.DataFrame(resolution_rows),
              "assignment_agreement.csv": pd.DataFrame(agreements),
              "ATAC_thinning_stability.csv": pd.DataFrame(perturbations),
              "clean_seed_stability.csv": pd.DataFrame(seed_rows),
              "embedding_block_norms.csv": pd.DataFrame(norm_tables)}
    for filename, table in tables.items():
        # Save structured environment information separately in each run's result.json.
        table.drop(columns=["versions"], errors="ignore").to_csv(summary_dir / filename, index=False)
    if view_rows:
        df = tables["modality_metrics.csv"]
        df.groupby(["dataset", "construction", "condition", "evaluation_modality", "population"]).agg(
            seeds=("seed", "nunique"), retention_mean=("neighbor_retention", "mean"),
            retention_sd=("neighbor_retention", "std"), cosine_mean=("within_anchor_cosine", "mean"),
            cosine_sd=("within_anchor_cosine", "std"), all_k_matched=("within_target_tolerance", "all")).to_csv(summary_dir / "mean_sd_across_seeds.csv")
        plot_results(tables, summary_dir)
    dump_json(summary_dir / "evaluation_manifest.json", {"datasets_requested": requested_names, "datasets_evaluated": names,
              "runs_evaluated": len(resolution_rows), "no_experiment_results_fabricated": True,
              "notes": ["The clean references are shared across every training condition.",
                        "Annotation recovery is descriptive and does not establish an independent regulatory state.",
                        "Block norm shares are scale diagnostics, not causal attribution or quality gates.",
                        "Check realized K and unmatched flags before claiming matched compression.",
                        "ATAC thinning is one specific degradation, not an exhaustive robustness test."]})


def plot_results(tables, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = tables["modality_metrics.csv"]
    for name in df.dataset.unique():
        subset = df[(df.dataset == name) & (df.condition == "clean") & (df.population == "all_cells")]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
        for ax, metric, title in zip(axes, ["neighbor_retention", "within_anchor_cosine"],
                                    ["Same-anchor fraction of 15 neighbors", "Within-anchor cosine similarity"]):
            table = subset.pivot_table(index="construction", columns="evaluation_modality", values=metric, aggfunc="mean")
            table = table.reindex(columns=[m for m in MODALITIES if m in table.columns])
            mesh = ax.imshow(table.to_numpy(), cmap="cividis", aspect="auto")
            ax.set_xticks(range(len(table.columns)), table.columns)
            ax.set_yticks(range(len(table.index)), table.index, fontsize=8)
            ax.set_title(title, fontsize=10)
            for i in range(len(table.index)):
                for j in range(len(table.columns)):
                    ax.text(j, i, f"{table.iloc[i,j]:.3f}", ha="center", va="center", fontsize=8,
                            color="white" if mesh.norm(table.iloc[i,j]) < .5 else "black")
            fig.colorbar(mesh, ax=ax, shrink=.7)
        fig.suptitle(name + " | Mean over completed seeds; check actual K in resolution.csv", fontsize=10)
        fig.savefig(out / f"{name}_modality_structure.png", dpi=220)
        fig.savefig(out / f"{name}_modality_structure.svg")
        plt.close(fig)

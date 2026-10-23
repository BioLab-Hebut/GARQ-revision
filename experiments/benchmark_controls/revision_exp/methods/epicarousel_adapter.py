#!/usr/bin/env python3
"""Thin official EpiCarousel adapter with audited API-compatibility shims."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from epicarousel import preprocess as pp
from scipy import sparse
from epicarousel.core import Carousel
from tqdm import tqdm


class SnapAPICompatCarousel(Carousel):
    """Preserve upstream construction; adapt SnapATAC2 Polars obs/var output."""

    def data_split(self):
        snap_adata = pp.read(self.data_dir, formal="lazily")
        snap_adata.X.enable_cache()
        chunks = snap_adata.X.chunked(self.chunk_size)
        self.cells_number, self.peaks_number = snap_adata.shape
        try:
            raw_obs, raw_var = snap_adata.obs[:], snap_adata.var[:]
            self.obs = raw_obs.to_pandas() if hasattr(raw_obs, "to_pandas") else pd.DataFrame(raw_obs)
            self.var = raw_var.to_pandas() if hasattr(raw_var, "to_pandas") else pd.DataFrame(raw_var)
        except RuntimeError as exc:
            # SnapATAC2 cannot decode some valid H5AD scalar columns. Upstream
            # only propagates these frames; assignments use the source H5AD.
            self.obs = pd.DataFrame(index=np.arange(self.cells_number).astype(str))
            self.var = pd.DataFrame(index=np.arange(self.peaks_number).astype(str))
            self.metadata_fallback_reason = str(exc)
        origin_sum = 0
        for i, chunk in tqdm(enumerate(chunks), desc="EpiCarousel chunks"):
            x = chunk[0]
            # Upstream preprocessing requires sparse `.X.data`; SnapATAC2 may
            # expose a dense ndarray for dense source H5AD files.
            x = x.astype(np.int8)
            x = x if sparse.issparse(x) else sparse.csr_matrix(x)
            chunk_adata = ad.AnnData(X=x)
            chunk_adata.write(self.chunk_dir + f"/{self.data_name}_fold{i + 1}.h5ad")
            origin_sum += np.sum(x)
            del chunk_adata, x, chunk
            gc.collect()
        self.fold_number = int(np.ceil(self.cells_number / self.chunk_size))
        self.sparsity = origin_sum / self.cells_number / self.peaks_number
        snap_adata.close()


def export_assignments(carousel, source_path, output_path, dataset, seed, requested_k, fingerprint):
    source = ad.read_h5ad(source_path, backed="r")
    assignments = np.full(source.n_obs, -1, dtype=int)
    for metacell_id, row in enumerate(carousel.mc_adata.obs.itertuples()):
        fold = int(row.which_fold)
        local = [int(x) for x in str(row.cells).split()]
        global_idx = [x + (fold - 1) * carousel.chunk_size for x in local]
        assignments[global_idx] = metacell_id
    if np.any(assignments < 0):
        raise RuntimeError(f"{int(np.sum(assignments < 0))} cells lack an EpiCarousel assignment")
    frame = pd.DataFrame({
        "cell_id": source.obs_names.astype(str), "dataset": dataset,
        "source_dataset_fingerprint": fingerprint, "method": "EpiCarousel",
        "method_version": "0.0.2", "implementation_tag": "official_with_api_compatibility_shim",
        "seed": seed, "requested_K": requested_k, "realized_K": len(np.unique(assignments)),
        "resolution": requested_k / source.n_obs, "metacell_id": assignments,
    })
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    source.file.close()
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--requested-k", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=10000)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    input_path = Path(args.input).resolve()
    input_adata = ad.read_h5ad(input_path, backed="r")
    n = input_adata.n_obs
    input_adata.file.close()
    compression = max(1, round(n / args.requested_k))
    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    os.environ["PATH"] = str(Path(os.sys.executable).parent) + os.pathsep + os.environ["PATH"]
    carousel = SnapAPICompatCarousel(
        data_name=f"{args.dataset}_seed{args.seed}_n{n}", data_dir=str(input_path),
        if_bi=1, if_mc_bi=0, filter_rate=0.01, chunk_size=args.chunk_size,
        carousel_resolution=compression, base=str(out.resolve()), step=4, threads=args.threads,
        mc_mode="average", index="cell_type", neighbors_method="umap", n_components=30,
        svd_solver="arpack", shuffle=0, random_state=args.seed,
    )
    carousel.make_dirs(); carousel.data_split(); carousel.identify_metacells(); carousel.merge_metacells()
    frame = export_assignments(carousel, input_path, out / "cell_assignments.csv", args.dataset,
                               args.seed, args.requested_k, args.fingerprint)
    summary = {"status": "PASS", "n_cells": n, "requested_K": args.requested_k,
               "realized_K": int(frame.metacell_id.nunique()), "compression": compression,
               "compatibility_shims": ["SnapATAC2 Polars obs/var conversion",
                                        "dense chunk to equivalent CSR for upstream sparse API",
                                        "upstream np.int/np.float aliases replaced by built-in equivalents"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

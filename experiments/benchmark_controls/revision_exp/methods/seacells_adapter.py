"""Thin adapter around the official SEACells 0.3.3 implementation."""

from __future__ import annotations

import argparse
import json
import random
import resource
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import SEACells


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation", type=Path, required=True)
    parser.add_argument("--cell-ids", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--requested-k", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-iter", type=int, default=100)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    random.seed(args.seed)
    matrix = np.load(args.representation, mmap_mode="r")
    ids = pd.read_csv(args.cell_ids)["cell_id"].astype(str).to_numpy()
    if matrix.shape[0] != len(ids):
        raise ValueError("Representation/cell ID length mismatch")
    work = ad.AnnData(np.zeros((len(ids), 1), dtype=np.float32))
    work.obs_names = ids
    # NNDescent (used above its small-data threshold) requires a writable,
    # C-contiguous array; copying a read-only mmap changes no numeric values.
    work.obsm["X_fixed"] = np.array(matrix, dtype=np.float32, order="C", copy=True)
    started = time.perf_counter()
    model = SEACells.core.SEACells(
        work,
        build_kernel_on="X_fixed",
        n_SEACells=args.requested_k,
        use_gpu=False,
        verbose=True,
    )
    model.construct_kernel_matrix()
    model.initialize_archetypes()
    model.fit(max_iter=args.max_iter, min_iter=10)
    hard = model.get_hard_assignments().loc[ids, "SEACell"].astype(str)
    frame = pd.DataFrame({
        "cell_id": ids,
        "dataset": args.dataset,
        "source_dataset_fingerprint": "see_phase2_registry_and_representation_provenance",
        "method": "SEACells",
        "method_version": "0.3.3",
        "implementation_tag": "official_SEACells_fixed_equal_weight_representation",
        "seed": args.seed,
        "requested_K": args.requested_k,
        "realized_K": hard.nunique(),
        "resolution": args.requested_k / len(ids),
        "metacell_id": hard.to_numpy(),
    })
    frame.to_csv(args.output_dir / "cell_assignments.csv", index=False)
    summary = {
        "status": "PASS",
        "dataset": args.dataset,
        "seed": args.seed,
        "requested_K": args.requested_k,
        "realized_K": int(hard.nunique()),
        "cells": len(ids),
        "wall_time_seconds": time.perf_counter() - started,
        "peak_cpu_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "official_version": "0.3.3",
        "representation": "fixed_equal_weight_PCA_LSI_CLR",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

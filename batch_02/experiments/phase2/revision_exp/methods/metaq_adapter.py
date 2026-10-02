"""Thin adapter around official metaq-sc 1.0.6 with disclosed import shim."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import resource
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-files", nargs="+", type=Path, required=True)
    parser.add_argument("--data-types", nargs="+", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--label-key", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--requested-k", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if len(args.data_files) != len(args.data_types):
        raise ValueError("data-files/data-types length mismatch")

    args.data_files = [path.resolve() for path in args.data_files]
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = ad.read_h5ad(args.data_files[0], backed="r")
    source_ids = source.obs_names.astype(str).to_numpy()
    for index, path in enumerate(args.data_files[1:], start=1):
        paired = ad.read_h5ad(path, backed="r")
        paired_ids = paired.obs_names.astype(str).to_numpy()
        if args.dataset == "D18" and index == len(args.data_files) - 1:
            paired_ids = np.asarray([re.sub(r"\.([0-9]+)$", r"-\1", value) for value in paired_ids])
        if not np.array_equal(source_ids, paired_ids):
            raise ValueError(f"Paired row order mismatch: {path}")

    version = importlib.metadata.version("metaq-sc")
    package_dir = Path(importlib.metadata.distribution("metaq-sc").locate_file("MetaQ"))
    sys.path.insert(0, str(package_dir))  # official 1.0.6 uses package-broken absolute imports
    import main as metaq_main

    prior = Path.cwd()
    os.chdir(args.output_dir)
    Path("figures").mkdir(exist_ok=True)
    save_name = f"{args.dataset}_seed{args.seed}"
    started = time.perf_counter()
    try:
        metaq_main.run_metaq(
            data_path=[str(path) for path in args.data_files],
            data_type=args.data_types,
            save_name=save_name,
            metacell_num=args.requested_k,
            # Labels are not training inputs. Skip the official package's broken
            # label plotting helper and evaluate labels externally and uniformly.
            type_key="__phase2_external_evaluator__",
            codebook_init="Random",
            train_epoch=300,
            batch_size=512,
            converge_threshold=10,
            random_seed=args.seed,
            device=args.device,
        )
    finally:
        os.chdir(prior)
    assignment_h5ad = args.output_dir / "save" / f"{save_name}_{args.requested_k}metacell_ids.h5ad"
    result = ad.read_h5ad(assignment_h5ad)
    if result.n_obs != len(source_ids):
        raise ValueError("Official MetaQ output/source length mismatch")
    labels = result.obs["metacell"].astype(str).to_numpy()
    frame = pd.DataFrame({
        "cell_id": source_ids,
        "dataset": args.dataset,
        "source_dataset_fingerprint": "see_phase2_registry",
        "method": "MetaQ",
        "method_version": version,
        "implementation_tag": "official_metaq_sc_1.0.6_import_compatibility_shim",
        "seed": args.seed,
        "requested_K": args.requested_k,
        "realized_K": np.unique(labels).size,
        "resolution": args.requested_k / len(source_ids),
        "metacell_id": labels,
    })
    frame.to_csv(args.output_dir / "cell_assignments.csv", index=False)
    summary = {
        "status": "PASS",
        "dataset": args.dataset,
        "seed": args.seed,
        "requested_K": args.requested_k,
        "realized_K": int(np.unique(labels).size),
        "cells": len(source_ids),
        "wall_time_seconds": time.perf_counter() - started,
        "peak_cpu_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "official_version": version,
        "compatibility_shim": "MetaQ package directory prepended to sys.path because 1.0.6 uses absolute intra-package imports",
        "label_plotting": "disabled with a nonexistent type_key; official plotting raises KeyError after assignment export on corrected D5; labels are evaluated externally",
        "cell_id_recovery": "source obs_names restored by verified invariant input/output row order",
        "paired_id_canonicalization": "D18 ADT .N suffix converted to -N for validation only" if args.dataset == "D18" else None,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

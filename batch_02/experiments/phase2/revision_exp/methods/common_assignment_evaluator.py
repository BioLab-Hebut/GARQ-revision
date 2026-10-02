"""Evaluate a baseline assignment CSV with the Phase 2 common evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from revision_exp.metrics.metacell import evaluate_assignments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--metadata-h5ad", type=Path, required=True)
    parser.add_argument("--label-key", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.assignments)
    required = {"cell_id", "dataset", "method", "seed", "requested_K", "metacell_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Assignment schema missing columns: {sorted(missing)}")
    metadata = ad.read_h5ad(args.metadata_h5ad, backed="r")
    source_ids = metadata.obs_names.astype(str)
    assignment_ids = pd.Index(frame["cell_id"].astype(str))
    if assignment_ids.has_duplicates:
        raise ValueError("Assignment cell IDs must be unique")
    missing_ids = assignment_ids.difference(source_ids)
    if len(missing_ids):
        raise ValueError(f"Assignment contains {len(missing_ids)} IDs absent from the frozen source")
    if args.label_key not in metadata.obs:
        raise KeyError(f"Missing label key {args.label_key!r}")
    labels = pd.Series(metadata.obs[args.label_key].astype(str).to_numpy(), index=source_ids)
    frame["cell_type"] = labels.loc[assignment_ids].to_numpy()
    requested_k = int(frame["requested_K"].iloc[0])
    realized_k = int(frame["metacell_id"].nunique())
    frame["realized_K"] = realized_k
    frame.to_csv(args.assignments, index=False)
    tables = evaluate_assignments(
        frame,
        requested_k=requested_k,
        dataset=str(frame["dataset"].iloc[0]),
        method=str(frame["method"].iloc[0]),
        seed=int(frame["seed"].iloc[0]),
        implementation_tag=str(frame.get("implementation_tag", pd.Series(["official_baseline"])).iloc[0]),
        resolution=requested_k / len(frame),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(args.output_dir / f"{name}.csv", index=False)
    print(json.dumps({"cells": len(frame), "requested_K": requested_k, "realized_K": realized_k}))


if __name__ == "__main__":
    main()

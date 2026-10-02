"""Combine corrected full-data GARQ and matched-K primary baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FOCAL = {
    "D5": {"Treg", "cDC2"},
    "D11": {"Plasma", "gdT"},
    "D17": {"Mast Cells"},
    "D18": {"T.DoubleNegative", "DC.Myeloid", "Platelets"},
}


def _successful_run_dirs(root: Path) -> list[Path]:
    candidates = []
    for path in root.glob("*/[A-Za-z]*/full_seed*_K002*/per_type_metrics_long.csv"):
        candidates.append(path.parent)
    selected: dict[tuple[str, str, int], Path] = {}
    for run in sorted(candidates):
        frame = pd.read_csv(run / "per_type_metrics_long.csv", nrows=1)
        key = (str(frame.dataset.iloc[0]), str(frame.method.iloc[0]), int(frame.seed.iloc[0]))
        selected[key] = run
    return [selected[key] for key in sorted(selected)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase2-root", type=Path, default=Path("revision_results/phase2"))
    args = parser.parse_args()
    out = args.phase2_root / "01_size_resolution"
    runs = _successful_run_dirs(out)
    if not runs:
        raise RuntimeError("No evaluated full-data runs found")
    per_type = pd.concat([pd.read_csv(run / "per_type_metrics_long.csv") for run in runs], ignore_index=True)
    sizes = pd.concat([pd.read_csv(run / "metacell_size_summary.csv") for run in runs], ignore_index=True)
    keys = ["dataset", "method", "seed", "implementation_tag", "resolution"]
    macro = per_type.groupby(keys, as_index=False).agg(
        cell_type_count=("cell_type", "nunique"),
        balanced_accuracy=("cell_recall", "mean"),
        macro_precision=("cell_precision", "mean"),
        macro_recall=("cell_recall", "mean"),
        macro_f1=("cell_f1", "mean"),
        mean_majority_retention=("majority_retention", "mean"),
        mean_high_purity_recovery=("high_purity_recovery", "mean"),
    )
    benchmark = sizes.merge(macro, on=keys, how="left", validate="one_to_one")
    per_type.to_csv(out / "per_type_metrics_long.csv", index=False)
    sizes.to_csv(out / "metacell_size_summary.csv", index=False)
    benchmark.to_csv(out / "full_benchmark_long.csv", index=False)

    focal = per_type[
        per_type.apply(lambda row: row.cell_type in FOCAL.get(row.dataset, set()), axis=1)
    ].copy()
    focal.to_csv(out / "matchedK_focal_rare_long.csv", index=False)
    summary = focal.groupby(["dataset", "cell_type", "method"], as_index=False).agg(
        seeds=("seed", "nunique"), abundance=("abundance", "first"),
        mean_f1=("cell_f1", "mean"), sd_f1=("cell_f1", "std"),
        mean_recall=("cell_recall", "mean"),
        mean_majority_retention=("majority_retention", "mean"),
        mean_high_purity_recovery=("high_purity_recovery", "mean"),
    )
    summary.to_csv(out / "matchedK_focal_rare_summary.csv", index=False)

    contrasts = []
    for (dataset, cell_type), block in focal.groupby(["dataset", "cell_type"]):
        pivot = block.pivot_table(index="seed", columns="method", values="cell_f1", aggfunc="first")
        if "GARQ" not in pivot:
            continue
        for method in sorted(set(pivot.columns) - {"GARQ"}):
            paired = pivot[["GARQ", method]].dropna()
            diff = paired.GARQ - paired[method]
            contrasts.append({
                "dataset": dataset, "cell_type": cell_type, "baseline": method,
                "paired_seeds": len(diff), "mean_f1_difference_GARQ_minus_baseline": diff.mean(),
                "min_difference": diff.min() if len(diff) else np.nan,
                "max_difference": diff.max() if len(diff) else np.nan,
            })
    pd.DataFrame(contrasts).to_csv(out / "matchedK_focal_rare_paired_contrasts.csv", index=False)
    print({"runs": len(runs), "methods": sorted(per_type.method.unique()), "focal_rows": len(focal)})


if __name__ == "__main__":
    main()

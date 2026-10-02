"""Summarize corrected full-data D18 thinning/permutation runs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from revision_exp.metrics.assignment import compare_assignments


def main() -> None:
    root = Path("revision_results/phase2/02_modality/noise_full_retry1/D18")
    baseline = Path("revision_results/phase2/02_modality/full/D18/RNA_ATAC_ADT")
    rows, per_type = [], []
    for run_dir in sorted(root.glob("*/seed*")):
        done_files = list(run_dir.glob("*.done.json"))
        if len(done_files) != 1:
            continue
        condition = run_dir.parent.name
        seed = int(run_dir.name.removeprefix("seed"))
        modality, detail = condition.split("_", 1)
        kind = "cell_permutation" if detail == "cell_permutation" else "binomial_thinning"
        retention = None if kind == "cell_permutation" else float(detail.removeprefix("thin_"))
        size = pd.read_csv(run_dir / "metacell_size_summary.csv").iloc[0]
        metrics = pd.read_csv(run_dir / "per_type_metrics_long.csv")
        assignments = pd.read_csv(run_dir / "cell_assignments.csv")["metacell_id"].to_numpy()
        baseline_ids = pd.read_csv(baseline / f"seed{seed}" / "cell_assignments.csv")["metacell_id"].to_numpy()
        agreement = compare_assignments(baseline_ids, assignments)
        done = json.loads(done_files[0].read_text())
        rows.append({
            "dataset": "D18", "condition": condition, "perturbed_modality": modality,
            "perturbation_kind": kind, "retention_probability": retention, "seed": seed,
            "requested_K": int(size["requested_K"]), "realized_K": int(size["realized_K"]),
            "empty_anchor_count": int(size["empty_anchor_count"]), "size_median": float(size["size_median"]),
            "size_max": float(size["size_max"]), "size_gini": float(size["size_gini"]),
            "macro_f1": float(metrics["cell_f1"].mean()), "macro_precision": float(metrics["cell_precision"].mean()),
            "macro_recall": float(metrics["cell_recall"].mean()), "mean_weighted_purity": float(metrics["purity_weighted"].mean()),
            **agreement, "wall_seconds": float(done["wall_time_seconds"]), "peak_cpu_rss_bytes": int(done["peak_cpu_rss_bytes"]),
            "source_dir": str(run_dir),
        })
        metrics.insert(0, "condition", condition)
        per_type.append(metrics)
    output = root.parent
    pd.DataFrame(rows).sort_values(["perturbation_kind", "perturbed_modality", "retention_probability", "seed"]).to_csv(
        output / "d18_noise_full_summary.csv", index=False
    )
    pd.concat(per_type, ignore_index=True).to_csv(output / "d18_noise_full_per_type.csv", index=False)


if __name__ == "__main__":
    main()

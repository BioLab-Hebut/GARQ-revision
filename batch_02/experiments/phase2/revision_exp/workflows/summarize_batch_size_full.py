"""Summarize D5/D11 full-data training batch-size stability."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from revision_exp.metrics.assignment import compare_assignments


def run_dir(dataset: str, batch: int, seed: int) -> Path:
    if batch == 256:
        return Path(f"revision_results/phase2/01_size_resolution/{dataset}/GARQ/full_seed{seed}_K002")
    return Path(
        f"revision_results/phase2/08_scalability/training_batch_size/{dataset}/"
        f"batch{batch}_seed{seed}_K002"
    )


def main() -> None:
    rows, stability = [], []
    for dataset in ("D5", "D11"):
        for seed in (0, 1, 2):
            base = pd.read_csv(run_dir(dataset, 256, seed) / "cell_assignments.csv")
            for batch in (256, 512, 1024, 2048):
                directory = run_dir(dataset, batch, seed)
                size = pd.read_csv(directory / "metacell_size_summary.csv").iloc[0]
                per_type = pd.read_csv(directory / "per_type_metrics_long.csv")
                done = json.loads(next(directory.glob("*.done.json")).read_text())
                stage = pd.read_csv(directory / "stage_profile.csv")
                rows.append({
                    "dataset": dataset, "seed": seed, "batch_size": batch,
                    "requested_K": int(size.requested_K), "realized_K": int(size.realized_K),
                    "empty_anchor_count": int(size.empty_anchor_count),
                    "size_median": float(size.size_median), "size_max": float(size.size_max),
                    "size_gini": float(size.size_gini), "macro_f1": float(per_type.cell_f1.mean()),
                    "mean_weighted_purity": float(per_type.purity_weighted.mean()),
                    "wall_time_seconds": float(done["wall_time_seconds"]),
                    "peak_cpu_rss_bytes": int(done["peak_cpu_rss_bytes"]),
                    "peak_gpu_allocated_bytes": int(stage.gpu_allocated_peak_bytes.max()),
                    "peak_gpu_reserved_bytes": int(stage.gpu_reserved_peak_bytes.max()),
                    "status": "PASS", "source_dir": str(directory),
                })
                if batch != 256:
                    other = pd.read_csv(directory / "cell_assignments.csv")
                    if not base.cell_id.equals(other.cell_id):
                        raise ValueError(f"cell ID/order mismatch: {dataset} seed{seed} batch{batch}")
                    stability.append({
                        "dataset": dataset, "seed": seed, "reference_batch_size": 256,
                        "batch_size": batch,
                        **compare_assignments(base.metacell_id.to_numpy(), other.metacell_id.to_numpy()),
                    })
    output = Path("revision_results/phase2/08_scalability")
    pd.DataFrame(rows).to_csv(output / "training_batch_size.csv", index=False)
    pd.DataFrame(stability).to_csv(output / "training_batch_size_assignment_stability.csv", index=False)


if __name__ == "__main__":
    main()

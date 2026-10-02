"""Collect E4 smoke traces and label-permutation-invariant comparisons."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from revision_exp.metrics.assignment import compare_assignments


def main() -> None:
    root = Path("revision_results/04_anchor")
    trace_tables = []
    per_type_tables = []
    summary_rows = []
    variants = (
        "legacy_continuous",
        "no_manual_reposition",
        "reposition_interval5",
        "reposition_interval10",
    )
    for dataset in ("D5", "D11", "D17"):
        for seed in range(5):
            suffix = "" if seed == 0 else f"_seed{seed}"
            legacy_dir = root / f"{dataset}_legacy_continuous_smoke{suffix}"
            legacy_ids = pd.read_csv(legacy_dir / "cell_assignments.csv")[
                "metacell_id"
            ].to_numpy()
            for variant in variants:
                run_dir = root / f"{dataset}_{variant}_smoke{suffix}"
                _append_run(
                    dataset,
                    seed,
                    variant,
                    run_dir,
                    legacy_ids,
                    trace_tables,
                    per_type_tables,
                    summary_rows,
                )
    pd.concat(trace_tables, ignore_index=True).to_csv(
        root / "anchor_dynamics_step.csv", index=False
    )
    pd.DataFrame(summary_rows).to_csv(
        root / "anchor_schedule_ablation.csv", index=False
    )
    pd.concat(per_type_tables, ignore_index=True).to_csv(
        root / "anchor_schedule_per_type_metrics.csv", index=False
    )


def _append_run(
    dataset: str,
    seed: int,
    variant: str,
    run_dir: Path,
    legacy_ids,
    trace_tables: list[pd.DataFrame],
    per_type_tables: list[pd.DataFrame],
    summary_rows: list[dict],
) -> None:
    trace = pd.read_csv(run_dir / "anchor_dynamics_step.csv")
    trace.insert(0, "run", run_dir.name)
    trace.insert(0, "seed", seed)
    trace.insert(0, "dataset", dataset)
    trace_tables.append(trace)
    per_type = pd.read_csv(run_dir / "per_type_metrics_long.csv")
    if set(per_type["dataset"]) != {dataset} or set(per_type["seed"]) != {seed}:
        raise ValueError(f"Per-type provenance mismatch in {run_dir}")
    per_type.insert(0, "run", run_dir.name)
    per_type.insert(0, "variant", variant)
    per_type_tables.append(per_type)
    assignments = pd.read_csv(run_dir / "cell_assignments.csv")[
        "metacell_id"
    ].to_numpy()
    comparison = compare_assignments(legacy_ids, assignments)
    size = pd.read_csv(run_dir / "metacell_size_summary.csv").iloc[0]
    summary_rows.append(
        {
            "dataset": dataset,
            "seed": seed,
            "variant": variant,
            "run": run_dir.name,
            "implementation_tag": size["implementation_tag"],
            "requested_K": int(size["requested_K"]),
            "realized_K": int(size["realized_K"]),
            "size_gini": float(size["size_gini"]),
            "size_median": float(size["size_median"]),
            "size_max": float(size["size_max"]),
            "ARI_vs_legacy": comparison["ARI"],
            "NMI_vs_legacy": comparison["NMI"],
            "VI_nats_vs_legacy": comparison["VI_nats"],
            "coassignment_agreement_rand_vs_legacy": comparison[
                "coassignment_agreement_rand"
            ],
            "quantized_steps": len(trace),
            "reposition_steps": int(trace["reposition_due"].sum()),
            "local_branch_steps": int(trace["local_branch_executed"].sum()),
            "manual_displacement_anchor_events": int(
                trace["manual_displacement_anchor_count"].sum()
            ),
            "final_usage_perplexity": float(trace["usage_perplexity"].iloc[-1]),
            "final_usage_gini": float(trace["usage_gini"].iloc[-1]),
            "anchor_nan_count_max": int(trace["anchor_nan_count"].max()),
            "anchor_inf_count_max": int(trace["anchor_inf_count"].max()),
        }
    )


if __name__ == "__main__":
    main()

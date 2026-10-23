"""Summarize E2 count-level modality perturbation diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from revision_exp.metrics.assignment import compare_assignments
from revision_exp.workflows.neighborhood_mapping import cosine_neighbors, neighbor_jaccard


def _blocks(run_dir: Path) -> dict[str, np.ndarray]:
    embeddings = np.load(run_dir / "cell_embeddings.npy")
    layout = json.loads((run_dir / "embedding_layout.json").read_text(encoding="utf-8"))
    return {
        item["data_type"]: embeddings[:, item["start"] : item["end"]]
        for item in layout["blocks"]
    }


def _condition_fields(condition: str) -> tuple[str, str, float | None]:
    if condition == "baseline_counts_p1":
        return ("none", "baseline_count_matrix", 1.0)
    modality, detail = condition.split("_", 1)
    if detail.startswith("thin_"):
        return (modality, "binomial_thinning", float(detail.removeprefix("thin_")))
    if detail == "cell_permutation":
        return (modality, "cell_permutation", None)
    raise ValueError(f"Unknown condition: {condition}")


def main() -> None:
    rows = []
    per_type_rows = []
    noise_root = Path("revision_results/02_modality/noise")
    for root in sorted(path for path in noise_root.iterdir() if path.is_dir()):
        dataset = root.name
        baseline_dir = root / "baseline_counts_p1"
        if not baseline_dir.exists():
            continue
        baseline_assignments = pd.read_csv(baseline_dir / "cell_assignments.csv")
        baseline_ids = baseline_assignments["metacell_id"].to_numpy()
        baseline_blocks = _blocks(baseline_dir)
        baseline_neighbors = {
            name: cosine_neighbors(block, 15)
            for name, block in baseline_blocks.items()
        }
        for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            condition = run_dir.name
            perturbed_modality, perturbation_kind, probability = _condition_fields(condition)
            assignments = pd.read_csv(run_dir / "cell_assignments.csv")
            comparison = compare_assignments(
                baseline_ids, assignments["metacell_id"].to_numpy()
            )
            size = pd.read_csv(run_dir / "metacell_size_summary.csv").iloc[0]
            contribution = pd.read_csv(
                run_dir / "modality_block_contribution.csv"
            ).set_index("data_type")
            blocks = _blocks(run_dir)
            for evaluated_modality, block in blocks.items():
                jaccard = neighbor_jaccard(
                    baseline_neighbors[evaluated_modality],
                    cosine_neighbors(block, 15),
                )
                rows.append(
                    {
                    "dataset": dataset,
                    "seed": 0,
                    "condition": condition,
                    "perturbed_modality": perturbed_modality,
                    "perturbation_kind": perturbation_kind,
                    "retention_probability": probability,
                    "evaluated_modality": evaluated_modality,
                    "evaluated_modality_is_perturbed": evaluated_modality == perturbed_modality,
                    "requested_K": int(size["requested_K"]),
                    "realized_K": int(size["realized_K"]),
                    "size_gini": float(size["size_gini"]),
                    "ARI_vs_p1": comparison["ARI"],
                    "NMI_vs_p1": comparison["NMI"],
                    "VI_nats_vs_p1": comparison["VI_nats"],
                    "coassignment_agreement_rand_vs_p1": comparison["coassignment_agreement_rand"],
                    "learned_block_knn15_jaccard_mean_vs_p1": float(jaccard.mean()),
                    "learned_block_knn15_jaccard_p10_vs_p1": float(np.quantile(jaccard, 0.1)),
                    "learned_block_knn15_jaccard_median_vs_p1": float(np.median(jaccard)),
                    "block_l2_norm_mean": float(contribution.loc[evaluated_modality, "l2_norm_mean"]),
                    "block_variance_trace": float(contribution.loc[evaluated_modality, "variance_trace"]),
                    "relative_abs_dot_contribution": float(contribution.loc[evaluated_modality, "relative_abs_dot_contribution"]),
                    }
                )
            per_type = pd.read_csv(run_dir / "per_type_metrics_long.csv")
            per_type.insert(0, "retention_probability", probability)
            per_type.insert(0, "perturbation_kind", perturbation_kind)
            per_type.insert(0, "perturbed_modality", perturbed_modality)
            per_type.insert(0, "condition", condition)
            per_type_rows.append(per_type)
    output_root = Path("revision_results/02_modality")
    pd.DataFrame(rows).to_csv(output_root / "modality_noise_perturbation.csv", index=False)
    pd.concat(per_type_rows, ignore_index=True).to_csv(output_root / "modality_noise_per_type.csv", index=False)


if __name__ == "__main__":
    main()

"""Summarize E2 explicit modality-weight diagnostics."""

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


def main() -> None:
    rows = []
    per_type_rows = []
    root = Path("revision_results/02_modality/weights")
    for dataset_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        baseline_dir = dataset_dir / "equal_weights"
        baseline = pd.read_csv(baseline_dir / "cell_assignments.csv")
        baseline_ids = baseline["metacell_id"].to_numpy()
        baseline_blocks = _blocks(baseline_dir)
        baseline_neighbors = {
            name: cosine_neighbors(block, 15)
            for name, block in baseline_blocks.items()
        }
        for run_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
            done = json.loads(next(run_dir.glob("*.done.json")).read_text())
            config = json.loads(Path(done["resolved_config"]).read_text())
            assignments = pd.read_csv(run_dir / "cell_assignments.csv")
            comparison = compare_assignments(
                baseline_ids, assignments["metacell_id"].to_numpy()
            )
            size = pd.read_csv(run_dir / "metacell_size_summary.csv").iloc[0]
            contribution = pd.read_csv(
                run_dir / "modality_block_contribution.csv"
            ).set_index("data_type")
            for evaluated_modality, block in _blocks(run_dir).items():
                jaccard = neighbor_jaccard(
                    baseline_neighbors[evaluated_modality],
                    cosine_neighbors(block, 15),
                )
                rows.append(
                    {
                        "dataset": dataset_dir.name,
                        "seed": 0,
                        "condition": run_dir.name,
                        "weighted_modality": config.get("weighted_modality"),
                        "weight_lambda": config["weight_lambda"],
                        "evaluated_modality": evaluated_modality,
                        "requested_K": int(size["requested_K"]),
                        "realized_K": int(size["realized_K"]),
                        "ARI_vs_equal_weights": comparison["ARI"],
                        "NMI_vs_equal_weights": comparison["NMI"],
                        "VI_nats_vs_equal_weights": comparison["VI_nats"],
                        "knn15_jaccard_mean_vs_equal_weights": float(jaccard.mean()),
                        "relative_abs_dot_contribution": float(
                            contribution.loc[
                                evaluated_modality, "relative_abs_dot_contribution"
                            ]
                        ),
                    }
                )
            per_type = pd.read_csv(run_dir / "per_type_metrics_long.csv")
            per_type.insert(0, "weight_lambda", config["weight_lambda"])
            per_type.insert(0, "weighted_modality", config.get("weighted_modality"))
            per_type.insert(0, "condition", run_dir.name)
            per_type_rows.append(per_type)
    output = Path("revision_results/02_modality")
    pd.DataFrame(rows).to_csv(output / "modality_weight_grid.csv", index=False)
    pd.concat(per_type_rows, ignore_index=True).to_csv(
        output / "modality_weight_per_type.csv", index=False
    )


if __name__ == "__main__":
    main()

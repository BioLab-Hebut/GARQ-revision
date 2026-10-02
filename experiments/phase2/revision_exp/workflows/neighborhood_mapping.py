"""E2 learned-modality neighborhood to shared-anchor diagnostics."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def neighbor_jaccard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape:
        raise ValueError("Neighbor arrays must have identical shape")
    return np.asarray([len(set(x) & set(y)) / len(set(x) | set(y)) for x, y in zip(a, b)])


def cosine_neighbors(block: np.ndarray, k: int) -> np.ndarray:
    model = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
    indices = model.fit(block).kneighbors(return_distance=False)
    return indices[:, 1:]


def anchor_compactness(block: np.ndarray, assignments: np.ndarray) -> pd.DataFrame:
    normalized = block / np.maximum(np.linalg.norm(block, axis=1, keepdims=True), 1e-12)
    rows = []
    for anchor in np.unique(assignments):
        mask = assignments == anchor
        center = normalized[mask].mean(axis=0)
        center /= max(np.linalg.norm(center), 1e-12)
        distances = 1.0 - normalized[mask] @ center
        rows.append({"metacell_id": int(anchor), "size": int(mask.sum()), "compactness_cosine_mean": float(distances.mean())})
    return pd.DataFrame(rows)


def analyze_run(run_dir: Path) -> tuple[list[dict], list[pd.DataFrame]]:
    layout = json.loads((run_dir / "embedding_layout.json").read_text())
    embeddings = np.load(run_dir / "cell_embeddings.npy")
    assignments_frame = pd.read_csv(run_dir / "cell_assignments.csv")
    assignments = assignments_frame["metacell_id"].to_numpy()
    dataset = str(assignments_frame["dataset"].iloc[0])
    blocks = {
        item["data_type"]: embeddings[:, item["start"] : item["end"]]
        for item in layout["blocks"]
    }
    rows = []
    compactness_frames = []
    for data_type, block in blocks.items():
        comp = anchor_compactness(block, assignments)
        comp.insert(0, "data_type", data_type)
        comp.insert(0, "dataset", dataset)
        compactness_frames.append(comp)
    combined_comp = pd.concat(compactness_frames, ignore_index=True)
    combined_comp["compactness_percentile"] = combined_comp.groupby("data_type")["compactness_cosine_mean"].rank(pct=True)
    pivot = combined_comp.pivot(index="metacell_id", columns="data_type", values="compactness_percentile")
    discordant = (pivot.max(axis=1) >= 0.75) & (pivot.min(axis=1) <= 0.25)
    discordant_ids = set(pivot.index[discordant])
    combined_comp["modality_conflict_anchor"] = combined_comp["metacell_id"].isin(discordant_ids)

    for k in (5, 15, 30):
        neighbors = {name: cosine_neighbors(block, k) for name, block in blocks.items()}
        for data_type, indices in neighbors.items():
            within = assignments[indices] == assignments[:, None]
            rows.append(
                {
                    "dataset": dataset,
                    "metric_scope": "modality_to_shared_anchor",
                    "data_type_a": data_type,
                    "data_type_b": "",
                    "k": k,
                    "within_anchor_edge_fraction": float(within.mean()),
                    "cell_knn_jaccard_mean": np.nan,
                    "cell_knn_jaccard_p10": np.nan,
                    "cell_knn_jaccard_median": np.nan,
                    "modality_conflict_anchor_count": len(discordant_ids),
                    "modality_conflict_anchor_rate": len(discordant_ids) / len(np.unique(assignments)),
                    "n_cells": len(assignments),
                    "realized_K": len(np.unique(assignments)),
                    "source_dir": str(run_dir),
                }
            )
        for name_a, name_b in itertools.combinations(blocks, 2):
            jaccard = neighbor_jaccard(neighbors[name_a], neighbors[name_b])
            rows.append(
                {
                    "dataset": dataset,
                    "metric_scope": "modality_pair_overlap",
                    "data_type_a": name_a,
                    "data_type_b": name_b,
                    "k": k,
                    "within_anchor_edge_fraction": np.nan,
                    "cell_knn_jaccard_mean": float(jaccard.mean()),
                    "cell_knn_jaccard_p10": float(np.quantile(jaccard, 0.10)),
                    "cell_knn_jaccard_median": float(np.median(jaccard)),
                    "modality_conflict_anchor_count": len(discordant_ids),
                    "modality_conflict_anchor_rate": len(discordant_ids) / len(np.unique(assignments)),
                    "n_cells": len(assignments),
                    "realized_K": len(np.unique(assignments)),
                    "source_dir": str(run_dir),
                }
            )
    return rows, [combined_comp]


def main() -> None:
    root = Path("revision_results/02_modality")
    rows = []
    compactness = []
    for run_dir in sorted((root / "blockdiag").glob("*")):
        if (run_dir / "cell_embeddings.npy").exists():
            run_rows, frames = analyze_run(run_dir)
            rows.extend(run_rows)
            compactness.extend(frames)
    pd.DataFrame(rows).to_csv(root / "neighborhood_anchor_mapping.csv", index=False)
    pd.concat(compactness, ignore_index=True).to_csv(root / "conflict_anchor_metrics.csv", index=False)


if __name__ == "__main__":
    main()

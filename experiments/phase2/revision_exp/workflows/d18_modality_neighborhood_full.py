"""Evaluate D18 full-data modality-block neighborhoods for the frozen grid."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from revision_exp.workflows.neighborhood_mapping import (
    anchor_compactness,
    cosine_neighbors,
    neighbor_jaccard,
)


def evaluate(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    layout = json.loads((run_dir / "embedding_layout.json").read_text())
    embeddings = np.load(run_dir / "cell_embeddings.npy", mmap_mode="r")
    assignments = pd.read_csv(run_dir / "cell_assignments.csv")["metacell_id"].to_numpy()
    blocks = {
        item["data_type"]: np.asarray(embeddings[:, item["start"] : item["end"]])
        for item in layout["blocks"]
    }
    parts = run_dir.parts
    combination, seed = parts[-2], int(parts[-1].removeprefix("seed"))
    compact = []
    for modality, block in blocks.items():
        frame = anchor_compactness(block, assignments)
        frame.insert(0, "modality", modality)
        compact.append(frame)
    compactness = pd.concat(compact, ignore_index=True)
    compactness.insert(0, "seed", seed)
    compactness.insert(0, "combination", combination)
    pivot = compactness.pivot(index="metacell_id", columns="modality", values="compactness_cosine_mean")
    discordant = set(pivot.index[(pivot.rank(pct=True).max(axis=1) >= 0.75) & (pivot.rank(pct=True).min(axis=1) <= 0.25)])
    compactness["modality_conflict_anchor"] = compactness["metacell_id"].isin(discordant)

    max_neighbors = {name: cosine_neighbors(block, 30) for name, block in blocks.items()}
    rows = []
    for k in (5, 15, 30):
        neighbors = {name: values[:, :k] for name, values in max_neighbors.items()}
        for modality, indices in neighbors.items():
            rows.append({
                "combination": combination, "seed": seed, "scope": "modality_to_shared_anchor",
                "modality_a": modality, "modality_b": "", "k": k,
                "within_anchor_edge_fraction": float((assignments[indices] == assignments[:, None]).mean()),
                "knn_jaccard_mean": np.nan, "knn_jaccard_p10": np.nan, "knn_jaccard_median": np.nan,
                "conflict_anchor_count": len(discordant), "conflict_anchor_rate": len(discordant) / len(np.unique(assignments)),
                "n_cells": len(assignments), "realized_K": len(np.unique(assignments)), "source_dir": str(run_dir),
            })
        for a, b in itertools.combinations(neighbors, 2):
            values = neighbor_jaccard(neighbors[a], neighbors[b])
            rows.append({
                "combination": combination, "seed": seed, "scope": "modality_pair_overlap",
                "modality_a": a, "modality_b": b, "k": k, "within_anchor_edge_fraction": np.nan,
                "knn_jaccard_mean": float(values.mean()), "knn_jaccard_p10": float(np.quantile(values, .1)),
                "knn_jaccard_median": float(np.median(values)), "conflict_anchor_count": len(discordant),
                "conflict_anchor_rate": len(discordant) / len(np.unique(assignments)), "n_cells": len(assignments),
                "realized_K": len(np.unique(assignments)), "source_dir": str(run_dir),
            })
    return pd.DataFrame(rows), compactness


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("revision_results/phase2/02_modality/full/D18"))
    parser.add_argument("--output", type=Path, default=Path("revision_results/phase2/02_modality/d18_modality_neighborhood_full.csv"))
    args = parser.parse_args()
    metric_frames, compactness_frames = [], []
    for run_dir in sorted(args.root.glob("*/seed*")):
        if (run_dir / "cell_embeddings.npy").exists():
            metrics, compactness = evaluate(run_dir)
            metric_frames.append(metrics)
            compactness_frames.append(compactness)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(metric_frames, ignore_index=True).to_csv(args.output, index=False)
    pd.concat(compactness_frames, ignore_index=True).to_csv(args.output.with_name("d18_modality_anchor_compactness_full.csv"), index=False)


if __name__ == "__main__":
    main()

"""Common, method-independent evaluation of cell-to-metacell assignments."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


def gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.any(values < 0):
        return float("nan")
    total = values.sum()
    if total == 0:
        return 0.0
    ordered = np.sort(values)
    index = np.arange(1, values.size + 1)
    return float(np.sum((2 * index - values.size - 1) * ordered) / (values.size * total))


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else float("nan")


def size_tables(
    assignments: pd.DataFrame,
    requested_k: int,
    metadata: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sizes = assignments.groupby("metacell_id", observed=True).size().rename("size").reset_index()
    realized_k = int(sizes.shape[0])
    n_cells = int(assignments.shape[0])
    sizes = sizes.assign(**metadata, requested_K=requested_k, realized_K=realized_k, n_cells=n_cells)
    array = sizes["size"].to_numpy(dtype=float)
    q1, median, q3 = np.percentile(array, [25, 50, 75])
    mean = float(array.mean())
    sd = float(array.std(ddof=1)) if array.size > 1 else 0.0
    summary = {
        **metadata,
        "n_cells": n_cells,
        "requested_K": requested_k,
        "realized_K": realized_k,
        "empty_anchor_count": max(0, requested_k - realized_k),
        "empty_rate": max(0, requested_k - realized_k) / requested_k if requested_k else float("nan"),
        "compression_ratio": n_cells / realized_k,
        "size_min": float(array.min()),
        "size_q1": float(q1),
        "size_median": float(median),
        "size_mean": mean,
        "size_q3": float(q3),
        "size_p90": _percentile(array, 90),
        "size_p95": _percentile(array, 95),
        "size_p99": _percentile(array, 99),
        "size_max": float(array.max()),
        "size_sd": sd,
        "size_cv": sd / mean if mean else float("nan"),
        "size_gini": gini(array),
        "singleton_rate": float(np.mean(array == 1)),
        "size_le_2_rate": float(np.mean(array <= 2)),
        "size_gt_2x_median_rate": float(np.mean(array > 2 * median)),
        "size_gt_5x_median_rate": float(np.mean(array > 5 * median)),
        "tukey_outlier_count": int(np.sum(array > q3 + 1.5 * (q3 - q1))),
        "max_median_ratio": float(array.max() / median) if median else float("nan"),
    }
    return sizes, pd.DataFrame([summary])


def per_type_table(assignments: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    required = {"cell_id", "metacell_id", "cell_type"}
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    table = assignments.copy()
    table["cell_type"] = table["cell_type"].astype(str)
    composition = (
        table.groupby(["metacell_id", "cell_type"], observed=True).size().rename("n_type").reset_index()
    )
    totals = table.groupby("metacell_id", observed=True).size().rename("metacell_size")
    composition = composition.join(totals, on="metacell_id")
    composition["purity_for_type"] = composition["n_type"] / composition["metacell_size"]
    dominant_idx = composition.groupby("metacell_id", observed=True)["n_type"].idxmax()
    dominant = composition.loc[dominant_idx, ["metacell_id", "cell_type", "purity_for_type"]].rename(
        columns={"cell_type": "majority_prediction", "purity_for_type": "majority_purity"}
    )
    table = table.merge(dominant, on="metacell_id", how="left")
    labels = sorted(table["cell_type"].unique())
    precision, recall, f1, support = precision_recall_fscore_support(
        table["cell_type"], table["majority_prediction"], labels=labels, zero_division=0
    )
    rows = []
    for label, p, r, f, n in zip(labels, precision, recall, f1, support):
        comp = composition[composition["cell_type"] == label]
        dominant_for_type = dominant[dominant["majority_prediction"] == label]
        involved_ids = comp["metacell_id"]
        involved_sizes = totals.reindex(involved_ids).to_numpy(dtype=float)
        abundance = n / len(table)
        cell_rows = table[table["cell_type"] == label]
        strict_recovery = float(np.mean(cell_rows["majority_purity"] == 1.0))
        rows.append(
            {
                **metadata,
                "cell_type": label,
                "abundance": abundance,
                "abundance_class": "Rare" if abundance < 0.01 else ("Intermediate" if abundance <= 0.10 else "Common"),
                "support": int(n),
                "cell_precision": float(p),
                "cell_recall": float(r),
                "cell_f1": float(f),
                "dominant_metacell_count": int(len(dominant_for_type)),
                "purity_unweighted": float(dominant_for_type["majority_purity"].mean()) if len(dominant_for_type) else float("nan"),
                "purity_weighted": float(
                    np.average(
                        dominant_for_type["majority_purity"],
                        weights=totals.reindex(dominant_for_type["metacell_id"]).to_numpy(),
                    )
                ) if len(dominant_for_type) else float("nan"),
                "involved_metacell_size_median": _percentile(involved_sizes, 50),
                "involved_metacell_size_p95": _percentile(involved_sizes, 95),
                "involved_metacell_size_max": float(involved_sizes.max()) if involved_sizes.size else float("nan"),
                "strict_recovery": strict_recovery,
                "majority_recovery": float(r),
                "majority_retention_metacell_count": int(np.sum(comp["purity_for_type"] > 0.5)),
                "majority_retention": float(np.mean(comp["purity_for_type"] > 0.5)) if len(comp) else float("nan"),
                "high_purity_recovery_metacell_count": int(np.sum(comp["purity_for_type"] >= 0.7)),
                "high_purity_recovery": float(np.mean(comp["purity_for_type"] >= 0.7)) if len(comp) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def evaluate_assignments(
    assignments: pd.DataFrame,
    *,
    requested_k: int,
    dataset: str,
    method: str,
    seed: int,
    implementation_tag: str,
    resolution: float | None = None,
) -> dict[str, pd.DataFrame]:
    required = {"cell_id", "metacell_id"}
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if assignments["cell_id"].duplicated().any():
        raise ValueError("cell_id must be unique")
    metadata = {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "implementation_tag": implementation_tag,
        "resolution": resolution,
    }
    sizes, summary = size_tables(assignments, requested_k, metadata)
    output = {"metacell_size_long": sizes, "metacell_size_summary": summary}
    if "cell_type" in assignments:
        output["per_type_metrics_long"] = per_type_table(assignments, metadata)
    return output

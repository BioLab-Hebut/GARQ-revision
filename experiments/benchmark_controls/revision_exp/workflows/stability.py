"""E8 same-checkpoint inference batch-size and cell-order diagnostics."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from revision_exp.metrics.assignment import compare_assignments
from revision_exp.metrics.metacell import per_type_table


@torch.no_grad()
def _assign(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Run only the released encoder + batch-local quantizer assignment path."""
    model.eval()
    chunks: list[np.ndarray] = []
    for data in loader:
        x_list = [item.to(device, non_blocking=True) for item in data["x"]]
        hiddens = model(x_list)
        ids, _, _ = model.quantize(hiddens, return_assignment=True)
        chunks.append(ids.detach().cpu().numpy())
    return np.concatenate(chunks)


def run_inference_stability(
    *,
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    reference_ids: np.ndarray,
    cell_ids: np.ndarray,
    cell_types: np.ndarray | None,
    config: dict[str, Any],
    output_dir: Path,
) -> None:
    """Compare partitions after restoring every permuted run to canonical cell order."""
    settings = config.get("inference_stability")
    if not settings:
        return
    n_cells = len(dataset)
    reference_ids = np.asarray(reference_ids)
    cell_ids = np.asarray(cell_ids, dtype=str)
    if reference_ids.shape != (n_cells,) or cell_ids.shape != (n_cells,):
        raise ValueError("Reference assignments and cell IDs must match dataset length")
    batch_sizes = [int(value) for value in settings["batch_sizes"]]
    order_seeds = [int(value) for value in settings["order_seeds"]]
    num_workers = int(settings.get("num_workers", 0))
    rows: list[dict[str, Any]] = []
    per_type_rows: list[pd.DataFrame] = []
    canonical = np.arange(n_cells)
    for batch_size in batch_sizes:
        for order_seed in order_seeds:
            order = canonical if order_seed < 0 else np.random.default_rng(order_seed).permutation(n_cells)
            loader = DataLoader(
                Subset(dataset, order.tolist()),
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers,
                pin_memory=True,
            )
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            ordered_ids = _assign(model, loader, device)
            wall_time = time.perf_counter() - started
            restored_ids = np.empty_like(ordered_ids)
            restored_ids[order] = ordered_ids
            comparison = compare_assignments(reference_ids, restored_ids)
            metadata = {
                "dataset": config["dataset"],
                "seed": int(config["seed"]),
                "implementation_tag": "instrumented_legacy",
                "reference": "legacy_eval_loader_assignment",
                "inference_batch_size": batch_size,
                "cell_order_seed": order_seed,
                "cell_order": "canonical" if order_seed < 0 else "permuted",
                "realized_K": int(np.unique(restored_ids).size),
                "wall_time_seconds": wall_time,
                "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if torch.cuda.is_available() else 0,
                **comparison,
            }
            rows.append(metadata)
            if cell_types is not None:
                assignments = pd.DataFrame(
                    {"cell_id": cell_ids, "metacell_id": restored_ids, "cell_type": cell_types}
                )
                per_type_rows.append(per_type_table(assignments, metadata))
    pd.DataFrame(rows).to_csv(output_dir / "batch_size_order_stability.csv", index=False)
    if per_type_rows:
        pd.concat(per_type_rows, ignore_index=True).to_csv(
            output_dir / "batch_size_order_per_type.csv", index=False
        )

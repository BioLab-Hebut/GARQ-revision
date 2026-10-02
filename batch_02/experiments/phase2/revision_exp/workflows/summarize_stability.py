"""Collect E8 stability outputs without discarding per-run evidence."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    root = Path("revision_results/08_scalability")
    stability = sorted(root.glob("*_inference_stability_smoke/batch_size_order_stability.csv"))
    per_type = sorted(root.glob("*_inference_stability_smoke/batch_size_order_per_type.csv"))
    if not stability:
        raise FileNotFoundError("No stability outputs found")
    pd.concat([pd.read_csv(path) for path in stability], ignore_index=True).to_csv(
        root / "batch_size_order_stability.csv", index=False
    )
    if per_type:
        pd.concat([pd.read_csv(path) for path in per_type], ignore_index=True).to_csv(
            root / "batch_size_order_per_type.csv", index=False
        )


if __name__ == "__main__":
    main()

"""Run frozen, label-free requested-K calibration for official MetaQ."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from revision_exp.workflows.matched_baseline_fixed import REGISTRY


# Frozen from seed-0 requested/realized K only; no biological labels or metrics.
CALIBRATION = {
    "D5": {"pilot_requested_K": 242, "pilot_realized_K": 230, "calibrated_requested_K": 255},
    "D11": {"pilot_requested_K": 193, "pilot_realized_K": 168, "calibrated_requested_K": 222},
    "D17": {"pilot_requested_K": 323, "pilot_realized_K": 323, "calibrated_requested_K": 323},
    "D18": {"pilot_requested_K": 510, "pilot_realized_K": 484, "calibrated_requested_K": 537},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(REGISTRY), required=True)
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--phase2-root", type=Path, default=Path("revision_results/phase2"))
    args = parser.parse_args()
    cfg = REGISTRY[args.dataset]
    cal = CALIBRATION[args.dataset]
    out = args.phase2_root / "01_size_resolution" / args.dataset / "MetaQ" / f"full_seed{args.seed}_K002_realizedmatched"
    out.mkdir(parents=True, exist_ok=True)
    provenance = {
        "dataset": args.dataset,
        "seed": args.seed,
        "target_K": cfg["requested_K"],
        **cal,
        "rule": "round(target_K * pilot_requested_K / pilot_realized_K), frozen from seed 0 without labels",
        "labels_used_for_calibration": False,
    }
    (out / "calibration_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    adapter = [
        sys.executable, "-m", "revision_exp.methods.metaq_adapter",
        "--data-files", *cfg["data_files"],
        "--data-types", *cfg["data_types"],
        "--dataset", args.dataset,
        "--label-key", cfg["label_key"],
        "--seed", str(args.seed),
        "--requested-k", str(cal["calibrated_requested_K"]),
        "--output-dir", str(out),
        "--device", args.device,
    ]
    subprocess.run(adapter, check=True)
    evaluator = [
        sys.executable, "-m", "revision_exp.methods.common_assignment_evaluator",
        "--assignments", str(out / "cell_assignments.csv"),
        "--metadata-h5ad", cfg["data_files"][0],
        "--label-key", cfg["label_key"],
        "--output-dir", str(out),
    ]
    subprocess.run(evaluator, check=True)


if __name__ == "__main__":
    main()

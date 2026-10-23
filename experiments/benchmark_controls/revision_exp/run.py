"""Unified resumable CLI for GARQ revision experiments."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time
from pathlib import Path

import yaml

from revision_exp.audit.e0 import run_e0
from revision_exp.workflows.fixed_representation import run_fixed_representation_benchmark
from revision_exp.workflows.legacy import run_legacy_experiment
from revision_exp.utils.provenance import config_hash, hardware_snapshot, peak_rss_bytes, write_json


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo_root), *args], check=True, capture_output=True, text=True).stdout.strip()


def run(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    digest = config_hash(config)
    config_id = config.get("config_id", config_path.stem)
    run_id = f"{config_id}-{digest[:12]}"
    repo_root = Path(__file__).resolve().parents[1]
    result_root = Path(config.get("result_root", repo_root / "revision_results"))
    resolved_dir = result_root / "configs_resolved"
    log_dir = result_root / "logs"
    manifest_dir = result_root / "manifests"
    for directory in (resolved_dir, log_dir, manifest_dir):
        directory.mkdir(parents=True, exist_ok=True)
    resolved_path = resolved_dir / f"{run_id}.json"
    if not resolved_path.exists():
        write_json(resolved_path, config)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_dir / f"{run_id}.log"), logging.StreamHandler()],
        force=True,
    )
    output_dir = result_root / config["output_subdir"]
    done_path = output_dir / f"{run_id}.done.json"
    if done_path.exists() and not config.get("force", False):
        logging.info("Cached run already complete: %s", done_path)
        return done_path
    started = time.time()
    before = hardware_snapshot()
    status = "PASS"
    error = None
    try:
        if config["task"] == "e0_audit":
            run_e0(output_dir, repo_root)
        elif config["task"] == "legacy_garq":
            run_legacy_experiment(config, output_dir, result_root)
        elif config["task"] == "fixed_representation_benchmark":
            run_fixed_representation_benchmark(config, output_dir, result_root)
        else:
            raise ValueError(f"Unsupported task: {config['task']}")
    except BaseException as caught:
        status = "FAIL"
        error = {"type": type(caught).__name__, "message": str(caught)}
        raise
    finally:
        ended = time.time()
        manifest = {
            "run_id": run_id,
            "config_id": config_id,
            "config_sha256": digest,
            "task": config["task"],
            "implementation_tag": config["implementation_tag"],
            "status": status,
            "error": error,
            "wall_time_seconds": ended - started,
            "peak_cpu_rss_bytes": peak_rss_bytes(),
            "git_commit": _git(repo_root, "rev-parse", "HEAD"),
            "git_branch": _git(repo_root, "branch", "--show-current"),
            "runtime_environment": {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            },
            "hardware_before": before,
            "hardware_after": hardware_snapshot(),
            "resolved_config": str(resolved_path),
        }
        write_json(manifest_dir / f"{run_id}.json", manifest)
        if status == "PASS":
            write_json(done_path, manifest)
    return done_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Preflight inputs, then forward the unchanged command to the public GARQ.py."""

import argparse
from pathlib import Path
import subprocess
import sys

from _garq_tool_utils import ValidationError, validate_paired_names, validate_training_capacity


def build_parser():
    parser = argparse.ArgumentParser(
        description="Validate paired row identity and the original initialization capacity, then run GARQ unchanged.",
        epilog="All supplied arguments are forwarded in their original order. Unknown options are left for GARQ.py to validate. Outputs remain relative to the current working directory.",
        allow_abbrev=False,
    )
    parser.add_argument("--data_file", nargs="+", required=True)
    parser.add_argument("--data_type", nargs="+", choices=["RNA", "ADT", "ATAC"], required=True)
    parser.add_argument("--save_name", required=True)
    parser.add_argument("--n_GARQs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=512)
    return parser


def core_script_path():
    return Path(__file__).resolve().parents[1] / "GARQ.py"


def read_input_names(paths):
    try:
        import anndata as ad
    except ImportError as error:
        raise ValidationError(
            "Input preflight requires anndata in the same Python environment used for GARQ. "
            "Activate the documented GARQ environment first."
        ) from error
    names_by_modality = []
    for value in paths:
        path = Path(value)
        if not path.is_file():
            raise ValidationError(f"Input file does not exist: {path}.")
        try:
            source = ad.read_h5ad(path, backed="r")
        except Exception as error:
            raise ValidationError(f"Cannot read input metadata from {path}: {error}") from error
        try:
            if source.n_vars < 1:
                raise ValidationError(f"Input has no features: {path}.")
            names_by_modality.append(tuple(source.obs_names))
        finally:
            source.file.close()
    return names_by_modality


def preflight(args):
    if len(args.data_file) != len(args.data_type):
        raise ValidationError("--data_file and --data_type must have the same length and order.")
    if not args.save_name:
        raise ValidationError("--save_name must not be empty.")
    # Validate K and batch bounds before opening any inputs.
    from _garq_tool_utils import effective_training_batch
    effective_training_batch(args.batch_size, args.n_GARQs)
    cell_names = validate_paired_names(read_input_names(args.data_file))
    effective_batch, points = validate_training_capacity(len(cell_names), args.batch_size, args.n_GARQs)
    return {"cells": len(cell_names), "train_batch": effective_batch, "initialization_points": points}


def main(argv=None):
    original_args = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args, _unknown_args = parser.parse_known_args(original_args)
    core = core_script_path()
    if not core.is_file():
        parser.error(f"Expected the unchanged core script at {core}.")
    try:
        summary = preflight(args)
    except ValidationError as error:
        parser.error(str(error))
    working_dir = Path.cwd()
    (working_dir / "save").mkdir(exist_ok=True)
    (working_dir / "figures").mkdir(exist_ok=True)
    print(
        "Preflight passed: "
        f"{summary['cells']} paired cells, effective training batch {summary['train_batch']}, "
        f"{summary['initialization_points']} points in the original initialization sample.",
        flush=True,
    )
    # Keep argument strings, their order, Python environment, and cwd unchanged.
    command = [sys.executable, str(core), *original_args]
    try:
        return subprocess.run(command, check=False).returncode
    except OSError as error:
        parser.error(f"Cannot start the core script: {error}")


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Restore names and anchor-row mapping into separate copies of original outputs."""

import argparse
import os
from pathlib import Path
import sys
import tempfile

from _garq_tool_utils import (
    ValidationError,
    matrix_fingerprint,
    metadata_output_path,
    sorted_anchor_ids,
    validate_output_paths,
    validate_paired_names,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Restore metadata in separate .metadata.h5ad copies; preserve original matrix values and membership IDs.",
        epilog="Supply the exact original construction inputs in their original modality and cell row order. This tool cannot infer provenance or rematch cells from an output that lost its original names.",
        allow_abbrev=False,
    )
    parser.add_argument("--data-file", "--data_file", nargs="+", required=True, dest="data_file")
    parser.add_argument("--assignment-file", "--assignment_file", required=True, dest="assignment_file")
    parser.add_argument("--metacell-file", "--metacell_file", nargs="+", required=True, dest="metacell_file")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir")
    return parser


def read_source_metadata(ad, paths):
    metadata = []
    for value in paths:
        source = ad.read_h5ad(value, backed="r")
        try:
            metadata.append({"obs_names": tuple(source.obs_names), "var": source.var.copy(deep=True)})
        finally:
            source.file.close()
    validate_paired_names([item["obs_names"] for item in metadata])
    return metadata


def require_files(paths):
    for value in paths:
        path = Path(value)
        if not path.is_file():
            raise ValidationError(f"File does not exist: {path}.")


def check_source_output_shapes(ad, assignment_path, metacell_paths, metadata):
    """Check every shape before writing any result; read only metadata here."""
    assignment = ad.read_h5ad(assignment_path, backed="r")
    try:
        if "metacell" not in assignment.obs:
            raise ValidationError("The assignment file has no obs['metacell'] membership column.")
        ids = tuple(assignment.obs["metacell"].tolist())
        anchor_ids = sorted_anchor_ids(ids)
        if assignment.n_obs != len(metadata[0]["obs_names"]):
            raise ValidationError("The assignment row count differs from the supplied original inputs.")
        assignment_shape = tuple(assignment.shape)
        id_dtype = str(assignment.obs["metacell"].dtype)
    finally:
        assignment.file.close()
    for value, source in zip(metacell_paths, metadata):
        metacell = ad.read_h5ad(value, backed="r")
        try:
            expected_shape = (len(anchor_ids), len(source["var"]))
            if tuple(metacell.shape) != expected_shape:
                raise ValidationError(
                    f"Metacell shape {tuple(metacell.shape)} in {value} differs from "
                    f"expected occupied-anchor/full-feature shape {expected_shape}."
                )
        finally:
            metacell.file.close()
    return anchor_ids, ids, id_dtype, assignment_shape


def write_verified_copy(ad, data, output, verify_extra, expected_fingerprint=None):
    """Write and verify a temporary copy, then publish without replacing any file."""
    before = expected_fingerprint if expected_fingerprint is not None else matrix_fingerprint(data.X)
    shape_before = tuple(data.shape)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=".garq_metadata_", suffix=".h5ad", dir=output.parent)
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        data.write_h5ad(temporary)
        restored = ad.read_h5ad(temporary, backed="r")
        try:
            if tuple(restored.shape) != shape_before or matrix_fingerprint(restored.X) != before:
                raise ValidationError("Writing metadata changed X shape, dtype or exact values; no result was published.")
            verify_extra(restored)
        finally:
            restored.file.close()
        # The temporary file is on the same filesystem. link() is atomic and
        # refuses an existing destination, unlike a replacing rename.
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise ValidationError(f"Refusing to overwrite existing output: {output}.") from error
        except OSError as error:
            raise ValidationError(
                f"Cannot publish the verified copy with a non-overwriting hard link at {output}: {error}. "
                "Use an output directory on a filesystem that supports hard links."
            ) from error
    finally:
        temporary.unlink(missing_ok=True)
    return output


def restore_metadata(args, ad):
    if len(args.data_file) != len(args.metacell_file):
        raise ValidationError("Supply one --metacell-file per --data-file, in matching modality order.")
    source_paths = [*args.data_file, args.assignment_file, *args.metacell_file]
    require_files(source_paths)
    if len({Path(path).resolve() for path in [args.assignment_file, *args.metacell_file]}) != 1 + len(args.metacell_file):
        raise ValidationError("Assignment and metacell source outputs must be distinct files.")
    output_paths = validate_output_paths(
        source_paths,
        [metadata_output_path(path, args.output_dir) for path in [args.assignment_file, *args.metacell_file]],
    )
    metadata = read_source_metadata(ad, args.data_file)
    anchor_ids, ids_before, id_dtype, expected_assignment_shape = check_source_output_shapes(
        ad, args.assignment_file, args.metacell_file, metadata
    )
    cell_names = metadata[0]["obs_names"]

    assignment = ad.read_h5ad(args.assignment_file)
    original_assignment_fingerprint = matrix_fingerprint(assignment.X)
    assignment.obs_names = list(cell_names)

    def verify_assignment(restored):
        if tuple(restored.obs_names) != cell_names:
            raise ValidationError("Assignment cell names did not round-trip correctly.")
        if tuple(restored.shape) != expected_assignment_shape:
            raise ValidationError("Assignment shape changed.")
        if tuple(restored.obs["metacell"].tolist()) != ids_before or str(restored.obs["metacell"].dtype) != id_dtype:
            raise ValidationError("Original assignment membership values or dtype changed.")
    published = [write_verified_copy(ad, assignment, output_paths[0], verify_assignment, original_assignment_fingerprint)]
    del assignment
    for path, output, source_metadata in zip(args.metacell_file, output_paths[1:], metadata):
        metacell = ad.read_h5ad(path)
        original_metacell_fingerprint = matrix_fingerprint(metacell.X)
        metacell.var = source_metadata["var"].copy(deep=True)
        metacell.obs["metacell"] = list(anchor_ids)
        metacell.obs_names = [str(anchor) for anchor in anchor_ids]
        expected_names = tuple(str(anchor) for anchor in anchor_ids)
        expected_var = source_metadata["var"]

        def verify_metacell(restored):
            if tuple(restored.obs_names) != expected_names or tuple(restored.obs["metacell"].tolist()) != anchor_ids:
                raise ValidationError("The sorted occupied-anchor row mapping changed.")
            if not restored.var.equals(expected_var):
                raise ValidationError("Feature metadata did not round-trip correctly.")
        published.append(write_verified_copy(ad, metacell, output, verify_metacell, original_metacell_fingerprint))
        del metacell
    return published


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        import anndata as ad
        outputs = restore_metadata(args, ad)
    except ImportError as error:
        parser.error(f"Metadata restoration requires anndata/numpy in the documented GARQ environment: {error}.")
    except (ValidationError, OSError) as error:
        parser.error(str(error))
    print("Verified separate copies; original files were kept unchanged:")
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

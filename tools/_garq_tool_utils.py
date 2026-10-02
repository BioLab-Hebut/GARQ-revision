"""Small validation helpers; importing this module needs only the standard library."""

import hashlib
import math
import numbers
from pathlib import Path


class ValidationError(ValueError):
    """A request cannot be checked or applied without changing original behavior."""


def effective_training_batch(requested_batch, anchors):
    if requested_batch < 1:
        raise ValidationError("--batch_size must be at least 1.")
    if anchors < 2:
        raise ValidationError("--n_GARQs must be at least 2: the core uses topk(..., 2).")
    return 4096 if anchors > 1000 and requested_batch <= 512 else requested_batch


def initialization_point_count(cell_count, effective_batch):
    """The public initializer takes at most two complete shuffled batches."""
    if effective_batch < 1:
        raise ValidationError("The effective batch size must be positive.")
    complete_batches = cell_count // effective_batch
    return min(2, complete_batches) * effective_batch


def validate_training_capacity(cell_count, requested_batch, anchors):
    effective_batch = effective_training_batch(requested_batch, anchors)
    points = initialization_point_count(cell_count, effective_batch)
    if points == 0:
        raise ValidationError(
            f"Only {cell_count} cells are available, but the original drop_last=True "
            f"training loader requires at least one complete batch of {effective_batch}. "
            "Choose a valid batch explicitly; the wrapper does not change it."
        )
    if anchors > points:
        raise ValidationError(
            f"The original first-two-batch initializer sees {points} cells, fewer "
            f"than K={anchors}. FAISS needs at least K training points. Choose a "
            "valid K/batch explicitly; the wrapper does not collect extra batches."
        )
    return effective_batch, points


def validate_paired_names(names_by_modality):
    """Require exact row identity/order; never match or reorder input cells."""
    if not names_by_modality:
        raise ValidationError("Supply at least one modality input.")
    reference = tuple(str(value) for value in names_by_modality[0])
    if not reference:
        raise ValidationError("Modality inputs contain no cells.")
    if len(set(reference)) != len(reference):
        raise ValidationError("The first modality has duplicate cell identifiers.")
    for index, names in enumerate(names_by_modality[1:], start=2):
        current = tuple(str(value) for value in names)
        if len(current) != len(reference):
            raise ValidationError(
                f"Modality {index} has {len(current)} cells; the first has {len(reference)}."
            )
        if len(set(current)) != len(current):
            raise ValidationError(f"Modality {index} has duplicate cell identifiers.")
        if current != reference:
            mismatch = next(i for i, pair in enumerate(zip(reference, current)) if pair[0] != pair[1])
            raise ValidationError(
                f"Modality {index} has different cell identity/order at row {mismatch + 1}: "
                f"{current[mismatch]!r} versus {reference[mismatch]!r}. Supply the original "
                "paired row order; this tool does not reorder cells."
            )
    return reference


def sorted_anchor_ids(values):
    """Validate saved membership values and recover the original aggregation row order."""
    normalized = []
    for row, value in enumerate(values, start=1):
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValidationError(f"Assignment row {row} is not a numeric anchor index: {value!r}.")
        if isinstance(value, numbers.Integral):
            integer = int(value)
        else:
            if not math.isfinite(value) or not float(value).is_integer():
                raise ValidationError(f"Assignment row {row} is not a finite integer anchor index: {value!r}.")
            integer = int(value)
        if integer < 0:
            raise ValidationError(f"Assignment row {row} has a negative anchor index: {integer}.")
        normalized.append(integer)
    if not normalized:
        raise ValidationError("The assignment file contains no memberships.")
    return tuple(sorted(set(normalized)))


def metadata_output_path(source, output_dir=None):
    source = Path(source)
    if source.suffix.lower() != ".h5ad":
        raise ValidationError(f"Expected an .h5ad output file: {source}.")
    parent = Path(output_dir) if output_dir is not None else source.parent
    return parent / (source.stem + ".metadata.h5ad")


def validate_output_paths(sources, outputs):
    """Reject aliases, collisions and overwrites before making any output."""
    source_paths = {Path(path).resolve() for path in sources}
    destinations = [Path(path).resolve() for path in outputs]
    if len(destinations) != len(set(destinations)):
        raise ValidationError("Output names collide; choose a different --output-dir or filenames.")
    for output in destinations:
        if output in source_paths:
            raise ValidationError(f"An output would replace a source file: {output}.")
        if output.exists():
            raise ValidationError(f"Refusing to overwrite existing output: {output}.")
    return tuple(destinations)


def matrix_fingerprint(matrix, max_chunk_bytes=16 * 1024 * 1024):
    """Hash exact matrix dtype/shape/values in bounded row chunks, including backed X."""
    import numpy as np

    if matrix is None or len(matrix.shape) != 2:
        raise ValidationError("Expected a two-dimensional X matrix.")
    shape = tuple(int(value) for value in matrix.shape)
    dtype = np.dtype(matrix.dtype)
    rows_per_chunk = max(1, max_chunk_bytes // max(1, shape[1] * dtype.itemsize))
    digest = hashlib.sha256()
    digest.update(repr(shape).encode("ascii"))
    digest.update(dtype.str.encode("ascii"))
    for start in range(0, shape[0], rows_per_chunk):
        block = matrix[start : start + rows_per_chunk, :]
        if hasattr(block, "toarray"):
            block = block.toarray()
        block = np.ascontiguousarray(block)
        if block.dtype != dtype:
            raise ValidationError("A matrix slice changed dtype during validation.")
        digest.update(block.tobytes(order="C"))
    return shape, dtype.str, digest.hexdigest()

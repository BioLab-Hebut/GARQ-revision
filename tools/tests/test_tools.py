"""Small synthetic checks; no real datasets or model training are used."""

import argparse
import contextlib
import hashlib
import importlib.util
import io
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest import mock
import uuid

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

from _garq_tool_utils import (
    ValidationError,
    effective_training_batch,
    initialization_point_count,
    matrix_fingerprint,
    metadata_output_path,
    sorted_anchor_ids,
    validate_output_paths,
    validate_paired_names,
    validate_training_capacity,
)
import restore_output_metadata
import run_garq


@contextlib.contextmanager
def workspace_test_directory():
    # Python 3.13's Windows mkdtemp(mode=0o700) can create inaccessible ACLs
    # under a restricted token. Use a normal mkdir in the workspace instead.
    base = TOOLS.parent / "audit" / "tests_temp"
    base.mkdir(parents=True, exist_ok=True)
    destination = base / ("tool_test_" + uuid.uuid4().hex)
    destination.mkdir()
    try:
        yield str(destination)
    finally:
        resolved = destination.resolve()
        if resolved.parent != base.resolve() or not resolved.name.startswith("tool_test_"):
            raise RuntimeError("Refusing to clean a path outside the test workspace.")
        shutil.rmtree(resolved)


class ValidationTests(unittest.TestCase):
    def test_effective_batch_and_original_two_batch_capacity(self):
        self.assertEqual(effective_training_batch(512, 1000), 512)
        self.assertEqual(effective_training_batch(512, 1001), 4096)
        self.assertEqual(effective_training_batch(256, 1001), 4096)
        self.assertEqual(effective_training_batch(1024, 1001), 1024)
        self.assertEqual(initialization_point_count(1500, 512), 1024)
        self.assertEqual(initialization_point_count(700, 512), 512)
        self.assertEqual(validate_training_capacity(1500, 512, 800), (512, 1024))

    def test_capacity_rejects_zero_full_batches_and_too_many_anchors(self):
        with self.assertRaisesRegex(ValidationError, "complete batch"):
            validate_training_capacity(511, 512, 100)
        # K>1000 triggers batch4096; do not silently collect extra batches.
        with self.assertRaisesRegex(ValidationError, "fewer than K"):
            validate_training_capacity(20000, 512, 9000)
        with self.assertRaisesRegex(ValidationError, "at least 2"):
            validate_training_capacity(1500, 512, 1)
        with self.assertRaisesRegex(ValidationError, "at least 1"):
            validate_training_capacity(1500, 0, 2)

    def test_paired_row_order_is_checked_without_reordering(self):
        rows = ("cell C", "cell A", "cell B")
        self.assertEqual(validate_paired_names([rows, rows]), rows)
        with self.assertRaisesRegex(ValidationError, "identity/order at row 1"):
            validate_paired_names([rows, tuple(sorted(rows))])
        with self.assertRaisesRegex(ValidationError, "duplicate"):
            validate_paired_names([("A", "A")])
        with self.assertRaisesRegex(ValidationError, "has 1 cells"):
            validate_paired_names([rows, ("A",)])

    def test_sparse_anchor_ids_restore_filtered_original_row_mapping(self):
        self.assertEqual(sorted_anchor_ids([9, 2, 9, 5, 2]), (2, 5, 9))
        for values in ([0, -1], [1.5, 2], [float("nan")], [True], ["2"], []):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                sorted_anchor_ids(values)

    def test_output_names_and_no_overwrite_preflight(self):
        self.assertEqual(metadata_output_path(Path("save/RNA.h5ad")), Path("save/RNA.metadata.h5ad"))
        with workspace_test_directory() as temporary:
            base = Path(temporary)
            source = base / "input.h5ad"
            source.write_bytes(b"synthetic")
            output = base / "input.metadata.h5ad"
            self.assertEqual(validate_output_paths([source], [output]), (output.resolve(),))
            with self.assertRaisesRegex(ValidationError, "source file"):
                validate_output_paths([source], [source])
            with self.assertRaisesRegex(ValidationError, "collide"):
                validate_output_paths([source], [output, output])
            output.write_bytes(b"existing")
            with self.assertRaisesRegex(ValidationError, "overwrite"):
                validate_output_paths([source], [output])
            self.assertEqual(source.read_bytes(), b"synthetic")
            self.assertEqual(output.read_bytes(), b"existing")

    def test_wrapper_forwards_exact_strings_order_cwd_and_child_exit_code(self):
        argv = [
            "--data_file", "folder with spaces/rna.h5ad", "adt.h5ad",
            "--data_type", "RNA", "ADT", "--save_name", "example",
            "--n_GARQs", "400", "--device", "cpu", "--seed", "7",
            "--epoch", "80", "--k_knn", "0", "--batch_size", "512",
            "--anchers_init", "original literal value",
        ]
        with workspace_test_directory() as temporary:
            base = Path(temporary)
            core = base / "GARQ.py"
            core.write_text("# synthetic placeholder, never executed\n", encoding="utf-8")
            summary = {"cells": 1500, "train_batch": 512, "initialization_points": 1024}
            with mock.patch.object(run_garq, "core_script_path", return_value=core), \
                 mock.patch.object(run_garq, "preflight", return_value=summary), \
                 mock.patch.object(run_garq.Path, "cwd", return_value=base), \
                 mock.patch.object(run_garq.subprocess, "run", return_value=subprocess.CompletedProcess([], 7)) as child, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_garq.main(argv), 7)
            child.assert_called_once_with([sys.executable, str(core), *argv], check=False)
            self.assertTrue((base / "save").is_dir())
            self.assertTrue((base / "figures").is_dir())
            self.assertEqual(core.read_text(encoding="utf-8"), "# synthetic placeholder, never executed\n")

    def test_wrapper_never_launches_core_after_preflight_failure(self):
        argv = ["--data_file", "rna.h5ad", "--data_type", "RNA", "--save_name", "x", "--n_GARQs", "400"]
        with mock.patch.object(run_garq, "core_script_path", return_value=Path(__file__)), \
             mock.patch.object(run_garq, "preflight", side_effect=ValidationError("row mismatch")), \
             mock.patch.object(run_garq.subprocess, "run") as child, \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exit_result:
                run_garq.main(argv)
            self.assertEqual(exit_result.exception.code, 2)
        child.assert_not_called()


@unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy is unavailable")
class MatrixFingerprintTests(unittest.TestCase):
    def test_exact_values_dtype_shape_and_chunk_independence(self):
        import numpy as np
        matrix = np.array([[1, 2, 3], [4, float("nan"), 6]], dtype=np.float32)
        baseline = matrix_fingerprint(matrix, max_chunk_bytes=12)
        self.assertEqual(baseline, matrix_fingerprint(matrix.copy(), max_chunk_bytes=1024))
        changed = matrix.copy()
        changed[0, 0] = 9
        self.assertNotEqual(baseline, matrix_fingerprint(changed))
        self.assertNotEqual(baseline, matrix_fingerprint(matrix.astype(np.float64)))
        self.assertNotEqual(baseline, matrix_fingerprint(matrix.reshape(3, 2)))

    def test_verified_copy_publishes_without_overwrite_and_cleans_rejected_temporary(self):
        import numpy as np

        class SyntheticReader:
            def __init__(self, array):
                self.X = array
                self.shape = array.shape
                self.file = mock.Mock()

        class SyntheticWriter(SyntheticReader):
            def write_h5ad(self, path):
                # This intentionally is a lightweight NumPy test fixture,
                # not an H5AD implementation or an AnnData integration test.
                with Path(path).open("wb") as stream:
                    np.save(stream, self.X, allow_pickle=False)

        class SyntheticBackend:
            @staticmethod
            def read_h5ad(path, backed=None):
                with Path(path).open("rb") as stream:
                    return SyntheticReader(np.load(stream, allow_pickle=False))

        matrix = np.array([[1, 2], [3, 4]], dtype=np.float32)
        with workspace_test_directory() as temporary:
            base = Path(temporary)
            output = base / "synthetic.metadata.h5ad"
            callback = mock.Mock()
            restore_output_metadata.write_verified_copy(SyntheticBackend, SyntheticWriter(matrix), output, callback)
            callback.assert_called_once()
            before = output.read_bytes()
            with self.assertRaisesRegex(ValidationError, "overwrite"):
                restore_output_metadata.write_verified_copy(SyntheticBackend, SyntheticWriter(matrix + 1), output, lambda _: None)
            self.assertEqual(output.read_bytes(), before)
            rejected = base / "rejected.metadata.h5ad"
            with self.assertRaisesRegex(ValidationError, "invalid metadata"):
                restore_output_metadata.write_verified_copy(
                    SyntheticBackend, SyntheticWriter(matrix), rejected,
                    mock.Mock(side_effect=ValidationError("invalid metadata")),
                )
            self.assertFalse(rejected.exists())
            self.assertEqual(list(base.glob(".garq_metadata_*")), [])


@unittest.skipUnless(importlib.util.find_spec("anndata"), "anndata is unavailable; synthetic H5AD round-trip not executed")
class SyntheticAnnDataTests(unittest.TestCase):
    def test_round_trip_preserves_original_files_values_and_nonconsecutive_ids(self):
        import anndata as ad
        import numpy as np
        with workspace_test_directory() as temporary:
            base = Path(temporary)
            original_input = base / "RNA.h5ad"
            assignment_path = base / "example_ids.h5ad"
            metacell_path = base / "example_RNA.h5ad"
            source = ad.AnnData(np.array([[1, 2, 3], [3, 2, 1], [0, 2, 1], [1, 0, 2]], dtype=np.float32))
            source.obs_names = ["cell C", "cell A", "cell B", "cell D"]
            source.var_names = ["gene C", "gene A", "gene B"]
            source.var["feature_annotation"] = ["C", "A", "B"]
            source.write_h5ad(original_input)
            embeddings = np.arange(8, dtype=np.float32).reshape(4, 2)
            assignment = ad.AnnData(embeddings)
            assignment.obs["metacell"] = np.array([9, 2, 9, 2], dtype=np.int64)
            assignment.obsm["X_umap"] = np.zeros((4, 2), dtype=np.float32)
            assignment.write_h5ad(assignment_path)
            profiles = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
            metacell = ad.AnnData(profiles)
            metacell.obs["celltype"] = ["type A", "type B"]
            metacell.write_h5ad(metacell_path)
            before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in [original_input, assignment_path, metacell_path]}
            args = argparse.Namespace(data_file=[str(original_input)], assignment_file=str(assignment_path), metacell_file=[str(metacell_path)], output_dir=None)
            outputs = restore_output_metadata.restore_metadata(args, ad)
            self.assertEqual([path.name for path in outputs], ["example_ids.metadata.h5ad", "example_RNA.metadata.h5ad"])
            restored_assignment = ad.read_h5ad(outputs[0])
            restored_metacell = ad.read_h5ad(outputs[1])
            self.assertEqual(list(restored_assignment.obs_names), list(source.obs_names))
            self.assertEqual(restored_assignment.obs["metacell"].tolist(), [9, 2, 9, 2])
            self.assertEqual(list(restored_metacell.obs_names), ["2", "9"])
            self.assertEqual(restored_metacell.obs["metacell"].tolist(), [2, 9])
            self.assertEqual(list(restored_metacell.var_names), list(source.var_names))
            np.testing.assert_array_equal(restored_assignment.X, embeddings)
            np.testing.assert_array_equal(restored_assignment.obsm["X_umap"], assignment.obsm["X_umap"])
            np.testing.assert_array_equal(restored_metacell.X, profiles)
            for path, digest in before.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            with self.assertRaisesRegex(ValidationError, "overwrite"):
                restore_output_metadata.restore_metadata(args, ad)


if __name__ == "__main__":
    unittest.main(verbosity=2)

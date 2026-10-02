from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse

from revision_exp.workflows.legacy import _prepare_uniform_subset


def test_binomial_thinning_is_count_level_and_reproducible(tmp_path: Path):
    matrix = sparse.csr_matrix(np.arange(60, dtype=np.float32).reshape(10, 6))
    source = tmp_path / "source.h5ad"
    ad.AnnData(matrix).write_h5ad(source)
    kwargs = dict(
        source_paths=[source],
        cache_dir=tmp_path / "cache",
        dataset="synthetic",
        n_cells=8,
        seed=3,
        perturbations=[{"kind": "binomial_thinning", "p": 0.5, "seed": 9}],
    )
    first, provenance = _prepare_uniform_subset(**kwargs)
    second, _ = _prepare_uniform_subset(**kwargs)
    first_matrix = ad.read_h5ad(first[0]).X
    second_matrix = ad.read_h5ad(second[0]).X
    assert (first_matrix != second_matrix).nnz == 0
    assert np.all(first_matrix.data >= 0)
    assert np.allclose(first_matrix.data, np.rint(first_matrix.data))
    assert provenance["modality_checks"][0]["perturbation"]["p"] == 0.5


def test_full_size_cell_permutation_is_materialized(tmp_path: Path):
    matrix = sparse.csr_matrix(np.arange(60, dtype=np.float32).reshape(10, 6))
    source = tmp_path / "source.h5ad"
    ad.AnnData(matrix).write_h5ad(source)
    outputs, provenance = _prepare_uniform_subset(
        source_paths=[source],
        cache_dir=tmp_path / "cache",
        dataset="synthetic",
        n_cells=10,
        seed=3,
        perturbations=[{"kind": "cell_permutation", "seed": 9}],
    )
    observed = ad.read_h5ad(outputs[0]).X
    assert Path(outputs[0]) != source
    assert (observed != matrix).nnz > 0
    assert provenance["subset_applied"]
    assert provenance["modality_checks"][0]["perturbation"]["kind"] == "cell_permutation"

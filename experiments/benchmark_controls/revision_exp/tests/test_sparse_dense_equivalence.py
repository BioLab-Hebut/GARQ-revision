from types import SimpleNamespace

import anndata as ad
import numpy as np
from scipy import sparse

from data_utils import preprocess


def test_legacy_sparse_input_matches_dense_input():
    rng = np.random.default_rng(11)
    counts = rng.poisson(1.5, size=(120, 80)).astype(np.float32)
    dense = ad.AnnData(counts.copy())
    sparse_data = ad.AnnData(sparse.csr_matrix(counts))
    x_dense, sf_dense, raw_dense, _ = preprocess(dense, "RNA")
    x_sparse, sf_sparse, raw_sparse, _ = preprocess(sparse_data, "RNA")
    np.testing.assert_allclose(x_dense, x_sparse, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(sf_dense, sf_sparse, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(raw_dense, raw_sparse, rtol=0, atol=0)

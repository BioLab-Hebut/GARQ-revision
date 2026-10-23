# Analysis environments

Use separate environments for GARQ, SEACells and the R downstream analyses.
The repository's `environment.yaml` describes the original Linux/CUDA workflow;
it is not a single installer for every comparison.

| Workflow | Required libraries |
|---|---|
| GARQ construction and repeated training | NumPy, pandas, SciPy, Scanpy, anndata, scikit-learn, PyTorch, FAISS, einops, psutil; the model's plotting dependencies |
| Controlled representations | GARQ dependencies; mofapy2 for MOFA+; SEACells and CuPy in a separate environment |
| Saved-assignment evaluation | NumPy, pandas, h5py, scikit-learn, threadpoolctl |
| v2 numerical and figure scripts | NumPy, pandas, SciPy, anndata, matplotlib; quantizer controls additionally use PyTorch/FAISS/einops |
| Trajectory analyses | R: Seurat, slingshot, Matrix, anndata, jsonlite, igraph; UCell for pathway scores |
| LinkPeaks | R: Seurat, Signac, Matrix, GenomicRanges, jsonlite, rhdf5, future, future.apply, Rcpp |
| Matched downstream analyses | mofapy2; R: glmnet, zellkonverter, ComplexHeatmap, clusterProfiler, org.Hs.eg.db and the plotting packages specified in the existing workflow |

Recorded GARQ/control environment: NumPy 1.26.0, pandas 2.3.2, SciPy 1.11.3,
scikit-learn 1.1.3, Scanpy 1.9.6, anndata 0.11.4, PyTorch 2.1.1+cu121,
einops 0.7.0, faiss-cpu 1.11.0.

Recorded SEACells environment: NumPy 1.23.5, pandas 2.0.3, SciPy 1.10.1,
scikit-learn 1.3.0, Scanpy 1.9.3, anndata 0.8.0, SEACells 0.3.3,
cupy-cuda12x 12.3.0. Do not mix these pins with the GARQ environment.

The earlier matched downstream environment used pandas 2.2.3 and mofapy2 0.7.2.
The kidney/v2 trajectory analyses used R 4.4.3, Seurat 5.3.0 and slingshot 2.12.0;
the fixed-pathway analysis used UCell 2.8.0. v2 D also recorded Matrix 1.7-2,
jsonlite 2.0.0, R anndata 0.8.0 and igraph 2.1.4. The older matched downstream
R environment is separate (R 4.5.2). See the workflow's environment record before
installation. Versions not recorded are not inferred here.

R `anndata` and Python `anndata` are distinct packages. Set `RETICULATE_PYTHON`
to a compatible Python interpreter for the H5AD workflows. `fcntl` and `resource`
are POSIX standard-library modules, not pip dependencies. Some training runners
therefore require Linux. Arial is needed for the supplied publication figure
scripts. Syntax/configuration checks do not establish a fresh end-to-end run
in a newly installed environment.

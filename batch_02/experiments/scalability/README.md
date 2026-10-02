# R2Q5/R1Q3 full-process profiling and batch sensitivity

This subtree uses the final GiB package's actual five-module `source_code` backend, placed at `source_server` to match the existing scripts' import layout. `scripts/build_deliverables.py` was restored from the existing September 15 package because the final bundle omitted it. This builder summarizes completed runs and produces historical report/figure artifacts; it does not fit models. Its historical reply wording should be reviewed separately before manuscript use.

Run on Linux with CUDA, `nvidia-smi`, `lscpu`, psutil and the archived Python environment. Recorded package versions are under `provenance/`. Python, PyTorch/CUDA, Scanpy, AnnData, NumPy, SciPy, pandas, scikit-learn, h5py, matplotlib and FAISS are needed. The preserved RAM/GPU preflight thresholds can be substantial; the launcher does not tune numerical settings.

```bash
python run_scalability.py --data-root /path/to/data --output-root /path/to/new-scalability-run --gpu 0 --stage prepare
```

The default `prepare` stage copies only code and resolved configurations. Choose `profiles`, `frozen`, `training`, `summary` or `all` explicitly to execute that existing analysis. `all` runs two full profiles, frozen diagnostics, the 18-configuration training grid and the original summary builder in order. Completed-run guards and the original preflight checks are retained. Every output root must be outside this source package.

Profiles use Ma/GSE140203 RNA+ATAC (K=644) and GSE164378 RNA+ADT (K=500), B=512 and seed 1. Frozen diagnostics reuse saved embeddings and anchors and vary inference B=512/1024/2048/4096 with canonical order plus seeds 1701-1710. Training sensitivity uses D5/D11, B=256/512/1024 and seeds 1-3, followed by common inference B=2048. D5 uses the archived corrected `D8` file paths. Full input paths are resolved from the specified data root.

Frozen/training evaluation requires the checkpoints, embeddings, assignments and annotation H5AD produced by the corresponding existing runs. These files are not distributed here. `run_scalability.py` is a new path/layout launcher. The original pipeline is retained as provenance and is not the portable entry point; its removed synthetic smoke input is not included.

The backend source hashes match the archived server source record. CPU RSS/PSS, PyTorch allocation/reservation and process CUDA samples are separate measurements. Preserve their definitions when interpreting outputs.

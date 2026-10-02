# Earlier instrumented revision workflows

Run from this directory inside the cloned Git repository. `run_revision.py` selects `vendor/phase2_backend` before importing `revision_exp`. This backend supports modality weights, anchor diagnostics and the extended initializer required by the archived driver; binding these workflows to the public baseline would change or break the protocol.

Use the experiment environment described by the repository and the archived workflow. Dependencies include Python, PyTorch/CUDA, Scanpy, AnnData, NumPy, SciPy, pandas, scikit-learn, PyYAML, psutil and FAISS. Some comparator adapters additionally require their named upstream package. No packages are bundled or installed by the launcher.

Prepare a resolved copy without running an experiment:

```bash
python run_revision.py --config revision_exp/configs/modality_full/p2_D18_RNA_ATAC_ADT_seed0.yaml --data-root /path/to/data --output-root /path/to/new-results
```

Inspect the selected filenames under `revision_exp/configs` before using a command: archived config names include retry/diagnostic variants. Add `--execute` to run the existing driver after reviewing its resolved configuration. The original numeric K, epochs, seeds, count perturbations and inference settings are retained. Outputs are written under the specified output root. D5 maps to the dataset files under `RNA+ADT/D8`; D11 maps to `RNA+ATAC/pbmc10k`, as in the corrected source registry.

Selected configurations cover `modality_full`, `noise_full`, `batch_size_full`, `inference_stability_full` and D13/D16 profiles. Smoke, failed/retry and diagnostic variants must be interpreted using their configuration tags; configuration presence is not evidence of completion. D18 count thinning requires verified nonnegative integer raw counts, canonical cell-ID reconciliation and the specified matrix source. Do not replace those inputs with normalized matrices.

The standalone earlier D18 15-factor MOFA and RNA/protein scripts use `GARQ_DATA_ROOT` for the data root and retain their native CLI. These analyses are separate from the later matched 10-factor MOFA protocol. The seven retained tests target existing backend/metric behavior; release preparation did not run them or any experiments.

Unified D13-D16 multibatch MOFA code is not supplied by this subtree. D13/D16 profiling does not establish batch integration. Annotation/assignment files required by optional post-hoc workflows must be supplied independently.

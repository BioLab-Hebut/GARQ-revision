# Optional validation and output metadata tools

The five original core scripts are retained unchanged. These optional tools check inputs before running the original `GARQ.py` and restore missing names into separate output copies. They do not train an alternative model, change batching, reinitialize anchors, or recalculate metacell values.

Use the same Python environment as GARQ. `anndata` and `numpy` are required for H5AD metadata work and are already dependencies of the documented GARQ environment. These tools do not install packages. Run the examples from the repository root.

## Validate and run the unchanged core

```bash
python tools/run_garq.py \
  --data_file datasets/RNA.h5ad datasets/ADT.h5ad \
  --data_type RNA ADT \
  --save_name example \
  --n_GARQs 400 \
  --batch_size 512 \
  --epoch 300 \
  --seed 1 \
  --device cuda
```

All supplied argument strings are passed to the original script in their original order, using the same Python interpreter, environment and working directory. Original optional arguments such as `--type_key`, `--anchers_init`, `--converge_threshold` and `--k_knn` are forwarded unchanged. The original script handles any unknown options. The wrapper returns its exit code.

Preflight opens input files in read-only backed mode to inspect cell/feature metadata; it does not normalize or load the count matrices into dense memory. It requires matching, unique cell identifiers in the same row order, corresponding `--data_file`/`--data_type` lists, at least two anchors, and a valid original initialization sample. With the original `drop_last=True` loader, the initializer sees `min(2, floor(N / B)) * B` cells. Here `B` is the original effective batch size: 4096 when `K > 1000` and the requested batch size is at most 512, otherwise the supplied batch size. The wrapper rejects an empty loader or `K` larger than this sample; it does not change the batch size or collect extra batches.

The wrapper creates `save/` and `figures/` in the current working directory before launching the core. This permits the original plotting calls to save figures when cell-type annotations are absent. It otherwise retains the original workflow, including optional plots and pairwise-correlation evaluation, which can have substantial time and memory requirements.

## Restore names into separate output copies

The original `compute_metacell` creates a new AnnData without feature names and excludes empty anchors. Its row order is the ascending order of occupied original anchor IDs. The original assignment AnnData similarly omits the input cell names. This postprocessing tool restores those metadata fields using the exact original inputs and saved memberships.

```bash
python tools/restore_output_metadata.py \
  --data-file datasets/RNA.h5ad datasets/ADT.h5ad \
  --assignment-file save/example_400metacell_k5_ids.h5ad \
  --metacell-file save/example_RNA_400metacell_k5.h5ad save/example_ADT_400metacell_k5.h5ad
```

Supply the original construction inputs in the same modality order and cell row order used in that run. The tool can check input pairing, shapes and membership values; an output that lost its names does not contain enough information to prove that a different same-sized input belongs to that run. The tool does not match or reorder cells.

The tool writes `*.metadata.h5ad` beside each original output. To use another destination, add `--output-dir save/metadata`. Existing destinations and filename collisions are rejected. Original inputs and outputs are read without modification.

The assignment copy receives the original cell `obs_names`, while its continuous cell embeddings, membership values and their dtype remain unchanged. Each modality's metacell copy receives the original full-feature `var` metadata, an `obs['metacell']` column containing the sorted occupied original anchor IDs, and those IDs as its row names. Existing annotations are retained. Matrices remain the original normalized, log-transformed member means; they are not converted into summed counts or reconstructed profiles.

Before publishing each copy, the tool rereads a temporary H5AD and checks exact X values, dtype and shape using row-chunked SHA256 fingerprints, plus membership values and restored metadata. It publishes with a non-overwriting hard link, so the output destination must support hard links, as ordinary local NTFS/Linux filesystems do. If a later file fails verification, any earlier published copies have already passed their checks; the original files are still intact. Input matrices are never materialized by the metadata reader; each saved output matrix is loaded one at a time for serialization.

## Lightweight checks

```bash
python -m unittest discover -s tools/tests -v
```

The tests use synthetic names, IDs and temporary files. They cover paired row-order rejection, the original batch/initialization bounds, nonconsecutive anchor IDs, destination collision and overwrite protection, exact argument forwarding, and failure before starting the core. A small X fingerprint test runs when NumPy is available. A synthetic H5AD round-trip test runs when AnnData is available and verifies original-file hashes, output matrix values and ID/name mapping. It is explicitly skipped when AnnData is unavailable. No test reads study datasets or trains GARQ.

The original early-stopping implementation is preserved: its quantized reconstruction monitor sums batch losses before averaging over modalities, while its alignment monitor averages across batches. Stable epochs compare to retained reference values that are updated only after an unstable epoch. Changing these statistics or the reference-update rule would change training behavior and is outside these tools.

# Modality construction and count thinning

Generation and evaluation workflows for the modality comparisons in R2Q1.
The recorded conditions are kept together; count thinning changes observed
counts, not modality loss weights.

| Workflow | Entry point | Settings |
|---|---|---|
| Original modality combinations | `run_modality_rerun.py` | D5 K403, D11 K321, kidney K403; D18 K637 with seven combinations |
| Updated kidney comparison | `r2c1.py` | D17 K403; joint, RNA, ATAC and joint ATAC-thin50 |
| D18 seven-combination evaluation | `evaluate_d18_k637.py` | Existing K637 assignments and common clean references |
| D18 count thinning | `run_d18_quality_k637.py`, `evaluate_d18_quality_k637.py` | K637 clean/RNA-thin50/ATAC-thin50/ADT-thin50 quartet |

The D18 quartet uses its own clean baseline. The original and updated kidney
inputs are distinct and are listed separately in `configs/`.

## Requirements

Generation requires Linux, CUDA/PyTorch, NumPy, pandas, SciPy, scikit-learn,
FAISS, einops, anndata, scanpy, h5py, matplotlib, seaborn and alive-progress.
Use compatible CUDA/PyTorch and FAISS builds. Model files are reused from
`../scalability/source_server/`; the model-source helper checks their hashes.

Supply the paired H5AD counts listed in the Python configurations. D18
evaluation also needs the original prepared counts, reference NPZ caches,
membership H5AD files and their run manifests. These data are not included.
Existing run manifests remain JSON and are read without rewriting their
bytes, so their identity checks are preserved.

Set `GARQ_INPUT_ROOT` to the input root. For relocated historical archives,
`GARQ_INPUT_PATH_MAP` may point to a private Python file containing
`CONFIG = {"old_prefix": "/new/prefix"}`. This changes paths in memory and
does not alter hashes or cell identities.

## D17

```bash
export GARQ_INPUT_ROOT=/path/to/inputs-root
python r2c1.py prepare --datasets D17 --config configs/config_D17_K403.py
python r2c1.py run --datasets D17 --config configs/config_D17_K403.py --gpu 0
python r2c1.py evaluate --datasets D17 --config configs/config_D17_K403.py
```

The default output is `results/D17`; use `--root` to select another new
directory. The preparation step copies the model source used by the workers.

## Original modality combinations

```bash
export GARQ_INPUT_ROOT=/path/to/inputs-root
export GARQ_HISTORICAL_WORK=/path/to/new-historical-output
mkdir -p "$GARQ_HISTORICAL_WORK"
cp ../scalability/source_server/{GARQ,model,engine,data_utils,eval_utils}.py "$GARQ_HISTORICAL_WORK/"
python run_modality_rerun.py --prepare
python run_modality_rerun.py --run
```

This runs all 16 recorded combinations with the original GPU scheduling and
resource checks. A private path-only configuration may be selected using
`GARQ_HISTORICAL_CONFIG`. Newly trained H5AD files are new artifacts; the
historical evaluator's expected hashes identify the archived outputs.

## D18 evaluation and count thinning

```bash
export GARQ_REFERENCE_ROOT=/path/to/original-reference-archive
export GARQ_HISTORICAL_OUTPUT=/path/to/original-modality-output
export GARQ_MODALITY_OUTPUT=/path/to/new-d18-output
python evaluate_d18_k637.py --output "$GARQ_MODALITY_OUTPUT/D18_K637_evaluation"
python prepare_source.py --root "$GARQ_MODALITY_OUTPUT"
python run_d18_quality_k637.py --gpu 0
```

The last command evaluates the completed quartet automatically. To evaluate
an existing quartet separately, run `python evaluate_d18_quality_k637.py`.
`prepare_source.py` copies the verified model and extracts the existing
training-through-inference function; it does not train a model. Evaluation
retains the original source, input, assignment and row-identity checks.

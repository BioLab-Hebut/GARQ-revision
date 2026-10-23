# Controlled representations and resolution comparison

Prepare D5 and D11 standard, GARQ-learned and MOFA+ representations and run
the original two-resolution comparison. The separate
[controlled comparison](../representation_benchmark/) evaluates the final
primary-resolution rerun; its evaluator does not accept the secondary grid.

## Requirements

Use the original paired H5AD counts and fixed-feature H5AD files listed in
`configs/preflight.py`. The fixed-feature files contain `obsm['X_fixed']`
and the original modality blocks. Set `GARQ_INPUT_ROOT` to their parent input
directory, or change only the paths in a private copy of the configuration
and set `GARQ_R2C2_PREFLIGHT` to that file.

Required packages are NumPy, pandas, SciPy, scikit-learn, anndata, scanpy,
PyTorch, FAISS, einops, matplotlib, seaborn, alive-progress and mofapy2.
The comparison also needs SEACells and its compatible Palantir/CuPy
dependencies; `GARQ_TRAIN_PYTHON` and `GARQ_SEACELLS_PYTHON` can select
separate training environments.
Learned representations require a compatible CUDA/PyTorch environment.
The model is reused from `../scalability/source_server/`; its five file
hashes are checked before use. An identical copy may be selected with
`GARQ_MODEL_SOURCE`.

## Run

Run from this directory, using a new output directory:

```bash
export GARQ_INPUT_ROOT=/path/to/inputs-root
export GARQ_R2C2_ROOT=/path/to/controlled-inputs
python r2c2_prepare.py
python r2c2_train_learned.py --dataset D5 --seed 1
python r2c2_train_learned.py --dataset D11 --seed 1
python r2c2_mofa.py --dataset D5
python r2c2_mofa.py --dataset D11
```

The standard representation uses PCA to 64 dimensions followed by cell-wise
L2 normalization. GARQ uses the recorded D5 K403 and D11 K321 settings.
MOFA+ uses 64 factors and the recorded scaling and training parameters.
Outputs include `inputs/{D5,D11}/standard.npz`, `MOFA.npz`, and
`learned_training/{D5,D11}_seed1/embedding.npz`.

For the original grid, run:

```bash
python r2c2_controller.py --dataset D5 --gpu 0
python r2c2_controller.py --dataset D11 --gpu 0
python r2c2_evaluate.py --posthoc
python r2c2_geometry.py --dataset D5
python r2c2_geometry.py --dataset D11
```

The grid contains D5 K403/242 and D11 K321/193, with seeds 0/1/2 and all
recorded representation/aggregation combinations. Its evaluator retains
both resolutions. Post-hoc sampling keeps memberships fixed and is distinct
from lowering abundance before rebuilding metacells.

Set `GARQ_CONTROL_INPUT` to the generated output directory (`$GARQ_R2C2_ROOT`) when using the
separate primary-resolution workflow. That evaluator also requires the
corresponding saved sampling and geometry inputs. The fixed-feature H5AD
generation code is not part of this stage; use the archived inputs to
reproduce the intended analysis. The original grid sources are retained;
their presence alone does not establish that every regenerated file is
identical to a final table's archived input.

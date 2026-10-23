# Revision analyses

Run scripts from this folder in the documented NumPy/pandas/SciPy/Scanpy/PyTorch/FAISS or R/Seurat/Slingshot environments. Scientific inputs remain external.

- `code/module_A_v2.py`: presentation of verified saved stability tables.
- `code/module_B_v2.py`: per-modality information retention from saved memberships.
- `code/module_C.py` and `code/module_C4_complete.py`: RNA-protein comparison and saved-link budget analysis.
- `code/module_D_v4.R`: independent gene-trend panel and root-only trajectory analysis.
- `code/module_E.py`: three-arm frozen-input quantizer comparison.

`configs/*.py` are literal scientific settings; `D_analysis_plan.R` is the equivalent R settings list. Edit input locations only. Set `GARQ_INPUT_ROOT`, `GARQ_V2_CONFIG_DIR` if using a private settings copy, and `GARQ_REVISION_OUTPUT` to a new output directory. Original external JSON configurations remain accepted as same-stem alternatives. Runtime JSON summaries remain enabled.

```bash
export GARQ_INPUT_ROOT=/your/inputs
export GARQ_REVISION_OUTPUT=/your/new/results
python code/module_B_v2.py
```

E additionally requires `GARQ_CONTROL_INPUTS`. C requires `GARQ_D18_RAW`, `GARQ_D18_ANALYSIS` and `GARQ_D18_METAQ_ASSIGNMENTS`; D requires `GARQ_TRAJECTORY_INPUTS`. A requires `GARQ_R79_RESULTS`. See script environment-variable readers for optional archived comparisons. Do not substitute different assignments, references or seeds for hash-bound inputs.

The original R79 manifest/probe files remain external and unchanged. `GARQ_LEGACY_PATH_MAP` may relocate their paths without bypassing hashes. The saved-assignment evaluator reuses the byte-identical public `experiments/scalability/scripts/analysis_metrics.py`; E reuses public `model.py`, both checked by SHA256. This directory does not duplicate those sources or bundle cell matrices/checkpoints.

# Kidney trajectory and molecular-program comparison

This workflow compares five saved trajectory orderings on the same original-cell
cohort, using common raw-RNA profiles, the fixed 15-gene panel and three pathway
signatures. It validates memberships before calculating the shared-reference
metrics. It does not refit the five original trajectories.

Set the input paths in `settings.py`. Required inputs are the original RNA matrix,
five memberships, matched metacell RNA exports and archived pseudotime CSVs.
For a relocated SuperCell input, add `input_file` under `ANALYSIS_PLAN["methods"]["SuperCell"]`;
barcode and expression validation remains active.

From this directory, run:

```bash
python prepare_settings.py --output /path/to/new_analysis --input-root /path/to/project_home
python prepare_memberships_v3.py /path/to/new_analysis
Rscript reference_and_programs.R /path/to/new_analysis
Rscript common_programs.R /path/to/new_analysis
Rscript calculate_metrics.R /path/to/new_analysis
```

The first command prepares the runtime plan files; it performs no scientific
calculation. Use a new output directory. The two runtime plans retain the common
normalization and membership definitions used by the analysis.

Python requirements: anndata, NumPy, pandas and SciPy. Recorded R environment:
R 4.4.3, Seurat 5.3.0, slingshot 2.12.0 and UCell 2.8.0, with Matrix, anndata and
jsonlite. Configure the reticulate Python environment for H5AD input. Raw data,
saved trajectories and original cell memberships are external inputs.

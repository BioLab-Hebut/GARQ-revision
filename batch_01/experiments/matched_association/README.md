# Matched construction and association workflows

Sources with `matched_archived_execution` labels match an archived execution-record hash before path sanitization. Other code comes from the final bins60 matched package. Three supplied sources (`atac.ipynb`, `adt.ipynb`, `compare.R`) preserve their source cells/settings; notebook outputs and metadata were removed. New path preparation does not execute them.

The existing membership workflow evaluates seven GARQ modality constructions and the saved SEACells/MetaCell V2/SuperCell partitions using all three paired measured assays. Matched sampling uses size 20, common eligible K and seeds 1-3. The existing downstream driver adapts input/output maps while retaining supplied Pearson/Spearman and glmnet settings. The later MOFA code uses common features and 10 factors/seed 1; it is a separate protocol from the earlier 15-factor phase2 and 64-factor fixed-representation analyses.

Create a JSON path map for the inputs you hold, for example:

```json
{
  "/workspace/garq/data": "/path/to/data",
  "/workspace/garq/vscode/GARQ20260905/save": "/path/to/GARQ-assignments",
  "/workspace/garq/results": "/path/to/comparator-results"
}
```

```bash
python prepare_workflow.py --path-map path-map.json --output-root /path/to/new-matched-run
```

Then inspect the prepared `code/` and run the selected existing script from that output root. `prepare_all_arms.py` reads paired raw count H5AD plus saved membership H5AD, validates ID/label order and writes new aggregation inputs. It creates `analysis/` with `exist_ok=False`, so use a fresh output directory. `evaluate_supplied_workflows.py full` or `matched_seed1`/`matched_seed2`/`matched_seed3` expects the corresponding prepared MOFA input manifest. `mofa_controls.py select` selects common features; fitting arguments are the batch tags. Native Python/R/Signac/MOFA dependencies and archived version metadata are provided in `provenance/`.

Broad peak-gene analysis additionally requires user-supplied `audit/original_peak_metadata.rds` (GC and sequence length), `audit/original_annotation.rds` (gene coordinates), and, for the single-cell arm, the independently archived processed Seurat object. The preparation helper copies only the supplied Signac source helpers into `audit/`; it supplies none of these RDS files. Add the processed-object and R-library neutral prefixes to your path map if that arm is used. The optional CPP/region matching helpers preserve the archived implementation whose pilot equivalence record is included.

The new helper resolves the old September 17 analysis reference to the current runtime `analysis/` and the September 18 root to your output root. Any remaining neutral path must be mapped before the relevant script is executed. `run_bimodal_downstream.py` also offers its original `--raw-root`, `--save-root` and `--output-root` flags. Source notebooks are input to that driver, not a turnkey clinical data package.

This code does not establish the original main Fig6 target/GO selection cutoff. The later matched GO helper's coefficient cutoff 0.1 and the nonzero stability cutoff 1e-6 have different purposes. No raw data, memberships, fitted coefficients, model objects, outputs or clinical records are included.

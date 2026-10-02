"""Prepare the preregistered fixed representation and matched-K KMeans control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from revision_exp.workflows.fixed_representation import run_fixed_representation_benchmark


REGISTRY = {
    "D5": {
        "data_files": ["/workspace/garq/data/RNA+ADT/D8/D8_rna.h5ad", "/workspace/garq/data/RNA+ADT/D8/D8_adt.h5ad"],
        "data_types": ["RNA", "ADT"], "label_key": "celltype", "requested_K": 242,
    },
    "D11": {
        "data_files": ["/workspace/garq/data/RNA+ATAC/pbmc10k/10x-Multiome-Pbmc10k-RNA.h5ad", "/workspace/garq/data/RNA+ATAC/pbmc10k/10x-Multiome-Pbmc10k-ATAC.h5ad"],
        "data_types": ["RNA", "ATAC"], "label_key": "cell_type", "requested_K": 193,
    },
    "D17": {
        "data_files": ["/workspace/garq/data/analysis/new/kidney_rna_updated.h5ad", "/workspace/garq/data/analysis/new/kidney_atac_updated.h5ad"],
        "data_types": ["RNA", "ATAC"], "label_key": "celltype", "requested_K": 323,
    },
    "D18": {
        "data_files": ["/workspace/garq/data/RNA_ATAC_ADT/GSE158013/GSE158013_rna.h5ad", "/workspace/garq/data/RNA_ATAC_ADT/GSE158013/GSE158013_atac.h5ad", "/workspace/garq/data/RNA_ATAC_ADT/GSE158013/GSE158013_adt.h5ad"],
        "data_types": ["RNA", "ATAC", "ADT"], "label_key": "celltype", "requested_K": 510,
        "obs_name_canonicalization": [None, None, {"pattern": r"\.([0-9]+)$", "replacement": r"-\1"}],
    },
    "D13": {
        "data_files": [
            "/workspace/garq/data/RNA+ADT/GSE164378/GSE164378_rna.h5ad",
            "/workspace/garq/data/RNA+ADT/GSE164378/GSE164378_adt.h5ad",
        ],
        "data_types": ["RNA", "ADT"], "label_key": "celltype", "requested_K": 3235,
    },
    "D16": {
        "data_files": [
            "/workspace/garq/data/RNA+ATAC/ma/Ma-2020-RNA.h5ad",
            "/workspace/garq/data/RNA+ATAC/ma/Ma-2020-ATAC.h5ad",
        ],
        "data_types": ["RNA", "ATAC"], "label_key": "cell_type", "requested_K": 645,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(REGISTRY), required=True)
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--phase2-root", type=Path, default=Path("revision_results/phase2"))
    args = parser.parse_args()
    output = args.phase2_root / "01_size_resolution" / args.dataset / "KMeans" / f"full_seed{args.seed}_K002"
    config = {
        "implementation_tag": "diagnostic_fixed_representation",
        "dataset": args.dataset,
        "seed": args.seed,
        "aggregation_methods": ["KMeans"],
        "result_root": "revision_results",
        **REGISTRY[args.dataset],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.json").write_text(json.dumps(config, indent=2) + "\n")
    run_fixed_representation_benchmark(config, output, Path("revision_results"))
    assignments = pd.read_csv(output / "cell_assignments_KMeans.csv")
    assignments["source_dataset_fingerprint"] = "see_phase2_registry_and_representation_provenance"
    assignments["method_version"] = "scikit-learn repository environment"
    assignments["implementation_tag"] = "diagnostic_fixed_representation"
    assignments["resolution"] = config["requested_K"] / len(assignments)
    assignments.to_csv(output / "cell_assignments.csv", index=False)
    pd.read_csv(output / "fixed_representation_benchmark.csv").to_csv(output / "metacell_size_summary.csv", index=False)
    pd.read_csv(output / "fixed_representation_per_type.csv").to_csv(output / "per_type_metrics_long.csv", index=False)


if __name__ == "__main__":
    main()

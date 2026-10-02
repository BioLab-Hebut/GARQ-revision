"""Generate preregistered full-data D5/D11 training batch-size configs."""

from pathlib import Path


TEMPLATE = """config_id: P2_E8_{dataset}_GARQ_batch{batch}_seed{seed}_K002
task: legacy_garq
implementation_tag: instrumented_legacy
dataset: {dataset}
data_files:
  - {file1}
  - {file2}
data_types: [{type1}, {type2}]
matrix_sources: [X, X]
label_key: {label_key}
requested_K: {requested_k}
epochs: 300
batch_size: {batch}
k_knn: 5
seed: {seed}
device: cuda
deterministic: true
anchors_init: Kmeans
converge_threshold: 10
min_available_ram_gb: 96
min_free_gpu_gb: 12
resource_poll_seconds: 0.1
output_subdir: phase2/08_scalability/training_batch_size/{dataset}/batch{batch}_seed{seed}_K002
result_root: revision_results
anchor_dynamics:
  trace: true
  manual_reposition_enabled: true
  reposition_interval: 1
force: false
"""


DATASETS = {
    "D5": dict(
        file1="/workspace/garq/data/RNA+ADT/D8/D8_rna.h5ad",
        file2="/workspace/garq/data/RNA+ADT/D8/D8_adt.h5ad",
        type1="RNA", type2="ADT", label_key="celltype", requested_k=242,
    ),
    "D11": dict(
        file1="/workspace/garq/data/RNA+ATAC/pbmc10k/10x-Multiome-Pbmc10k-RNA.h5ad",
        file2="/workspace/garq/data/RNA+ATAC/pbmc10k/10x-Multiome-Pbmc10k-ATAC.h5ad",
        type1="RNA", type2="ATAC", label_key="cell_type", requested_k=193,
    ),
}


def main() -> None:
    out = Path("revision_exp/configs/batch_size_full")
    out.mkdir(parents=True, exist_ok=True)
    for dataset, values in DATASETS.items():
        for batch in (512, 1024, 2048):
            for seed in (0, 1, 2):
                text = TEMPLATE.format(dataset=dataset, batch=batch, seed=seed, **values)
                (out / f"p2_{dataset}_batch{batch}_seed{seed}.yaml").write_text(text)


if __name__ == "__main__":
    main()

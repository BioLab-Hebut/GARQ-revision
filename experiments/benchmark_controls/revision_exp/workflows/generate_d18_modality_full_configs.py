"""Generate preregistered corrected-full D18 modality-combination configs."""

from pathlib import Path

ROOT = Path("revision_exp/configs/modality_full")
FILES = {
    "RNA": "/workspace/garq/data/RNA_ATAC_ADT/GSE158013/GSE158013_rna.h5ad",
    "ATAC": "/workspace/garq/data/RNA_ATAC_ADT/GSE158013/GSE158013_atac.h5ad",
    "ADT": "/workspace/garq/data/RNA_ATAC_ADT/GSE158013/GSE158013_adt.h5ad",
}
COMBINATIONS = [
    ("RNA",), ("ATAC",), ("ADT",), ("RNA", "ATAC"),
    ("RNA", "ADT"), ("ATAC", "ADT"), ("RNA", "ATAC", "ADT"),
]


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    for modalities in COMBINATIONS:
        name = "_".join(modalities)
        for seed in range(3):
            data_files = "\n".join(f"  - {FILES[m]}" for m in modalities)
            canonical = "\n".join(
                "  - {pattern: '\\.([0-9]+)$', replacement: '-\\1'}" if m == "ADT" else "  - null"
                for m in modalities
            )
            text = f"""config_id: P2_E7_D18_modality_{name}_seed{seed}_K002
task: legacy_garq
implementation_tag: instrumented_legacy
dataset: D18
modality_combination: {name.replace('_', '+')}
data_files:
{data_files}
data_types: [{', '.join(modalities)}]
matrix_sources: [{', '.join('X' for _ in modalities)}]
obs_name_canonicalization:
{canonical}
label_key: celltype
requested_K: 510
epochs: 300
batch_size: 256
k_knn: 5
seed: {seed}
device: cuda
deterministic: true
anchors_init: Kmeans
converge_threshold: 10
min_available_ram_gb: 128
min_free_gpu_gb: 12
resource_poll_seconds: 0.1
output_subdir: phase2/02_modality/full/D18/{name}/seed{seed}
result_root: revision_results
anchor_dynamics:
  trace: true
  manual_reposition_enabled: true
  reposition_interval: 1
force: false
"""
            (ROOT / f"p2_D18_{name}_seed{seed}.yaml").write_text(text)


if __name__ == "__main__":
    main()

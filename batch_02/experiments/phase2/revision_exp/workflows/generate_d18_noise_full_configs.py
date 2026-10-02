"""Generate the preregistered corrected full-data D18 perturbation grid."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


MODALITIES = ("RNA", "ATAC", "ADT")
LEVELS = (0.25, 0.5, 0.75)
SEEDS = (0, 1, 2)


def main() -> None:
    source = Path("revision_exp/configs/modality_full/p2_D18_RNA_ATAC_ADT_seed0.yaml")
    output = Path("revision_exp/configs/noise_full")
    output.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(source.read_text(encoding="utf-8"))
    for seed in SEEDS:
        for modality_index, modality in enumerate(MODALITIES):
            conditions: list[tuple[str, dict]] = [
                (f"thin_{level}", {"kind": "binomial_thinning", "p": level})
                for level in LEVELS
            ] + [("cell_permutation", {"kind": "cell_permutation"})]
            for condition_index, (condition, perturbation) in enumerate(conditions):
                config = deepcopy(base)
                perturbation["seed"] = 8101 + seed * 1000 + modality_index * 100 + condition_index
                perturbations = [None, None, None]
                perturbations[modality_index] = perturbation
                config.update({
                    "config_id": f"P2_E2_D18_noise_{modality}_{condition}_seed{seed}_retry1",
                    "implementation_tag": "diagnostic_variant_modality_noise",
                    "seed": seed,
                    "modality_perturbations": perturbations,
                    "noise_condition": f"{modality}_{condition}",
                    "output_subdir": f"phase2/02_modality/noise_full_retry1/D18/{modality}_{condition}/seed{seed}",
                })
                target = output / f"p2_D18_noise_{modality}_{condition}_seed{seed}.yaml"
                target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()

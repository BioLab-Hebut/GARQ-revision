"""Expand the existing E4 schedule smoke templates to seeds 1--4."""

from pathlib import Path

import yaml


DATASETS = ("D5", "D11", "D17")
VARIANTS = (
    "legacy_continuous",
    "no_manual_reposition",
    "reposition_interval5",
    "reposition_interval10",
)


def main() -> None:
    template_root = Path("revision_exp/configs")
    output_root = template_root / "anchor_multiseed"
    output_root.mkdir(parents=True, exist_ok=True)
    written = 0
    for dataset in DATASETS:
        for variant in VARIANTS:
            template_path = template_root / f"e4_{dataset}_{variant}.yaml"
            template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
            expected_id = f"E4_{dataset}_{variant}_smoke_seed0_v1"
            if template["config_id"] != expected_id or template["seed"] != 0:
                raise ValueError(f"Unexpected seed-0 template: {template_path}")
            for seed in range(1, 5):
                config = dict(template)
                config["config_id"] = expected_id.replace("seed0", f"seed{seed}")
                config["seed"] = seed
                config["output_subdir"] = f"{template['output_subdir']}_seed{seed}"
                output_path = output_root / f"e4_{dataset}_{variant}_seed{seed}.yaml"
                output_path.write_text(
                    yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
                )
                written += 1
    if written != 48:
        raise AssertionError(f"Expected 48 configs, wrote {written}")


if __name__ == "__main__":
    main()

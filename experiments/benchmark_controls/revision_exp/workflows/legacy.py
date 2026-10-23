"""Instrumented execution of the released GARQ numerical workflow."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse

from data_utils import compute_metacell, load_data
from engine import inference, init_gart_anchors, train_one_epoch, warm_one_epoch
from model import GARQ
from revision_exp.metrics.metacell import evaluate_assignments
from revision_exp.utils.provenance import data_fingerprint, write_json
from revision_exp.utils.resource_monitor import ResourceMonitor, enforce_resource_floor
from revision_exp.workflows.stability import run_inference_stability


def _args(config: dict[str, Any], paths: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        data_file=paths,
        data_type=config["data_types"],
        save_name=config["dataset"],
        n_GARQs=int(config["requested_K"]),
        type_key=config.get("label_key", "celltype"),
        anchers_init=config.get("anchors_init", "Kmeans"),
        epoch=int(config["epochs"]),
        batch_size=int(config["batch_size"]),
        converge_threshold=int(config.get("converge_threshold", 10)),
        seed=int(config["seed"]),
        device=config.get("device", "cuda"),
        k_knn=int(config.get("k_knn", 5)),
    )


def _seed(seed: int, deterministic: bool) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(deterministic)
    return {
        "python_seed": seed,
        "numpy_seed": seed,
        "torch_seed": seed,
        "faiss_seed": "released faiss.Kmeans default; not set by legacy code",
        "deterministic_algorithms": deterministic,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }


def _canonicalize_obs_names(names: np.ndarray, rule: dict[str, str] | None) -> np.ndarray:
    names = np.asarray(names, dtype=str)
    if rule is None:
        return names
    canonical = np.asarray(
        [re.sub(rule["pattern"], rule["replacement"], value) for value in names]
    )
    if len(np.unique(canonical)) != len(canonical):
        raise ValueError("obs_name canonicalization produced duplicate identifiers")
    return canonical


def _select_targeted_abundance_indices(
    labels: np.ndarray,
    n_cells: int,
    target_label: str,
    target_abundance: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    labels = np.asarray(labels, dtype=str)
    if not 0 < target_abundance < 1:
        raise ValueError("target_abundance must be strictly between 0 and 1")
    if not 0 < n_cells <= len(labels):
        raise ValueError("n_cells must be in [1, len(labels)]")
    target_pool = np.flatnonzero(labels == target_label)
    other_pool = np.flatnonzero(labels != target_label)
    source_abundance = len(target_pool) / len(labels)
    if target_abundance > source_abundance + 1e-12:
        raise ValueError(
            f"target abundance {target_abundance:g} exceeds source abundance "
            f"{source_abundance:g}; rare-cell upsampling is prohibited"
        )
    target_count = int(round(target_abundance * n_cells))
    other_count = n_cells - target_count
    if target_count < 1:
        raise ValueError("target abundance yields zero target cells")
    if target_count > len(target_pool) or other_count > len(other_pool):
        raise ValueError("requested targeted subset cannot be sampled without replacement")
    rng = np.random.default_rng(seed)
    indices = np.sort(
        np.concatenate(
            [
                rng.choice(target_pool, size=target_count, replace=False),
                rng.choice(other_pool, size=other_count, replace=False),
            ]
        )
    )
    return indices, {
        "sampling": "targeted_abundance_without_replacement",
        "target_label": target_label,
        "target_abundance_requested": target_abundance,
        "target_abundance_realized": target_count / n_cells,
        "target_count": target_count,
        "source_target_count": int(len(target_pool)),
        "source_target_abundance": source_abundance,
    }


def _prepare_uniform_subset(
    source_paths: list[Path],
    cache_dir: Path,
    dataset: str,
    n_cells: int,
    seed: int,
    canonicalization_rules: list[dict[str, str] | None] | None = None,
    matrix_sources: list[str] | None = None,
    perturbations: list[dict[str, Any] | None] | None = None,
    rare_subsampling: dict[str, Any] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    canonicalization_rules = canonicalization_rules or [None] * len(source_paths)
    matrix_sources = matrix_sources or ["X"] * len(source_paths)
    perturbations = perturbations or [None] * len(source_paths)
    if len(canonicalization_rules) != len(source_paths):
        raise ValueError("obs_name_canonicalization must match data_files length")
    if len(matrix_sources) != len(source_paths) or len(perturbations) != len(source_paths):
        raise ValueError("matrix_sources and modality_perturbations must match data_files length")
    first = ad.read_h5ad(source_paths[0], backed="r")
    total = first.n_obs
    first_names = _canonicalize_obs_names(
        first.obs_names.to_numpy().astype(str), canonicalization_rules[0]
    )
    first_labels = None
    if rare_subsampling is not None:
        label_key = rare_subsampling["label_key"]
        if label_key not in first.obs:
            raise KeyError(f"Missing rare-subsampling label column {label_key!r}")
        first_labels = first.obs[label_key].astype(str).to_numpy()
    first.file.close()
    if n_cells >= total and rare_subsampling is None and not any(perturbations) and all(
        source == "X" for source in matrix_sources
    ):
        return [str(path) for path in source_paths], {"subset_applied": False, "n_cells": total}
    rare_info: dict[str, Any] = {}
    sampling_variant = ""
    if rare_subsampling is not None:
        indices, rare_info = _select_targeted_abundance_indices(
            first_labels,
            n_cells,
            str(rare_subsampling["target_label"]),
            float(rare_subsampling["target_abundance"]),
            int(rare_subsampling.get("seed", seed)),
        )
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", rare_info["target_label"]).strip("-")
        if not safe_label:
            raise ValueError("target_label does not yield a safe cache identifier")
        sampling_variant = (
            f"_rare-{safe_label}-abundance{rare_info['target_abundance_requested']:g}"
            f"-seed{int(rare_subsampling.get('seed', seed))}"
        )
    elif n_cells >= total:
        indices = np.arange(total)
    else:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(total, size=n_cells, replace=False))
    selected_names = first_names[indices]
    output_paths = []
    modality_checks = []
    for source, rule, matrix_source, perturbation in zip(
        source_paths, canonicalization_rules, matrix_sources, perturbations
    ):
        variant = sampling_variant
        if matrix_source != "X":
            variant += f"_matrix-{matrix_source}"
        if perturbation:
            kind = perturbation["kind"]
            if kind == "binomial_thinning":
                variant += f"_thin-{float(perturbation['p']):g}-seed{int(perturbation['seed'])}"
            elif kind == "cell_permutation":
                variant += f"_permute-seed{int(perturbation['seed'])}"
            else:
                raise ValueError(f"Unsupported modality perturbation: {kind}")
        if not re.fullmatch(r"[A-Za-z0-9_.-]*", variant):
            raise ValueError(f"Unsafe cache variant: {variant}")
        output = cache_dir / f"{dataset}_{source.stem}_{n_cells}cells_seed{seed}{variant}.h5ad"
        backed = ad.read_h5ad(source, backed="r")
        source_names_original = backed.obs_names.to_numpy().astype(str)
        source_names = _canonicalize_obs_names(source_names_original, rule)
        same_order = bool(np.array_equal(source_names, first_names))
        if not same_order:
            backed.file.close()
            raise ValueError(f"Paired obs_names/order mismatch: {source}")
        needs_write = not output.exists()
        if output.exists():
            cached = ad.read_h5ad(output, backed="r")
            cached_names = cached.obs_names.to_numpy().astype(str)
            cached.file.close()
            needs_write = not np.array_equal(cached_names, source_names[indices])
        if needs_write:
            subset = backed[indices].to_memory()
            subset.obs_names = source_names[indices]
            if matrix_source != "X":
                if matrix_source not in subset.layers:
                    raise KeyError(f"Missing requested layer {matrix_source!r} in {source}")
                subset.X = subset.layers[matrix_source].copy()
            for layer_name in list(subset.layers.keys()):
                del subset.layers[layer_name]
            if perturbation:
                perturb_rng = np.random.default_rng(int(perturbation["seed"]))
                if perturbation["kind"] == "binomial_thinning":
                    probability = float(perturbation["p"])
                    if not 0 <= probability <= 1:
                        raise ValueError("Binomial thinning probability must be in [0, 1]")
                    if sparse.issparse(subset.X):
                        matrix = sparse.csr_matrix(subset.X, copy=True)
                        rounded = np.rint(matrix.data)
                        if np.any(matrix.data < 0) or np.any(np.abs(matrix.data - rounded) > 1e-6):
                            raise ValueError("Count-level thinning requires non-negative integer matrix values")
                        matrix.data = perturb_rng.binomial(rounded.astype(np.int64), probability).astype(matrix.data.dtype)
                        matrix.eliminate_zeros()
                        subset.X = matrix
                    else:
                        matrix = np.asarray(subset.X)
                        rounded = np.rint(matrix)
                        if np.any(matrix < 0) or np.any(np.abs(matrix - rounded) > 1e-6):
                            raise ValueError("Count-level thinning requires non-negative integer matrix values")
                        subset.X = perturb_rng.binomial(rounded.astype(np.int64), probability).astype(matrix.dtype)
                elif perturbation["kind"] == "cell_permutation":
                    permutation = perturb_rng.permutation(subset.n_obs)
                    subset.X = subset.X[permutation].copy()
            subset.write_h5ad(output, compression="gzip")
        backed.file.close()
        output_paths.append(str(output))
        modality_checks.append(
            {
                "source": str(source),
                "canonicalization_rule": rule,
                "names_changed": bool(np.any(source_names != source_names_original)),
                "same_obs_names_and_order": same_order,
                "subset": str(output),
                "matrix_source": matrix_source,
                "perturbation": perturbation,
            }
        )
    return output_paths, {
        "subset_applied": True,
        "sampling": rare_info.get("sampling", "uniform_without_replacement_no_labels_used"),
        "subset_seed": seed,
        "source_n_cells": total,
        "n_cells": n_cells,
        "selected_obs_names_sha256": __import__("hashlib").sha256("\n".join(selected_names).encode()).hexdigest(),
        "modality_checks": modality_checks,
        **rare_info,
    }


def _write_tables(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
        try:
            table.to_parquet(output_dir / f"{name}.parquet", index=False)
        except (ImportError, ModuleNotFoundError):
            pass


def run_legacy_experiment(config: dict[str, Any], output_dir: Path, result_root: Path) -> None:
    allowed_tags = {
        "instrumented_legacy",
        "diagnostic_variant_no_dynamic_update",
        "diagnostic_variant_reposition_interval",
        "diagnostic_variant_modality_noise",
        "diagnostic_variant_modality_weight",
        "diagnostic_variant_rare_subsampling",
    }
    if config["implementation_tag"] not in allowed_tags:
        raise ValueError(f"Unsupported implementation_tag: {config['implementation_tag']}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_paths = [Path(path) for path in config["data_files"]]
    for path in source_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    write_json(output_dir / "data_fingerprints.json", [data_fingerprint(path) for path in source_paths])
    monitor = ResourceMonitor(float(config.get("resource_poll_seconds", 0.1)))
    with monitor.stage("resource_preflight"):
        preflight = enforce_resource_floor(
            float(config.get("min_available_ram_gb", 32)),
            float(config.get("min_free_gpu_gb", 8)),
        )
    write_json(output_dir / "resource_preflight.json", preflight)
    paths = [str(path) for path in source_paths]
    subset_info = {"subset_applied": False}
    needs_materialization = bool(
        config.get("cell_limit")
        or config.get("rare_subsampling")
        or any(config.get("modality_perturbations") or [])
        or any(source != "X" for source in (config.get("matrix_sources") or []))
    )
    if needs_materialization:
        cell_limit = config.get("cell_limit")
        if cell_limit is None:
            first_backed = ad.read_h5ad(source_paths[0], backed="r")
            cell_limit = int(first_backed.n_obs)
            first_backed.file.close()
        with monitor.stage("subset_materialization"):
            paths, subset_info = _prepare_uniform_subset(
                source_paths,
                result_root / "cache" / "subsets",
                config["dataset"],
                int(cell_limit),
                int(config.get("subset_seed", config["seed"])),
                config.get("obs_name_canonicalization"),
                config.get("matrix_sources"),
                config.get("modality_perturbations"),
                config.get("rare_subsampling"),
            )
    write_json(output_dir / "subset_provenance.json", subset_info)
    args = _args(config, paths)
    seed_info = _seed(args.seed, bool(config.get("deterministic", True)))
    write_json(output_dir / "randomness.json", seed_info)
    device = torch.device(args.device)
    with monitor.stage("h5ad_loading_and_legacy_preprocessing"):
        adata_list, dataloader_train, dataloader_eval, input_dims = load_data(args)
    write_json(
        output_dir / "preprocessed_shapes.json",
        {
            "input_dims": input_dims,
            "n_cells": len(dataloader_train.dataset),
            "x_dtypes": [str(x.dtype) for x in dataloader_train.dataset.x_list],
            "raw_dtypes": [str(x.dtype) for x in dataloader_train.dataset.raw_list],
            "train_batch_size_after_legacy_override": args.batch_size,
            "train_drop_last": True,
            "eval_batch_size": args.batch_size * 4,
        },
    )
    model = GARQ(
        input_dims=input_dims,
        data_types=args.data_type,
        entry_num=args.n_GARQs,
        k_knn=args.k_knn,
        modality_weights=config.get("modality_weights"),
    ).to(device)
    anchor_settings = config.get("anchor_dynamics", {})
    model.quantizer.configure_anchor_diagnostics(
        enabled=bool(anchor_settings.get("trace", False)),
        dynamic_update_enabled=bool(anchor_settings.get("manual_reposition_enabled", True)),
        reposition_interval=int(anchor_settings.get("reposition_interval", 1)),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    parameters_before = [name for name, _ in model.named_parameters()]
    with monitor.stage("anchor_initialization_before_warmup"):
        init_gart_anchors(model, args.data_type, dataloader_train, device, args.anchers_init)
    if len(adata_list) == 1:
        model.copy_decoder_q()
    warm_epochs = min(50, int(args.epoch * 0.2))
    history = []
    loss_rec_his = loss_vq_his = 1e7
    stable_epochs = 0
    with monitor.stage("warmup_and_full_training"):
        for epoch in range(args.epoch):
            if epoch < warm_epochs:
                model.quantizer.diagnostic_epoch = epoch
                model.quantizer.diagnostic_phase = "warmup_no_quantization"
                loss_rec = warm_one_epoch(model, args.data_type, dataloader_train, optimizer, epoch, device)
                history.append({"epoch": epoch, "phase": "warmup", "loss_rec": loss_rec, "loss_vq": ""})
            else:
                model.quantizer.diagnostic_epoch = epoch
                model.quantizer.diagnostic_phase = "quantized_training"
                loss_rec, loss_vq = train_one_epoch(model, args.data_type, dataloader_train, optimizer, epoch, device)
                history.append({"epoch": epoch, "phase": "quantized", "loss_rec": loss_rec, "loss_vq": loss_vq})
                converge = abs(loss_vq_his - loss_vq) <= 1e-5 and abs(loss_rec_his - loss_rec) <= 1e-5
                if converge:
                    stable_epochs += 1
                    if stable_epochs >= args.converge_threshold:
                        break
                else:
                    stable_epochs = 0
                    loss_rec_his, loss_vq_his = loss_rec, loss_vq
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    if model.quantizer.diagnostics_enabled:
        pd.DataFrame(model.quantizer.diagnostic_trace).to_csv(
            output_dir / "anchor_dynamics_step.csv", index=False
        )
    with monitor.stage("inference"):
        embeds, ids, delta_confs, rec_q_percent, loss_anchor = inference(
            model, args.data_type, dataloader_eval, device
        )
    np.save(output_dir / "cell_embeddings.npy", embeds.astype(np.float32))
    block_dim = model.quantizer.entry_dim // model.omics_num
    layout = []
    contribution_rows = []
    abs_dot_energy = []
    rng = np.random.default_rng(args.seed + 911)
    pair_count = min(10000, len(embeds) * 5)
    pair_a = rng.integers(0, len(embeds), size=pair_count)
    pair_b = rng.integers(0, len(embeds), size=pair_count)
    for block_index, data_type in enumerate(args.data_type):
        start = block_index * block_dim
        end = start + block_dim
        block = embeds[:, start:end]
        norms = np.linalg.norm(block, axis=1)
        variance_trace = float(np.var(block, axis=0, ddof=1).sum())
        dots = np.sum(block[pair_a] * block[pair_b], axis=1)
        mean_abs_dot = float(np.mean(np.abs(dots)))
        abs_dot_energy.append(mean_abs_dot)
        layout.append({"data_type": data_type, "start": start, "end": end})
        contribution_rows.append(
            {
                "dataset": config["dataset"],
                "seed": args.seed,
                "modality_combination": config.get("modality_combination", "+".join(args.data_type)),
                "data_type": data_type,
                "block_dim": block_dim,
                "l2_norm_mean": float(norms.mean()),
                "l2_norm_median": float(np.median(norms)),
                "l2_norm_p95": float(np.quantile(norms, 0.95)),
                "variance_trace": variance_trace,
                "variance_per_dim_mean": float(variance_trace / block_dim),
                "pairwise_squared_distance_energy": float(2.0 * variance_trace),
                "sampled_pair_mean_abs_dot": mean_abs_dot,
                "normalization_before_concat": False,
                "modality_weight": (
                    model.modality_weights[block_index]
                    if model.modality_weights is not None
                    else 1.0
                ),
            }
        )
    energy_total = sum(abs_dot_energy)
    for row, energy in zip(contribution_rows, abs_dot_energy):
        row["relative_abs_dot_contribution"] = energy / energy_total if energy_total else np.nan
    pd.DataFrame(contribution_rows).to_csv(output_dir / "modality_block_contribution.csv", index=False)
    write_json(output_dir / "embedding_layout.json", {"shape": list(embeds.shape), "blocks": layout, "pair_sample_count": pair_count})
    assignments = pd.DataFrame(
        {
            "cell_id": adata_list[0].obs_names.astype(str),
            "method": "GARQ",
            "dataset": config["dataset"],
            "seed": args.seed,
            "requested_K": args.n_GARQs,
            "realized_K": len(np.unique(ids)),
            "metacell_id": ids.astype(int),
            "delta_confidence": delta_confs,
        }
    )
    if args.type_key in adata_list[0].obs:
        assignments["cell_type"] = adata_list[0].obs[args.type_key].astype(str).to_numpy()
    assignments.to_csv(output_dir / "cell_assignments.csv", index=False)
    with monitor.stage("same_checkpoint_inference_stability"):
        run_inference_stability(
            model=model,
            dataset=dataloader_eval.dataset,
            device=device,
            reference_ids=ids,
            cell_ids=adata_list[0].obs_names.astype(str).to_numpy(),
            cell_types=(
                adata_list[0].obs[args.type_key].astype(str).to_numpy()
                if args.type_key in adata_list[0].obs
                else None
            ),
            config=config,
            output_dir=output_dir,
        )
    with monitor.stage("common_evaluation"):
        tables = evaluate_assignments(
            assignments,
            requested_k=args.n_GARQs,
            dataset=config["dataset"],
            method="GARQ",
            seed=args.seed,
            implementation_tag=config["implementation_tag"],
            resolution=args.n_GARQs / len(assignments),
        )
        _write_tables(tables, output_dir)
    with monitor.stage("metacell_aggregation"):
        scale_evidence = []
        for data_type, adata in zip(args.data_type, adata_list):
            metacell = compute_metacell(adata, ids, args)
            scale_evidence.append(
                {
                    "data_type": data_type,
                    "source_matrix": "adata_ copy after normalize_total+log1p and before HVG selection/scale",
                    "shape": list(metacell.shape),
                    "x_min": float(np.min(metacell.X)),
                    "x_max": float(np.max(metacell.X)),
                    "x_mean": float(np.mean(metacell.X)),
                }
            )
        write_json(output_dir / "metacell_output_scale.json", scale_evidence)
    pd.DataFrame(monitor.records).to_csv(output_dir / "stage_profile.csv", index=False)
    write_json(
        output_dir / "legacy_runtime_summary.json",
        {
            "implementation_tag": config["implementation_tag"],
            "modality_weights": model.modality_weights,
            "parameters_before_first_forward": parameters_before,
            "parameters_after_inference": [name for name, _ in model.named_parameters()],
            "warm_epochs": warm_epochs,
            "epochs_completed": len(history),
            "rec_q_percent": rec_q_percent,
            "loss_anchor": loss_anchor,
            "realized_K": int(len(np.unique(ids))),
            "empty_anchor_count": int(args.n_GARQs - len(np.unique(ids))),
            "anchor_usage": model.quantizer.anchor_usage.detach().cpu().tolist(),
            "anchor_dynamics": {
                "trace_enabled": model.quantizer.diagnostics_enabled,
                "manual_reposition_enabled": model.quantizer.dynamic_update_enabled,
                "reposition_interval": model.quantizer.reposition_interval,
                "quantized_training_steps": model.quantizer._quantized_training_step,
            },
        },
    )

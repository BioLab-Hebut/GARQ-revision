"""Run focused GARQ modality ablations with the user's existing GARQ environment.

Usage: python r2c1.py plan --config config.py
       python r2c1.py run --config config.py --datasets D18 --gpu 0
       python r2c1.py evaluate --config config.py --datasets D18
"""
import argparse
import datetime as dt
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from release_paths import load_json, PACKAGE_ROOT

from r2c1_core import (MODALITIES, align_order, assignment_size, canonical_ids,
                       dump_json, extract_training_module, ids_hash, sha_file,
                       thin_count_values)


def read_json(path):
    return load_json(path)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def modalities(ds):
    return [m for m in MODALITIES if m in ds]


def combinations(ds):
    mods = modalities(ds)
    return [mods] + [list(x) for k in range(1, len(mods)) for x in itertools.combinations(mods, k)]


def batch_plan(config, datasets):
    jobs = [{"dataset": name, "seed": seed, "modalities": combo, "condition": condition,
             "target_k": config["datasets"][name]["expected_k"]}
            for name in datasets for seed in config["seeds"]
            for combo, condition in ([(x, "clean") for x in combinations(config["datasets"][name])]
                                     + [(modalities(config["datasets"][name]), "ATAC_thin50")])]
    # Reuse verified historical runs; this schedules only genuinely missing jobs.
    allowlist = config.get("job_allowlist")
    if allowlist is not None:
        unknown = set(allowlist) - {job_name(j) for j in jobs}
        if unknown:
            raise ValueError(f"Unknown jobs in allowlist: {sorted(unknown)}")
        jobs = [j for j in jobs if job_name(j) in allowlist]
    return jobs


def file_metadata(path):
    stat = Path(path).stat()
    return {"path": str(Path(path).resolve()), "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sha256": sha_file(path)}


def inspect_dataset(config, name):
    import anndata as ad
    import numpy as np
    import pandas as pd
    ds = config["datasets"][name]
    info = {"dataset": name, "checked_at": now(), "modalities": {}, "errors": []}
    ref_ids = None
    reference_labels = None
    for mod in modalities(ds):
        path = ds[mod]
        if not path or not Path(path).is_file():
            info["errors"].append(f"{mod}: exact manuscript input is missing: {path}")
            continue
        a = ad.read_h5ad(path, backed="r")
        try:
            cell_ids = canonical_ids(a.obs_names, ds.get("normalize_terminal_barcode_separator", False))
            if ref_ids is None:
                ref_ids = cell_ids
                if ds["label_key"] in a.obs:
                    reference_labels = pd.Series(a.obs[ds["label_key"]].to_numpy(), index=cell_ids)
            order = align_order(ref_ids, cell_ids)
            entry = {"path": path, "n_cells": a.n_obs, "n_features": a.n_vars,
                     "label_columns": list(a.obs.columns), "canonical_cell_ids_sha256": ids_hash(cell_ids),
                     "reordering_required": bool(np.any(order != np.arange(a.n_obs)))}
            info["modalities"][mod] = entry
            if a.n_obs != ds["expected_n_cells"]:
                info["errors"].append(f"{mod}: {a.n_obs} cells; manuscript requires {ds['expected_n_cells']}")
        except Exception as exc:
            info["errors"].append(f"{mod}: {exc}")
        finally:
            a.file.close()
    if ds.get("labels_csv") and ref_ids is not None:
        labels = pd.read_csv(ds["labels_csv"], dtype=str)
        labels["cell_id"] = canonical_ids(labels["cell_id"], ds.get("normalize_terminal_barcode_separator", False))
        order = align_order(ref_ids, labels["cell_id"])
        reference_labels = pd.Series(labels.iloc[order]["celltype"].to_numpy(), index=ref_ids)
    if reference_labels is None or reference_labels.isna().any() or reference_labels.astype(str).str.strip().eq("").any():
        info["errors"].append("Complete fixed celltype annotations are required for the rare-population comparison")
    else:
        info["label_counts"] = {str(k): int(v) for k, v in reference_labels.value_counts().items()}
        info["annotation_status"] = ds["annotation_status"]
    info["ready"] = not info["errors"]
    return info, ref_ids, reference_labels


def snapshot_code(config, root):
    src = Path(config["code_dir"])
    dest = root / "source"
    files = ("GARQ.py", "model.py", "engine.py", "data_utils.py", "eval_utils.py")
    observed = {f: sha_file(src / f) for f in files}
    if (dest / "source_manifest.json").is_file():
        previous = read_json(dest / "source_manifest.json")
        if previous["sha256"] != observed:
            raise RuntimeError("GARQ source changed. Use a new result_dir to keep experiments comparable")
        return previous
    dest.mkdir(parents=True, exist_ok=True)
    for file in files:
        shutil.copy2(src / file, dest / file)
    transformed, prefix_digest = extract_training_module((dest / "GARQ.py").read_text(encoding="utf-8"))
    (dest / "training_through_inference.py").write_text(transformed, encoding="utf-8")
    try:
        git = subprocess.check_output(["git", "-C", str(src), "rev-parse", "HEAD"], text=True,
                                      stderr=subprocess.DEVNULL).strip()
    except Exception:
        git = None
    result = {"source_dir": str(src), "sha256": observed, "git_commit": git,
              "training_prefix_ast_sha256": prefix_digest,
              "transformed_sha256": sha_file(dest / "training_through_inference.py"),
              "change": "Original main through inference; replace post-inference plotting and aggregation with explicit return",
              "snapshot_at": now()}
    dump_json(dest / "source_manifest.json", result)
    return result


def prepare_dataset(config, name, root):
    import anndata as ad
    import numpy as np
    import pandas as pd
    from scipy import sparse
    ds = config["datasets"][name]
    output = root / "prepared" / name
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        if manifest["dataset_config"] != ds:
            raise RuntimeError(f"{name}: data settings changed; use a new result_dir")
        if (manifest["seeds"] != config["seeds"] or
                manifest["thinning_probability"] != config["thinning_probability"] or
                manifest["perturbation_seed_base"] != config["perturbation_seed_base"]):
            raise RuntimeError(f"{name}: perturbation settings changed; use a new result_dir")
        for entry in manifest["inputs"].values():
            if entry != file_metadata(entry["path"]):
                raise RuntimeError(f"Source input changed: {entry['path']}")
        for filename, digest in manifest["prepared_sha256"].items():
            if sha_file(output / filename) != digest:
                raise RuntimeError(f"Prepared input changed: {filename}")
        return manifest
    info, reference, labels = inspect_dataset(config, name)
    dump_json(output / "input_audit.json", info)
    if not info["ready"]:
        raise RuntimeError("; ".join(info["errors"]))
    pd.DataFrame({"cell_id": reference, "celltype": labels.astype(str).to_numpy()}).to_csv(
        output / "cells.csv", index=False)
    source_info = {}
    for mod in modalities(ds):
        print(f"{name}: prepare {mod}; preserve X values, align by cell ID", flush=True)
        a = ad.read_h5ad(ds[mod])
        cell_ids = canonical_ids(a.obs_names, ds.get("normalize_terminal_barcode_separator", False))
        a = a[align_order(reference, cell_ids)].copy()
        x = sparse.csr_matrix(a.X)
        if not np.isfinite(x.data).all() or (x.data < 0).any():
            raise ValueError(f"{name}/{mod}: X must be finite nonnegative counts, not scaled inputs")
        if not np.allclose(x.data, np.rint(x.data), atol=1e-6, rtol=0):
            raise ValueError(f"{name}/{mod}: X is noninteger. Identify the manuscript count source; do not guess or round")
        if (np.asarray(x.sum(axis=1)).ravel() == 0).any():
            raise ValueError(f"{name}/{mod}: zero-count cells; do not silently filter the paired population")
        obs = pd.DataFrame({"celltype": labels.astype(str).to_numpy()}, index=pd.Index(reference, name="cell_id"))
        clean = ad.AnnData(X=x, obs=obs, var=a.var.copy())
        clean.write_h5ad(output / f"{mod}_clean.h5ad", compression="gzip")
        source_info[mod] = file_metadata(ds[mod])
        if mod == "ATAC":
            for seed in config["seeds"]:
                thinned = x.copy()
                thinned.data = thin_count_values(x.data, config["thinning_probability"],
                                                config["perturbation_seed_base"] + seed).astype(x.dtype)
                thinned.eliminate_zeros()
                zeros = int(np.sum(np.asarray(thinned.sum(axis=1)).ravel() == 0))
                if zeros:
                    raise ValueError(f"ATAC thinning creates {zeros} zero-count cells. Review severity; preserve the cell set")
                ad.AnnData(X=thinned, obs=obs.copy(), var=a.var.copy()).write_h5ad(
                    output / f"ATAC_thin50_seed{seed}.h5ad", compression="gzip")
        del a, clean, x
    if ds.get("labels_csv"):
        source_info["labels_csv"] = file_metadata(ds["labels_csv"])
    prepared_files = {p.name: sha_file(p) for p in output.glob("*.h5ad")}
    manifest = {"dataset_config": ds, "inputs": source_info, "prepared_sha256": prepared_files,
                "cell_ids_sha256": ids_hash(reference), "n_cells": len(reference),
                "seeds": config["seeds"], "thinning_probability": config["thinning_probability"],
                "perturbation_seed_base": config["perturbation_seed_base"],
                "prepared_at": now(), "input_matrix": "X; verified nonnegative integer-valued counts"}
    dump_json(manifest_path, manifest)
    return manifest


def job_name(job):
    return f"{job['dataset']}_{'_'.join(job['modalities'])}_{job['condition']}_seed{job['seed']}"


def worker(args):
    import importlib.util
    import numpy as np
    import pandas as pd
    import torch
    job_path = Path(args.job).resolve()
    job = read_json(job_path)
    run_dir = job_path.parent
    source = Path(job["source_dir"])
    sys.path.insert(0, str(source))
    os.chdir(run_dir)
    random.seed(job["seed"])
    np.random.seed(job["seed"])
    torch.manual_seed(job["seed"])
    torch.random.manual_seed(job["seed"])
    torch.cuda.manual_seed_all(job["seed"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in this environment")
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    spec = importlib.util.spec_from_file_location("garq_r2c1_training", source / "training_through_inference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kw = {key: job[key] for key in ("seed", "epoch", "batch_size", "k_knn", "converge_threshold")}
    kw.update(data_file=job["data_files"], data_type=job["modalities"], save_name=job["name"],
              n_GARQs=job["requested_k"], type_key="celltype", anchers_init="Kmeans", device="cuda")
    embeds, ids, confidence, rec, loss, net, adata_list = module.main(SimpleNamespace(**kw))
    cells = pd.read_csv(job["cells_csv"], dtype=str)
    if len(cells) != len(ids) or not np.isfinite(embeds).all() or not np.isfinite(confidence).all():
        raise RuntimeError("Incomplete or non-finite inference output")
    if not np.isfinite(ids).all() or not np.all(ids == np.rint(ids)) or np.min(ids) < 0 or np.max(ids) >= job["requested_k"]:
        raise RuntimeError("Invalid anchor IDs")
    for a in adata_list:
        if not np.array_equal(cells.cell_id.to_numpy(), a.obs_names.to_numpy()):
            raise RuntimeError("Inference cell order differs from the recorded paired input")
    ids = ids.astype(np.int64)
    cells["metacell"] = ids
    cells.to_csv(run_dir / "assignments.csv.gz", index=False)
    np.savez_compressed(run_dir / "embedding.npz", embedding=embeds,
                        anchors=net.quantizer.anchors.weight.detach().cpu().numpy(),
                        delta_confidence=confidence)
    # Encoding blocks are saved for scale diagnostics; they are not used to build evaluation neighborhoods.
    widths = [getattr(enc, "input_proj").out_features for enc in net.encoders]
    if sum(widths) != embeds.shape[1]:
        raise RuntimeError("Unexpected encoder block sizes")
    norm2 = np.stack([np.sum(block ** 2, axis=1) for block in np.split(embeds, np.cumsum(widths)[:-1], axis=1)], axis=1)
    shares = np.divide(norm2, norm2.sum(axis=1, keepdims=True), out=np.zeros_like(norm2),
                       where=norm2.sum(axis=1, keepdims=True) > 0)
    blocks = pd.DataFrame({"cell_id": cells.cell_id})
    for i, mod in enumerate(job["modalities"]):
        blocks[mod + "_norm"] = np.sqrt(norm2[:, i])
        blocks[mod + "_squared_norm_share"] = shares[:, i]
    blocks.to_csv(run_dir / "embedding_block_norms.csv.gz", index=False)
    size, _, _, _ = assignment_size(ids, job["requested_k"], job["target_k"])
    versions = {}
    for package in ("numpy", "torch", "scanpy", "anndata", "scipy", "scikit-learn", "faiss-gpu"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    result = {**size, "finished_at": now(), "elapsed_seconds": time.time() - start,
              "peak_torch_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
              "gpu_name": torch.cuda.get_device_name(0), "versions": versions,
              "quantized_reconstruction_ratio": float(rec), "anchor_loss": float(loss),
              "cell_ids_sha256": ids_hash(cells.cell_id), "job_sha256": sha_file(job_path)}
    dump_json(run_dir / "result.json", result)


def run_grid(config, names, root, gpu):
    import numpy as np
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "controller.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"A controller may already be running: {lock}. Verify its PID before removing a stale lock")
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        snapshot_code(config, root)
        dump_json(root / "protocol.json", config)
        ready = []
        for name in names:
            try:
                prepare_dataset(config, name, root)
                ready.append(name)
            except Exception as exc:
                dump_json(root / "prepared" / name / "blocked.json", {"error": str(exc), "at": now()})
                print(f"BLOCKED {name}: {exc}", flush=True)
        if not ready:
            raise RuntimeError("No manuscript dataset passed input checks")
        plan = batch_plan(config, ready)
        previous_plan = read_json(root / "run_plan.json") if (root / "run_plan.json").exists() else []
        combined_plan = {job_name(j): j for j in previous_plan + plan}
        dump_json(root / "run_plan.json", list(combined_plan.values()))
        failures = []
        for job in plan:
            name = job_name(job)
            output = root / "runs" / name
            output.mkdir(parents=True, exist_ok=True)
            settings = {**job, **{k: config[k] for k in ("epoch", "batch_size", "k_knn", "converge_threshold")},
                        "source_dir": str((root / "source").resolve()),
                        "source_manifest_sha256": sha_file(root / "source" / "source_manifest.json"),
                        "prepared_manifest_sha256": sha_file(root / "prepared" / job["dataset"] / "manifest.json"),
                        "cells_csv": str((root / "prepared" / job["dataset"] / "cells.csv").resolve()), "name": name}
            if (output / "settings.json").is_file() and read_json(output / "settings.json") != settings:
                raise RuntimeError(f"Run settings changed for {name}; use a new result_dir")
            dump_json(output / "settings.json", settings)
            if (output / "selected.json").is_file():
                print(f"RESUME skip completed {name}", flush=True)
                continue
            requested = job["target_k"]
            rounds = config["max_resolution_attempts"]
            if job["condition"] != "clean":
                clean = {**job, "condition": "clean"}
                selection = root / "runs" / job_name(clean) / "selected.json"
                if not selection.exists():
                    failures.append(name + ": no successful clean full-modality run")
                    continue
                requested = read_json(selection)["requested_k"]
                rounds = 1  # Quality perturbation holds the clean requested K fixed.
            prepared = root / "prepared" / job["dataset"]
            for attempt in range(rounds):
                attempt_dir = output / f"attempt{attempt + 1}_K{requested}"
                attempt_dir.mkdir(parents=True, exist_ok=True)
                data_files = [str((prepared / (f"ATAC_thin50_seed{job['seed']}.h5ad"
                                   if mod == "ATAC" and job["condition"] != "clean" else f"{mod}_clean.h5ad")).resolve())
                              for mod in job["modalities"]]
                resolved = {**settings, "requested_k": requested, "data_files": data_files}
                job_path = attempt_dir / "job.json"
                if job_path.exists() and read_json(job_path) != resolved:
                    raise RuntimeError("Attempt settings conflict with an existing run")
                dump_json(job_path, resolved)
                dump_json(root / "status.json", {"state": "running", "job": name, "attempt": attempt + 1,
                                                 "gpu": gpu, "pid": os.getpid(), "updated_at": now()})
                if not (attempt_dir / "result.json").is_file():
                    print(f"START {name} requested_K={requested} GPU={gpu}", flush=True)
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
                    command = [sys.executable, str(Path(__file__).resolve()), "worker", "--job", str(job_path.resolve())]
                    with open(attempt_dir / "training.log", "a", encoding="utf-8") as log:
                        completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                    if completed.returncode:
                        failures.append(name + f": exit {completed.returncode}; see {attempt_dir / 'training.log'}")
                        dump_json(attempt_dir / "failed.json", {"exit_code": completed.returncode, "at": now()})
                        break
                result = read_json(attempt_dir / "result.json")
                deviation = abs(result["realized_k"] - job["target_k"]) / job["target_k"]
                matched = deviation <= config["realized_k_tolerance"]
                if matched or job["condition"] != "clean" or attempt == rounds - 1:
                    selected = {"attempt_dir": str(attempt_dir.resolve()), "requested_k": requested,
                                "realized_k": result["realized_k"], "target_k": job["target_k"],
                                "within_target_tolerance": matched,
                                "selection_rule": f"First attempt within {100 * config['realized_k_tolerance']:g}% of target K; only realized K is consulted. Failed matching remains flagged."}
                    dump_json(output / "selected.json", selected)
                    print(f"DONE {name} realized_K={result['realized_k']} matched={matched}", flush=True)
                    break
                proposed = int(np.ceil(requested * job["target_k"] / result["realized_k"]))
                requested = max(2, proposed if proposed != requested else requested + (1 if result["realized_k"] < job["target_k"] else -1))
        dump_json(root / "status.json", {"state": "completed" if not failures and len(ready) == len(names) else "partial",
                                         "failures": failures, "ready_datasets": ready,
                                         "blocked_datasets": [n for n in names if n not in ready], "updated_at": now()})
        return ready
    finally:
        lock.unlink(missing_ok=True)


def locate_inputs(roots):
    """Read h5ad headers only; candidates are not automatically treated as D17."""
    import anndata as ad
    count = 0
    for root in roots:
        if not Path(root).is_dir():
            continue
        for path in Path(root).rglob("*.h5ad"):
            if count >= 1500:
                print("Stopped after 1500 candidate headers", flush=True)
                return
            count += 1
            a = None
            try:
                a = ad.read_h5ad(path, backed="r")
                if a.n_obs == 16143:
                    print(json.dumps({"candidate": str(path), "shape": list(a.shape),
                                      "obs_columns": list(a.obs.columns), "var_examples": list(map(str, a.var_names[:3])),
                                      "cell_ids_sha256": ids_hash(a.obs_names)}, ensure_ascii=False), flush=True)
            except Exception as exc:
                print(f"HEADER_READ_FAILED {path}: {type(exc).__name__}", flush=True)
            finally:
                if a is not None:
                    a.file.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "preflight", "prepare", "run", "evaluate", "worker", "locate", "status"])
    parser.add_argument("--config", default=str(PACKAGE_ROOT/"configs/config_D17_K403.py"))
    parser.add_argument("--datasets", default="D17,D18")
    parser.add_argument("--root", help="Override the result directory")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--job")
    parser.add_argument("--search-roots", nargs="+", default=[])
    args = parser.parse_args()
    if args.command == "worker":
        worker(args)
        return
    if args.command == "locate":
        locate_inputs(args.search_roots)
        return
    config = read_json(args.config)
    root = Path(args.root or config["result_dir"]).resolve()
    names = args.datasets.split(",")
    if args.command == "plan":
        plan = batch_plan(config, names)
        print(json.dumps({"base_jobs": len(plan), "jobs": plan}, indent=2))
    elif args.command == "status":
        print(json.dumps(read_json(root / "status.json"), indent=2))
    elif args.command == "preflight":
        for name in names:
            print(json.dumps(inspect_dataset(config, name)[0], indent=2, ensure_ascii=False))
    elif args.command == "prepare":
        snapshot_code(config, root)
        for name in names:
            prepare_dataset(config, name, root)
    elif args.command == "run":
        ready = run_grid(config, names, root, args.gpu)
        from r2c1_evaluate import evaluate
        evaluate(config, ready, root)
    elif args.command == "evaluate":
        from r2c1_evaluate import evaluate
        evaluate(config, names, root)


if __name__ == "__main__":
    main()

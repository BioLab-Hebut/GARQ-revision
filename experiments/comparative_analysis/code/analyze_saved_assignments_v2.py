#!/usr/bin/env python3
"""R78 E1: read saved partitions; write only aggregate biological-stability tables.

No model import, fit, reassignment, raw count matrix, or N-by-N matrix is used.
See ../README.md for the manifest contract and external-input requirements.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import itertools
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid
from settings_io import public_file

# Set limits before importing numerical libraries, also enforced at runtime below.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import h5py
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import adjusted_rand_score, f1_score
from threadpoolctl import threadpool_limits

VERSION = "r78-e1-v2-explicit-saved-row-schema"
HELPER_SHA256 = "e527745450c6950ddd3ac1664a79b33aaccd71708ca103b850daf0c21c620a37"
ROW_SCHEMA_SOURCE_HASHES = {
    "GARQ.py": "49c97de6440cd8ed76eca677a9225214752856461cc27930a0d0a465ae3179ab",
    "profile_runner.py": "14eb221ae146802692df037db1c98bb5139e1650167238c53633773367806623",
}
UNASSIGNED = "__R78_UNASSIGNED__"
TABLES = ("per_run_quality", "population_quality", "pairwise_stability",
          "population_pairwise_stability", "confusion_counts", "local_summary_stability")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(obj):
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def digest_obj(obj):
    return hashlib.sha256(json_bytes(obj)).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def cell_digest(cells):
    return hashlib.sha256(("\n".join(cells) + "\n").encode("utf-8")).hexdigest()


def immutable(path, data):
    path = Path(path)
    if path.exists():
        require(path.read_bytes() == data, f"Existing output conflicts; use a new --out: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(data)
    os.replace(temporary, path)


def status_write(path, obj):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_bytes(json_bytes(obj))
    os.replace(temporary, path)


def resolved(value, base):
    # Optional relocation map changes only file lookup, not hash-bound content.
    value = str(value)
    mapping_file = os.environ.get("GARQ_LEGACY_PATH_MAP")
    if mapping_file:
        mapping = read_json(mapping_file)
        for prefix in sorted(mapping, key=len, reverse=True):
            if value == prefix or value.startswith(prefix.rstrip("/\\") + "/") or value.startswith(prefix.rstrip("/\\") + "\\"):
                value = mapping[prefix] + value[len(prefix):]
                break
    path = Path(os.path.expandvars(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def checked_file(path, expected=None):
    path = Path(path)
    require(path.is_file(), f"Missing input: {path}")
    actual = sha256(path)
    if expected:
        require(actual == expected, f"Input SHA256 differs: {path}")
    return {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}


def decode(values):
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values]


def read_obs_values(item):
    if isinstance(item, h5py.Dataset):
        return decode(item[:])
    require("categories" in item and "codes" in item, "Unsupported saved obs encoding")
    categories = decode(item["categories"][:])
    codes = item["codes"][:]
    require(np.all(codes >= 0), "Missing values in saved obs column")
    return [categories[int(code)] for code in codes]


def read_index(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require({"row", "cell_id", "cell_type"}.issubset(reader.fieldnames or []),
                f"Index requires row/cell_id/cell_type: {path}")
        rows = list(reader)
    cells = [r["cell_id"] for r in rows]
    labels = [r["cell_type"] for r in rows]
    require(bool(cells) and len(set(cells)) == len(cells) and all(cells),
            f"Missing/duplicate cell IDs: {path}")
    require([int(r["row"]) for r in rows] == list(range(len(rows))),
            f"Noncanonical index row column: {path}")
    missing = sum(not x or x.lower() in ("nan", "none", "na") for x in labels)
    require(missing == 0, f"Missing annotations: {missing}/{len(labels)} in {path}; quality not evaluated")
    require(UNASSIGNED not in labels, "Annotation collides with reserved unassigned marker")
    return cells, labels


def read_annotation_metadata(path, label_key):
    """Read only obs; never load X or another expression matrix."""
    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        index_key = obs.attrs.get("_index", "_index")
        if isinstance(index_key, bytes):
            index_key = index_key.decode("utf-8")
        require(index_key in obs, f"Annotation H5AD lacks obs index: {path}")
        cells = decode(obs[index_key][:])
        item = obs[label_key]
        if isinstance(item, h5py.Group) and "categories" in item:
            categories, codes = decode(item["categories"][:]), item["codes"][:]
            annotations = pd.Series(pd.Categorical.from_codes(
                codes, categories=categories, ordered=bool(item.attrs.get("ordered", False))))
            identity = {"categories": categories, "codes": codes.tolist(),
                        "ordered": bool(item.attrs.get("ordered", False))}
        else:
            annotations = pd.Series(decode(item[:]))
            identity = {"string_annotations": annotations.tolist()}
        if "metacell" in obs:
            identity["saved_metacell_values"] = read_obs_values(obs["metacell"])
    require(not annotations.isna().any(), f"Missing H5AD annotations: {path}")
    identity["cell_ids"] = cells
    return cells, annotations, digest_obj(identity)


def validate_annotation_identity(spec, dataset, base, annotation_path, annotation_cells,
                                 annotations, cells, labels, ids):
    """Accept only explicit original-ID or evidence-bound default-row schemas."""
    schema = dataset.get("saved_annotation_schema", {"kind": "original_cell_ids"})
    require(annotations.astype(str).tolist() == labels,
            "H5AD annotations differ from original cell_index rows")
    if schema["kind"] == "original_cell_ids":
        require(annotation_cells == cells,
                "Saved H5AD index differs from original assignment row IDs")
        return {"kind": "original_cell_ids", "ordered_original_ids_equal": True}
    require(schema["kind"] == "default_row_index_with_input_proof", "Unknown saved annotation schema")
    require(dataset["cell_identity"]["kind"] == "row_index_with_input_proof",
            "Default saved rows require explicit row_index_with_input_proof identity")
    canonical_hash = dataset["cell_identity"]["expected_cell_id_sha256"]
    require(cell_digest(cells) == canonical_hash, "This run is not the verified canonical original ID order")
    require(annotation_cells == [str(i) for i in range(len(cells))],
            "Saved H5AD index must be EXACT default strings 0..N-1, not merely same length")
    require(schema.get("metacell_key", "metacell") == "metacell", "Original source writes only obs/metacell")
    with h5py.File(annotation_path, "r") as handle:
        require("metacell" in handle["obs"], "Saved row schema requires original obs/metacell")
        saved_groups = read_obs_values(handle["obs"]["metacell"])
    require(saved_groups == [str(int(value)) for value in ids],
            "Saved H5AD metacell rows differ from hash-bound assignments.npy")
    proof_info = checked_file(resolved(schema["probe_path"], base), schema["probe_sha256"])
    proof = read_json(proof_info["path"])
    require(proof.get("all_pass") is True, "Saved-row identity probe did not pass")
    matching = [row for row in proof["datasets"] if row["dataset"] == dataset["id"]]
    require(len(matching) == 1, "Identity probe has missing/duplicate dataset")
    proof_dataset = matching[0]
    proof_runs = proof_dataset["runs"]
    expected_run_ids = {run["run_id"] for run in dataset["runs"]}
    require(len(proof_runs) == len(expected_run_ids) and {row["run"] for row in proof_runs} == expected_run_ids,
            "Identity probe does not cover the exact fixed run set")
    for row in proof_runs:
        require(row["ids_sha256"] == canonical_hash,
                "Each run must have the raw-input-verified canonical ID digest")
        require(all(row.get(key) is True for key in ("default_row_index", "label_rows_equal", "metacell_rows_equal")),
                "Saved-row identity probe contains an unsuccessful run check")
    raw_proof = proof_dataset["raw_input_ids"]
    expected_raw = set(spec["expected_config"]["data_files"])
    require(set(raw_proof) == expected_raw and len(expected_raw) >= 1, "Probe raw inputs differ from original run config")
    require(all(row.get("ordered_ids_equal") is True and row.get("label_rows_equal") is True
                for row in raw_proof.values()), "Raw input ordered IDs/labels were not verified")
    source_files = schema["source_files"]
    require(set(source_files) == set(ROW_SCHEMA_SOURCE_HASHES), "Both original save-chain source files are required")
    source_evidence = {}
    for name, expected_hash in ROW_SCHEMA_SOURCE_HASHES.items():
        item = source_files[name]
        require(item["sha256"] == expected_hash, "Unrecognized original save-chain source")
        source_evidence[name] = checked_file(resolved(item["path"], base), expected_hash)
    return {"kind": schema["kind"], "default_row_index_exact": True,
            "saved_metacell_rows_equal_hash_bound_assignments": True,
            "saved_annotation_rows_equal_cell_index": True, "canonical_original_id_sha256": canonical_hash,
            "probe": proof_info, "source_files": source_evidence,
            "interpretation": "GARQ creates AnnData(embeds) without original obs_names; profile runner saves original IDs and the same inference assignment vector separately"}


def load_helper(path):
    checked_file(path, HELPER_SHA256)
    spec = importlib.util.spec_from_file_location("r78_archived_analysis_metrics", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    # Only majority_statistics is called; refresh_native_quality is NEVER called.
    return helper


def metric_arrays(ids, annotations, label_set, helper, strict=None):
    """Original tie rule on original row order; unassigned cells count as errors."""
    ids = np.asarray(ids)
    labels = annotations.astype(str).to_numpy()
    assigned = ids >= 0
    predictions = np.full(len(ids), UNASSIGNED, dtype=object)
    group_stats = {"size": np.asarray([], int), "purity": np.asarray([], float)}
    ties = 0
    if assigned.any():
        _, pred, group_stats, ties = helper.majority_statistics(
            ids[assigned], annotations.loc[assigned].reset_index(drop=True))
        predictions[assigned] = pred
    correct = predictions == labels
    counts = {label: int(np.sum(labels == label)) for label in label_set}
    require(set(label_set) == set(labels), "Fixed label set must equal all observed original cohort types")
    populations = []
    strict_recalled = {label: 0 for label in label_set}
    if strict is not None:
        threshold = float(strict["threshold"])
        operator = strict["comparison"]
        require(0 <= threshold <= 1 and operator in (">", ">="), "Invalid prespecified strict-majority threshold")
        require(bool(strict.get("provenance")), "Strict-majority threshold needs provenance")
        for group in np.unique(ids[assigned]):
            mask = ids == group
            for label in label_set:
                n = int(np.sum(mask & (labels == label)))
                fraction = n / int(mask.sum())
                qualifies = fraction > threshold if operator == ">" else fraction >= threshold
                if qualifies:
                    strict_recalled[label] += n
    for label in label_set:
        actual, predicted = labels == label, predictions == label
        tp = int(np.sum(actual & predicted))
        fp = int(np.sum(~actual & predicted))
        fn = int(np.sum(actual & ~predicted))
        population = {"cell_type": label, "n_original_cells": counts[label],
                      "original_fraction": counts[label] / len(ids),
                      "rare_population_le_1pct": counts[label] / len(ids) <= .01,
                      "n_assigned": int(np.sum(actual & assigned)),
                      "assignment_coverage": float(np.mean(assigned[actual])),
                      "true_positive": tp, "false_positive": fp, "false_negative": fn,
                      "dominant_label_recall": tp / counts[label],
                      "precision": tp / (tp + fp) if tp + fp else 0.0,
                      "F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                      "strict_majority_recall": strict_recalled[label] / counts[label] if strict else None,
                      "strict_majority_status": "EVALUATED" if strict else "SKIPPED_NO_PRESPECIFIED_THRESHOLD"}
        populations.append(population)
    sizes = np.asarray(group_stats["size"])
    quality = {"n_cells": len(ids), "n_missing_annotations": 0,
               "n_assigned": int(assigned.sum()), "n_unassigned": int((~assigned).sum()),
               "coverage": float(assigned.mean()), "actual_K": len(sizes),
               "singleton_metacells": int(np.sum(sizes == 1)),
               "cells_in_singletons": int(np.sum(sizes[sizes == 1])),
               "balanced_accuracy": float(np.mean([p["dominant_label_recall"] for p in populations])),
               "macro_F1": float(f1_score(labels, predictions, labels=label_set, average="macro", zero_division=0)),
               "cell_weighted_correctness_full_cohort": float(correct.mean()),
               "cell_weighted_purity_assigned_only": float(correct[assigned].mean()) if assigned.any() else None,
               "mean_metacell_purity_equal_group_weight": float(np.mean(group_stats["purity"])) if len(sizes) else None,
               "majority_tie_groups": ties, "n_cell_types": len(label_set)}
    confusion = []
    for actual in label_set:
        for predicted in label_set + ([UNASSIGNED] if (~assigned).any() else []):
            count = int(np.sum((labels == actual) & (predictions == predicted)))
            if count:
                confusion.append({"true_label": actual, "inherited_label": predicted, "n_cells": count})
    return quality, populations, predictions, confusion


def pair_statistics(ids_a, ids_b, pred_a, pred_b, labels, frozen=False):
    """Never compare raw anchor numbers across independently trained models."""
    assigned_a, assigned_b = ids_a >= 0, ids_b >= 0
    both = assigned_a & assigned_b
    ca, cb = pred_a == labels, pred_b == labels
    result = {"n_cells": len(labels), "n_both_assigned": int(both.sum()),
              "both_assigned_coverage": float(both.mean()),
              "ARI_assignment_shared_assigned": float(adjusted_rand_score(ids_a[both], ids_b[both])) if both.any() else None,
              "ARI_scope": "full_cohort" if both.all() else "shared_assigned_subset_with_coverage",
              "dominant_label_agreement_full_cohort": float(np.mean(both & (pred_a == pred_b))),
              "dominant_label_agreement_both_assigned": float(np.mean(pred_a[both] == pred_b[both])) if both.any() else None,
              "n_dominant_label_changed_both_assigned": int(np.sum(both & (pred_a != pred_b))),
              "dominant_label_changed_full_cohort": float(np.mean(both & (pred_a != pred_b))),
              "correct_both": int(np.sum(ca & cb)), "correct_a_wrong_b": int(np.sum(ca & ~cb)),
              "wrong_a_correct_b": int(np.sum(~ca & cb)), "wrong_both": int(np.sum(~ca & ~cb)),
              "n_exactly_one_unassigned": int(np.sum(assigned_a ^ assigned_b)),
              "n_both_unassigned": int(np.sum(~assigned_a & ~assigned_b)),
              "anchor_comparison_status": "EVALUATED_VERIFIED_FROZEN_STATE" if frozen else "NOT_DEFINED_ACROSS_TRAINING"}
    for name in ("correct_both", "correct_a_wrong_b", "wrong_a_correct_b", "wrong_both"):
        result[name + "_fraction"] = result[name] / len(labels)
    if frozen:
        changed = ids_a != ids_b
        result["anchor_changed_full_cohort"] = float(np.mean(changed))
        result["anchor_changed_label_unchanged_full_cohort"] = float(np.mean(changed & both & (pred_a == pred_b)))
    return result


def loo_means(ids, coordinates, chunk_size=4096):
    """O(N*d + K*d) storage; singleton/unassigned cells remain NaN."""
    n, dimensions = coordinates.shape
    assigned = ids >= 0
    groups, inverse, counts = np.unique(ids[assigned], return_inverse=True, return_counts=True)
    sums = np.zeros((len(groups), dimensions), dtype=np.float64)
    positions = np.flatnonzero(assigned)
    for start in range(0, len(positions), chunk_size):
        pos = positions[start:start + chunk_size]
        values = np.asarray(coordinates[pos], dtype=np.float64)
        require(np.isfinite(values).all(), "Nonfinite fixed reference coordinates")
        np.add.at(sums, inverse[start:start + chunk_size], values)
    means = np.full((n, dimensions), np.nan, dtype=np.float64)
    for start in range(0, len(positions), chunk_size):
        pos = positions[start:start + chunk_size]
        group = inverse[start:start + chunk_size]
        valid = counts[group] > 1
        good_pos, good_group = pos[valid], group[valid]
        means[good_pos] = ((sums[good_group] - np.asarray(coordinates[good_pos], dtype=np.float64)) /
                           (counts[good_group] - 1)[:, None])
    return means


def numeric_summary(values, prefix):
    if not len(values):
        return {prefix + key: None for key in ("mean", "median", "p95", "max")}
    return {prefix + "mean": float(np.mean(values)), prefix + "median": float(np.median(values)),
            prefix + "p95": float(np.quantile(values, .95)), prefix + "max": float(np.max(values))}


def validate_final_quality(spec, dataset, base, quality, cells):
    """Compare against the final categorical-tie summary, never legacy completed BA."""
    reference = spec.get("final_quality_reference")
    if not reference:
        require(not dataset.get("require_final_quality_reference", False),
                "Hash-bound final quality reference required")
        return {"status": "NOT_SUPPLIED"}
    require(reference.get("sha256"), "Final quality reference SHA256 required")
    path = resolved(reference["path"], base)
    evidence = checked_file(path, reference["sha256"])
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [r for r in csv.DictReader(handle)
                if r["dataset"] == dataset["id"] and int(r["seed"]) == spec["seed"]]
    require(len(rows) == 1, "Final summary must contain exactly one dataset/seed row")
    row = rows[0]
    for column, expected in (("assignment_sha256", spec["expected_sha256"]["assignments"]),
                             ("cell_index_sha256", spec["expected_sha256"]["cell_index"]),
                             ("cell_id_sha256", cell_digest(cells))):
        require(row[column] == expected, f"Final summary input identity differs: {column}")
    for column, expected in (("n_cells", quality["n_cells"]),
                             ("requested_K", dataset["requested_K"]),
                             ("actual_K", quality["actual_K"]),
                             ("majority_tie_groups", quality["majority_tie_groups"])):
        require(int(row[column]) == expected, f"Final summary count differs: {column}")
    comparisons = {"balanced_accuracy": "balanced_accuracy",
                   "cell_weighted_purity": "cell_weighted_purity_assigned_only",
                   "mean_metacell_purity": "mean_metacell_purity_equal_group_weight"}
    for column, calculated in comparisons.items():
        require(quality[calculated] is not None and
                abs(float(row[column]) - quality[calculated]) < 1e-12,
                f"Final categorical-tie metric does not reproduce: {column}")
    return {"status": "REPRODUCED", "file": evidence, "dataset": dataset["id"],
            "seed": spec["seed"], "matched_metrics": {key: float(row[key]) for key in comparisons}}


def load_run(spec, dataset, base, helper):
    path = resolved(spec["path"], base)
    paths = {name: resolved(spec.get(name + "_path", str(path / filename)), base)
             for name, filename in (("assignments", "assignments.npy"), ("cell_index", "cell_index.csv"),
                                    ("config", "effective_config.json"), ("completed", "completed.json"),
                                    ("resource", "resource_summary.json"))}
    expected = spec.get("expected_sha256", {})
    require(all(expected.get(key) for key in ("assignments", "cell_index")),
            f"Archived assignment/cell-index hashes required: {spec['run_id']}")
    provenance = {key: checked_file(value, expected.get(key)) for key, value in paths.items()}
    config, completed, resource = (read_json(paths[key]) for key in ("config", "completed", "resource"))
    require(completed.get("status") == "complete" and resource.get("returncode") == 0,
            f"Incomplete run: {path}")
    required_config = {**dataset.get("expected_config", {}), **spec.get("expected_config", {})}
    required_config.update({"K": dataset["requested_K"], "seed": spec["seed"], "label_key": dataset.get("label_key", "celltype")})
    for key, value in required_config.items():
        require(config.get(key) == value, f"Config differs for {spec['run_id']} {key}: {config.get(key)!r} != {value!r}")
    ids = np.load(paths["assignments"], allow_pickle=False)
    require(ids.ndim == 1 and np.issubdtype(ids.dtype, np.integer), f"Invalid partition vector: {path}")
    require(len(ids) == dataset["n_cells"] and len(ids) > 0, f"Partition N differs: {path}")
    require(np.all(ids >= -1) and np.all(ids < dataset["requested_K"]), f"Invalid/unrecognized assignment IDs: {path}")
    cells, labels = read_index(paths["cell_index"])
    require(len(cells) == len(ids), f"Index/partition length differs: {path}")
    if spec.get("annotation_h5ad_path"):
        annotation_path = resolved(spec["annotation_h5ad_path"], base)
    else:
        candidates = list((path / "save").glob("*_ids.h5ad"))
        require(len(candidates) == 1, f"Need exactly one saved annotation H5AD: {path}")
        annotation_path = candidates[0]
    annotation_cells, annotations, obs_digest = read_annotation_metadata(annotation_path, dataset.get("label_key", "celltype"))
    schema_evidence = validate_annotation_identity(spec, dataset, base, annotation_path,
        annotation_cells, annotations, cells, labels, ids)
    categories = list(annotations.cat.categories) if isinstance(annotations.dtype, pd.CategoricalDtype) else None
    provenance["annotations"] = {"path": str(annotation_path), "obs_sha256": obs_digest,
                                  "categories": categories, "scope": "obs only; X not read"}
    provenance["saved_annotation_schema"] = schema_evidence
    if expected.get("annotation_obs"):
        require(obs_digest == expected["annotation_obs"], f"Annotation obs digest differs: {path}")
    label_set = dataset.get("fixed_label_set", list(dict.fromkeys(labels)))
    quality, populations, predictions, confusion = metric_arrays(ids, annotations, label_set, helper, dataset.get("strict_majority"))
    quality["requested_K"] = dataset["requested_K"]
    quality["empty_anchors"] = dataset["requested_K"] - quality["actual_K"]
    if "quality" in completed:
        q = completed["quality"]
        if "K_actual" in q:
            require(int(q["K_actual"]) == quality["actual_K"], f"Saved actual K differs: {path}")
        if "n_cells" in q:
            require(int(q["n_cells"]) == len(ids), f"Saved N differs: {path}")
        provenance["legacy_completed_quality"] = {
            "values": q, "scope": "Historical profile_runner metrics; not the final categorical-tie quality reference"}
    provenance["final_quality_reference"] = validate_final_quality(spec, dataset, base, quality, cells)
    if dataset.get("comparison_kind", "same_setting_training") == "frozen_inference":
        artifacts = spec.get("frozen_artifacts", {})
        require(set(artifacts) == {"checkpoint", "embeddings", "anchors", "internal_state"},
                "Frozen anchor comparisons require checked checkpoint/embedding/anchor/internal-state files")
        provenance["frozen_artifacts"] = {key: checked_file(resolved(value["path"], base), value["sha256"])
                                          for key, value in artifacts.items()}
    return {"spec": spec, "config": config, "ids": ids, "cells": cells, "labels": labels, "categories": categories,
            "predictions": predictions, "quality": quality, "populations": populations,
            "confusion": confusion, "provenance": provenance}


def align_runs(runs, dataset, base):
    reference_cells, reference_labels = runs[0]["cells"], runs[0]["labels"]
    identity = dataset.get("cell_identity", {})
    require(identity.get("kind") in ("original_ids", "row_index_with_input_proof"), "Explicit cell-identity kind required")
    require(identity.get("evidence_path") and identity.get("evidence_sha256"), "Hashed input/order identity evidence required")
    evidence = checked_file(resolved(identity["evidence_path"], base), identity["evidence_sha256"])
    require(identity.get("expected_cell_id_sha256") == cell_digest(reference_cells), "Canonical original-ID digest differs")
    for run in runs:
        require(set(run["cells"]) == set(reference_cells), f"Missing/extra original cell IDs: {run['spec']['run_id']}")
        index = {value: i for i, value in enumerate(run["cells"])}
        order = np.asarray([index[value] for value in reference_cells], dtype=np.int64)
        require([run["labels"][i] for i in order] == reference_labels, "Labels differ after exact-ID alignment")
        require(run["categories"] == runs[0]["categories"], "Saved annotation category order differs across runs")
        run["aligned_ids"] = run["ids"][order]
        run["aligned_predictions"] = run["predictions"][order]
        run["provenance"]["original_cell_id_sha256"] = cell_digest(run["cells"])
        run["provenance"]["aligned_cell_id_sha256"] = cell_digest(reference_cells)
        run["provenance"]["alignment_reordered_rows"] = int(np.sum(order != np.arange(len(order))))
    return reference_cells, np.asarray(reference_labels), evidence


def reference_results(dataset, runs, cells, base):
    rows, provenance = [], []
    references = dataset.get("references", [])
    if not references:
        return [{"dataset": dataset["id"], "status": "SKIPPED_NO_FIXED_INDEPENDENT_REFERENCE",
                 "reason": "No archived independent PCA/LSI/ADT coordinates supplied; no new reference fitted"}], []
    for reference in references:
        require(reference.get("independent_of_runs") is True and bool(reference.get("feature_coordinate_identity")),
                "Fixed reference must have verified independent feature/coordinate identity")
        evidence = checked_file(resolved(reference["provenance_path"], base), reference["provenance_sha256"])
        path = resolved(reference["path"], base)
        info = checked_file(path, reference["sha256"])
        ref_index = resolved(reference["cell_index_path"], base)
        index_info = checked_file(ref_index, reference["cell_index_sha256"])
        ref_cells, _ = read_index(ref_index)
        require(ref_cells == cells, "Fixed reference must use verified canonical original cell order")
        coordinates = np.load(path, mmap_mode="r", allow_pickle=False)
        require(coordinates.ndim == 2 and coordinates.shape[0] == len(cells), "Fixed reference shape differs")
        require(coordinates.shape[1] <= int(reference.get("maximum_dimensions", 256)), "Reference dimension exceeds explicit memory guard")
        require(np.issubdtype(coordinates.dtype, np.number), "Reference coordinates must be numeric")
        estimated = len(runs) * coordinates.shape[0] * coordinates.shape[1] * 8
        require(estimated <= int(reference.get("maximum_loo_bytes", 512 * 1024**2)), "LOO arrays exceed explicit memory guard")
        for start in range(0, len(cells), 4096):
            require(np.isfinite(coordinates[start:start + 4096]).all(), "Fixed reference has nonfinite coordinates")
        means = [loo_means(run["aligned_ids"], coordinates) for run in runs]
        scale = None
        if reference.get("neighbor_scale_path"):
            scale_path = resolved(reference["neighbor_scale_path"], base)
            info["neighbor_scale"] = checked_file(scale_path, reference["neighbor_scale_sha256"])
            require(bool(reference.get("neighbor_scale_definition")), "Saved neighbor-scale definition required")
            scale = np.load(scale_path, mmap_mode="r", allow_pickle=False)
            require(scale.ndim == 1 and len(scale) == len(cells), "Neighbor scale shape differs")
        for a, b in itertools.combinations(range(len(runs)), 2):
            valid = np.isfinite(means[a]).all(1) & np.isfinite(means[b]).all(1)
            distances = np.linalg.norm(means[a][valid] - means[b][valid], axis=1)
            row = {"dataset": dataset["id"], "reference": reference["name"],
                   "run_a": runs[a]["spec"]["run_id"], "run_b": runs[b]["spec"]["run_id"],
                   "status": "EVALUATED" if valid.any() else "NO_ELIGIBLE_NON_SINGLETON_CELLS",
                   "n_cells": len(cells), "n_valid_loo_pairs": int(valid.sum()),
                   "loo_coverage": float(valid.mean()),
                   "n_invalid_a_singleton_or_unassigned": int(np.sum(~np.isfinite(means[a]).all(1))),
                   "n_invalid_b_singleton_or_unassigned": int(np.sum(~np.isfinite(means[b]).all(1))),
                   **numeric_summary(distances, "distance_")}
            if scale is None:
                row["normalized_status"] = "SKIPPED_NO_SAVED_NEIGHBOR_SCALE"
            else:
                good_scale = np.isfinite(scale) & (scale > 0)
                mask = valid & good_scale
                normalized = np.linalg.norm(means[a][mask] - means[b][mask], axis=1) / scale[mask]
                row.update({"normalized_status": "EVALUATED" if mask.any() else "NO_VALID_POSITIVE_NEIGHBOR_SCALE",
                            "n_valid_normalized": int(mask.sum()),
                            "n_nonpositive_or_missing_scale": int(np.sum(~good_scale)),
                            "normalized_coverage": float(mask.mean()), **numeric_summary(normalized, "normalized_distance_")})
            rows.append(row)
        provenance.append({"reference": reference["name"], "coordinates": info, "index": index_info, "evidence": evidence})
        del means, coordinates
    return rows, provenance


def analyze_dataset(dataset, base, helper):
    require(re.fullmatch(r"[A-Za-z0-9_.-]+", dataset["id"]) is not None, "Unsafe dataset identifier")
    require(len(dataset["runs"]) >= 2, "At least two saved runs required per dataset")
    run_ids = [spec["run_id"] for spec in dataset["runs"]]
    require(len(set(run_ids)) == len(run_ids), "Duplicate run identifiers")
    comparison = dataset.get("comparison_kind", "same_setting_training")
    require(comparison in ("same_setting_training", "training_batch", "frozen_inference"), "Unknown comparison protocol")
    require(len({(str(resolved(x["path"], base)), str(x.get("assignments_path", "assignments.npy"))) for x in dataset["runs"]}) == len(run_ids),
            "Duplicate saved assignment sources must not be counted as new runs")
    if comparison == "same_setting_training":
        require(len({x["seed"] for x in dataset["runs"]}) == len(run_ids), "Repeated training requires distinct prespecified seeds")
    if dataset.get("expected_seeds") is not None:
        require(sorted(x["seed"] for x in dataset["runs"]) == sorted(dataset["expected_seeds"]), "Prespecified seed set differs")
    runs = [load_run(spec, dataset, base, helper) for spec in dataset["runs"]]
    if comparison == "same_setting_training":
        required_settings = ("K", "epoch", "train_batch_size", "eval_batch_size", "label_key")
        optional_settings = ("batch_size", "modalities", "train_shuffle", "train_drop_last", "eval_shuffle",
                             "eval_drop_last", "num_workers", "pin_memory", "objective", "initialization_rule")
        for key in required_settings:
            require(all(key in run["config"] for run in runs), f"Missing same-setting protocol field: {key}")
        for key in required_settings + optional_settings:
            if any(key in run["config"] for run in runs):
                require(all(run["config"].get(key) == runs[0]["config"].get(key) for run in runs),
                        f"Same-setting training protocol differs between runs: {key}")
    cells, labels, evidence = align_runs(runs, dataset, base)
    tables = {name: [] for name in TABLES}
    for run in runs:
        prefix = {"dataset": dataset["id"], "run_id": run["spec"]["run_id"], "seed": run["spec"]["seed"],
                  "comparison_kind": comparison}
        tables["per_run_quality"].append({**prefix, **run["quality"]})
        tables["population_quality"].extend({**prefix, **row} for row in run["populations"])
        tables["confusion_counts"].extend({**prefix, **row} for row in run["confusion"])
    for a, b in itertools.combinations(runs, 2):
        if comparison == "frozen_inference":
            require({k: v["sha256"] for k, v in a["provenance"]["frozen_artifacts"].items()} ==
                    {k: v["sha256"] for k, v in b["provenance"]["frozen_artifacts"].items()},
                    "Frozen state differs; anchor-ID change cannot be interpreted")
        prefix = {"dataset": dataset["id"], "run_a": a["spec"]["run_id"], "run_b": b["spec"]["run_id"],
                  "seed_a": a["spec"]["seed"], "seed_b": b["spec"]["seed"], "comparison_kind": comparison}
        arrays = (a["aligned_ids"], b["aligned_ids"], a["aligned_predictions"], b["aligned_predictions"])
        tables["pairwise_stability"].append({**prefix, **pair_statistics(*arrays, labels, comparison == "frozen_inference")})
        for label in dict.fromkeys(labels):
            mask = labels == label
            tables["population_pairwise_stability"].append({**prefix, "cell_type": label,
                "original_fraction": float(mask.mean()), "rare_population_le_1pct": float(mask.mean()) <= .01,
                **pair_statistics(*(x[mask] for x in arrays), labels[mask], comparison == "frozen_inference")})
    tables["local_summary_stability"], refs = reference_results(dataset, runs, cells, base)
    provenance = {"input_identity_evidence": evidence, "canonical_cell_id_sha256": cell_digest(cells),
                  "runs": {run["spec"]["run_id"]: run["provenance"] for run in runs}, "references": refs}
    return {"dataset": dataset["id"], "status": "COMPLETE", "tables": tables, "provenance": provenance,
            "scientific_scope": "Saved partitions only; inherited labels are descriptive in-cohort annotations, not held-out prediction"}


def tsv_bytes(rows):
    if not rows:
        return b""
    fields = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows({key: "" if value is None else value for key, value in row.items()} for row in rows)
    return stream.getvalue().encode("utf-8")


def verify_cached_dataset(path, marker, dataset):
    """Revalidate exact saved inputs before reusing a completed aggregate stage."""
    complete = read_json(marker)
    require(sha256(path) == complete["result_sha256"], f"Cached aggregate was changed: {path}")
    result = read_json(path)
    require(result["dataset"] == dataset["id"] and result["status"] == "COMPLETE", "Invalid cached stage identity")

    def visit(value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                checked_file(value["path"], value["sha256"])
            if "path" in value and "obs_sha256" in value:
                _, _, actual = read_annotation_metadata(value["path"], dataset.get("label_key", "celltype"))
                require(actual == value["obs_sha256"], f"Cached annotation identity changed: {value['path']}")
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(result["provenance"])
    return result


def run_manifest(manifest_path, out, threads=1):
    require(1 <= threads <= 8, "Thread limit must be between 1 and 8")
    manifest_path, out = Path(manifest_path).resolve(), Path(out).resolve()
    manifest, base = read_json(manifest_path), manifest_path.parent
    require(manifest.get("schema_version") == 1, "Manifest schema_version must be 1")
    require(manifest.get("analysis_metrics_sha256") == HELPER_SHA256, "Use the hash-bound original analysis helper")
    datasets = manifest["datasets"]
    require(len({x["id"] for x in datasets}) == len(datasets), "Duplicate dataset identifiers")
    require(len(datasets) == manifest["expected_datasets"], "Dataset count differs from fixed manifest expectation")
    require(sum(len(x["runs"]) for x in datasets) == manifest["expected_runs"], "Run count differs from fixed manifest expectation")
    # Prevent accidental writes anywhere inside an input run tree.
    for dataset in datasets:
        for spec in dataset["runs"]:
            path = resolved(spec["path"], base)
            require(out != path and path not in out.parents, f"--out must not be inside an original run: {path}")
    helper_path = public_file("experiments/scalability/scripts/analysis_metrics.py", HELPER_SHA256)
    helper = load_helper(helper_path)
    identity = {"version": VERSION, "script_sha256": sha256(__file__), "manifest_sha256": sha256(manifest_path),
                "helper_sha256": sha256(helper_path), "threads": threads,
                "versions": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
                             "h5py": h5py.__version__, "sklearn": sklearn.__version__}}
    out.mkdir(parents=True, exist_ok=True)
    immutable(out / "analysis_identity.json", json_bytes(identity))
    lock = out / ".analysis.lock"
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        raise RuntimeError(f"Existing analyzer lock; verify the recorded process before removing: {lock}")
    with os.fdopen(fd, "w") as handle:
        handle.write(json.dumps({"pid": os.getpid(), "started_utc_epoch": time.time()}))
    status = {"status": "RUNNING", "completed_datasets": [], "failed_datasets": [], "identity": identity}
    try:
        with threadpool_limits(limits=threads):
            for dataset in datasets:
                status["active_dataset"] = dataset["id"]
                status_write(out / "status.json", status)
                print(f"ANALYZE {dataset['id']} saved_runs={len(dataset['runs'])}", flush=True)
                try:
                    dest = out / "datasets" / (dataset["id"] + ".json")
                    marker = out / "datasets" / (dataset["id"] + ".complete.json")
                    if marker.exists():
                        verify_cached_dataset(dest, marker, dataset)
                        print(f"REUSED {dataset['id']} input hashes and obs identity match", flush=True)
                    else:
                        result = analyze_dataset(dataset, base, helper)
                        immutable(dest, json_bytes(result))
                        immutable(marker, json_bytes({"dataset": dataset["id"], "result_sha256": sha256(dest)}))
                    status["completed_datasets"].append(dataset["id"])
                except Exception as error:
                    status["failed_datasets"].append({"dataset": dataset["id"], "error": str(error)})
                    print(f"FAILED {dataset['id']}: {error}", file=sys.stderr, flush=True)
                status_write(out / "status.json", status)
        require(not status["failed_datasets"], "One or more datasets failed; see status.json; completed datasets preserved")
        results = [read_json(out / "datasets" / (dataset["id"] + ".json")) for dataset in datasets]
        outputs = {}
        for name in TABLES:
            rows = [row for result in results for row in result["tables"][name]]
            path = out / (name + ".tsv")
            immutable(path, tsv_bytes(rows))
            outputs[path.name] = {"sha256": sha256(path), "rows": len(rows)}
        complete = {"status": "COMPLETE", "version": VERSION, "datasets": len(datasets),
                    "runs": manifest["expected_runs"], "new_model_fits": 0, "new_assignments": 0,
                    "formal_manuscripts_modified": 0, "identity": identity, "outputs": outputs,
                    "rare_definition": "fixed original full-cohort annotated abundance <= 1%",
                    "tie_definition": "archived annotations[mask].value_counts().idxmax(); original category and row order",
                    "macro_F1_definition": "per-type F1, fixed full-cohort observed type set, zero_division=0",
                    "strict_majority": "only evaluated if an explicit threshold/comparison/provenance is supplied",
                    "limitations": ["Pairwise comparisons share runs; pair counts are not independent experiments",
                        "Inherited labels use the same original annotations for group labeling and evaluation",
                        "No cross-training raw anchor-ID comparisons or new reference fits",
                        "Local summary is skipped when no fixed independent reference is supplied"]}
        immutable(out / "COMPLETE.json", json_bytes(complete))
        status["status"] = "COMPLETE"
        status.pop("active_dataset", None)
        status_write(out / "status.json", status)
        print(f"COMPLETE datasets={len(datasets)} runs={manifest['expected_runs']} new_fits=0", flush=True)
        return complete
    except Exception:
        status["status"] = "PARTIAL" if status["completed_datasets"] else "FAILED"
        status_write(out / "status.json", status)
        raise
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    try:
        run_manifest(args.manifest, args.out, args.threads)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

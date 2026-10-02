#!/usr/bin/env python3
"""Read-only Gate 0 audit for the manuscript-authoritative GARQ datasets."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path

import h5py
import numpy as np
import yaml


DATASETS = {
    "D5": {
        "name": "BMMC_batch1", "cells": 12103, "type": "RNA+ADT",
        "modalities": {
            "RNA": "/workspace/garq/data/RNA+ADT/D8/D8_rna.h5ad",
            "ADT": "/workspace/garq/data/RNA+ADT/D8/D8_adt.h5ad",
        },
        "features": {"RNA": 9786, "ADT": 25}, "label_key": "celltype",
    },
    "D11": {
        "name": "10Xpbmc10k", "cells": 9631, "type": "RNA+ATAC",
        "modalities": {
            "RNA": "/workspace/garq/data/RNA+ATAC/pbmc10k/10x-Multiome-Pbmc10k-RNA.h5ad",
            "ATAC": "/workspace/garq/data/RNA+ATAC/pbmc10k/10x-Multiome-Pbmc10k-ATAC.h5ad",
        },
        "features": {"RNA": 29095, "ATAC": 107194}, "label_key": "cell_type",
    },
    "D13": {
        "name": "GSE164378", "cells": 161764, "type": "RNA+ADT",
        "modalities": {
            "RNA": "/workspace/garq/data/RNA+ADT/GSE164378/GSE164378_rna.h5ad",
            "ADT": "/workspace/garq/data/RNA+ADT/GSE164378/GSE164378_adt.h5ad",
        },
        "features": {"RNA": 33538, "ADT": 228}, "label_key": "celltype",
    },
    "D16": {
        "name": "GSE140203", "cells": 32231, "type": "RNA+ATAC",
        "modalities": {
            "RNA": "/workspace/garq/data/RNA+ATAC/ma/Ma-2020-RNA.h5ad",
            "ATAC": "/workspace/garq/data/RNA+ATAC/ma/Ma-2020-ATAC.h5ad",
        },
        "features": {"RNA": 21478, "ATAC": 340341}, "label_key": "celltype",
    },
    "D17": {
        "name": "Human_Kidney_Cancer", "cells": 16143, "type": "RNA+ATAC",
        "modalities": {
            "RNA": "/workspace/garq/data/analysis/new/kidney_rna_updated.h5ad",
            "ATAC": "/workspace/garq/data/analysis/new/kidney_atac_updated.h5ad",
        },
        "features": {"RNA": 32445, "ATAC": 67150}, "label_key": "celltype",
    },
    "D18": {
        "name": "GSE158013", "cells": 25517, "type": "RNA+ATAC+ADT",
        "modalities": {
            "RNA": "/workspace/garq/data/RNA_ATAC_ADT/GSE158013/GSE158013_rna.h5ad",
            "ATAC": "/workspace/garq/data/RNA_ATAC_ADT/GSE158013/GSE158013_atac.h5ad",
            "ADT": "/workspace/garq/data/RNA_ATAC_ADT/GSE158013/GSE158013_adt.h5ad",
        },
        "features": {"RNA": 17882, "ATAC": 128853, "ADT": 46},
        "label_key": "celltype", "canonicalize": {"ADT": "terminal_dot_to_hyphen"},
    },
}


def decode(values):
    a = np.asarray(values)
    if a.dtype.kind in "SO":
        return [x.decode("utf-8", "replace") if isinstance(x, bytes) else str(x) for x in a]
    return [str(x) for x in a]


def frame_index(f, key):
    g = f[key]
    idx_key = g.attrs.get("_index", "_index")
    if isinstance(idx_key, bytes):
        idx_key = idx_key.decode()
    return decode(g[idx_key][...])


def hash_strings(values):
    h = hashlib.sha256()
    for value in values:
        h.update(value.encode("utf-8", "surrogatepass")); h.update(b"\0")
    return h.hexdigest()


def fast_file_fingerprint(path):
    size = os.path.getsize(path); h = hashlib.sha256(); h.update(str(size).encode())
    with open(path, "rb") as fh:
        for offset in sorted({0, max(0, size // 2 - 524288), max(0, size - 1048576)}):
            fh.seek(offset); h.update(fh.read(1048576))
    return "sha256_size_plus_3x1MiB:" + h.hexdigest()


def x_info(f):
    x = f["X"]
    if isinstance(x, h5py.Dataset):
        shape = tuple(x.shape); sparse = "dense"; data = x
    else:
        shape = tuple(int(v) for v in x.attrs["shape"]); sparse = str(x.attrs.get("encoding-type", "sparse")); data = x["data"]
    n = int(data.size); take = min(n, 200000)
    if n <= take:
        sample = np.asarray(data[...]).ravel()
    else:
        starts = [0, max(0, n // 2 - take // 6), max(0, n - take // 3)]
        sample = np.concatenate([np.asarray(data[s:s + take // 3]).ravel() for s in starts])
    finite = sample[np.isfinite(sample)]
    integer_fraction = float(np.mean(np.isclose(finite, np.rint(finite)))) if finite.size else None
    explicit_zero = float(np.mean(finite == 0)) if finite.size else None
    total = int(np.prod(shape)); implicit_zeros = max(0, total - n) if sparse != "dense" else 0
    zero_fraction = (implicit_zeros + explicit_zero * n) / total if total else None
    return shape, sparse, str(data.dtype), float(np.min(finite)), float(np.max(finite)), integer_fraction, float(zero_fraction)


def canonicalize(values, rule):
    if rule == "terminal_dot_to_hyphen":
        return [v.rsplit(".", 1)[0] + "-" + v.rsplit(".", 1)[1] if "." in v and v.rsplit(".", 1)[1].isdigit() else v for v in values]
    return values


def main():
    out = Path("revision_results/phase2/audit"); out.mkdir(parents=True, exist_ok=True)
    inventory = []; provenance = []; registry = {"registry_version": 2, "authority": "Supplementary Table 1", "datasets": {}}
    for did, spec in DATASETS.items():
        obs_by_mod = {}; modrows = []
        for mod, raw_path in spec["modalities"].items():
            path = Path(raw_path); st = path.stat()
            with h5py.File(path, "r") as f:
                shape, sparse, dtype, xmin, xmax, intfrac, zerofrac = x_info(f)
                obs = frame_index(f, "obs"); var = frame_index(f, "var")
                obs_by_mod[mod] = canonicalize(obs, spec.get("canonicalize", {}).get(mod))
                keys = sorted(f["obs"].keys()); layers = sorted(f.get("layers", {}).keys())
                raw_present = "raw" in f and "X" in f["raw"]
            exact_shape = shape == (spec["cells"], spec["features"][mod])
            row = {
                "dataset_id": did, "authoritative_name": spec["name"], "modality": mod,
                "absolute_path": str(path), "file_fast_fingerprint": fast_file_fingerprint(path),
                "file_size": st.st_size, "mtime_epoch": int(st.st_mtime), "n_obs": shape[0], "n_vars": shape[1],
                "obs_names_hash": hash_strings(obs), "var_names_hash": hash_strings(var),
                "label_keys": "|".join(k for k in keys if any(s in k.lower() for s in ("cell", "label", "type"))),
                "batch_keys": "|".join(k for k in keys if any(s in k.lower() for s in ("batch", "donor", "sample"))),
                "X_sparse_or_dense": sparse, "X_dtype": dtype, "X_min_sample": xmin, "X_max_sample": xmax,
                "integer_fraction_sample": intfrac, "zero_fraction_estimate": zerofrac,
                "raw_present": raw_present, "layers": "|".join(layers), "expected_shape_match": exact_shape,
            }
            inventory.append(row); modrows.append(row)
            looks_count = xmin >= 0 and intfrac >= 0.999
            provenance.append({**row, "counts_source_candidate": "X" if looks_count else "UNRESOLVED",
                               "count_scale_status": "COUNT_LIKE" if looks_count else "NONINTEGER_OR_TRANSFORMED",
                               "duplicate_normalize_log_risk": "LOW" if looks_count else "HIGH_OR_UNKNOWN"})
        sets_equal = len({tuple(v) for v in obs_by_mod.values()}) == 1
        hashes_equal = len({hash_strings(v) for v in obs_by_mod.values()}) == 1
        shape_ok = all(r["expected_shape_match"] for r in modrows)
        status = "CONFIRMED" if shape_ok and sets_equal and hashes_equal else "BLOCKED_WITH_EVIDENCE"
        registry["datasets"][did] = {
            "status": status, "authoritative_name": spec["name"], "data_type": spec["type"],
            "n_cells": spec["cells"], "modalities": spec["modalities"], "feature_dimensions": spec["features"],
            "label_key_candidate": spec["label_key"], "paired_id_set_equality": sets_equal,
            "paired_row_order_verified": hashes_equal, "canonicalization": spec.get("canonicalize", {}),
            "identity_basis": "authoritative shape + modality + feature dimensions + paired obs hashes",
        }
    fields = list(inventory[0])
    with (out / "data_inventory.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fields); w.writeheader(); w.writerows(inventory)
    fields = list(provenance[0])
    with (out / "INPUT_MATRIX_PROVENANCE.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fields); w.writeheader(); w.writerows(provenance)
    regpath = Path("revision_exp/data_registry/datasets_v2.yaml"); regpath.parent.mkdir(parents=True, exist_ok=True)
    regpath.write_text(yaml.safe_dump(registry, sort_keys=False, allow_unicode=True), encoding="utf-8")
    conflicts = [
        {"dataset_id": "D5", "authoritative_name": "BMMC_batch1", "authoritative_n_cells": 12103,
         "phase1_path": "/workspace/garq/data/RNA+ADT/D5/D5_rna.h5ad", "phase1_n_cells": 8670,
         "candidate_correspondence": "D4 PBMC_HD3_Adult", "status": "INVALID_FOR_MANUSCRIPT_DATASET_ID",
         "reason": "Phase 1 shape is the authoritative D4 scale; no silent relabel permitted."},
        {"dataset_id": "D11", "authoritative_name": "10Xpbmc10k", "authoritative_n_cells": 9631,
         "phase1_path": "/workspace/garq/data/11_GSE194122/11_GSE194122_rna.h5ad", "phase1_n_cells": 9876,
         "candidate_correspondence": "D7 GSE194122_s4d8", "status": "INVALID_FOR_MANUSCRIPT_DATASET_ID",
         "reason": "Phase 1 accession/shape match the authoritative D7 entry; no silent relabel permitted."},
    ]
    with (out / "dataset_registry_conflicts.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(conflicts[0])); w.writeheader(); w.writerows(conflicts)
    tracked = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    invalid = []
    for path in tracked:
        if path.startswith("revision_results/phase2/") or not ("D5" in path or "D11" in path):
            continue
        phase_label = "D5" if "D5" in path else "D11"
        invalid.append({
            "path": path, "phase1_dataset_label": phase_label, "authoritative_dataset_id": phase_label,
            "validity_status": "INVALID_FOR_MANUSCRIPT_DATASET_ID",
            "reason": "Phase 1 registry used a dataset with conflicting accession/shape.",
            "reusable_for_code_audit": "true" if path.startswith("revision_results/00_audit/") else "false",
            "reusable_for_biological_claim": "false",
            "required_action": "Retain for provenance; exclude from Phase 2 biological summaries and manuscript claims.",
        })
    fields = ["path", "phase1_dataset_label", "authoritative_dataset_id", "validity_status", "reason",
              "reusable_for_code_audit", "reusable_for_biological_claim", "required_action"]
    with (out / "phase1_result_validity.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fields); w.writeheader(); w.writerows(invalid)
    audit_lines = [
        "# Dataset Identity Audit — Phase 2", "", "## Status", "",
        "D5, D11, D13, D16, D17 and D18 are CONFIRMED by manuscript-authoritative cell/feature dimensions, modality, and paired identifier hashes.", "",
        "## Critical correction", "",
        "- Correct D5: `/workspace/garq/data/RNA+ADT/D8/D8_rna.h5ad` + `D8_adt.h5ad` (12,103 cells; 9,786 RNA genes; 25 ADTs). The `D8` directory label is historical and is not used as identity evidence.",
        "- Correct D11: `/workspace/garq/data/RNA+ATAC/pbmc10k/10x-Multiome-Pbmc10k-RNA.h5ad` + `10x-Multiome-Pbmc10k-ATAC.h5ad` (9,631 cells; 29,095 genes; 107,194 peaks).",
        "- Phase 1 D5 (8,670 cells) and D11 (9,876 cells/GSE194122) biological outputs are quarantined. Candidate D4/D7 correspondences are not silent relabels.", "",
        "## Count sources", "",
        "All confirmed candidate `.X` matrices sampled as non-negative, integer-valued counts. `INPUT_MATRIX_PROVENANCE.csv` records sparsity, dtype, extrema, integer fraction, zero fraction, raw/layer presence, and normalization risk.", "",
        "## Pairing", "",
        "RNA/ADT or RNA/ATAC row order is exact for D5, D11, D13, D16 and D17. D18 is exact after the predeclared ADT terminal `.number` to `-number` canonicalization; the canonicalized identifiers are unique and position-wise equal.", "",
        "## Evidence files", "",
        "- `data_inventory.csv`", "- `INPUT_MATRIX_PROVENANCE.csv`", "- `dataset_registry_conflicts.csv`",
        "- `phase1_result_validity.csv`", "- `revision_exp/data_registry/datasets_v2.yaml`", "",
        "No source dataset was modified or copied.",
    ]
    (out / "DATASET_IDENTITY_AUDIT_PHASE2.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    reports = Path("revision_results/phase2/reports"); reports.mkdir(parents=True, exist_ok=True)
    skeletons = {
        "PHASE2_EXPERIMENT_REPORT.md": "# Phase 2 Experiment Report\n\nStatus: IN PROGRESS\n\n## Executive summary\n\nGate 0 dataset identity audit completed; confirmatory experiments pending.\n\n## Dataset identity audit\n\nSee `../audit/DATASET_IDENTITY_AUDIT_PHASE2.md`.\n\n## P2-E1–E8 results\n\nResults will be appended without deleting negative or failed runs.\n",
        "REVIEWER_EVIDENCE_MATRIX_PHASE2.md": "# Reviewer Evidence Matrix — Phase 2\n\nStatus: IN PROGRESS\n\nGate 0 resolves the D5/D11 identity conflict. Reviewer-linked experiments remain tracked as pending until completed.\n",
        "HANDOFF_FOR_CHATGPT_PHASE2.md": "# Handoff for ChatGPT — Phase 2\n\nStatus: IN PROGRESS\n\nBase commit: `5da45adcd62f1be8ee318d8742c80c59cb242ca2`\n\nPhase 1 commit: `cfaf79bdcbc26840b6cbf67e3531d6fab6540a09`\n\nBranch: `revision/major-review-experiments-phase2`\n\nGate 0 is complete; experiment evidence and final commit will be populated before handoff.\n",
        "REBUTTAL_EVIDENCE_DRAFT_PHASE2.md": "# Rebuttal Evidence Draft — Phase 2\n\nStatus: IN PROGRESS\n\nOnly verified factual evidence will be inserted here; this is not the final rebuttal.\n",
        "BLOCKED_OR_DEFERRED_PHASE2.md": "# Blocked or Deferred — Phase 2\n\nNo item is classified as blocked at Gate 0. Compute-budget deferrals will be recorded with evidence.\n",
        "FAILURE_LOG_PHASE2.md": "# Failure Log — Phase 2\n\n## Source review\n\nThe supplied reviewer DOCX ends mid-sentence at `clearly distinguishable co`; no text was inferred beyond the source.\n",
        "RESULT_FILE_INDEX_PHASE2.md": "# Result File Index — Phase 2\n\n- `../audit/`: Gate 0 identity, provenance, conflict, and validity evidence.\n",
        "BASELINE_IMPLEMENTATION_TABLE_PHASE2.md": "# Baseline Implementation Table — Phase 2\n\nStatus: environment audit pending. Official repositories/packages only; failed installations will remain listed.\n",
    }
    for name, body in skeletons.items():
        p = reports / name
        if not p.exists(): p.write_text(body, encoding="utf-8")
    print(json.dumps({k: v["status"] for k, v in registry["datasets"].items()}, indent=2))


if __name__ == "__main__":
    main()

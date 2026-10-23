"""Small, independently testable calculations for GARQ reviewer 2 comment 1."""
import ast
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

MODALITIES = ("RNA", "ATAC", "ADT")


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ids_hash(ids):
    return hashlib.sha256("\n".join(map(str, ids)).encode()).hexdigest()


def canonical_ids(ids, terminal_separator=False):
    result = [re.sub(r"\.(\d+)$", r"-\1", str(x)) if terminal_separator else str(x)
              for x in ids]
    if len(set(result)) != len(result):
        raise ValueError("Non-unique cell IDs after the specified normalization")
    return np.asarray(result, dtype=str)


def align_order(reference_ids, other_ids):
    reference = pd.Index(reference_ids)
    other = pd.Index(other_ids)
    if not reference.is_unique or not other.is_unique:
        raise ValueError("Cell IDs must be unique")
    order = other.get_indexer(reference)
    if len(reference) != len(other) or (order < 0).any():
        raise ValueError("Modalities must contain exactly the same cells")
    return order


def thin_count_values(values, probability, seed):
    values = np.asarray(values)
    if not (0 < probability <= 1):
        raise ValueError("Thinning probability must be in (0, 1]")
    if (not np.isfinite(values).all() or (values < 0).any()
            or not np.allclose(values, np.rint(values), atol=1e-6, rtol=0)):
        raise ValueError("Binomial thinning requires finite, nonnegative integer counts; do not round normalized values")
    return np.random.default_rng(seed).binomial(np.rint(values).astype(np.int64), probability)


def assignment_size(ids, requested_k, target_k):
    unique, inverse, sizes = np.unique(ids, return_inverse=True, return_counts=True)
    n = len(ids)
    return {
        "n_cells": n, "requested_k": int(requested_k), "target_k": int(target_k),
        "realized_k": len(unique), "empty_anchors": int(requested_k - len(unique)),
        "cells_per_nonempty_anchor": float(n / len(unique)),
        "size_min": int(sizes.min()), "size_median": float(np.median(sizes)),
        "size_p95": float(np.quantile(sizes, .95)), "size_max": int(sizes.max()),
        "singleton_anchor_fraction": float(np.mean(sizes == 1)),
    }, unique, inverse, sizes


def neighborhood_metrics(ids, neighbors):
    """Directed kNN edges kept inside an anchor, plus a size-conditioned null."""
    ids = np.asarray(ids)
    neighbors = np.asarray(neighbors)
    if neighbors.ndim != 2 or neighbors.shape[0] != len(ids):
        raise ValueError("Neighbor shape mismatch")
    if (neighbors == np.arange(len(ids))[:, None]).any():
        raise ValueError("Self neighbors must be excluded")
    if neighbors.min() < 0 or neighbors.max() >= len(ids):
        raise ValueError("Neighbor index out of range")
    _, inverse, sizes = np.unique(ids, return_inverse=True, return_counts=True)
    retention = np.mean(ids[neighbors] == ids[:, None], axis=1)
    chance = (sizes[inverse] - 1) / (len(ids) - 1)
    adjusted = np.divide(retention - chance, 1 - chance,
                         out=np.full(len(ids), np.nan), where=chance < 1)
    return retention, chance, adjusted


def within_anchor_cosine(z, ids):
    """Per-cell mean cosine to other members; singleton/zero-vector values are NA."""
    z = np.asarray(z, dtype=np.float64)
    norm = np.linalg.norm(z, axis=1)
    good = norm > 1e-12
    z = np.divide(z, norm[:, None], out=np.zeros_like(z), where=good[:, None])
    _, inverse = np.unique(ids, return_inverse=True)
    sums = np.zeros((int(inverse.max()) + 1, z.shape[1]))
    np.add.at(sums, inverse, z)
    counts = np.bincount(inverse, weights=good.astype(float))
    other_n = counts[inverse] - good
    numerator = np.einsum("ij,ij->i", z, sums[inverse]) - good
    value = np.divide(numerator, other_n, out=np.full(len(ids), np.nan),
                      where=good & (other_n > 0))
    return np.clip(value, -1, 1)


def type_metrics(ids, labels, rare_threshold=.01, purity_threshold=.7):
    """Label recovery by majority vote. These are descriptive annotation metrics."""
    labels = np.asarray(labels, dtype=str)
    ids = np.asarray(ids)
    table = pd.crosstab(pd.Series(ids, name="anchor"), pd.Series(labels, name="type"))
    sizes = table.sum(axis=1)
    # Stable lexical tie-break; ties are recorded, not counted as pure.
    winner = table.idxmax(axis=1)
    purity = table.max(axis=1) / sizes
    tied = table.eq(table.max(axis=1), axis=0).sum(axis=1) > 1
    prediction = pd.Series(ids).map(winner).to_numpy()
    result = []
    for label in table.columns:
        truth = labels == label
        predicted = prediction == label
        tp = int(np.sum(truth & predicted))
        precision = tp / np.sum(predicted) if np.sum(predicted) else 0.
        recall = tp / np.sum(truth)
        dominated = winner.eq(label)
        retained = dominated & purity.ge(purity_threshold) & ~tied
        retained_tp = int(table.loc[retained, label].sum())
        result.append({
            "celltype": label, "n_cells": int(truth.sum()),
            "frequency": float(truth.mean()), "rare_le_1pct": bool(truth.mean() <= rare_threshold),
            "precision": float(precision), "recall": float(recall),
            "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.,
            "retained_recall_purity_ge_0_7": float(retained_tp / truth.sum()),
            "dominant_anchor_count": int(dominated.sum()),
            "dominant_anchor_mean_purity": float(purity[dominated].mean()) if dominated.any() else np.nan,
            "dominant_anchor_median_size": float(sizes[dominated].median()) if dominated.any() else np.nan,
        })
    anchors = pd.DataFrame({"anchor": table.index, "size": sizes.to_numpy(),
                            "majority_label": winner.to_numpy(), "purity": purity.to_numpy(),
                            "majority_tie": tied.to_numpy()})
    return pd.DataFrame(result), anchors


def extract_training_module(source):
    """Preserve the original main() through inference; omit post-training plots.

    No encoder, loss, quantizer, warm-up, optimizer or stopping rule is rewritten.
    The source format is checked and unexpected versions fail explicitly.
    """
    tree = ast.parse(source)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    candidates = [(i, n) for i, n in enumerate(main.body)
                  if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                  and isinstance(n.value.func, ast.Name) and n.value.func.id == "inference"]
    if len(candidates) != 1:
        raise ValueError("Unsupported GARQ.py: expected one direct inference assignment in main")
    index, node = candidates[0]
    if not isinstance(node.targets[0], (ast.Tuple, ast.List)) or len(node.targets[0].elts) != 5:
        raise ValueError("Unsupported inference return signature")
    names = [e.id for e in node.targets[0].elts]
    prefix = ast.dump(ast.Module(body=main.body[:index + 1], type_ignores=[]), include_attributes=False)
    main.body = main.body[:index + 1] + [ast.Return(value=ast.Tuple(
        elts=[ast.Name(id=x, ctx=ast.Load()) for x in names + ["net", "adata_list"]], ctx=ast.Load()))]
    for node in tree.body:
        if isinstance(node, ast.If) and ast.unparse(node.test) != "__name__ == '__main__'":
            raise ValueError("Unsupported GARQ.py: unexpected conditional at module level")
    tree.body = [n for n in tree.body if not isinstance(n, ast.If)
                 and not (isinstance(n, ast.ImportFrom) and n.module == "eval_utils")]
    # The only module-level if in the inspected implementation is the CLI entry point.
    ast.fix_missing_locations(tree)
    transformed = ast.unparse(tree)
    return transformed, hashlib.sha256(prefix.encode()).hexdigest()

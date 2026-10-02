"""Label-permutation invariant assignment comparison metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, rand_score


def variation_of_information(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Return VI in nats; zero means identical partitions."""
    a, a_inv = np.unique(np.asarray(labels_a), return_inverse=True)
    b, b_inv = np.unique(np.asarray(labels_b), return_inverse=True)
    if len(a_inv) != len(b_inv):
        raise ValueError("Assignment vectors must have the same length")
    contingency = np.zeros((len(a), len(b)), dtype=np.int64)
    np.add.at(contingency, (a_inv, b_inv), 1)
    joint = contingency / contingency.sum()
    pa = joint.sum(axis=1)
    pb = joint.sum(axis=0)
    h_a = -sum(float(p) * math.log(float(p)) for p in pa if p > 0)
    h_b = -sum(float(p) * math.log(float(p)) for p in pb if p > 0)
    mutual_information = 0.0
    for i, j in zip(*np.nonzero(joint)):
        mutual_information += float(joint[i, j]) * math.log(float(joint[i, j] / (pa[i] * pb[j])))
    return h_a + h_b - 2.0 * mutual_information


def compare_assignments(labels_a: np.ndarray, labels_b: np.ndarray) -> dict[str, Any]:
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    if labels_a.shape != labels_b.shape:
        raise ValueError(f"Assignment shapes differ: {labels_a.shape} vs {labels_b.shape}")
    return {
        "n_cells": int(labels_a.size),
        "ARI": float(adjusted_rand_score(labels_a, labels_b)),
        "NMI": float(normalized_mutual_info_score(labels_a, labels_b)),
        "VI_nats": float(variation_of_information(labels_a, labels_b)),
        "coassignment_agreement_rand": float(rand_score(labels_a, labels_b)),
    }

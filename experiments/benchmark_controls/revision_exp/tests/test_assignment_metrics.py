import numpy as np

from revision_exp.metrics.assignment import compare_assignments


def test_assignment_metrics_ignore_cluster_label_permutation():
    a = np.array([0, 0, 1, 1, 2, 2])
    b = np.array([7, 7, 3, 3, 9, 9])
    result = compare_assignments(a, b)
    assert result["ARI"] == 1.0
    assert result["NMI"] == 1.0
    assert abs(result["VI_nats"]) < 1e-12
    assert result["coassignment_agreement_rand"] == 1.0


def test_assignment_metrics_detect_partition_change():
    result = compare_assignments(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))
    assert result["ARI"] < 1.0
    assert result["VI_nats"] > 0.0

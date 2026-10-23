import torch

from model import GARQuantizer


def test_no_manual_reposition_records_zero_displacement():
    torch.manual_seed(17)
    quantizer = GARQuantizer(entry_num=6, entry_dim=4, k_knn=2)
    quantizer.configure_anchor_diagnostics(
        enabled=True, dynamic_update_enabled=False, reposition_interval=1
    )
    quantizer.train()
    inputs = torch.randn(12, 4)
    for _ in range(3):
        quantizer(inputs, return_assignment=False)
    assert all(row["reposition_due"] is False for row in quantizer.diagnostic_trace)
    assert all(row["manual_displacement_anchor_count"] == 0 for row in quantizer.diagnostic_trace)


def test_reposition_interval_is_counted_on_quantized_training_calls():
    torch.manual_seed(18)
    quantizer = GARQuantizer(entry_num=6, entry_dim=4, k_knn=2)
    quantizer.configure_anchor_diagnostics(
        enabled=True, dynamic_update_enabled=True, reposition_interval=5
    )
    quantizer.train()
    inputs = torch.randn(12, 4)
    for _ in range(6):
        quantizer(inputs, return_assignment=False)
    assert [row["reposition_due"] for row in quantizer.diagnostic_trace] == [
        False,
        False,
        False,
        False,
        True,
        False,
    ]

import pytest
import torch

from model import GARQ


def test_default_combination_is_the_released_concatenation() -> None:
    model = GARQ([3, 4], ["RNA", "ADT"], entry_num=2)
    blocks = [torch.randn(5, 32), torch.randn(5, 32)]
    combined = model.combine_hiddens(blocks)
    assert torch.equal(combined, torch.cat(blocks, dim=1))


def test_explicit_weights_scale_only_the_requested_blocks() -> None:
    model = GARQ([3, 4], ["RNA", "ADT"], entry_num=2, modality_weights=[0.0, 2.0])
    blocks = [torch.ones(5, 32), torch.full((5, 32), 3.0)]
    combined = model.combine_hiddens(blocks)
    assert torch.equal(combined[:, :32], torch.zeros(5, 32))
    assert torch.equal(combined[:, 32:], torch.full((5, 32), 6.0))


@pytest.mark.parametrize("weights", [[1.0], [-1.0, 1.0], [0.0, 0.0]])
def test_invalid_modality_weights_are_rejected(weights: list[float]) -> None:
    with pytest.raises(ValueError):
        GARQ([3, 4], ["RNA", "ADT"], entry_num=2, modality_weights=weights)

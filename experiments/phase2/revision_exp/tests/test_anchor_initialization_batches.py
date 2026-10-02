import torch

from engine import init_gart_anchors


class _Quantizer:
    entry_num = 645


class _Model:
    omics_num = 1
    quantizer = _Quantizer()

    def eval(self):
        return self

    def train(self):
        return self

    def __call__(self, x_list):
        return x_list

    def combine_hiddens(self, hiddens):
        return hiddens[0]

    def init_quantizer(self, hidden, method):
        self.initialization_shape = tuple(hidden.shape)


def test_anchor_initialization_collects_enough_points_for_large_k():
    model = _Model()
    batches = [{"x": [torch.zeros(256, 4)]} for _ in range(4)]
    init_gart_anchors(model, ["RNA"], batches, torch.device("cpu"), "Kmeans")
    assert model.initialization_shape == (768, 4)


def test_anchor_initialization_preserves_two_batch_minimum():
    model = _Model()
    model.quantizer.entry_num = 100
    batches = [{"x": [torch.zeros(256, 4)]} for _ in range(4)]
    init_gart_anchors(model, ["RNA"], batches, torch.device("cpu"), "Kmeans")
    assert model.initialization_shape == (512, 4)

"""
M1 placeholder test — proves pytest is wired up correctly before M2/M3
add real tokenizer and data-generation tests here.
"""
import torch


def test_torch_importable():
    assert torch.__version__ is not None


def test_basic_tensor_math():
    x = torch.tensor([1.0, 2.0, 3.0])
    assert x.sum().item() == 6.0

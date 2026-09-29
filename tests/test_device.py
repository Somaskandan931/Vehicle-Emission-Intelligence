import pytest
import torch

from src.common import get_device


def test_cpu_and_auto_never_raise():
    assert get_device({"compute": {"device": "cpu"}}).type == "cpu"
    assert get_device({"compute": {"device": "auto"}}).type in ("cuda", "mps", "cpu")


def test_gpu_request_fails_loudly_without_gpu():
    mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    if torch.cuda.is_available() or mps:
        pytest.skip("a GPU is present")
    with pytest.raises(RuntimeError, match="no GPU"):
        get_device({"compute": {"device": "gpu"}})

import numpy as np
import torch

from src.data.sequences import make_windows
from src.evaluation.metrics import regression_metrics
from src.models import build_model


def test_windows_no_leak_and_shape():
    F = np.arange(30, dtype=float).reshape(30, 1)
    T = np.arange(30, dtype=float).reshape(30, 1)
    X, y, idx = make_windows(F, T, length=5, horizon=1)
    assert X.shape == (25, 5, 1)
    assert y[0, 0] == 5 and X[0, -1, 0] == 4      # target is strictly after the window


def test_metrics_perfect_and_known():
    y = np.array([1.0, 2.0, 3.0])
    m = regression_metrics(y, y)
    assert m["MAE"] == 0 and m["R2"] == 1
    assert abs(regression_metrics(y, y + 1)["MAE"] - 1) < 1e-9


def test_models_forward():
    cfg = {"model": {"hidden_units": 8, "layers": 1, "dropout": 0.0}}
    x = torch.randn(4, 12, 7)
    for k in ("uni", "bi", "hybrid"):
        assert build_model(k, 7, 6, cfg, lam=0.5)(x).shape == (4, 6)


def test_hybrid_lambda_extremes_differ():
    cfg = {"model": {"hidden_units": 8, "layers": 1, "dropout": 0.0}}
    torch.manual_seed(0)
    m = build_model("hybrid", 3, 2, cfg, lam=0.0)
    x = torch.randn(2, 5, 3)
    a = m(x); m.lam = 1.0; b = m(x)
    assert not torch.allclose(a, b)

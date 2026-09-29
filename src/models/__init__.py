from .bi_lstm import BiLSTM
from .hybrid_lstm import HybridLSTM
from .uni_lstm import UniLSTM


def build_model(kind, n_features, n_out, cfg, lam=None):
    m = cfg["model"]
    args = (n_features, m["hidden_units"], m["layers"], m["dropout"], n_out)
    if kind == "uni":
        return UniLSTM(*args)
    if kind == "bi":
        return BiLSTM(*args)
    if kind == "hybrid":
        if lam is None:
            raise ValueError("hybrid model needs a lambda value")
        return HybridLSTM(*args, lam=lam)
    raise ValueError(kind)

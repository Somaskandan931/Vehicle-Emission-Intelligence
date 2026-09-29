"""Shared helpers: config loading (merged), seeding, device, paths."""
import copy
import random
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIGS = {"uni": "uni_lstm", "bi": "bi_lstm", "hybrid": "hybrid_lstm"}


def _read(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with open(path) as f:
        return yaml.safe_load(f) or {}


def deep_update(base, new):
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(model=None, extra=None, overrides=None):
    """config.yaml + configs/data.yaml + configs/<model>_lstm.yaml (+ extra yaml, + overrides dict).

    model: None | 'uni' | 'bi' | 'hybrid'. Data/split stages need no model config.
    """
    cfg = _read("config.yaml")
    cfg = deep_update(cfg, _read("configs/data.yaml"))
    if model:
        cfg = deep_update(cfg, _read(f"configs/{MODEL_CONFIGS[model]}.yaml"))
    if extra:
        cfg = deep_update(cfg, _read(extra))
    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _has_mps():
    return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()


def get_device(cfg=None):
    """Resolve compute.device from the config (default: gpu). GPU requests never fall back to CPU."""
    want = ((cfg or {}).get("compute", {}) or {}).get("device", "gpu")
    if want == "cpu":
        return torch.device("cpu")
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "mps" if _has_mps() else "cpu")
    if want in ("gpu", "cuda") and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False          # reproducibility over speed
        torch.backends.cudnn.deterministic = True
        return torch.device("cuda")
    if want in ("gpu", "mps") and _has_mps():
        return torch.device("mps")
    raise RuntimeError(
        f"compute.device='{want}' but no GPU is available to PyTorch "
        f"(torch {torch.__version__}, built for CUDA: {torch.version.cuda}). "
        "Install a CUDA build of PyTorch (see README, 'GPU setup') and update the NVIDIA driver, "
        "or set compute.device: cpu in config.yaml for a smoke test."
    )


def device_name(dev):
    if dev.type == "cuda":
        return torch.cuda.get_device_name(0)
    return {"mps": "Apple MPS", "cpu": "CPU"}[dev.type]


def rpath(cfg, key):
    p = ROOT / cfg["paths"][key]      # absolute paths in cfg are respected
    p.mkdir(parents=True, exist_ok=True)
    return p

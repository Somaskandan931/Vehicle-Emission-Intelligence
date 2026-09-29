"""Prediction + evaluation in original (unscaled) units."""
import json
import pickle

import numpy as np
import pandas as pd
import torch

from src.common import ROOT
from src.evaluation.metrics import metrics_table


def load_split(cfg, name):
    return np.load(ROOT / cfg["paths"]["processed"] / f"{name}.npz")


def load_scalers_meta(cfg):
    p = ROOT / cfg["paths"]["processed"]
    with open(p / "scalers.pkl", "rb") as f:
        sc = pickle.load(f)
    return sc, json.loads((p / "meta.json").read_text())


@torch.no_grad()
def predict(model, X, device, bs=512):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(model(torch.from_numpy(X[i:i + bs]).to(device)).cpu().numpy())
    return np.concatenate(out)


def evaluate_split(cfg, model, split, device):
    """Returns (metrics_table, predictions_dataframe), both in original units."""
    data = load_split(cfg, split)
    sc, meta = load_scalers_meta(cfg)
    pred = sc["y"].inverse_transform(predict(model, data["X"], device))
    true = sc["y"].inverse_transform(data["y"])
    table = metrics_table(true, pred, meta["targets"])
    pdf = pd.DataFrame({"vehicle_id": data["vid"], "step": data["step"]})
    for i, t in enumerate(meta["targets"]):
        pdf[f"{t}_true"], pdf[f"{t}_pred"] = true[:, i], pred[:, i]
    return table, pdf

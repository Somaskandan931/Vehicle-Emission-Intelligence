"""Persistence baseline: predict that the pollutant stays at its last observed value.

  python -m src.evaluation.baseline
For horizon h the forecast for t+h is the value at t (last row of the input window). Any learned
model has to beat this to show it learns more than autocorrelation. Needs the past pollutant
values to be model inputs (data.include_targets_as_inputs: true).

Writes results/metrics/persistence.csv and results/predictions/persistence.csv (test set), and
experiments/persistence/metrics_val.csv (validation set).
"""
import argparse

import numpy as np
import pandas as pd

from src.common import load_config, rpath
from src.evaluation.evaluate import load_scalers_meta, load_split
from src.evaluation.metrics import metrics_table


def persistence_predictions(cfg, split):
    sc, meta = load_scalers_meta(cfg)
    feats, targets = meta["features"], meta["targets"]
    # the target columns are appended last, so take their LAST occurrence in the feature list
    idx = [len(feats) - 1 - feats[::-1].index(t) for t in targets if t in feats]
    if len(idx) != len(targets):
        raise RuntimeError("Persistence baseline needs the pollutant columns among the model inputs "
                           "(set data.include_targets_as_inputs: true).")
    data = load_split(cfg, split)
    last = data["X"][:, -1, idx].astype(float)
    pred = last * sc["x"].scale_[idx] + sc["x"].mean_[idx]          # back to original units
    true = sc["y"].inverse_transform(data["y"])
    pdf = pd.DataFrame({"vehicle_id": data["vid"], "step": data["step"]})
    for i, t in enumerate(targets):
        pdf[f"{t}_true"], pdf[f"{t}_pred"] = true[:, i], pred[:, i]
    return metrics_table(true, pred, targets), pdf


def main(cfg=None):
    cfg = cfg or load_config()
    val, _ = persistence_predictions(cfg, "val")
    test, pred = persistence_predictions(cfg, "test")
    exp = rpath(cfg, "experiments") / "persistence"
    exp.mkdir(parents=True, exist_ok=True)
    val.to_csv(exp / "metrics_val.csv", index=False)
    test.to_csv(rpath(cfg, "metrics") / "persistence.csv", index=False)
    pred.to_csv(rpath(cfg, "predictions") / "persistence.csv", index=False)
    print(test.round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="optional extra yaml merged on top")
    main(load_config(extra=ap.parse_args().config))

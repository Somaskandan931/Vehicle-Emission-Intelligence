"""DEHS = 100 * exp(-beta * VEEI), plus the driver that scores the hybrid model's predictions.

  python -m src.emission.dehs
Reads   results/predictions/hybrid_lstm.csv
Writes  results/tables/veei_dehs.csv (VEEI/DEHS from predicted and from true values)
        results/tables/table_veei_parameters.csv (limits, weights, units, sources, beta)
Refuses to run while limits / weights / beta are null in config.yaml.
"""
import argparse

import numpy as np
import pandas as pd

from src.common import load_config, rpath
from src.emission.veei import compute_veei


def compute_dehs(veei, cfg):
    beta = cfg["dehs"]["beta"]
    if beta is None:
        raise ValueError("dehs.beta is null in config.yaml - set it and document how it was chosen.")
    return 100.0 * np.exp(-beta * veei)


def main(cfg=None):
    cfg = cfg or load_config()
    pred = pd.read_csv(rpath(cfg, "predictions") / "hybrid_lstm.csv")
    out = pred[["vehicle_id", "step"]].copy()
    out["VEEI_pred"] = compute_veei(pred, cfg, "_pred")
    out["VEEI_true"] = compute_veei(pred, cfg, "_true")
    out["DEHS_pred"] = compute_dehs(out["VEEI_pred"], cfg)
    out["DEHS_true"] = compute_dehs(out["VEEI_true"], cfg)
    tables = rpath(cfg, "tables")
    out.to_csv(tables / "veei_dehs.csv", index=False)

    v = cfg["veei"]
    pd.DataFrame([{"pollutant": p, "limit": v["limits"][p], "weight": v["weights"][p],
                   "unit": v.get("units", {}).get(p, ""), "source": v["sources"].get(p, "")}
                  for p in cfg["data"]["target_cols"]]
                 ).assign(beta=cfg["dehs"]["beta"], beta_justification=cfg["dehs"]["justification"]
                          ).to_csv(tables / "table_veei_parameters.csv", index=False)
    print(out.describe().round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="optional extra yaml merged on top")
    main(load_config(extra=ap.parse_args().config))

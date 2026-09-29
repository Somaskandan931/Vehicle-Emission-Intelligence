"""Collect Uni / Bi / Hybrid test metrics into the paper tables.

Reads   results/metrics/{uni_lstm,bi_lstm,hybrid_lstm}.csv
Writes  results/tables/table_metrics.csv          (Table III: per pollutant, per model)
        results/ablation/ablation_results.csv     (Table IV: macro summary, unit-free)
        results/tables/table_model_config.csv     (Table II: hyperparameters actually used)
"""
import argparse
import json

import pandas as pd

from src.common import ROOT, load_config, rpath

MODELS = {"Uni-LSTM": ("uni", "uni_lstm"), "Bi-LSTM": ("bi", "bi_lstm"), "Hybrid": ("hybrid", "hybrid_lstm")}
BASELINES = {"Persistence": "persistence"}          # non-learned reference, listed first in the tables


def model_config_table(cfg, lam=None):
    rows = []
    for label, (kind, _) in MODELS.items():
        c = load_config(kind)
        row = {"model": label, **c["model"], **c["training"]}
        if kind == "hybrid":
            row["lambda"] = lam if lam is not None else c["hybrid"]["lambda"]
        rows.append(row)
    d = cfg["data"]
    return pd.DataFrame(rows).assign(sequence_length=d["sequence_length"],
                                     prediction_horizon=d["prediction_horizon"])


def main(cfg=None):
    cfg = cfg or load_config()
    metrics_dir = rpath(cfg, "metrics")
    tables, abl = rpath(cfg, "tables"), rpath(cfg, "ablation")
    frames = []
    for label, (_, name) in {**{k: (None, v) for k, v in BASELINES.items()}, **MODELS}.items():
        f = metrics_dir / f"{name}.csv"
        if f.exists():
            df = pd.read_csv(f)
            df.insert(0, "model", label)
            frames.append(df)
    if not frames:
        raise RuntimeError("No metrics found in results/metrics - train the models first.")
    allm = pd.concat(frames, ignore_index=True)
    allm.to_csv(tables / "table_metrics.csv", index=False)
    macro = allm[allm.pollutant == "MACRO_AVG"].drop(columns=["pollutant", "MAE", "RMSE"]).round(4)
    if "Persistence" in set(macro["model"]):       # error relative to the persistence baseline
        ref = macro.loc[macro.model == "Persistence"].iloc[0]
        macro["NRMSE_vs_persistence"] = (macro["NRMSE"] / ref["NRMSE"]).round(4)   # <1 = beats it
    macro.to_csv(abl / "ablation_results.csv", index=False)

    lam_file = ROOT / cfg["paths"]["experiments"] / "lambda_search" / "best_lambda.json"
    lam = json.loads(lam_file.read_text())["best_lambda"] if lam_file.exists() else None
    model_config_table(cfg, lam).to_csv(tables / "table_model_config.csv", index=False)
    print(macro.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="optional extra yaml merged on top")
    main(load_config(extra=ap.parse_args().config))

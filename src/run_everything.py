"""Every experiment for the paper, in priority order. Safe to re-run: finished lambda runs are reused.

  python -m src.run_everything                # all phases
  python -m src.run_everything --phases 1 2   # selected phases
  python -m src.run_everything --seeds 43 44

Phase 1  seed 42, within-trip split: prep, persistence, uni, bi, lambda search (11 values), hybrid, tables, plots
Phase 2  extra seeds (default 43, 44) with the chosen lambda; results in results_seed<N>/ ; mean +/- std table
Phase 3  by-trip split (whole trips held out) with the chosen lambda; results in results_bytrip/
VEEI/DEHS ('emission' stage) is not run here: fill the sourced parameters in config.yaml first.
"""
import argparse
import json
import time

import pandas as pd

from src.common import ROOT, load_config
from src.run_all import run_pipeline

MODELS = {"Uni-LSTM": "uni_lstm", "Bi-LSTM": "bi_lstm", "Hybrid": "hybrid_lstm"}


def best_lambda():
    return json.loads((ROOT / "experiments/lambda_search/best_lambda.json").read_text())["best_lambda"]


def seed_paths(s):
    return {k: f"{v}_seed{s}" for k, v in {
        "experiments": "experiments", "checkpoints": "models/checkpoints", "metrics": "results/metrics",
        "predictions": "results/predictions", "ablation": "results/ablation",
        "tables": "results/tables", "figures": "results/figures"}.items()}


def phase1():
    run_pipeline(["prep", "persistence", "uni", "bi", "lambda", "hybrid", "ablation", "plots"])


def phase2(seeds):
    lam = best_lambda()
    for s in seeds:
        print(f"===== seed {s} (lambda {lam}) =====", flush=True)
        run_pipeline(["persistence", "uni", "bi", "hybrid", "ablation"],
                     overrides={"seed": s, "paths": seed_paths(s), "hybrid": {"lambda": lam}})
    aggregate([42] + list(seeds))


def aggregate(seeds):
    rows = []
    for s in seeds:
        d = ROOT / ("results/metrics" if s == 42 else f"results/metrics_seed{s}")
        for label, name in MODELS.items():
            f = d / f"{name}.csv"
            if f.exists():
                df = pd.read_csv(f)
                df.insert(0, "seed", s); df.insert(0, "model", label)
                rows.append(df)
    if not rows:
        return
    a = pd.concat(rows)
    cols = [c for c in ("MAE", "RMSE", "MAPE", "R2", "NMAE", "NRMSE") if c in a.columns]
    g = a.groupby(["model", "pollutant"])[cols].agg(["mean", "std"])
    g.columns = [f"{m}_{k}" for m, k in g.columns]
    out = ROOT / "results/tables"
    out.mkdir(parents=True, exist_ok=True)
    g.reset_index().to_csv(out / "table_metrics_seeds.csv", index=False)
    print(g.xs("MACRO_AVG", level="pollutant")[["NRMSE_mean", "NRMSE_std", "R2_mean", "R2_std"]].round(4))


def phase3():
    run_pipeline(["prep", "persistence", "uni", "bi", "hybrid", "ablation", "plots"],
                 extra="configs/by_trip.yaml", overrides={"hybrid": {"lambda": best_lambda()}})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", nargs="*", type=int, default=[1, 2, 3])
    ap.add_argument("--seeds", nargs="*", type=int, default=[43, 44])
    a = ap.parse_args()
    t0 = time.time()
    if 1 in a.phases: phase1()
    if 2 in a.phases: phase2(a.seeds)
    if 3 in a.phases: phase3()
    print(f"done in {(time.time() - t0) / 3600:.1f} h")

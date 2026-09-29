"""Actual vs predicted emissions (hybrid model, test set).  -> results/figures/fig_actual_vs_predicted.png"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.common import load_config, rpath


def main(cfg=None, n=200):
    cfg = cfg or load_config()
    f = rpath(cfg, "predictions") / "hybrid_lstm.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    vid = df["vehicle_id"].iloc[0]
    df = df[df.vehicle_id == vid].sort_values("step").head(n)
    tg = cfg["data"]["target_cols"]
    fig, axes = plt.subplots(2, (len(tg) + 1) // 2, figsize=(14, 6))
    for ax, t in zip(axes.ravel(), tg):
        ax.plot(df["step"], df[f"{t}_true"], label="actual")
        ax.plot(df["step"], df[f"{t}_pred"], "--", label="predicted (hybrid)")
        ax.set_title(f"{t} (vehicle {vid})")
    axes.ravel()[0].legend()
    fig.tight_layout()
    fig.savefig(rpath(cfg, "figures") / "fig_actual_vs_predicted.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()

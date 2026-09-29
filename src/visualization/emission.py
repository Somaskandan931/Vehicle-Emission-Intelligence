"""VEEI and DEHS over time, actual vs predicted.  -> results/figures/fig_veei_dehs.png"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.common import load_config, rpath


def main(cfg=None):
    cfg = cfg or load_config()
    f = rpath(cfg, "tables") / "veei_dehs.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    d = df[df.vehicle_id == df["vehicle_id"].iloc[0]].sort_values("step")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].plot(d["step"], d["VEEI_true"], label="actual")
    axes[0].plot(d["step"], d["VEEI_pred"], "--", label="predicted")
    axes[0].set_title("VEEI")
    axes[0].legend()
    axes[1].plot(d["step"], d["DEHS_true"])
    axes[1].plot(d["step"], d["DEHS_pred"], "--")
    axes[1].set_title("DEHS")
    fig.tight_layout()
    fig.savefig(rpath(cfg, "figures") / "fig_veei_dehs.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()

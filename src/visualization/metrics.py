"""Model comparison bars (unit-free macro metrics).  -> results/figures/fig_model_comparison.png"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.common import load_config, rpath


def main(cfg=None):
    cfg = cfg or load_config()
    f = rpath(cfg, "ablation") / "ablation_results.csv"
    if not f.exists():
        return
    df = pd.read_csv(f).set_index("model")
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    for ax, m in zip(axes, ["NMAE", "NRMSE", "MAPE", "R2"]):
        df[m].plot.bar(ax=ax, rot=20)
        ax.set_title(m)
    fig.tight_layout()
    fig.savefig(rpath(cfg, "figures") / "fig_model_comparison.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()

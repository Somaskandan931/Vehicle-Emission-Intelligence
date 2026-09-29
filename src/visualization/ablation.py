"""Lambda vs validation error.  -> results/figures/fig_lambda_vs_error.png"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.common import ROOT, load_config, rpath


def main(cfg=None):
    cfg = cfg or load_config()
    f = ROOT / cfg["paths"]["experiments"] / "lambda_search" / "lambda_results.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(df["lambda"], df["val_mse"], "o-")
    ax.set_xlabel("lambda (0 = Uni-LSTM only, 1 = Bi-LSTM only)")
    ax.set_ylabel("validation MSE (standardised)")
    fig.tight_layout()
    fig.savefig(rpath(cfg, "figures") / "fig_lambda_vs_error.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()

"""Dataset profiling: produces the numbers for the paper's dataset table."""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.common import load_config, rpath
from src.data.load_data import load_raw


def main(cfg):
    df = load_raw(cfg)
    d = cfg["data"]
    tables, figs = rpath(cfg, "tables"), rpath(cfg, "figures")

    summary = {
        "rows": len(df),
        "columns": df.shape[1],
        "vehicles": df[d["vehicle_col"]].nunique() if d["vehicle_col"] in df else 1,
        "duplicate_rows": int(df.duplicated().sum()),
        "total_missing_cells": int(df.isna().sum().sum()),
    }
    if d["time_col"] in df:
        t = df[d["time_col"]]
        summary["time_first"], summary["time_last"] = str(t.min()), str(t.max())
    pd.Series(summary).to_csv(tables / "dataset_summary.csv", header=["value"])
    df.describe(include="all").T.to_csv(tables / "dataset_statistics.csv")
    df.isna().sum().rename("missing").to_csv(tables / "missing_values.csv")
    for c in d["categorical_cols"]:
        if c in df:
            df[c].value_counts().to_csv(tables / f"counts_{c}.csv")

    tg = df[d["target_cols"]]
    fig, axes = plt.subplots(2, (len(tg.columns) + 1) // 2, figsize=(12, 6))
    for ax, c in zip(axes.ravel(), tg.columns):
        ax.hist(tg[c].dropna(), bins=40)
        ax.set_title(c)
    fig.tight_layout()
    fig.savefig(figs / "pollutant_distribution.png", dpi=200)
    plt.close(fig)

    corr = df.select_dtypes("number").corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90)
    ax.set_yticks(range(len(corr)), corr.columns)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(figs / "correlation_matrix.png", dpi=200)
    plt.close(fig)
    print(pd.Series(summary))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="optional extra yaml merged on top")
    main(load_config(extra=ap.parse_args().config))

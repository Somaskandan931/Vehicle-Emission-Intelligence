"""End-to-end smoke test on SYNTHETIC data written to a temp dir.

Checks that every stage runs and writes its outputs. The numbers are meaningless and
nothing is written into the repository's data/ or results/ folders.
"""
import numpy as np
import pandas as pd
import pytest

from src.common import load_config
from src.run_all import run_pipeline
from src.emission.veei import compute_veei

POLL = ["CO", "HC", "NOx", "CO2", "SO2", "PM"]


def _synthetic(path, vehicles=6, n=200):
    rng = np.random.default_rng(0)
    rows = []
    for v in range(vehicles):
        t = pd.date_range("2024-01-01", periods=n, freq="5min")
        speed = np.clip(40 + 20 * np.sin(np.arange(n) / 30) + rng.normal(0, 5, n), 0, None)
        load = np.clip(0.5 + 0.3 * np.sin(np.arange(n) / 17) + rng.normal(0, 0.05, n), 0, 1)
        d = {"vehicle_id": f"V{v:02d}", "timestamp": t, "vehicle_type": rng.choice(["petrol", "diesel"]),
             "speed": speed, "engine_load": load}
        for k, base in zip(POLL, [1.0, 0.4, 0.8, 120, 0.05, 0.02]):
            s = base * (0.5 + load) * (1 + speed / 100)
            d[k] = np.clip(s + 0.6 * np.roll(s, 1) + rng.normal(0, base * 0.05, n), 0, None)
        rows.append(pd.DataFrame(d))
    pd.concat(rows).to_csv(path, index=False)


def test_full_pipeline(tmp_path):
    raw = tmp_path / "raw.csv"
    _synthetic(raw)
    p = lambda name: str(tmp_path / name)
    ov = {
        "data": {"raw_file": str(raw), "categorical_cols": ["vehicle_type"], "vehicle_col": "vehicle_id",
                 "target_cols": POLL, "feature_cols": None, "drop_negative_targets": True},
        "paths": {"processed": p("processed"), "experiments": p("experiments"),
                  "checkpoints": p("ckpt"), "metrics": p("metrics"), "predictions": p("pred"),
                  "ablation": p("ablation"), "tables": p("tables"), "figures": p("figures")},
        "compute": {"device": "cpu"},          # smoke test only; real runs use the GPU
        "training": {"epochs": 3, "patience": 2, "batch_size": 128},
        "model": {"hidden_units": 8},
        "hybrid": {"lambda_grid": [0.0, 1.0]},
        # placeholder VEEI/DEHS values - smoke test only
        "veei": {"limits": {k: 1.0 for k in POLL}, "weights": {k: 1 / 6 for k in POLL}},
        "dehs": {"beta": 0.5},
    }
    run_pipeline(overrides=ov)
    for rel in ["tables/dataset_summary.csv", "tables/cleaning_report.csv", "tables/split_report.csv",
                "metrics/persistence.csv", "metrics/uni_lstm.csv", "metrics/bi_lstm.csv", "metrics/hybrid_lstm.csv",
                "pred/hybrid_lstm.csv", "ckpt/hybrid_lstm.pt", "experiments/lambda_search/lambda_results.csv",
                "ablation/ablation_results.csv", "tables/table_model_config.csv", "tables/veei_dehs.csv",
                "figures/fig_actual_vs_predicted.png", "figures/fig_lambda_vs_error.png"]:
        assert (tmp_path / rel).exists(), rel


def test_veei_refuses_null_parameters():
    cfg = load_config()
    df = pd.DataFrame({f"{k}_pred": [1.0] for k in POLL})
    with pytest.raises(ValueError):
        compute_veei(df, cfg)

"""Minimal dashboard: shows per-vehicle VEEI / DEHS computed from saved model predictions.
Run AFTER the pipeline:  python dashboard/app.py
"""
import sys
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
app = Flask(__name__)


def _load():
    f = ROOT / "results/tables/veei_dehs.csv"
    if not f.exists():
        return None
    return pd.read_csv(f)


@app.route("/")
def index():
    df = _load()
    if df is None:
        return "Run `python run_all.py` first (results/tables/veei_dehs.csv missing).", 404
    summ = df.groupby("vehicle_id").agg(VEEI=("VEEI_pred", "mean"), DEHS=("DEHS_pred", "mean")).round(3)
    return render_template("dashboard.html", rows=summ.reset_index().to_dict("records"))


@app.route("/api/vehicle/<vid>")
def vehicle(vid):
    df = _load()
    d = df[df.vehicle_id == vid].sort_values("step")
    return jsonify(d.to_dict("list"))


if __name__ == "__main__":
    app.run(debug=True)

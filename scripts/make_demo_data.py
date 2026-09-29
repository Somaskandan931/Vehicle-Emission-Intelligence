"""Generate a SYNTHETIC dataset to smoke-test the pipeline.

!! Results obtained from this file are meaningless and must NEVER appear in the paper. !!
Writes data/raw/demo_emissions.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from src.common import ROOT

rng = np.random.default_rng(0)
rows = []
for v in range(12):
    vtype = rng.choice(["petrol", "diesel", "cng"])
    age = rng.integers(1, 15)
    n = 500
    t = pd.date_range("2024-01-01", periods=n, freq="5min")
    speed = np.clip(40 + 20 * np.sin(np.arange(n) / 30) + rng.normal(0, 5, n), 0, None)
    load = np.clip(0.5 + 0.3 * np.sin(np.arange(n) / 17) + rng.normal(0, 0.05, n), 0, 1)
    temp = 25 + 5 * np.sin(np.arange(n) / 100) + rng.normal(0, 0.5, n)
    base = 1 + 0.05 * age + (0.3 if vtype == "diesel" else 0)
    def series(k, noise):
        s = base * k * (0.5 + load) * (1 + speed / 100)
        s = s + 0.6 * np.roll(s, 1)  # temporal dependence
        return np.clip(s + rng.normal(0, noise, n), 0, None)
    rows.append(pd.DataFrame({
        "vehicle_id": f"V{v:02d}", "timestamp": t, "vehicle_type": vtype, "age_years": age,
        "speed": speed, "engine_load": load, "ambient_temp": temp,
        "CO": series(1.0, 0.05), "HC": series(0.4, 0.02), "NOx": series(0.8, 0.04),
        "CO2": series(120, 3), "SO2": series(0.05, 0.005), "PM": series(0.02, 0.002)}))
df = pd.concat(rows)
(ROOT / "data/raw").mkdir(parents=True, exist_ok=True)
df.to_csv(ROOT / "data/raw/demo_emissions.csv", index=False)
print("wrote data/raw/demo_emissions.csv", df.shape)

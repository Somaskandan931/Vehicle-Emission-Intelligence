import numpy as np
import pandas as pd


def regression_metrics(y_true, y_pred, eps=1e-8, mape_floor=0.01):
    """MAE, RMSE, MAPE(%), R2 for 1-D arrays.

    MAPE is computed only where |y| > mape_floor * mean(|y|) (default 1% of the mean level).
    Emissions are often ~0 and float round-trips through the scaler turn exact zeros into ~1e-7,
    so a fixed tiny cutoff lets near-zero samples explode the percentage. State this rule in the
    paper; lead with NMAE / NRMSE / R2."""
    e = y_true - y_pred
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e ** 2)))
    mask = np.abs(y_true) > max(eps, mape_floor * float(np.mean(np.abs(y_true))))
    mape = float(np.mean(np.abs(e[mask] / y_true[mask])) * 100) if mask.any() else float("nan")
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - np.sum(e ** 2) / ss_tot) if ss_tot > 0 else float("nan")
    sd = float(np.std(y_true))
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2,
            "NMAE": mae / sd if sd > 0 else float("nan"),     # MAE / std(y): unit-free
            "NRMSE": rmse / sd if sd > 0 else float("nan")}


def metrics_table(y_true, y_pred, target_names):
    """Per-pollutant metrics plus a MACRO_AVG row. Pollutants have different units/scales
    (CO2 >> PM), so raw MAE/RMSE are NOT averaged (left NaN); the macro row averages the
    unit-free NMAE, NRMSE, MAPE and R2 instead."""
    rows = []
    for i, name in enumerate(target_names):
        rows.append({"pollutant": name, **regression_metrics(y_true[:, i], y_pred[:, i])})
    df = pd.DataFrame(rows)
    macro = df.drop(columns="pollutant").mean().to_dict()
    macro["MAE"] = macro["RMSE"] = float("nan")
    return pd.concat([df, pd.DataFrame([{"pollutant": "MACRO_AVG", **macro}])], ignore_index=True)

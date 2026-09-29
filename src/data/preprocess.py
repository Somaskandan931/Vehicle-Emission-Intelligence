"""Cleaning -> chronological split -> scaling (fit on train only) -> sequences.

Outputs (data/processed): train.npz, val.npz, test.npz, scalers.pkl, meta.json
and results/tables/cleaning_report.csv, split_report.csv.
"""
import argparse
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.common import ROOT, load_config, rpath
from src.data.load_data import load_raw
from src.data.sequences import make_windows
from src.data.split import by_trip_split, chronological_split


def clean(df: pd.DataFrame, cfg):
    d = cfg["data"]
    vcol, tcol, targets = d["vehicle_col"], d["time_col"], d["target_cols"]
    log = [("raw", len(df))]

    if tcol in df.columns:
        if not pd.api.types.is_numeric_dtype(df[tcol]):
            df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
        df = df.dropna(subset=[tcol])
        log.append(("valid_timestamp", len(df)))
    sort_cols = [c for c in (vcol, tcol) if c and c in df.columns]
    df = df.sort_values(sort_cols).drop_duplicates().reset_index(drop=True)
    log.append(("no_duplicates", len(df)))

    num_cols = [c for c in df.columns if c not in sort_cols and c not in d["categorical_cols"]]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # missing values: interpolate within each vehicle, then drop what remains
    if vcol and vcol in df.columns:
        df[num_cols] = df.groupby(vcol)[num_cols].transform(
            lambda s: s.interpolate(limit=3, limit_direction="both"))
    else:
        df[num_cols] = df[num_cols].interpolate(limit=3, limit_direction="both")
    df = df.dropna(subset=num_cols + d["categorical_cols"]).reset_index(drop=True)
    log.append(("after_missing_handling", len(df)))

    if d["drop_negative_targets"]:
        df = df[(df[targets] >= 0).all(axis=1)].reset_index(drop=True)
        log.append(("no_negative_targets", len(df)))

    if d["outlier_method"] == "iqr":
        q1, q3 = df[targets].quantile(0.25), df[targets].quantile(0.75)
        iqr, k = q3 - q1, d["iqr_factor"]
        keep = ((df[targets] >= q1 - k * iqr) & (df[targets] <= q3 + k * iqr)).all(axis=1)
        df = df[keep].reset_index(drop=True)
        log.append((f"iqr_outliers_k{k}", len(df)))
    return df, log


def build_feature_frame(df, cfg):
    d = cfg["data"]
    vcol, tcol, targets = d["vehicle_col"], d["time_col"], d["target_cols"]
    if d["categorical_cols"]:
        df = pd.get_dummies(df, columns=d["categorical_cols"], dtype=float)
    exclude = {c for c in (vcol, tcol) if c}
    if d["feature_cols"] is None:
        feats = [c for c in df.columns if c not in exclude and c not in targets
                 and pd.api.types.is_numeric_dtype(df[c])]
    else:
        feats = list(d["feature_cols"])
        for c in d["categorical_cols"]:
            feats += [x for x in df.columns if x.startswith(c + "_")]
    if d["include_targets_as_inputs"]:
        feats = feats + list(targets)
    return df, feats


def main(cfg):
    d, s = cfg["data"], cfg["split"]
    assert abs(s["train"] + s["validation"] + s["test"] - 1) < 1e-6, "split must sum to 1"
    out = ROOT / cfg["paths"]["processed"]
    out.mkdir(parents=True, exist_ok=True)
    tables = rpath(cfg, "tables")

    df, log = clean(load_raw(cfg), cfg)
    pd.DataFrame(log, columns=["step", "rows"]).to_csv(tables / "cleaning_report.csv", index=False)
    strat_col = cfg["split"].get("stratify_col", "vehicle")
    strat_src = df[strat_col] if strat_col in df.columns else None   # before one-hot encoding
    df, feats = build_feature_frame(df, cfg)
    if strat_src is not None:
        df[strat_col] = strat_src.values
    targets, L, H = d["target_cols"], d["sequence_length"], d["prediction_horizon"]
    vcol = d["vehicle_col"] if d["vehicle_col"] in df.columns else None
    groups = df.groupby(vcol, sort=False) if vcol else [("all", df)]

    # 1. split
    mode = s.get("mode", "within_trip")
    if mode == "by_trip":
        strat_col = s.get("stratify_col", "vehicle")
        strata = ({str(k): v for k, v in df.groupby(vcol)[strat_col].first().items()}
                  if strat_col in df.columns else None)
        segs, assign = by_trip_split(groups, feats, targets, s, L + H, cfg.get("seed", 42), strata)
        skipped = 0
        pd.DataFrame(assign).to_csv(tables / "trip_assignment.csv", index=False)
    elif mode == "within_trip":
        segs, skipped = chronological_split(groups, feats, targets, s, L + H)
    else:
        raise ValueError(f"split.mode must be within_trip or by_trip, got {mode!r}")
    print(f"split mode: {mode}")
    if not segs["train"]:
        raise RuntimeError("No vehicle has enough rows for the requested split/sequence length.")
    if skipped:
        print(f"[warn] skipped {skipped} vehicles with too few rows")

    # 2. scalers fitted on TRAIN rows only
    fx, ft = StandardScaler(), StandardScaler()
    fx.fit(np.vstack([f for _, f, _ in segs["train"]]))
    ft.fit(np.vstack([t for _, _, t in segs["train"]]))

    # 3. windows
    rep = []
    for name, items in segs.items():
        Xs, ys, vids, steps = [], [], [], []
        for vid, F, T in items:
            X, y, tgt = make_windows(fx.transform(F), ft.transform(T), L, H)
            Xs.append(X); ys.append(y); vids += [vid] * len(y); steps.append(tgt)
        X, y = np.concatenate(Xs), np.concatenate(ys)
        np.savez(out / f"{name}.npz", X=X, y=y, vid=np.array(vids), step=np.concatenate(steps))
        rep.append({"split": name, "rows": sum(len(f) for _, f, _ in items),
                    "sequences": len(y), "vehicles": len(items)})
    pd.DataFrame(rep).to_csv(tables / "split_report.csv", index=False)
    with open(out / "scalers.pkl", "wb") as f:
        pickle.dump({"x": fx, "y": ft}, f)
    meta = {"split_mode": mode, "features": feats, "targets": targets, "sequence_length": L,
            "prediction_horizon": H, "n_features": len(feats)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(pd.DataFrame(rep).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="optional extra yaml merged on top")
    main(load_config(extra=ap.parse_args().config))

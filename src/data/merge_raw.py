"""Merge the University of Pretoria RDE files into the single CSV the pipeline reads.

Input : public-{etios,figo,rrv}.csv from Mendeley Data y9pjtt5ngc. Looked for, in order, in --src,
        data/raw/pretoria/ and data/raw/ (so it works wherever you unzipped them).
Output: data/raw/emissions.csv   + results/tables/merge_report.csv

What it does (every step is counted in the report):
  1. parses the timestamp (the +0200 offset is identical in all files and is dropped);
  2. resamples every trip to 1 Hz (mean per second) because trips were logged at 1 Hz *and* 5 Hz;
  3. gives each trip its own series id  <vehicle>_t<trip>_l<load>  and splits a trip in two
     if a gap longer than --max-gap seconds occurs, so no window ever spans a gap;
  4. keeps ONLY vehicle-side and ambient inputs plus the pollutant targets. Exhaust-side
     measurements (concentrations, exhaust flow/humidity/temperature, *_mass_cor) are dropped
     because the pollutant mass rate is computed from them: using them would leak the target;
  5. renames targets CO_mass, CO2_mass, NOx_mass -> CO, CO2, NOx (g/s) and clips tiny negative
     analyser readings (zero drift) to 0.
HC, SO2 and PM are NOT measured in this dataset.

  python -m src.data.merge_raw
"""
import argparse

import numpy as np
import pandas as pd

from src.common import ROOT

VEHICLES = {"etios": "public-etios.csv", "figo": "public-figo.csv", "rrv": "public-rrv.csv"}
TARGETS = {"CO_mass": "CO", "CO2_mass": "CO2", "NOx_mass": "NOx"}
INPUTS = ["load", "coldStart", "gps_alt", "gps_speed", "humidity", "pressure", "temp", "rpm",
          "speed_vehicle", "throttle", "manifold_pressure", "manifold_temp", "coolant_temp",
          "fuel_flow", "fuel_rate", "air_fuel_ratio"]


def load_vehicle(path, name):
    d = pd.read_csv(path, usecols=["date", "trip", "load", "coldStart"] + INPUTS[2:] + list(TARGETS))
    n_raw = len(d)
    ts = pd.to_datetime(d["date"].str.replace(r"\s\+\d{4}$", "", regex=True),
                        format="%m/%d/%Y %H:%M:%S.%f", errors="coerce")
    d = d.drop(columns="date").assign(timestamp=ts.dt.floor("s"))
    d = d.dropna(subset=["timestamp"])
    d["coldStart"] = d["coldStart"].astype(str).str.upper().eq("TRUE").astype(float)
    d = d.rename(columns=TARGETS)
    keys = ["trip", "load", "timestamp"]
    d = d.groupby(keys, sort=True).mean().reset_index()          # 1 Hz
    d.insert(0, "vehicle", name)
    return d, n_raw


def add_series_id(d, max_gap):
    out, n_split = [], 0
    for (trip, load), g in d.groupby(["trip", "load"], sort=True):
        g = g.sort_values("timestamp")
        seg = (g["timestamp"].diff().dt.total_seconds() > max_gap).cumsum()
        n_split += int(seg.max())
        sid = [f"{g['vehicle'].iloc[0]}_t{int(trip)}_l{int(load)}" + (f"_s{s}" if s else "") for s in seg]
        out.append(g.assign(series_id=sid))
    return pd.concat(out), n_split


def find_source_dir(src_dir=None):
    """First directory that contains all three public-*.csv files."""
    cands = [src_dir] if src_dir else []
    cands += ["data/raw/pretoria", "data/raw"]
    for c in cands:
        d = ROOT / c
        if all((d / fn).exists() for fn in VEHICLES.values()):
            return d
    missing = {c: [fn for fn in VEHICLES.values() if not (ROOT / c / fn).exists()] for c in cands}
    raise FileNotFoundError(
        "Could not find all three Pretoria files (public-etios.csv, public-figo.csv, public-rrv.csv).\n"
        + "\n".join(f"  {ROOT / c}: missing {m}" for c, m in missing.items())
        + "\nUnzip the Mendeley download and put the three files in data/raw/pretoria/ (or data/raw/).")


def main(src_dir=None, out_file="data/raw/emissions.csv", max_gap=5):
    src, frames, rep = find_source_dir(src_dir), [], []
    print(f"reading Pretoria files from {src}")
    for name, fn in VEHICLES.items():
        p = src / fn
        d, n_raw = load_vehicle(p, name)
        d, n_split = add_series_id(d, max_gap)
        frames.append(d)
        rep.append({"vehicle": name, "rows_target_clipped_to_0": int((d[list(TARGETS.values())] < 0).any(axis=1).sum()), "raw_rows": n_raw, "rows_1hz": len(d), "trips": d[["trip", "load"]]
                    .drop_duplicates().shape[0], "series": d["series_id"].nunique(), "gap_splits": n_split,
                    "first": d["timestamp"].min(), "last": d["timestamp"].max()})
    df = pd.concat(frames, ignore_index=True)
    tg = list(TARGETS.values())
    df[tg] = df[tg].clip(lower=0)
    cols = ["series_id", "timestamp", "vehicle"] + INPUTS + tg
    df = df[cols].sort_values(["series_id", "timestamp"])
    out = ROOT / out_file
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    rep = pd.DataFrame(rep)
    (ROOT / "results/tables").mkdir(parents=True, exist_ok=True)
    rep.to_csv(ROOT / "results/tables/merge_report.csv", index=False)
    print(rep.to_string(index=False))
    print(f"\nwrote {out}  ({len(df)} rows, {df['series_id'].nunique()} series)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None, help="folder with the three CSVs (default: auto-detect)")
    ap.add_argument("--out", default="data/raw/emissions.csv")
    ap.add_argument("--max-gap", type=float, default=5, help="seconds; longer gaps split a trip")
    a = ap.parse_args()
    main(a.src, a.out, a.max_gap)

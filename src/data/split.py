"""Train / validation / test splits.

within_trip (default): every series is cut in time order into train | val | test.
by_trip: whole trips are assigned to a split, so the test set contains trips the model never saw
         (stricter; a trip's gap-split fragments `..._s1`, `..._s2` always stay together).
"""
import re

import numpy as np

_FRAGMENT = re.compile(r"_s\d+$")


def trip_key(series_id):
    return _FRAGMENT.sub("", str(series_id))


def by_trip_split(groups, feats, targets, split_cfg, min_rows, seed=42, strata=None):
    """groups: iterable of (series_id, DataFrame sorted by time). strata: {series_id: label}
    (e.g. vehicle) so every vehicle contributes trips to each split. Returns (segs, assignment_df_rows)."""
    frac = {"train": split_cfg["train"], "val": split_cfg["validation"], "test": split_cfg["test"]}
    trips = {}
    for sid, g in groups:
        k = trip_key(sid)
        e = trips.setdefault(k, {"stratum": (strata or {}).get(sid, "all"), "parts": []})
        e["parts"].append((str(sid), g))
    by_stratum = {}
    for k, e in trips.items():
        by_stratum.setdefault(e["stratum"], []).append(k)
    rng = np.random.default_rng(seed)
    segs = {"train": [], "val": [], "test": []}
    assign = []
    for st in sorted(by_stratum):
        keys = sorted(by_stratum[st])
        rng.shuffle(keys)
        total = sum(len(g) for k in keys for _, g in trips[k]["parts"])
        got = {n: 0 for n in frac}
        for k in keys:
            n = sum(len(g) for _, g in trips[k]["parts"])
            # give the trip to the split that is furthest below its target share
            name = max(frac, key=lambda x: (frac[x] * total - got[x], x == "train"))
            got[name] += n
            for sid, g in trips[k]["parts"]:
                segs[name].append((sid, g[feats].to_numpy(float), g[targets].to_numpy(float)))
            assign.append({"stratum": st, "trip": k, "split": name, "rows": n})
    return segs, assign


def chronological_split(groups, feats, targets, split_cfg, min_rows):
    """groups: iterable of (vehicle_id, DataFrame sorted by time).

    Each vehicle's rows are cut in time order into train | validation | test, so no future
    information reaches training. Vehicles whose validation or test part would be shorter
    than `min_rows` (sequence_length + horizon) are skipped.
    Returns ({'train': [...], 'val': [...], 'test': [...]}, n_skipped) where each item is
    (vehicle_id, features_array, targets_array).
    """
    segs = {"train": [], "val": [], "test": []}
    skipped = 0
    for vid, g in groups:
        n = len(g)
        n_tr, n_va = int(n * split_cfg["train"]), int(n * split_cfg["validation"])
        if n_va < min_rows or (n - n_tr - n_va) < min_rows:
            skipped += 1
            continue
        F, T = g[feats].to_numpy(float), g[targets].to_numpy(float)
        segs["train"].append((str(vid), F[:n_tr], T[:n_tr]))
        segs["val"].append((str(vid), F[n_tr:n_tr + n_va], T[n_tr:n_tr + n_va]))
        segs["test"].append((str(vid), F[n_tr + n_va:], T[n_tr + n_va:]))
    return segs, skipped

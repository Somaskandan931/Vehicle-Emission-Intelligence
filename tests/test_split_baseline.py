import numpy as np
import pandas as pd
import pytest

from src.data.split import by_trip_split, trip_key
from src.evaluation.baseline import persistence_predictions


def _groups():
    out = []
    for v in ("a", "b"):
        for t in range(6):
            n = 100
            df = pd.DataFrame({"f": np.arange(n, dtype=float), "y": np.arange(n, dtype=float)})
            out.append((f"{v}_t{t}_l0", df))
            if t == 0:                                   # a gap-split fragment of trip 0
                out.append((f"{v}_t{t}_l0_s1", df.copy()))
    return out


def test_trip_key_merges_fragments():
    assert trip_key("rrv_t3_l1_s4") == trip_key("rrv_t3_l1") == "rrv_t3_l1"


def test_by_trip_split_keeps_trips_and_fragments_together():
    g = _groups()
    strata = {sid: sid.split("_")[0] for sid, _ in g}
    segs, assign = by_trip_split(g, ["f"], ["y"], {"train": .6, "validation": .2, "test": .2}, 5, 0, strata)
    ids = {n: {trip_key(s) for s, _, _ in segs[n]} for n in segs}
    assert not (ids["train"] & ids["val"]) and not (ids["train"] & ids["test"]) and not (ids["val"] & ids["test"])
    assert sum(len(v) for v in segs.values()) == len(g)              # nothing dropped or duplicated
    for v in ("a", "b"):                                             # each vehicle appears in every split
        for n in segs:
            assert any(s.startswith(v) for s, _, _ in segs[n])
    assert {a["trip"] for a in assign} == {trip_key(s) for s, _ in g}


def test_by_trip_split_is_deterministic():
    g = _groups()
    a = by_trip_split(g, ["f"], ["y"], {"train": .6, "validation": .2, "test": .2}, 5, 7)[1]
    b = by_trip_split(g, ["f"], ["y"], {"train": .6, "validation": .2, "test": .2}, 5, 7)[1]
    assert a == b


def test_persistence_returns_last_input_value(tmp_path):
    import json, pickle
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(0)
    F = rng.normal(5, 2, (200, 3))                    # columns: speed, CO, NOx (targets last)
    fx = StandardScaler().fit(F)
    T = F[:, 1:]
    ft = StandardScaler().fit(T)
    X = fx.transform(F)[:-1].reshape(199, 1, 3).astype(np.float32)
    proc = tmp_path / "proc"; proc.mkdir()
    for n in ("val", "test"):
        np.savez(proc / f"{n}.npz", X=X, y=ft.transform(T[1:]).astype(np.float32),
                 vid=np.array(["s"] * 199), step=np.arange(199))
    (proc / "meta.json").write_text(json.dumps({"features": ["speed", "CO", "NOx"], "targets": ["CO", "NOx"]}))
    pickle.dump({"x": fx, "y": ft}, open(proc / "scalers.pkl", "wb"))
    table, pdf = persistence_predictions({"paths": {"processed": str(proc)}}, "test")
    assert np.allclose(pdf["CO_pred"], F[:-1, 1]) and np.allclose(pdf["NOx_pred"], F[:-1, 2])
    assert np.allclose(pdf["CO_true"], F[1:, 1])


def test_persistence_needs_targets_as_inputs(tmp_path):
    import json, pickle
    from sklearn.preprocessing import StandardScaler
    proc = tmp_path / "proc"; proc.mkdir()
    (proc / "meta.json").write_text(json.dumps({"features": ["speed"], "targets": ["CO"]}))
    pickle.dump({"x": StandardScaler().fit([[1.0], [2.0]]), "y": StandardScaler().fit([[1.0], [2.0]])},
                open(proc / "scalers.pkl", "wb"))
    with pytest.raises(RuntimeError, match="include_targets_as_inputs"):
        persistence_predictions({"paths": {"processed": str(proc)}}, "test")

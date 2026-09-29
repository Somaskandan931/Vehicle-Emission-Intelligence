"""Load the raw dataset and validate it against config.yaml."""
import pandas as pd

from src.common import ROOT


def load_raw(cfg) -> pd.DataFrame:
    path = ROOT / cfg["data"]["raw_file"]
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}.\n"
            "Pretoria data: put public-etios/figo/rrv.csv in data/raw/ (or data/raw/pretoria/) and run\n"
            "  python -m src.data.merge_raw\n"
            "Other data: see data/raw/README.md. Pipeline smoke test with synthetic data:  pytest -q"
        )
    df = pd.read_csv(path)
    d = cfg["data"]
    missing = [c for c in d["target_cols"] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Target columns {missing} are not in the dataset. Columns found: {list(df.columns)}.\n"
            "If the dataset lacks some pollutants, change data.target_cols in config.yaml "
            "AND adjust the paper's claims accordingly."
        )
    return df

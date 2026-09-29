"""Hybrid Bi-LSTM + Uni-LSTM:  H = lambda * H_bi + (1 - lambda) * H_uni.

  python -m src.training.train_hybrid --search        # choose lambda on VALIDATION only
  python -m src.training.train_hybrid                 # final run (lambda from config or search)
  python -m src.training.train_hybrid --lam 0.5       # final run with an explicit lambda

Lambda search criterion (predefined): lowest best-epoch validation MSE in standardised units.
Writes experiments/lambda_search/lambda_results.csv and best_lambda.json. Test metrics are never
computed during the search. lambda=0 is Uni-only and lambda=1 is Bi-only.
"""
import argparse
import json

import pandas as pd

from src.common import ROOT, load_config
from src.training.train_utils import run


def _search_dir(cfg):
    p = ROOT / cfg["paths"]["experiments"] / "lambda_search"
    p.mkdir(parents=True, exist_ok=True)
    return p


def lambda_search(cfg):
    out = _search_dir(cfg)
    rows = []
    for lam in cfg["hybrid"]["lambda_grid"]:
        info = run(cfg, "hybrid", lam, name=f"lambda_search/lam_{lam:.1f}",
                   evaluate_test=False, verbose=False)
        rows.append({"lambda": lam, "val_mse": info["best_val_loss_scaled_mse"],
                     "best_epoch": info["best_epoch"]})
        print(rows[-1])
    df = pd.DataFrame(rows)
    df.to_csv(out / "lambda_results.csv", index=False)
    best = float(df.loc[df["val_mse"].idxmin(), "lambda"])
    (out / "best_lambda.json").write_text(
        json.dumps({"best_lambda": best, "criterion": "min validation MSE (standardised units)"}))
    print("best lambda:", best)
    return best


def resolve_lambda(cfg, lam=None):
    if lam is not None:
        return lam
    if cfg["hybrid"]["lambda"] is not None:
        return cfg["hybrid"]["lambda"]
    p = _search_dir(cfg) / "best_lambda.json"
    if not p.exists():
        raise RuntimeError("lambda is null in configs/hybrid_lstm.yaml and no search result found; "
                           "run:  python -m src.training.train_hybrid --search")
    return json.loads(p.read_text())["best_lambda"]


def main(cfg=None, lam=None):
    cfg = cfg or load_config("hybrid")
    return run(cfg, "hybrid", resolve_lambda(cfg, lam))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", action="store_true", help="run the validation-only lambda search")
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--config", default=None, help="optional extra yaml merged on top")
    a = ap.parse_args()
    c = load_config("hybrid", extra=a.config)
    print(lambda_search(c) if a.search else main(c, a.lam))

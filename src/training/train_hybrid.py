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
import yaml

from src.common import ROOT, load_config
from src.training.train_utils import run


def _search_dir(cfg):
    p = ROOT / cfg["paths"]["experiments"] / "lambda_search"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _finished(run_dir, cfg, lam):
    """A search run counts as done when its history.json exists and was made with the same
    model / training / data settings and the same lambda. Returns its summary or None."""
    hist, conf = run_dir / "history.json", run_dir / "config.yaml"
    if not (hist.exists() and conf.exists()):
        return None
    try:
        summ = json.loads(hist.read_text())["summary"]
        old = yaml.safe_load(conf.read_text())
    except Exception:
        return None
    same = (abs(float(summ.get("lambda", -1)) - lam) < 1e-9
            and all(old.get(k) == cfg.get(k) for k in ("model", "training"))
            and old.get("data", {}).get("sequence_length") == cfg["data"]["sequence_length"]
            and old.get("data", {}).get("target_cols") == cfg["data"]["target_cols"])
    return summ if same else None


def lambda_search(cfg, grid=None, resume=True):
    """Validation-only search. Finished runs found under experiments/lambda_search/ are reused
    (resume=True), so an interrupted search continues instead of starting over."""
    out = _search_dir(cfg)
    rows = []
    grid = list(grid or cfg["hybrid"]["lambda_grid"])
    for i, lam in enumerate(grid, 1):
        name = f"lambda_search/lam_{lam:.1f}"
        prev = _finished(ROOT / cfg["paths"]["experiments"] / name, cfg, lam) if resume else None
        if prev is not None:
            print(f"[lambda search {i}/{len(grid)}] lambda={lam:.2f} already done "
                  f"(val_mse {prev['best_val_loss_scaled_mse']:.5f}) - skipping", flush=True)
            info = prev
        else:
            print(f"[lambda search {i}/{len(grid)}] lambda={lam:.2f} (validation only; early stopping applies)", flush=True)
            info = run(cfg, "hybrid", lam, name=name, evaluate_test=False, verbose=True)
        rows.append({"lambda": lam, "val_mse": info["best_val_loss_scaled_mse"],
                     "best_epoch": info["best_epoch"]})
        print(rows[-1])
        pd.DataFrame(rows).to_csv(out / "lambda_results.csv", index=False)   # progress survives a crash
    df = pd.DataFrame(rows)
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
    ap.add_argument("--grid", type=float, nargs="*", default=None, help="override lambda_grid, e.g. --grid 0 0.3 0.5 1")
    ap.add_argument("--no-resume", action="store_true", help="retrain search runs that are already finished")
    a = ap.parse_args()
    c = load_config("hybrid", extra=a.config)
    print(lambda_search(c, a.grid, not a.no_resume) if a.search else main(c, a.lam))

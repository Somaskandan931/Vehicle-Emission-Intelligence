"""Run the whole experiment pipeline end to end.

  python -m src.run_all                          # everything
  python -m src.run_all --stages prep uni bi     # selected stages
Stages: merge analyze prep persistence uni bi lambda hybrid ablation emission plots
  python -m src.run_all --config configs/by_trip.yaml   # stricter split: whole trips held out
If data/raw/emissions.csv is missing, the merge stage runs automatically when the three Pretoria
CSVs are present (data/raw/pretoria/ or data/raw/).
"""
import argparse

from src.common import ROOT, load_config

ORDER = ["merge", "analyze", "prep", "persistence", "uni", "bi", "lambda", "hybrid", "ablation",
         "emission", "plots"]
DEFAULT = [x for x in ORDER if x != "merge"]      # merge is explicit, or automatic when emissions.csv is missing
NEEDS_RAW = {"analyze", "prep"}
NEEDS_PROCESSED = {"persistence", "uni", "bi", "lambda", "hybrid"}


def _preflight(st, cfg):
    raw = ROOT / cfg["data"]["raw_file"]
    if NEEDS_RAW & set(st) and "merge" not in st and not raw.exists():
        from src.data.merge_raw import main as merge
        print(f"{raw} not found - running the merge stage first")
        merge()
    if NEEDS_PROCESSED & set(st) and "prep" not in st and not (ROOT / cfg["paths"]["processed"] / "train.npz").exists():
        raise SystemExit(f"No processed data in {ROOT / cfg['paths']['processed']}. Add 'prep' to --stages "
                         "(or run: python -m src.data.preprocess) before training.")


def run_pipeline(stages=DEFAULT, extra=None, overrides=None):
    st = [s for s in ORDER if s in stages]
    base = load_config(extra=extra, overrides=overrides)
    kw = dict(extra=extra, overrides=overrides)
    _preflight(st, base)
    if "merge" in st:
        from src.data.merge_raw import main as f; f()
    if "analyze" in st:
        from src.data.analyze_data import main as f; f(base)
    if "prep" in st:
        from src.data.preprocess import main as f; f(base)
    if "persistence" in st:
        from src.evaluation.baseline import main as f; f(base)
    if "uni" in st:
        from src.training.train_uni import main as f; print(f(load_config("uni", **kw)))
    if "bi" in st:
        from src.training.train_bi import main as f; print(f(load_config("bi", **kw)))
    if "lambda" in st or "hybrid" in st:
        from src.training import train_hybrid as h
        hc = load_config("hybrid", **kw)
        if "lambda" in st and hc["hybrid"]["lambda"] is None:
            h.lambda_search(hc)
        if "hybrid" in st:
            print(h.main(hc))
    if "ablation" in st:
        from src.evaluation.ablation import main as f; f(base)
    if "emission" in st:
        from src.emission.dehs import main as f; f(base)
    if "plots" in st:
        from src.visualization import main as f; f(base)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="optional extra yaml merged on top")
    ap.add_argument("--stages", nargs="*", default=DEFAULT, choices=ORDER)
    a = ap.parse_args()
    run_pipeline(a.stages, a.config)

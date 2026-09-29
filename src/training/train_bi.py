"""Train the Bi-LSTM baseline.   python -m src.training.train_bi"""
import argparse

from src.common import load_config
from src.training.train_utils import run


def main(cfg=None):
    cfg = cfg or load_config("bi")
    return run(cfg, "bi")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="optional extra yaml merged on top")
    print(main(load_config("bi", extra=ap.parse_args().config)))

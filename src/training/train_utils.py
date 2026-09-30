"""Shared training loop for uni | bi | hybrid models.

Outputs of a *final* run (evaluate_test=True), named <name> = uni_lstm | bi_lstm | hybrid_lstm:
  models/checkpoints/<name>.pt          best-validation weights
  results/metrics/<name>.csv            test metrics (per pollutant + MACRO_AVG)
  results/predictions/<name>.csv        test predictions (original units)
  experiments/<name>/                   config.yaml, training.log, history.json, metrics_val.csv
A lambda-search run (evaluate_test=False) never touches the test set and writes only under
experiments/lambda_search/lam_<x>/ (including its checkpoint).
"""
import json
import time

import torch
import yaml

from src.common import ROOT, device_name, get_device, rpath, set_seed
from src.evaluation.evaluate import evaluate_split, load_split
from src.models import build_model

EVAL_BS = 4096                                          # validation batch size (does not change the result)
DEFAULT_NAMES = {"uni": "uni_lstm", "bi": "bi_lstm", "hybrid": "hybrid_lstm"}


def _tensors(split, cfg, dev):
    """Whole split as tensors. On a GPU they live on the device (train ~300 MB, val/test ~65 MB each),
    so an epoch needs no DataLoader, no per-batch collation and no host->device copies."""
    d = load_split(cfg, split)
    X, y = torch.from_numpy(d["X"]), torch.from_numpy(d["y"])
    if dev.type != "cpu" and cfg.get("compute", {}).get("data_on_device", True):
        X, y = X.to(dev), y.to(dev)
    return X, y


def _batches(X, y, bs, shuffle):
    n = len(X)
    order = torch.randperm(n, device=X.device) if shuffle else None
    for i in range(0, n, bs):
        if order is None:
            yield X[i:i + bs], y[i:i + bs]
        else:
            idx = order[i:i + bs]
            yield X[idx], y[idx]


def run(cfg, kind, lam=None, name=None, evaluate_test=True, verbose=True):
    t = cfg["training"]
    set_seed(cfg["seed"])
    dev = get_device(cfg)
    amp = bool(cfg.get("compute", {}).get("amp", False)) and dev.type == "cuda"
    name = name or DEFAULT_NAMES[kind]
    out = ROOT / cfg["paths"]["experiments"] / name
    out.mkdir(parents=True, exist_ok=True)
    ckpt = (rpath(cfg, "checkpoints") / f"{name}.pt") if evaluate_test else (out / "model.pt")

    Xtr, ytr = _tensors("train", cfg, dev)
    Xva, yva = _tensors("val", cfg, dev)
    nf, no = Xtr.shape[-1], ytr.shape[-1]
    model = build_model(kind, nf, no, cfg, lam).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"],
                           fused=(dev.type == "cuda" and not amp))
    lossf = torch.nn.MSELoss()

    best, best_ep, bad, hist = float("inf"), 0, 0, []
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    log = open(out / "training.log", "w")
    log.write(f"device {dev} ({device_name(dev)}); amp {amp}; torch {torch.__version__}\n")
    if verbose:
        print(f"training on {device_name(dev)}")
    start = time.time()
    for ep in range(1, t["epochs"] + 1):
        model.train()
        tl = torch.zeros((), device=dev)               # accumulate on device: no per-batch GPU sync
        for xb, yb in _batches(Xtr, ytr, t["batch_size"], True):
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=amp):
                loss = lossf(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            scaler.step(opt)
            scaler.update()
            tl += loss.detach().float() * len(xb)
        tl = tl.item() / len(Xtr)
        model.eval()
        vl = torch.zeros((), device=dev)
        with torch.no_grad(), torch.autocast(device_type=dev.type, enabled=amp):
            for x, y in _batches(Xva, yva, EVAL_BS, False):
                vl += lossf(model(x.to(dev)).float(), y.to(dev)).detach() * len(x)
        vl = vl.item() / len(Xva)
        hist.append({"epoch": ep, "train_loss": tl, "val_loss": vl})
        msg = f"epoch {ep:3d}  train {tl:.5f}  val {vl:.5f}"
        log.write(msg + "\n")
        if verbose:
            print(msg)
        if vl < best - 1e-6:
            best, best_ep, bad = vl, ep, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= t["patience"]:
                break
    log.write(f"best_val_loss {best:.6f} at epoch {best_ep}; {time.time() - start:.1f}s\n")
    log.close()

    model.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True))
    val_table, _ = evaluate_split(cfg, model, "val", dev)
    val_table.to_csv(out / "metrics_val.csv", index=False)
    if evaluate_test:
        test_table, test_pred = evaluate_split(cfg, model, "test", dev)
        test_table.to_csv(rpath(cfg, "metrics") / f"{name}.csv", index=False)
        test_pred.to_csv(rpath(cfg, "predictions") / f"{name}.csv", index=False)
    info = {"model": kind, "lambda": lam, "best_val_loss_scaled_mse": best, "best_epoch": best_ep,
            "params": sum(p.numel() for p in model.parameters()),
            "device": device_name(dev), "amp": amp, "torch": str(torch.__version__)}
    (out / "history.json").write_text(json.dumps({"summary": info, "history": hist}, indent=2))
    (out / "config.yaml").write_text(yaml.safe_dump({**cfg, "run": info}))
    return info

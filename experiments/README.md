# Experiments

One folder per run, written by `src/training/`:

| Folder | Contents |
|---|---|
| `uni_lstm/`, `bi_lstm/`, `hybrid_lstm/` | `config.yaml`, `training.log`, `history.json`, `metrics_val.csv` |
| `persistence/` | `metrics_val.csv` (non-learned baseline; test metrics in `results/metrics/persistence.csv`) |
| `lambda_search/` | `lambda_results.csv`, `best_lambda.json`, per-lambda run folders (git-ignored) |

Checkpoints go to `models/checkpoints/`, test metrics to `results/metrics/`, test predictions to
`results/predictions/`. Only numbers from these files may appear in the paper.

## Experiment log
Record every real run (synthetic/smoke-test output is never logged here).

| Date | Experiment | Config changes | Result file | Notes |
|---|---|---|---|---|
| | | | | |

## Traceability
Paper table/figure -> `results/tables/*.csv`, `results/ablation/*.csv` or `results/figures/*`
-> `results/metrics/<model>.csv` -> `results/predictions/<model>.csv`

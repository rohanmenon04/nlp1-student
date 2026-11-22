# %% point 3 continued: embed size and optim settings for embed-and-pool
#   run: uv run python -m part2.pool_tuning

# %% Point 3, continued: embedding size and optimisation settings for embed-and-pool
# Starts from the best step-2 system (twitter + trainable + mean) and changes ONE setting at a time,
# like part2/rnn.py does for the LSTM. Same folds, dev split and early stopping as part2/pool.py.
#   run: uv run python -m part2.pool_tuning | tee part2/results/pool_tuning_log.txt
#
# Cache trap: systems are cached by name. If you change BASE itself, use a new CACHE filename.

import os
import pickle

import pandas as pd

from part2.pool import CACHE as POOL_CACHE
from part2.pool import SHORT as POOL_SHORT
from part2.pool import run_system
from part2.rnn import compare, name

CACHE = "part2/pool_tuning_results.pkl"
RESULTS_DIR = "part2/results"

# all four glove-twitter sizes are trained on the same tweets with the same 1.2M-word vocabulary,
# so only the vector size differs (coverage is identical)
SHORT = {**POOL_SHORT, "glove-twitter-25": "twitter25", "glove-twitter-100": "twitter100",
         "glove-twitter-200": "twitter200"}

BASE = dict(name="glove-twitter-50", freeze=False, pooling="mean",
            lr=1e-3, weight_decay=1e-4, dropout=0.0, hidden=50)
SETTINGS = ["lr", "weight_decay", "dropout", "hidden"]


def variant(**changes):
    return {**BASE, **changes}


SYSTEMS = [
    BASE,   # rerun here as a reproducibility check
    variant(lr=3e-4), variant(lr=3e-3),    # learning rate
    variant(weight_decay=1e-2),      # stronger weight decay (base 1e-4 is very weak)
    variant(dropout=0.2), variant(dropout=0.5),  # dropout in the classifier head
    variant(hidden=25), variant(hidden=100),   # classifier head size
    variant(name="glove-twitter-25"),   # embedding size
    variant(name="glove-twitter-100"),
    variant(name="glove-twitter-200"), # slowest by far: 1.2M x 200 trainable parameters
]


def key(cfg):
    """BASE is ('twitter', 'trainable-mean'), same as in pool.py. Every setting that differs from BASE
    gets a tag, e.g. ('twitter', 'trainable-mean-lr0.003') or ('twitter200', 'trainable-mean')."""
    desc = f"{'frozen' if cfg['freeze'] else 'trainable'}-{cfg['pooling']}"
    tags = {"lr": "lr", "weight_decay": "wd", "dropout": "drop", "hidden": "h"}
    for setting, tag in tags.items():
        if cfg[setting] != BASE[setting]:
            desc += f"-{tag}{cfg[setting]:g}"
    return (SHORT[cfg["name"]], desc)


def run_all():
    results = {}
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            results = pickle.load(f)
    for cfg in SYSTEMS:
        k = key(cfg)
        if k in results:
            print(f"skip {name(k)} (cached)")
            continue
        print(f"=== {name(k)}", flush=True)
        settings = {s: cfg[s] for s in SETTINGS}
        results[k] = run_system(cfg["name"], cfg["freeze"], cfg["pooling"], **settings)
        with open(CACHE, "wb") as f:
            pickle.dump(results, f)
    return results


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = run_all()

    table = pd.DataFrame([{"system": name(k),
                           "f1_mean": r["f1"].mean(), "f1_std": r["f1"].std(ddof=1),
                           "acc_mean": r["acc"].mean(), "best_epoch_mean": r["best_epoch"].mean()}
                          for k, r in results.items()]).sort_values("f1_mean", ascending=False)
    print(table.round(4).to_string(index=False))
    table.to_csv(f"{RESULTS_DIR}/pool_tuning.csv", index=False)


    base = key(BASE)
    with open(POOL_CACHE, "rb") as f:
        old = pickle.load(f)
    print(f"\nBASE f1 here: {results[base]['f1'].mean():.4f}   in pool_results.pkl: {old[base]['f1'].mean():.4f}")

    print("\n=== BASE vs one change (macro-F1, ROPE 0.005) ===")
    for cfg in SYSTEMS[1:]:
        compare(results, base, key(cfg))

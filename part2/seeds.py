# %% Point 5: experimental variance across random seeds
# Every earlier result used seed 42 only. Here the key systems are rerun with extra seeds on the SAME folds.
# The seed changes weight init, batch order, the dev split for early stopping (and word dropout), not the folds,
# so fold scores stay paired across systems and the correlated t-test can use runs = number of seeds.
#   run: uv run python -m part2.seeds | tee part2/results/seeds_log.txt

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from part1.analysis import BenavoliCorrelatedT
from part1.data import pool_labels
from part2 import pool, rnn
from part2.pool_tuning import CACHE as TUNING_CACHE

CACHE = "part2/seeds_results.pkl"
RESULTS_DIR = "part2/results"
ROPE = 0.005
SEEDS = [42, 1, 2]   # 42 is already in the earlier caches and is reused, not retrained

# name -> (kind, config). Pool configs use pool.fit_predict, LSTM configs use rnn.fit_predict.
SYSTEMS = {
    "pool twitter50 trainable-mean":  ("pool", dict(name="glove-twitter-50", freeze=False, pooling="mean")),
    "pool twitter100 trainable-mean": ("pool", dict(name="glove-twitter-100", freeze=False, pooling="mean")),
    "lstm twitter50 frozen-h50":      ("lstm", rnn.BASE),
    "lstm twitter50 trainable-h50":   ("lstm", rnn.variant(freeze=False)),
}

# where the seed-42 results already live
SEED42 = {
    "pool twitter50 trainable-mean":  (pool.CACHE, ("twitter", "trainable-mean")),
    "pool twitter100 trainable-mean": (TUNING_CACHE, ("twitter100", "trainable-mean")),
    "lstm twitter50 frozen-h50":      (rnn.CACHE, rnn.key(rnn.BASE)),
    "lstm twitter50 trainable-h50":   (rnn.CACHE, rnn.key(rnn.variant(freeze=False))),
}

COMPARISONS = [
    ("pool twitter50 trainable-mean", "pool twitter100 trainable-mean"),   # embedding size (P = 0.81 with one seed)
    ("lstm twitter50 frozen-h50", "lstm twitter50 trainable-h50"),         # frozen vs trainable LSTM
    ("pool twitter50 trainable-mean", "lstm twitter50 trainable-h50"),     # pool vs LSTM, same embeddings
    ("pool twitter100 trainable-mean", "lstm twitter50 trainable-h50"),    # best pool vs best LSTM
]


def run_seed(kind, cfg, seed):
    oof = np.empty(len(pool_labels), dtype=int)
    f1s, accs, epochs = [], [], []
    for fold, (_, val_idx) in enumerate(pool.folds):
        if kind == "pool":
            preds, best_epoch = pool.fit_predict(fold, cfg["name"], cfg["freeze"], cfg["pooling"], seed=seed)
        else:
            preds, best_epoch, _, _ = rnn.fit_predict(fold, cfg, seed=seed)
        oof[val_idx] = preds
        y = pool_labels[val_idx]
        f1s.append(f1_score(y, preds, average="macro"))
        accs.append(accuracy_score(y, preds))
        epochs.append(best_epoch)
        print(f"  fold {fold + 1}: f1={f1s[-1]:.4f}  (best epoch {best_epoch})", flush=True)
    return {"f1": np.array(f1s), "acc": np.array(accs), "oof": oof, "best_epoch": np.array(epochs)}


def load(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}


def seed42(system):
    path, k = SEED42[system]
    c = load(path)
    return (c["cv"] if "cv" in c else c)[k]


def run_all():
    cache = load(CACHE)
    for system, (kind, cfg) in SYSTEMS.items():
        cache[(system, 42)] = seed42(system)
        for seed in SEEDS[1:]:
            if (system, seed) in cache:
                print(f"skip {system} seed {seed} (cached)")
                continue
            print(f"=== {system}  seed {seed}", flush=True)
            cache[(system, seed)] = run_seed(kind, cfg, seed)
            with open(CACHE, "wb") as f:
                pickle.dump(cache, f)
    return cache


def scores(cache, system):
    """Fold scores for all seeds, seed-major, as baycomp expects for runs > 1."""
    return np.concatenate([cache[(system, s)]["f1"] for s in SEEDS])


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cache = run_all()

    rows = []
    for system in SYSTEMS:
        means = [cache[(system, s)]["f1"].mean() for s in SEEDS]
        rows.append({"system": system, **{f"f1_seed{s}": m for s, m in zip(SEEDS, means)},
                     "f1_mean": np.mean(means), "seed_sd": np.std(means, ddof=1),
                     "fold_sd": cache[(system, 42)]["f1"].std(ddof=1)})
    table = pd.DataFrame(rows)
    print("\n=== Mean macro-F1 per seed (5 folds each) ===")
    print(table.round(4).to_string(index=False))
    table.to_csv(f"{RESULTS_DIR}/seeds.csv", index=False)

    print(f"\n=== Comparisons over {len(SEEDS)} seeds x 5 folds (ROPE {ROPE}) vs seed 42 only ===")
    for a, b in COMPARISONS:
        multi = BenavoliCorrelatedT(scores(cache, a), scores(cache, b), runs=len(SEEDS), names=[a, b])
        single = BenavoliCorrelatedT(cache[(a, 42)]["f1"], cache[(b, 42)]["f1"], names=[a, b])
        pm, ps = multi.probs(ROPE), single.probs(ROPE)
        print(f"A={a:<31} B={b:<31}")
        print(f"   3 seeds: P(A)={pm[0]:.3f} P(tie)={pm[1]:.3f} P(B)={pm[2]:.3f}  mean diff {multi.mean_diff:+.4f}")
        print(f"   seed 42: P(A)={ps[0]:.3f} P(tie)={ps[1]:.3f} P(B)={ps[2]:.3f}  mean diff {single.mean_diff:+.4f}")

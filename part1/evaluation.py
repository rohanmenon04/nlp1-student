"""
Add evaluation code here:

1. Cross-validation over the 9 systems
2. results table
3. Bayesian ROPE comparison and error analysis
"""

import os
import pickle
import itertools

import matplotlib.pyplot as plt
from part1.analysis import BenavoliCorrelatedT, plot_box_and_points

import numpy as np
import pandas as pd

from part1.data import *
from part1.models import encode, fit_predict_logistic, bpe_tokenizer, fit_predict_neural, build_bigram_vocab

from scipy import stats
from scipy.sparse import csr_matrix

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score, f1_score

N_FOLDS = 5
SEED = 42 #meaning of life
CACHE = "part1/cv_results.pkl" # so if it crashes we dont lose finished systems. delete part1/cv_results.pkl if you want a fresh run

ENCODERS = ["indicator", "bow", "bigram"]
ARCHS = ["logistic", "mlp", "deep_mlp"]


### For ROPE 
ROPE = 0.005 # +-0.5 F1 points
FIG_DIR = "part1/figures"

texts = np.array(pool_texts)

####### BELOW IS A QUICK TEST
#created once
folds = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(texts, pool_labels))
# folds is a list of 5 pairs (train_indices, val_indices)
# every system should loop over same list

"""
print(len(folds[0][0]), len(folds[0][1])) #train on [0][0], validate on [0][1] (train_indices, val_indices)

texts = np.array(pool_texts)

train_idx, val_idx = folds[0] #just on one fold for now. 
X_tr = encode("bow", texts[train_idx])
X_val = encode("bow", texts[val_idx])
y_tr, y_val = pool_labels[train_idx], pool_labels[val_idx]

preds, _ = fit_predict_logistic(X_tr, y_tr, X_val, seed=0)
print(accuracy_score(y_val, preds), f1_score(y_val, preds, average="macro"))
"""

def run_system(encoder, arch):
    """Cross-validate one (encoder, architecture) pair."""
    accs, f1s = [], []
    oof_preds = np.zeros(len(texts), dtype=int) # out of fold predictions, one per tweet

    for k, (train_idx, val_idx) in enumerate(folds):
        # bigram vocab from train fold (otherwise there would be leakage from validation)
        bigram_vocab = build_bigram_vocab(texts[train_idx], bpe_tokenizer)
        X_tr = encode(encoder, texts[train_idx], bigram_vocab)
        X_val = encode(encoder, texts[val_idx], bigram_vocab)
        y_tr, y_val = pool_labels[train_idx], pool_labels[val_idx]

        if arch == "logistic":
            preds, _ = fit_predict_logistic(X_tr, y_tr, X_val, seed=SEED)
        else:
            preds, _ = fit_predict_neural(arch, csr_matrix(X_tr), y_tr, csr_matrix(X_val), seed=SEED)

        accs.append(accuracy_score(y_val, preds))
        f1s.append(f1_score(y_val, preds, average="macro")) #macro because of class imbalance in dataset.
        oof_preds[val_idx] = preds
        print(f" [{encoder}+{arch}] fold {k+1}/{N_FOLDS}: acc={accs[-1]:.4f} macro-F1={f1s[-1]:.4f}", flush=True)

    return {"acc": np.array(accs), "f1": np.array(f1s), "oof": oof_preds}

def run_all():
    results = {}
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            results = pickle.load(f)

    for encoder, arch in itertools.product(ENCODERS, ARCHS):
        if (encoder, arch) in results:
            print(f"skipping {encoder}+{arch} (cached)")
            continue
        print(f"running {encoder}+{arch}", flush=True)
        results[(encoder, arch)] = run_system(encoder, arch)
        with open(CACHE, "wb") as f:
            pickle.dump(results, f)

    return results

def results_table(results):
    rows = []
    for (encoder, arch), r in results.items():
        rows.append({
            "encoder": encoder, "architecture": arch,
            "acc_mean": r["acc"].mean(), "acc_std": r["acc"].std(ddof=1),
            "f1_mean": r["f1"].mean(), "f1_std": r["f1"].std(ddof=1),
        })
    return pd.DataFrame(rows).sort_values("f1_mean", ascending=False).reset_index(drop=True)

def fold_scores_df(results, metric="f1"):
    """One per row (system,fold), same format as tutorials plot_box_and_points"""
    rows =[]
    for system, r in results.items():
        for fold, score in enumerate(r[metric]):
            rows.append({"model": name(system), metric: score, "run": 0, "fold": fold})
    return pd.DataFrame(rows)

def name(system):
    return f"{system[0]}+{system[1]}"

def compare(results, a, b, metric="f1", rope=ROPE):
    """Bayesian correlated t-test between systems a and b on their per fold scores"""
    post = BenavoliCorrelatedT(results[a][metric], results[b][metric], names=[name(a), name(b)])
    p_a, p_tie, p_b = post.probs(rope)
    print(f"A={name(a):<20} B={name(b):<20} P(A wins)={p_a:.3f} P(tie)={p_tie:.3f} "
          f"P(B wins):{p_b:.3f} effect={post.effect_size:.2f}")
    fig, _ = post.plot(rope)
    fig.savefig(f"{FIG_DIR}/rope_{name(a)}_vs_{name(b)}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

def error_analysis(results, system, n_examples=8):
    preds = results[system]["oof"]
    print(f"\nConfusion matrix for {name(system)} (rows = true, columns = predicted)")
    print(pd.crosstab(pd.Series([labels[i] for i in pool_labels], name="true"),
                      pd.Series([labels[i] for i in preds], name="predicted")))

    wrong = np.flatnonzero(preds != pool_labels)
    print(f"\n{len(wrong)} / {len(texts)} missclassified. Random examples:")
    for i in np.random.default_rng(0).choice(wrong, n_examples, replace=False):
        print(f" true={labels[pool_labels[i]]:<9} predicted={labels[preds[i]]:<9} >>> {texts[i]}")

    n_words = np.array([len(t.split()) for t in texts])
    short = n_words <= np.median(n_words)
    err = preds != pool_labels
    print(f"\nError rate, short tweets: {err[short].mean():.3f}")
    print(f"Error rate, long tweets:  {err[~short].mean():.3f}")


if __name__ == "__main__":
    results = run_all()
    os.makedirs(FIG_DIR, exist_ok=True)

    table = results_table(results)
    print("\n=== Results (mean and std over 5 folds), sorted by macro-F1 ===")
    print(table.round(4).to_string())
    table.to_csv("part1/results_table.csv", index=False)

    best = (table.loc[0, "encoder"], table.loc[0, "architecture"])

    print(f"\n=== Best system vs the rest (macro-F1, ROPE = {ROPE}) ===")
    for system in results:
        if system != best:
            compare(results, best, system)

    print("\n=== Does the bigram representation help? (bow vs bigram, same architecture) ===")
    for arch in ARCHS:
        compare(results, ("bow", arch), ("bigram", arch))

    print("\n=== Does the deep MLP help? (mlp vs deep_mlp, same encoder) ===")
    for enc in ENCODERS:
        compare(results, (enc, "mlp"), (enc, "deep_mlp"))

    error_analysis(results, best)

    plot_box_and_points(fold_scores_df(results, "f1"), metric="f1")   # opens a window; close it to finish

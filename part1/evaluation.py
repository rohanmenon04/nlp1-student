"""
Add evaluation code here:

1. Cross-validation over the 9 systems
2. results table
3. Bayesian ROPE comparison and error analysis
"""

import os
import pickle
import itertools

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
        # bigram vocab from train fold (no leakage from val)
        bigram_vocab = build_bigram_vocab(texts[train_idx], bpe_tokenizer)
        X_tr = encode(encoder, texts[train_idx], bigram_vocab)
        X_val = encode(encoder, texts[val_idx], bigram_vocab)
        y_tr, y_val = pool_labels[train_idx], pool_labels[val_idx]

        if arch == "logistic":
            preds, _ = fit_predict_logistic(X_tr, y_tr, X_val, seed=SEED)
        else:
            preds, _ = fit_predict_neural(arch, csr_matrix(X_tr), y_tr, csr_matrix(X_val), seed=SEED)

        accs.append(accuracy_score(y_val, preds))
        f1s.append(f1_score(y_val, preds, average="macro"))
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

if __name__ == "__main__":
    print(run_system("bow", "logistic"))
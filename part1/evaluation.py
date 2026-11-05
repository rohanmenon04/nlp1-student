"""
Add evaluation code here:

1. Cross-validation over the 9 systems
2. results table
3. Bayesian ROPE comparison and error analysis
"""

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



####### BELOW IS A QUICK TEST
#created once
folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(pool_texts, pool_labels))
# folds is a list of 5 pairs (train_indices, val_indices)
# every system should loop over same list

print(len(folds[0][0]), len(folds[0][1])) #train on [0][0], validate on [0][1] (train_indices, val_indices)

texts = np.array(pool_texts)

train_idx, val_idx = folds[0] #just on one fold for now. 
X_tr = encode("bow", texts[train_idx])
X_val = encode("bow", texts[val_idx])
y_tr, y_val = pool_labels[train_idx], pool_labels[val_idx]

preds, _ = fit_predict_logistic(X_tr, y_tr, X_val, seed=0)
print(accuracy_score(y_val, preds), f1_score(y_val, preds, average="macro"))
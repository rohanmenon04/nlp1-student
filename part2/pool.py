# %% Step 1: embed-and-pool on ONE fold, just to see it work
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score

from part1.data import pool_texts, pool_labels
from part2 import clslib, experiment
from part2.embeddings import load_for_tutorial, make_tokenizer

SEED = 42
torch.manual_seed(SEED)

# %% Step 2: the step 1 code as functions, run over all 5 folds and 8 settings
import os
import pickle

CACHE = "part2/pool_results.pkl"    # finished systems are saved here; delete it for a fresh run
SHORT = {"glove-twitter-50": "twitter", "glove-wiki-gigaword-50": "wiki"}

# same 5 folds as part 1
folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(pool_texts, pool_labels))


class PoolEncoder(nn.Module):
    """Pooling from my thesis Classifier2, padding-safe, with the course's interface so SoftmaxRegressor accepts it."""
    def __init__(self, embedding_matrix, pooling="mean", padding_idx=0, freeze_embeddings=True):
        super().__init__()
        self.pooling = pooling
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=freeze_embeddings, padding_idx=padding_idx)

    @property
    def enc_dim(self):
        return self.embedding.embedding_dim

    @property
    def pad_idx(self):
        return self.embedding.padding_idx

    def forward(self, x):        # x: (B, T) word ids
        e = self.embedding(x)        # (B, T, E)
        mask = x.ne(self.pad_idx).unsqueeze(-1)   # (B, T, 1), True for real words
        if self.pooling == "mean":
            return (e * mask).sum(1) / mask.sum(1).clamp_min(1)
        if self.pooling == "max":
            h = e.masked_fill(~mask, float("-inf")).max(dim=1)[0]
            return torch.where(torch.isinf(h), torch.zeros_like(h), h)  # a tweet with no words -> zeros, not -inf
        raise ValueError(self.pooling)


_prepared = {}

def prepare(name):
    """Load one embedding model and turn every tweet into word ids. Done once per model, then reused."""
    if name not in _prepared:
        emb = load_for_tutorial(name)
        tok = make_tokenizer(name, emb.vocab, keep_punct=False)
        ids = [tok.ids(t) for t in pool_texts]
        _prepared[name] = (emb, ids)
    return _prepared[name]


def make_loader(ids, idx, pad_id, shuffle):
    data = experiment.LabelledCorpus([ids[i] for i in idx], pool_labels[idx])
    return DataLoader(data, batch_size=64, shuffle=shuffle,
                      collate_fn=lambda b: experiment.collate_labelled_texts(b, padding_value=pad_id))


def fit_predict(fold, name, freeze, pooling, seed=SEED):
    """Train on one fold (with a 10% dev split for early stopping), return predictions for the held-out fold."""
    train_idx, val_idx = folds[fold]
    fit_idx, dev_idx = train_test_split(train_idx, test_size=0.1, stratify=pool_labels[train_idx], random_state=seed)
    emb, ids = prepare(name)
    pad_id = emb.vocab.index(emb.pad_token)

    torch.manual_seed(seed)
    fit_dl = make_loader(ids, fit_idx, pad_id, True)
    dev_dl = make_loader(ids, dev_idx, pad_id, False)
    val_dl = make_loader(ids, val_idx, pad_id, False)

    encoder = PoolEncoder(torch.tensor(emb.vectors), pooling=pooling, padding_idx=pad_id, freeze_embeddings=freeze)
    model = clslib.SoftmaxRegressor(hidden_size1=50, hidden_size2=50, num_classes=3, encoder=encoder)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_loss, best_state, best_epoch, patience = float("inf"), None, 0, 0
    for epoch in range(20):
        experiment.training_one_epoch(model, fit_dl, optimizer, criterion)
        dev_loss = experiment.evaluate(model, dev_dl, criterion)["loss"]
        if dev_loss < best_loss:
            best_loss, best_state, best_epoch, patience = dev_loss, copy.deepcopy(model.state_dict()), epoch + 1, 0
        else:
            patience += 1
            if patience == 2:
                break

    model.load_state_dict(best_state)
    preds = np.concatenate(experiment.evaluate(model, val_dl, criterion)["predictions"])
    return preds, best_epoch


def run_system(name, freeze, pooling):
    """All 5 folds for one setting. Same output format as part 1's run_system."""
    oof = np.empty(len(pool_labels), dtype=int)  # out-of-fold predictions, filled in fold by fold
    accs, f1s = [], []
    for fold, (_, val_idx) in enumerate(folds):
        preds, best_epoch = fit_predict(fold, name, freeze, pooling)
        oof[val_idx] = preds
        y = pool_labels[val_idx]
        accs.append(accuracy_score(y, preds))
        f1s.append(f1_score(y, preds, average="macro"))
        print(f"  fold {fold + 1}: acc={accs[-1]:.4f}  f1={f1s[-1]:.4f}  (best epoch {best_epoch})", flush=True)
    return {"acc": np.array(accs), "f1": np.array(f1s), "oof": oof}


def key(name, freeze, pooling):
    """Name a system like part 1 does: ('twitter', 'trainable-mean') prints as twitter+trainable-mean."""
    return (SHORT[name], f"{'frozen' if freeze else 'trainable'}-{pooling}")


SYSTEMS = [(name, freeze, pooling) for name in SHORT for freeze in (True, False) for pooling in ("mean", "max")]


def run_all():
    results = {}
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            results = pickle.load(f)
    for name, freeze, pooling in SYSTEMS:
        k = key(name, freeze, pooling)
        if k in results:
            print(f"skip {k[0]}+{k[1]} (cached)")
            continue
        print(f"=== {k[0]}+{k[1]}", flush=True)
        results[k] = run_system(name, freeze, pooling)
        with open(CACHE, "wb") as f:
            pickle.dump(results, f)
    return results


if __name__ == "__main__":
    results = run_all()
    for k, r in results.items():
        print(f"{k[0]}+{k[1]:<16} f1 {r['f1'].mean():.4f} ± {r['f1'].std():.4f}")

# %% Step 3: compare with part 1 on the same folds
from part1.evaluation import compare, results_table

with open("part1/cv_results.pkl", "rb") as f:
    part1_results = pickle.load(f)
all_results = {**part1_results, **results}

print(results_table(all_results))

best = ("twitter", "trainable-mean")
compare(all_results, best, ("bow", "logistic"))
compare(all_results, best, ("bigram", "logistic"))
compare(all_results, best, ("wiki", "trainable-mean"))
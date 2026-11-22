# %% Point 4: recurrent encoder (LSTM)
# Same folds, dev split, classifier head, optimiser and early stopping as part2/pool.py,
# so LSTM and embed-and-pool scores can be compared fold by fold (and with part 1).
# The course gives us an LSTM encoder (clslib.LSTMEncoder) and no GRU, so we use the LSTM.
#   run: uv run python -m part2.rnn
#
# What it answers (assignment point 4):
#   - which design choices matter: embedding model, hidden size, word dropout (one change at a time from BASE)
#   - embedding parameters: pretrained frozen vs pretrained trainable vs random init (trainable)
#   - data size: BASE trained on 10/25/50/100% of each fold's training part
#   - over-fitting: train vs dev loss per epoch, without early stopping, on fold 1
#   - is it better than embed-and-pool: best LSTM vs best pool system with the ROPE test from part 1

import copy
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from part1.analysis import BenavoliCorrelatedT
from part1.data import NUM_CLASSES, pool_labels, pool_texts
from part2 import clslib, experiment
from part2.embeddings import load_for_tutorial, make_tokenizer
from part2.pool import CACHE as POOL_CACHE
from part2.pool import SEED, SHORT, folds, make_loader

CACHE = "part2/rnn_results.pkl"    # finished systems are saved here; delete it for a fresh run
RESULTS_DIR = "part2/results"
FIG_DIR = "part2/figures"
ROPE = 0.005                        # +-0.5 F1 points, same as part 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# %% Systems
# BASE uses glove twitter because it had the best coverage in point 1; glove wiki is one of the variations.

BASE = dict(name="glove-twitter-50", init="pretrained", freeze=True, hidden=50, word_dropout=0.0, data=1.0)

def variant(**changes):
    return {**BASE, **changes}

DESIGN = [
    BASE,
    variant(name="glove-wiki-gigaword-50"),    # embedding model
    variant(freeze=False),                     # fine-tune the pretrained vectors
    variant(init="random", freeze=False),      # no pretraining at all (same vocab, random vectors)
    variant(hidden=25),                        # hidden size
    variant(hidden=100),
    variant(word_dropout=0.2),                 # regularisation (the course encoder supports word dropout)
]
DATA_SIZES = [variant(data=f) for f in (0.1, 0.25, 0.5)]   # BASE is the 100% point
CURVES = [BASE, variant(freeze=False)]                     # trainable embeddings are the most likely to over-fit


def key(cfg):
    """Name a system like pool.py does, e.g. ('twitter', 'lstm-frozen-h50') prints as twitter+lstm-frozen-h50."""
    emb = "random" if cfg["init"] == "random" else ("frozen" if cfg["freeze"] else "trainable")
    desc = ["lstm", emb, f"h{cfg['hidden']}"]
    if cfg["word_dropout"]:
        desc.append(f"wdrop{cfg['word_dropout']}")
    if cfg["data"] < 1:
        desc.append(f"data{cfg['data']:.0%}")
    return (SHORT[cfg["name"]], "-".join(desc))


def name(system):
    return f"{system[0]}+{system[1]}"


# %% Data and model

_prepared = {}

def prepare(name):
    """Load one embedding model and turn every tweet into word ids. Done once per model, then reused."""
    if name not in _prepared:
        emb = load_for_tutorial(name)
        tok = make_tokenizer(name, emb.vocab, keep_punct=True)   # the LSTM reads word order, so punctuation stays in
        unk_id = emb.vocab.index(emb.unk_token)
        # a tweet can end up empty (e.g. only a mention and a link for glove wiki), and the LSTM can't pack length 0
        ids = [tok.ids(t) or [unk_id] for t in pool_texts]
        _prepared[name] = (emb, ids)
    return _prepared[name]


def embedding_matrix(emb, init, seed):
    vectors = torch.tensor(emb.vectors)
    if init == "random":
        # same shape and scale as the pretrained table, so only the pretraining differs
        g = torch.Generator().manual_seed(seed)
        vectors = torch.randn(vectors.shape, generator=g) * vectors[2:].std()
        vectors[emb.vocab.index(emb.pad_token)] = 0
    return vectors


def fit_predict(fold, cfg, seed=SEED, epochs=20, patience=2):
    """Train on one fold (10% dev split for early stopping), return held-out predictions, best epoch and the loss history.
    patience=None trains for all epochs (used for the over-fitting curves)."""
    train_idx, val_idx = folds[fold]
    fit_idx, dev_idx = train_test_split(train_idx, test_size=0.1, stratify=pool_labels[train_idx], random_state=seed)
    if cfg["data"] < 1:
        # smaller training set, same dev split and same held-out fold
        fit_idx, _ = train_test_split(fit_idx, train_size=cfg["data"], stratify=pool_labels[fit_idx], random_state=seed)
    emb, ids = prepare(cfg["name"])
    pad_id = emb.vocab.index(emb.pad_token)

    torch.manual_seed(seed)
    fit_dl = make_loader(ids, fit_idx, pad_id, True)
    dev_dl = make_loader(ids, dev_idx, pad_id, False)
    val_dl = make_loader(ids, val_idx, pad_id, False)

    encoder = clslib.LSTMEncoder(cfg["hidden"], embedding_matrix(emb, cfg["init"], seed), padding_idx=pad_id,
                                 freeze_embeddings=cfg["freeze"], word_dropout=cfg["word_dropout"])
    model = clslib.SoftmaxRegressor(hidden_size1=50, hidden_size2=50, num_classes=NUM_CLASSES, encoder=encoder).to(DEVICE)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    history = []
    best_loss, best_state, best_epoch, bad = float("inf"), None, 0, 0
    for epoch in range(1, epochs + 1):
        train = experiment.training_one_epoch(model, fit_dl, optimizer, criterion, device=DEVICE)  # measured during the epoch
        dev = experiment.evaluate(model, dev_dl, criterion, device=DEVICE)
        history.append({"epoch": epoch, "train_loss": train["loss"], "train_acc": train["accuracy"],
                        "dev_loss": dev["loss"], "dev_acc": dev["accuracy"]})
        if dev["loss"] < best_loss:
            best_loss, best_state, best_epoch, bad = dev["loss"], copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
            if patience is not None and bad == patience:
                break

    model.load_state_dict(best_state)
    preds = np.concatenate(experiment.evaluate(model, val_dl, criterion, device=DEVICE)["predictions"])
    return preds, best_epoch, history, len(fit_idx)


# %% Cross-validation

def run_system(cfg):
    """All 5 folds for one setting. Same output format as pool.py's run_system, plus best epochs and training size."""
    oof = np.empty(len(pool_labels), dtype=int)
    accs, f1s, best_epochs = [], [], []
    for fold, (_, val_idx) in enumerate(folds):
        preds, best_epoch, _, n_train = fit_predict(fold, cfg)
        oof[val_idx] = preds
        y = pool_labels[val_idx]
        accs.append(accuracy_score(y, preds))
        f1s.append(f1_score(y, preds, average="macro"))
        best_epochs.append(best_epoch)
        print(f"  fold {fold + 1}: acc={accs[-1]:.4f}  f1={f1s[-1]:.4f}  (best epoch {best_epoch})", flush=True)
    return {"acc": np.array(accs), "f1": np.array(f1s), "oof": oof,
            "best_epoch": np.array(best_epochs), "n_train": n_train, "cfg": cfg}


def load_cache():
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    return {"cv": {}, "curves": {}}


def save_cache(cache):
    with open(CACHE, "wb") as f:
        pickle.dump(cache, f)


def run_all(cfgs, cache):
    for cfg in cfgs:
        k = key(cfg)
        if k in cache["cv"]:
            print(f"skip {name(k)} (cached)")
            continue
        print(f"=== {name(k)}", flush=True)
        cache["cv"][k] = run_system(cfg)
        save_cache(cache)
    return {key(cfg): cache["cv"][key(cfg)] for cfg in cfgs}


def run_curves(cfgs, cache, fold=0, epochs=20):
    """Train for all epochs (no early stopping) on one fold and keep the train/dev losses, to look at over-fitting."""
    for cfg in cfgs:
        k = key(cfg)
        if k in cache["curves"]:
            print(f"skip curves {name(k)} (cached)")
            continue
        print(f"=== curves {name(k)} (fold {fold + 1}, {epochs} epochs)", flush=True)
        _, best_epoch, history, _ = fit_predict(fold, cfg, epochs=epochs, patience=None)
        cache["curves"][k] = {"history": history, "best_epoch": best_epoch}
        save_cache(cache)
    return {key(cfg): cache["curves"][key(cfg)] for cfg in cfgs}


# %% Reporting

def results_table(results):
    rows = []
    for system, r in results.items():
        rows.append({
            "system": name(system),
            "acc_mean": r["acc"].mean(), "acc_std": r["acc"].std(ddof=1),
            "f1_mean": r["f1"].mean(), "f1_std": r["f1"].std(ddof=1),
            "best_epoch_mean": r["best_epoch"].mean(), "n_train": r["n_train"],
        })
    return pd.DataFrame(rows).sort_values("f1_mean", ascending=False).reset_index(drop=True)


def curves_table(curves):
    return pd.DataFrame([{"system": name(k), **h} for k, c in curves.items() for h in c["history"]])


def compare(results, a, b, metric="f1", rope=ROPE):
    """Bayesian correlated t-test between systems a and b on their per-fold scores (as in part 1)."""
    post = BenavoliCorrelatedT(results[a][metric], results[b][metric], names=[name(a), name(b)])
    p_a, p_tie, p_b = post.probs(rope)
    print(f"A={name(a):<28} B={name(b):<28} P(A wins)={p_a:.3f} P(tie)={p_tie:.3f} "
          f"P(B wins)={p_b:.3f} effect={post.effect_size:.2f}")
    fig, _ = post.plot(rope)
    fig.savefig(f"{FIG_DIR}/rope_{name(a)}_vs_{name(b)}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    cache = load_cache()
    results = run_all(DESIGN + DATA_SIZES, cache)
    curves = run_curves(CURVES, cache)

    table = results_table(results)
    print("\n=== LSTM results (mean and std over 5 folds), sorted by macro-F1 ===")
    print(table.round(4).to_string())
    table.to_csv(f"{RESULTS_DIR}/rnn.csv", index=False)

    curves_table(curves).to_csv(f"{RESULTS_DIR}/rnn_overfitting.csv", index=False)
    print(f"\nTrain/dev loss per epoch saved to {RESULTS_DIR}/rnn_overfitting.csv")
    for k, c in curves.items():
        h = c["history"]
        print(f"  {name(k):<28} best dev loss at epoch {c['best_epoch']}, "
              f"epoch 20: train {h[-1]['train_loss']:.3f} vs dev {h[-1]['dev_loss']:.3f}")

    print(f"\n=== Design choices: BASE vs one change (macro-F1, ROPE = {ROPE}) ===")
    for cfg in DESIGN[1:]:
        compare(results, key(BASE), key(cfg))

    print("\n=== Data size: BASE vs BASE on less data ===")
    for cfg in DATA_SIZES:
        compare(results, key(BASE), key(cfg))

    print("\n=== Best LSTM vs best embed-and-pool ===")
    if os.path.exists(POOL_CACHE):
        with open(POOL_CACHE, "rb") as f:
            pool = pickle.load(f)
        best_lstm = max((key(cfg) for cfg in DESIGN), key=lambda k: results[k]["f1"].mean())
        best_pool = max(pool, key=lambda k: pool[k]["f1"].mean())
        compare({**results, **pool}, best_lstm, best_pool)
    else:
        print(f"no {POOL_CACHE} found, run `uv run python -m part2.pool` first")

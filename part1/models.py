"""
Representations (BPE tokenizer + 3 encoders) and architectures (3 classifiers).

In order to run from the repo root: python -m part1.models
"""
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Metaspace

from part1 import smlp
from part1.data import ds, NUM_CLASSES


# ---------------------------------------------------------------------------
# Tokenizer BPE
# ---------------------------------------------------------------------------

def train_tokenizer(corpus, vocab_size, min_frequency, pretokenizer=Metaspace(), unk_token="[UNK]"):
    """
    corpus: a list of texts (each a string)
    vocab_size: number of distinct BPE tokens
    min_frequency: we only merge token pairs that are frequent enough
    pretokenizer: a simple tokenizer that we use before BPE tokenization

        You probably don't need to change the pretokenizer, unless you are working with
        languages that really don't use any form of blank space
        (then you could, for example, pretokenize at character level)
    """
    # Create an empty BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token=unk_token)) #unk_token used for characters that cannot be encoded

    # First split the corpus into words
    tokenizer.pre_tokenizer = pretokenizer

    # Tell BPE how large the vocabulary should be
    trainer = BpeTrainer(
        vocab_size=vocab_size, # The size of the vocabulary to learn.
        min_frequency=min_frequency, # Only perform a merge if the pair occurs at least min_frequency times in the training corpus.
        special_tokens=[unk_token]
    )

    # Learn the vocabulary and merge rules
    tokenizer.train_from_iterator(corpus, trainer=trainer)

    return tokenizer


train_text = ds["train"]["text"]

vocab_size = 5000
min_frequency = 2

bpe_tokenizer = train_tokenizer(train_text, vocab_size=5000, min_frequency=2)


# ---------------------------------------------------------------------------
# Representation 1 token indicator
# ---------------------------------------------------------------------------

def token_indicator(text, tokenizer):
    vector = np.zeros(tokenizer.get_vocab_size(), dtype=np.float32)
    token_ids = tokenizer.encode(text).ids
    vector[token_ids] = 1
    return vector


# ---------------------------------------------------------------------------
# Representation 2 bag of words
# ---------------------------------------------------------------------------

def bag_of_words(text, tokenizer):
    vector = np.zeros(tokenizer.get_vocab_size(), dtype=np.float32)
    token_ids = tokenizer.encode(text).ids
    for token_id in token_ids:
        vector[token_id] += 1
    return vector


# ---------------------------------------------------------------------------
# Representation 3 bi-gram
# ---------------------------------------------------------------------------

def build_bigram_vocab(corpus, tokenizer, max_bigrams=5000):
    counter = {}

    for tweet in corpus:
        ids = tokenizer.encode(tweet).ids
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])
            counter[pair] = counter.get(pair, 0) + 1

    # Sort pairs from most to least frequent and keep the top ones
    top_pairs = sorted(counter, key=counter.get, reverse=True)[:max_bigrams]

    # Give each kept pair its own index
    return {pair: index for index, pair in enumerate(top_pairs)}


def bi_gram(text, tokenizer, bigram_vocab):
    vocab_size = tokenizer.get_vocab_size()
    vector = np.zeros(vocab_size + len(bigram_vocab), dtype=np.float32)
    ids = tokenizer.encode(text).ids

    # Token counts go in the first vocab_size slots
    for token_id in ids:
        vector[token_id] += 1

    # Bigram counts go in the slots after that
    for i in range(len(ids) - 1):
        pair = (ids[i], ids[i + 1])
        if pair in bigram_vocab:
            vector[vocab_size + bigram_vocab[pair]] += 1

    return vector


def encode(encoder_name, texts, bigram_vocab=None):
    """
    Apply encoders to a list of texts
    """
    if encoder_name == "indicator":
        rows = [token_indicator(t, bpe_tokenizer) for t in texts]
    elif encoder_name == "bow":
        rows = [bag_of_words(t, bpe_tokenizer) for t in texts]
    elif encoder_name == "bigram":
        rows = [bi_gram(t, bpe_tokenizer, bigram_vocab) for t in texts]
    else:
        raise ValueError(encoder_name)
    return np.stack(rows)


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------

class SparseDeepMLP(nn.Module):
    """
    Third architecture: the tutorial's SparseMLP with two hidden layers and more dropout.
    """
    def __init__(self, input_dim, hidden_dims=(256, 128), num_classes=3, dropout=0.3):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dim, hidden_dims[0]))
        self.bias = nn.Parameter(torch.zeros(hidden_dims[0]))
        nn.init.xavier_uniform_(self.weight)
        layers = [nn.ReLU(), nn.Dropout(dropout)]
        for d_in, d_out in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers += [nn.Linear(d_in, d_out), nn.ReLU(), nn.Dropout(dropout)]
        layers += [nn.Linear(hidden_dims[-1], num_classes), nn.LogSoftmax(dim=-1)]
        self.rest = nn.Sequential(*layers)

    def forward(self, x):
        return self.rest(torch.sparse.mm(x, self.weight) + self.bias)

ARCH_CONFIG = {
    "logistic": {"max_iter": 100},                     # same solver as part1/loglin.py
    "mlp":      {"hidden_dim": 128, "dropout": 0.1},   # tutorial SparseMLP
    "deep_mlp": {"hidden_dims": (256, 128), "dropout": 0.3},
}
TRAIN_CONFIG = {"batch_size": 64, "learning_rate": 1e-3, "weight_decay": 1e-4,
                "max_epochs": 10, "patience": 2, "dev_fraction": 0.1}


# ---------------------------------------------------------------------------
# Training: fit on a training fold, predict labels for a validation fold
# ---------------------------------------------------------------------------

def fit_predict_logistic(X_tr, y_tr, X_val, seed):
    model = LogisticRegression(solver="saga", max_iter=ARCH_CONFIG["logistic"]["max_iter"], random_state=seed)
    model.fit(X_tr, y_tr)
    return model.predict(X_val), {}


def fit_predict_neural(arch, X_tr, y_tr, X_val, seed):
    cfg = TRAIN_CONFIG
    torch.manual_seed(seed)
    # Inner dev split from the TRAINING fold, used to choose the best epoch.
    # (The tutorial picks the epoch on the validation fold it then scores, which is optimistic.)
    perm = np.random.default_rng(seed).permutation(X_tr.shape[0])
    n_dev = int(cfg["dev_fraction"] * len(perm))
    dev_idx, fit_idx = perm[:n_dev], perm[n_dev:]

    def loader(X, y, shuffle):
        return DataLoader(smlp.SparseDataset(X, list(y)), batch_size=cfg["batch_size"],
                          shuffle=shuffle, collate_fn=smlp.sparse_collate)
    train_dl = loader(X_tr[fit_idx], y_tr[fit_idx], True)
    dev_dl = loader(X_tr[dev_idx], y_tr[dev_idx], False)
    val_dl = loader(X_val, np.zeros(X_val.shape[0], dtype=int), False)   # dummy labels, we only keep preds

    if arch == "mlp":
        model = smlp.SparseMLP(X_tr.shape[1], num_classes=NUM_CLASSES, **ARCH_CONFIG["mlp"])
    else:
        model = SparseDeepMLP(X_tr.shape[1], num_classes=NUM_CLASSES, **ARCH_CONFIG["deep_mlp"])
    criterion = nn.NLLLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])

    best_loss, best_state, best_epoch, bad = float("inf"), None, -1, 0
    for epoch in range(cfg["max_epochs"]):
        smlp.train_one_epoch(model, train_dl, optimizer, criterion)
        dev_metrics, _ = smlp.evaluate(model, dev_dl, criterion)
        if dev_metrics["loss"] < best_loss:
            best_loss, best_state, best_epoch, bad = dev_metrics["loss"], copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= cfg["patience"]:   # early stopping
                break
    model.load_state_dict(best_state)
    _, preds = smlp.evaluate(model, val_dl, criterion)
    return preds, {"best_epoch": best_epoch}


if __name__ == "__main__":
    # Sanity check of the three representations on one example.
    # Only the non-zero entries are shown (the vectors are mostly zeros).
    example = "so so good"
    ids = bpe_tokenizer.encode(example).ids
    print(f'Example: "{example}"')
    print(f"BPE tokens: {bpe_tokenizer.encode(example).tokens}   ids: {ids}")

    # Representation 3 needs a bigram vocabulary (most frequent adjacent token pairs in train)
    bigram_vocab = build_bigram_vocab(train_text, bpe_tokenizer)
    id_to_bigram = {j: pair for pair, j in bigram_vocab.items()}
    V = bpe_tokenizer.get_vocab_size()

    def feature_name(i):
        """Readable name of dimension i: a token, or a token pair for the bigram slots."""
        if i < V:
            return repr(bpe_tokenizer.id_to_token(int(i)))
        a, b = id_to_bigram[i - V]
        return f"({bpe_tokenizer.id_to_token(a)!r}, {bpe_tokenizer.id_to_token(b)!r})"

    def show(title, vector, expected_sum, meaning):
        print(f"{title}")
        print(f"  dimensions: {vector.shape[0]}   non-zero: {np.count_nonzero(vector)}")
        for i in np.flatnonzero(vector):
            print(f"    {feature_name(i):<24} = {vector[i]:g}")
        ok = "OK" if vector.sum() == expected_sum else "MISMATCH"

    # Representation 1 token indicator
    show("Representation 1: token indicator (1 if the token occurs)",
         token_indicator(example, bpe_tokenizer), len(set(ids)), "number of distinct tokens")

    # Representation 2 bag of words
    show("Representation 2: bag of words (how often each token occurs)",
         bag_of_words(example, bpe_tokenizer), len(ids), "number of tokens")

    # Representation 3 bi-gram
    n_known_pairs = sum(pair in bigram_vocab for pair in zip(ids, ids[1:]))
    show("Representation 3: bag of words + bigram counts",
         bi_gram(example, bpe_tokenizer, bigram_vocab), len(ids) + n_known_pairs,
         "tokens + adjacent pairs that are in the bigram vocab")

    print(f"Bigram vocab: {len(bigram_vocab)} pairs. Most frequent 5:")
    for pair in list(bigram_vocab)[:5]:
        print(f"    ({bpe_tokenizer.id_to_token(pair[0])!r}, {bpe_tokenizer.id_to_token(pair[1])!r})")

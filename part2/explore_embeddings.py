# points 1 and 2 of the assignment: vocab coverage and what the embeddings capture
#   python -m part2.explore_embeddings   (or: python part2/explore_embeddings.py)

import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from part2.embeddings import MODELS, load_kv, make_tokenizer, is_special

RESULTS_DIR = Path(__file__).parent / "results"


def load_pool():
    # same pool as project 1 (tweet_eval train + validation)
    from part1.data import pool_texts, pool_labels
    return pool_texts, pool_labels


# ---- step 1: coverage ----
# word coverage = only tokens with a letter and no <special> tokens,
# fairer since numberbatch has no punctuation and only glove twitter has the specials

def is_word(t):
    return any(c.isalpha() for c in t) and not is_special(t)


def coverage_table(models=MODELS, texts=None, top_oov=10):
    if texts is None:
        texts, _ = load_pool()
    rows = []
    for name in models:
        kv = load_kv(name)
        tok = make_tokenizer(name)
        counts = Counter(t for text in texts for t in tok.tokens(text))
        words = {t: c for t, c in counts.items() if is_word(t)}
        known = {t for t in counts if t in kv}
        rows.append({
            "model": name,
            "token_coverage": sum(counts[t] for t in known) / sum(counts.values()),
            "word_token_coverage": sum(c for t, c in words.items() if t in known) / sum(words.values()),
            "word_type_coverage": sum(t in known for t in words) / len(words),
            "top_oov": " ".join([t for t, _ in counts.most_common() if t not in known][:top_oov]),
        })
    return pd.DataFrame(rows)


# ---- step 2: polarity gap with permutation test ----

def opinion_words():
    import nltk
    nltk.download("opinion_lexicon", quiet=True)
    from nltk.corpus import opinion_lexicon
    return list(opinion_lexicon.positive()), list(opinion_lexicon.negative())


def group_means(X, labels):
    # rows have unit length, so sum of all pairwise dots = ||sum of rows||^2,
    # minus n for the diagonal. way faster than building the n x n matrix
    A, B = X[labels == 1], X[labels == 0]
    sa, sb = A.sum(0), B.sum(0)
    na, nb = len(A), len(B)
    pp = (sa @ sa - na) / (na * (na - 1))
    nn = (sb @ sb - nb) / (nb * (nb - 1))
    pn = (sa @ sb) / (na * nb)
    return pp, nn, pn


def polarity_gap(name, pos_words, neg_words, n_perm=1000, seed=0):
    kv = load_kv(name)
    X = np.vstack([np.stack([kv[w] for w in pos_words]),
                   np.stack([kv[w] for w in neg_words])]).astype(np.float64)
    X = X - X.mean(0)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    labels = np.array([1] * len(pos_words) + [0] * len(neg_words))

    pp, nn, pn = group_means(X, labels)
    gap = (pp + nn) / 2 - pn

    rng = np.random.default_rng(seed)
    fake = []
    for _ in range(n_perm):
        a, b, c = group_means(X, rng.permutation(labels))
        fake.append((a + b) / 2 - c)
    fake = np.array(fake)

    return {"model": name, "PP": pp, "NN": nn, "PN": pn, "gap": gap,
            "null_mean": fake.mean(), "null_std": fake.std(),
            "z": (gap - fake.mean()) / fake.std(), "p": np.mean(fake >= gap),
            "n_pos": len(pos_words), "n_neg": len(neg_words)}


def polarity_table(models=MODELS, n_perm=1000):
    pos, neg = opinion_words()
    # only words that every model has, so it's the same comparison for all
    pos = [w for w in pos if all(w in load_kv(m) for m in models)]
    neg = [w for w in neg if all(w in load_kv(m) for m in models)]
    return pd.DataFrame([polarity_gap(m, pos, neg, n_perm) for m in models])


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    pd.set_option("display.width", 200, "display.max_colwidth", 80)

    cov = coverage_table()
    print(cov.to_string(index=False, float_format="{:.1%}".format))
    cov.to_csv(RESULTS_DIR / "coverage.csv", index=False)
    print()

    pol = polarity_table()
    print(pol.round(4).to_string(index=False))
    pol.to_csv(RESULTS_DIR / "polarity_gap.csv", index=False)
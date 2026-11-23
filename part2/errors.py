# %% Point 6: error analysis
# Which properties of a tweet predict whether a system gets it right, and where do the major design choices differ?
# Uses the out-of-fold (OOF) predictions already stored in the caches, so nothing is retrained.
#   1. per-tweet features (length, unknown/rare words, negation, contrast, emoticons, emphasis, lexicon words, gold class)
#   2. descriptive: accuracy of each system on tweets with vs without each binary feature (all 47,615 tweets)
#   3. the course's Bayesian correctness model (part2/analysis.py) on a fixed random subset of tweets,
#      with one weight per (system, feature); differences between systems' weights answer
#      "is this feature more harmful/helpful for system A than for system B?"
#   run: uv run python -m part2.errors | tee part2/results/errors_log.txt

import os
import pickle
from collections import Counter

import arviz as az
import numpy as np
import pandas as pd

from part1.data import pool_labels, pool_texts
from part2 import rnn
from part2.analysis import fit_bda_model, posterior_to_arviz
from part2.embeddings import load_kv, make_tokenizer, is_special

RESULTS_DIR = "part2/results"
N_SUBSET = 8000      # tweets per system in the Bayesian model (memory: the model stores one logit per observation)
SEED = 0

NEGATION = {"not", "no", "never", "n't", "nothing", "nobody", "none", "nor", "cannot", "without", "neither"}
CONTRAST = {"but", "however", "though", "although", "yet"}
EMOTICON = {"<smile>", "<sadface>", "<lolface>", "<neutralface>", "<heart>"}
EMPHASIS = {"<allcaps>", "<elong>", "<repeat>"}


def load_systems():
    with open("part1/cv_results.pkl", "rb") as f:
        p1 = pickle.load(f)
    with open("part2/pool_results.pkl", "rb") as f:
        pl = pickle.load(f)
    with open("part2/rnn_results.pkl", "rb") as f:
        rn = pickle.load(f)["cv"]
    return {
        "bow+logistic": p1[("bow", "logistic")]["oof"],
        "pool frozen": pl[("twitter", "frozen-mean")]["oof"],
        "pool trainable": pl[("twitter", "trainable-mean")]["oof"],
        "lstm frozen": rn[rnn.key(rnn.BASE)]["oof"],
        "lstm trainable": rn[rnn.key(rnn.variant(freeze=False))]["oof"],
    }


def features():
    import nltk
    nltk.download("opinion_lexicon", quiet=True)
    from nltk.corpus import opinion_lexicon
    pos_lex, neg_lex = set(opinion_lexicon.positive()), set(opinion_lexicon.negative())

    tok = make_tokenizer("glove-twitter-50", keep_punct=True)
    kv = load_kv("glove-twitter-50")
    toks = [tok.tokens(t) for t in pool_texts]
    words = [[w for w in ts if not is_special(w) and any(c.isalpha() for c in w)] for ts in toks]
    freq = Counter(w for ws in words for w in ws)

    rows = []
    for ts, ws, y in zip(toks, words, pool_labels):
        tset, n = set(ts), max(len(ws), 1)
        npos, nneg = sum(w in pos_lex for w in ws), sum(w in neg_lex for w in ws)
        rows.append({
            "length": len(ts),
            "unk_share": sum(w not in kv for w in ws) / n,
            "rare_share": sum(freq[w] <= 2 for w in ws) / n,     # seen at most twice in the whole pool
            "negation": int(bool(tset & NEGATION)),
            "contrast": int(bool(tset & CONTRAST)),
            "emoticon": int(bool(tset & EMOTICON)),
            "emphasis": int(bool(tset & EMPHASIS)),
            "hashtag": int("<hashtag>" in tset),
            "lexicon_word": int(npos + nneg > 0),
            "mixed_lexicon": int(npos > 0 and nneg > 0),       # both a positive and a negative opinion word
            "gold_negative": int(y == 0),                        # gold class, neutral is the reference
            "gold_positive": int(y == 2),
        })
    return pd.DataFrame(rows)


def descriptive(feats, systems):
    """Accuracy with vs without each binary feature, per system (no model, all tweets)."""
    rows = []
    for f in [c for c in feats.columns if set(feats[c].unique()) <= {0, 1}]:
        on = feats[f].values == 1
        row = {"feature": f, "share": on.mean()}
        for s, oof in systems.items():
            ok = oof == pool_labels
            row[f"{s} | with"] = ok[on].mean()
            row[f"{s} | without"] = ok[~on].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def bayesian(feats, systems):
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(pool_labels), size=N_SUBSET, replace=False)   # the same tweets for every system
    names = list(systems)
    X = np.concatenate([feats.values[idx]] * len(names)).astype(float)
    sys_id = np.repeat(np.arange(len(names)), len(idx))
    correct = np.concatenate([(systems[s][idx] == pool_labels[idx]).astype(int) for s in names])
    mcmc = fit_bda_model(len(names), sys_id, X, correct, seed=SEED, num_warmup=500, num_samples=500, num_chains=2)
    return posterior_to_arviz(mcmc, names, list(feats.columns)), names


def weight_table(post, names, features_):
    w = post.posterior["w"]   # chain, draw, system, feature
    rows = []
    for s in names:
        for f in features_:
            d = w.sel(system=s, feature=f).values.ravel()
            lo, hi = np.quantile(d, [0.03, 0.97])
            rows.append({"system": s, "feature": f, "mean": d.mean(), "hdi_3%": lo, "hdi_97%": hi,
                         "P(>0)": (d > 0).mean()})
    return pd.DataFrame(rows)


def diff_table(post, pairs, features_):
    """Posterior of w[B] - w[A] per feature: > 0 means the feature helps B more (or hurts it less) than A."""
    w = post.posterior["w"]
    rows = []
    for a, b in pairs:
        for f in features_:
            d = (w.sel(system=b, feature=f) - w.sel(system=a, feature=f)).values.ravel()
            lo, hi = np.quantile(d, [0.03, 0.97])
            rows.append({"A": a, "B": b, "feature": f, "mean": d.mean(), "hdi_3%": lo, "hdi_97%": hi,
                         "P(B-A>0)": (d > 0).mean()})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.set_option("display.width", 250)
    systems = load_systems()
    feats = features()
    feats.to_csv(f"{RESULTS_DIR}/error_features.csv", index=False)
    print("feature summary (mean over all tweets):")
    print(feats.mean().round(3).to_string())

    desc = descriptive(feats, systems)
    desc.to_csv(f"{RESULTS_DIR}/errors_descriptive.csv", index=False)
    print("\n=== Accuracy with / without each binary feature ===")
    print(desc.round(3).to_string(index=False))

    post, names = bayesian(feats, systems)
    summ = az.summary(post, var_names=["sys_shift", "w"])
    print(f"\nMCMC diagnostics: max r_hat {summ['r_hat'].max():.3f}, min ess_bulk {summ['ess_bulk'].min():.0f}")
    acc = post.posterior["accuracy"]
    print("\nmodel-implied accuracy on the subset:")
    for s in names:
        print(f"  {s:<16} {acc.sel(system=s).values.mean():.3f}")

    wt = weight_table(post, names, list(feats.columns))
    wt.to_csv(f"{RESULTS_DIR}/errors_weights.csv", index=False)
    print("\n=== Feature weights per system (logit per 1 sd of the feature) ===")
    print(wt.pivot(index="feature", columns="system", values="mean")[names].round(2).to_string())

    pairs = [("pool trainable", "lstm trainable"),   # does word order help where we expect it to?
             ("pool frozen", "pool trainable"),      # what does fine-tuning fix?
             ("lstm frozen", "lstm trainable"),
             ("bow+logistic", "pool trainable")]  # sparse counts vs embeddings
    dt = diff_table(post, pairs, list(feats.columns))
    dt.to_csv(f"{RESULTS_DIR}/errors_differences.csv", index=False)
    print("\n=== Weight differences B - A (94% interval, 3%-97% quantiles); interval excluding 0 is marked * ===")
    for (a, b), g in dt.groupby(["A", "B"], sort=False):
        print(f"\n{b}  minus  {a}")
        for _, r in g.iterrows():
            star = "*" if r["hdi_3%"] > 0 or r["hdi_97%"] < 0 else " "
            print(f"  {star} {r['feature']:<15} {r['mean']:+.3f}  [{r['hdi_3%']:+.3f}, {r['hdi_97%']:+.3f}]  "
                  f"P(>0)={r['P(B-A>0)']:.2f}")

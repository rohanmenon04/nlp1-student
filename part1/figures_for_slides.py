import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from part1.data import ds, labels, pool_labels

CACHE = "part1/cv_results.pkl"
OUT = "part1/figures"
os.makedirs(OUT, exist_ok=True)

with open(CACHE, "rb") as f:
    results = pickle.load(f)

plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False})


test_labels = np.array(ds["test"]["label"])
pool_share = np.bincount(pool_labels, minlength=len(labels)) / len(pool_labels)
test_share = np.bincount(test_labels, minlength=len(labels)) / len(test_labels)

x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(7, 4.5))
b1 = ax.bar(x - 0.2, pool_share * 100, width=0.4, label="CV pool (train + val)")
b2 = ax.bar(x + 0.2, test_share * 100, width=0.4, label="Test")
ax.bar_label(b1, fmt="%.1f%%", fontsize=11)
ax.bar_label(b2, fmt="%.1f%%", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Share of tweets (%)")
ax.set_ylim(0, 55)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
fig.savefig(f"{OUT}/slide1_label_shares.png", dpi=200, bbox_inches="tight")
plt.close(fig)


order = sorted(results, key=lambda s: -results[s]["f1"].mean())
names = [f"{enc}+{arch}" for enc, arch in order]
scores = [results[s]["f1"] for s in order]

fig, ax = plt.subplots(figsize=(11, 5))
ax.boxplot(scores, widths=0.55, showfliers=False,
           boxprops=dict(color="0.6"), whiskerprops=dict(color="0.6"),
           capprops=dict(color="0.6"), medianprops=dict(color="0.3"))
xs = np.arange(1, len(names) + 1)
fold_scores = np.array(scores)  # shape: systems x folds
colors = plt.cm.tab10(np.arange(fold_scores.shape[1]))
for k in range(fold_scores.shape[1]):
    offset = (k - 2) * 0.06  # small shift so folds don't sit on top of each other
    ax.scatter(xs + offset, fold_scores[:, k], color=colors[k], s=28, zorder=3, label=f"fold {k + 1}")
ax.set_xticks(xs)
ax.set_xticklabels(names, rotation=30, ha="right")
ax.set_ylabel("Macro-F1 per fold")
ax.legend(frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0))
fig.savefig(f"{OUT}/slide4_boxplot_f1.png", dpi=200, bbox_inches="tight")
plt.close(fig)


best = ("bigram", "logistic")
fig, ax = plt.subplots(figsize=(5.5, 4.5))
ConfusionMatrixDisplay.from_predictions(
    pool_labels, results[best]["oof"],
    display_labels=labels, normalize="true", values_format=".0%",
    cmap="Blues", colorbar=False, ax=ax,
)
ax.set_title("bigram + logistic (out-of-fold, row %)")
fig.savefig(f"{OUT}/slide4_confusion_bl.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("saved to", OUT)
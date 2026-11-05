"""
Data loading and statistical analysis of tweet_eval (sentiment).

Run from the repo root:  python -m part1.data
"""

# LOAD DATASET
from datasets import load_dataset
import pandas as pd
import numpy as np

subtask = "sentiment"
ds = load_dataset("cardiffnlp/tweet_eval", subtask)
labels = ds["train"].features["label"].names

# CV pool = train + validation; test stays untouched for a final check later
pool_texts = list(ds["train"]["text"]) + list(ds["validation"]["text"])
pool_labels = np.array(list(ds["train"]["label"]) + list(ds["validation"]["label"]))
NUM_CLASSES = len(labels)


if __name__ == "__main__":
    dfs = {split: ds[split].to_pandas() for split in ds}
    for split, df in dfs.items():
        df["label_name"] = df["label"].map(lambda i: labels[i])

    counts = pd.DataFrame({s: df["label_name"].value_counts() for s, df in dfs.items()})
    props = counts / counts.sum()

    print(counts, "\n")
    print(props.round(3), "\n")

    for s in dfs:
      print (s, "majority baseline:", props[s].max())

    for df in dfs.values():
        df["n_chars"] = df["text"].str.len()
        df["n_words"] = df["text"].str.split().str.len()

    all_df = pd.concat(dfs, names=["split"])
    print(all_df[["n_chars", "n_words"]].describe().round(1).drop(index="count"))
    print(all_df.groupby("label_name")[["n_chars", "n_words"]].mean().round(1)) # Checking if length differs by class

    print(len(pool_texts), "tweets in CV pool")

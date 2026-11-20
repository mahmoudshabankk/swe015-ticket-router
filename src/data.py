"""Download CLINC150 ('plus' config) and save normalized train/val/test CSVs.

All four models use exactly these splits; the test split (4,500 in-scope + 1,000 out-of-scope)
is the single held-out test set for every number in the results table.
"""
import json

import pandas as pd
from datasets import load_dataset

from config import DATA_DIR, INTENT2DOMAIN
from preprocess import normalize


def build():
    ds = load_dataset("clinc/clinc_oos", "plus")
    names = ds["train"].features["intent"].names
    missing = set(names) ^ set(INTENT2DOMAIN)
    assert not missing, f"domain map mismatch: {missing}"
    labels = sorted(names)  # stable label order
    (DATA_DIR / "labels.json").write_text(json.dumps(labels, indent=1))
    for split in ("train", "validation", "test"):
        df = ds[split].to_pandas()
        df["intent"] = df["intent"].map(lambda i: names[i])
        df["label"] = df["intent"].map(labels.index)
        df["clean"] = df["text"].map(lambda t: normalize(t)["text"])
        df.to_csv(DATA_DIR / f"{split}.csv", index=False)
        print(split, len(df), "oos:", (df.intent == "oos").sum())


def load(split):
    df = pd.read_csv(DATA_DIR / f"{split}.csv", keep_default_na=False)
    return df


def labels():
    return json.loads((DATA_DIR / "labels.json").read_text())


if __name__ == "__main__":
    build()

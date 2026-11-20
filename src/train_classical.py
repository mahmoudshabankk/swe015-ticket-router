"""Model 1 (area B): tf-idf + logistic regression baseline.
Model 2 (area A): feedforward network on averaged GloVe word embeddings, trained with SGD + backprop."""
import pickle
import time

import gensim.downloader as api
import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from config import MODEL_DIR, SEED
from data import labels, load

torch.manual_seed(SEED)
np.random.seed(SEED)


def train_tfidf(tr, va):
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    Xtr = vec.fit_transform(tr["clean"])
    clf = LogisticRegression(C=20, max_iter=3000)
    clf.fit(Xtr, tr["label"])
    acc = (clf.predict(vec.transform(va["clean"])) == va["label"]).mean()
    print(f"tf-idf+LR val acc {acc:.4f}")
    with open(MODEL_DIR / "tfidf_lr.pkl", "wb") as f:
        pickle.dump({"vec": vec, "clf": clf}, f)


class FFN(nn.Module):
    def __init__(self, d_in, n_out, hidden=512, p=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.net(x)


def avg_embed(texts, kv, dim):
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        vs = [kv[w] for w in t.split() if w in kv]
        if vs:
            out[i] = np.mean(vs, axis=0)
    return out


def train_ffn(tr, va, n_labels):
    kv = api.load("glove-wiki-gigaword-100")
    # keep only the vocabulary we need -> small file for the app
    # training vocabulary + the 30k most frequent GloVe words (no test data involved)
    vocab = {w for split in (tr, va) for t in split["clean"] for w in t.split()} | set(kv.index_to_key[:30000])
    small = {w: kv[w] for w in vocab if w in kv}
    print("embedding vocab", len(small))
    Xtr = torch.tensor(avg_embed(tr["clean"], small, 100))
    Xva = torch.tensor(avg_embed(va["clean"], small, 100))
    ytr, yva = torch.tensor(tr["label"].values), torch.tensor(va["label"].values)
    model = FFN(100, n_labels)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)
    loss_fn = nn.CrossEntropyLoss()
    best, t0 = 0, time.time()
    for ep in range(80):
        model.train()
        perm = torch.randperm(len(Xtr))
        for s in range(0, len(perm), 64):
            b = perm[s:s + 64]
            loss = loss_fn(model(Xtr[b]), ytr[b])
            opt.zero_grad()
            loss.backward()          # backpropagation
            opt.step()               # SGD update
        sched.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Xva).argmax(-1) == yva).float().mean().item()
        if acc > best:
            best = acc
            torch.save({"state": model.state_dict(), "emb": small}, MODEL_DIR / "ffn_glove.pt")
        if ep % 10 == 0:
            print(f"ffn ep{ep} loss {loss.item():.3f} val acc {acc:.4f}", flush=True)
    print(f"FFN best val acc {best:.4f} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    tr, va = load("train"), load("validation")
    train_tfidf(tr, va)
    train_ffn(tr, va, len(labels()))

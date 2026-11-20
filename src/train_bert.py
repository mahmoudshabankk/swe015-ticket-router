"""Technique area C: fine-tune BERT for 151-way intent classification (150 intents + oos).

    python train_bert.py teacher    # bert-base-uncased, cross-entropy
    python train_bert.py student    # 4-layer BERT-small, knowledge distillation from the teacher (area D)
"""
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from config import MODEL_DIR, SEED
from data import labels, load

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(8)
MAX_LEN = 48

CFG = {
    "teacher": dict(base="bert-base-uncased", out="bert-base-teacher", epochs=3, lr=4e-5, bs=32),
    "student": dict(base="google/bert_uncased_L-4_H-512_A-8", out="bert-small-student", epochs=8, lr=1e-4,
                    bs=32, T=2.0, alpha=0.7),
}


def batches(texts, ys, bs, tok, shuffle):
    idx = np.random.permutation(len(texts)) if shuffle else np.arange(len(texts))
    for s in range(0, len(idx), bs):
        b = idx[s:s + bs]
        enc = tok([texts[i] for i in b], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        yield b, enc, torch.tensor(ys[b])


@torch.no_grad()
def predict_logits(model, tok, texts, bs=128):
    model.eval()
    out = []
    for _, enc, _ in batches(texts, np.zeros(len(texts), dtype=int), bs, tok, False):
        out.append(model(**enc).logits)
    return torch.cat(out)


def main(kind):
    cfg = CFG[kind]
    labs = labels()
    tr, va = load("train"), load("validation")
    Xtr, ytr = tr["clean"].tolist(), tr["label"].values
    Xva, yva = va["clean"].tolist(), va["label"].values
    tok = AutoTokenizer.from_pretrained(cfg["base"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["base"], num_labels=len(labs), id2label=dict(enumerate(labs)), label2id={l: i for i, l in enumerate(labs)})

    teacher_logits = None
    if kind == "student":
        tdir = MODEL_DIR / CFG["teacher"]["out"]
        teacher = AutoModelForSequenceClassification.from_pretrained(tdir)
        ttok = AutoTokenizer.from_pretrained(tdir)
        t0 = time.time()
        teacher_logits = predict_logits(teacher, ttok, Xtr)
        print(f"teacher logits computed in {time.time() - t0:.0f}s", flush=True)
        del teacher

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    steps = cfg["epochs"] * ((len(Xtr) + cfg["bs"] - 1) // cfg["bs"])
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    best = -1
    for ep in range(cfg["epochs"]):
        model.train()
        t0, tot = time.time(), 0.0
        for step, (b, enc, y) in enumerate(batches(Xtr, ytr, cfg["bs"], tok, True)):
            logits = model(**enc).logits
            loss = F.cross_entropy(logits, y)
            if teacher_logits is not None:
                T, a = cfg["T"], cfg["alpha"]
                kd = F.kl_div(F.log_softmax(logits / T, -1), F.softmax(teacher_logits[b] / T, -1),
                              reduction="batchmean") * T * T
                loss = a * kd + (1 - a) * loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            if step % 100 == 0:
                print(f"ep{ep} step{step} loss {loss.item():.3f} ({time.time() - t0:.0f}s)", flush=True)
        acc = (predict_logits(model, tok, Xva).argmax(-1).numpy() == yva).mean()
        print(f"epoch {ep}: train loss {tot / (step + 1):.4f} val acc {acc:.4f} time {time.time() - t0:.0f}s",
              flush=True)
        if acc > best:
            best = acc
            model.save_pretrained(MODEL_DIR / cfg["out"])
            tok.save_pretrained(MODEL_DIR / cfg["out"])
    print("best val acc", best)


if __name__ == "__main__":
    main(sys.argv[1])

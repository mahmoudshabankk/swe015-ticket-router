"""Evaluate every model of the ladder on the same held-out CLINC150 test set (5,500 messages).

* The confidence threshold for routing to "human agent" (predict oos) is tuned on the validation set.
* Metrics: overall accuracy (151 classes incl. oos), in-scope accuracy, out-of-scope recall,
  routing (team) accuracy.
* Size = size of the weight files on disk; latency = one message end to end (tokenize + model),
  batch size 1, averaged over 300 test messages after 20 warm-up runs.
"""
import json
import os
import platform
import time

import numpy as np
import pandas as pd
import torch

from config import INTENT2DOMAIN, OOS, RESULTS_DIR
from data import labels, load
from models import APP_MODEL, all_models, size_mb

LABELS = labels()
OOS_ID = LABELS.index(OOS)
DOM = np.array([INTENT2DOMAIN[l] for l in LABELS])


def apply_threshold(p, tau):
    pred = p.argmax(1)
    pred[p.max(1) < tau] = OOS_ID
    return pred


def tune_threshold(p, y):
    grid = np.round(np.arange(0.0, 0.96, 0.02), 2)
    accs = [(apply_threshold(p, t) == y).mean() for t in grid]
    return float(grid[int(np.argmax(accs))])


def metrics(pred, y):
    ins = y != OOS_ID
    return {
        "accuracy": (pred == y).mean(),
        "in_scope_acc": (pred[ins] == y[ins]).mean(),
        "oos_recall": (pred[~ins] == OOS_ID).mean(),
        "routing_acc": (DOM[pred] == DOM[y]).mean(),
    }


def latency_ms(model, texts, n=300, warm=20):
    for t in texts[:warm]:
        model.proba([t])
    t0 = time.perf_counter()
    for t in texts[:n]:
        model.proba([t])
    return (time.perf_counter() - t0) / n * 1000


def cpu_name():
    try:
        import subprocess
        out = subprocess.run(["wmic", "cpu", "get", "name"], capture_output=True, text=True).stdout.splitlines()
        return [line.strip() for line in out if line.strip() and line.strip() != "Name"][0]
    except Exception:
        return platform.processor()


def main():
    va, te = load("validation"), load("test")
    rows, thresholds, preds = [], {}, {}
    rng = np.random.default_rng(0)
    lat_texts = te["clean"].values[rng.permutation(len(te))].tolist()
    for m in all_models():
        tau = tune_threshold(m.proba(va["clean"].tolist()), va["label"].values)
        p = m.proba(te["clean"].tolist())
        pred = apply_threshold(p, tau)
        preds[m.name] = [LABELS[i] for i in pred]
        r = {"model": m.name, "precision": m.precision, **metrics(pred, te["label"].values),
             "threshold": tau, "size_mb": size_mb(m), "latency_ms": latency_ms(m, lat_texts)}
        thresholds[m.name] = tau
        rows.append(r)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "results.csv", index=False)
    (RESULTS_DIR / "thresholds.json").write_text(json.dumps(thresholds, indent=1))
    out = te[["text", "intent"]].copy()
    for k, v in preds.items():
        out[k] = v
    out.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)
    # failure examples of the app model: wrong in-scope intent, missed oos, in-scope sent to human
    app = APP_MODEL
    wrong = out[out[app] != out["intent"]]
    picks = pd.concat([wrong[(wrong.intent != OOS) & (wrong[app] != OOS)].head(2),
                       wrong[wrong.intent == OOS].head(1),
                       wrong[(wrong.intent != OOS) & (wrong[app] == OOS)].head(1)])
    (RESULTS_DIR / "failures.json").write_text(json.dumps(
        [{"text": t, "true": i, "pred": p} for t, i, p in zip(picks.text, picks.intent, picks[app])], indent=1))
    (RESULTS_DIR / "data_stats.json").write_text(json.dumps(
        {s: {"n": len(d), "oos": int((d.intent == OOS).sum())}
         for s, d in [("train", load("train")), ("validation", va), ("test", te)]}))
    import onnxruntime
    hw = (f"{cpu_name()}, {os.cpu_count()} logical cores | {platform.system()} {platform.release()} | "
          f"torch {torch.__version__}, onnxruntime {onnxruntime.__version__} | "
          f"{torch.get_num_threads()} inference threads | CPU only")
    (RESULTS_DIR / "hardware.txt").write_text(hw)
    fmt = df.copy()
    for c in ["accuracy", "in_scope_acc", "oos_recall", "routing_acc"]:
        fmt[c] = (fmt[c] * 100).map("{:.2f}".format)
    fmt["size_mb"] = fmt["size_mb"].map("{:.1f}".format)
    fmt["latency_ms"] = fmt["latency_ms"].map("{:.2f}".format)
    (RESULTS_DIR / "results.md").write_text(fmt.to_markdown(index=False) + f"\n\nHardware: {hw}\n")
    print(fmt.to_markdown(index=False))
    print(hw)


if __name__ == "__main__":
    main()

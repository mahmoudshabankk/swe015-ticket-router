"""Uniform inference wrappers: every model maps a list of normalized texts -> class probabilities."""
import pickle
import warnings

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import MODEL_DIR
from data import labels

warnings.filterwarnings("ignore")
torch.set_num_threads(4)
MAX_LEN = 48


class TfidfLR:
    name, precision = "tf-idf + LogReg", "fp64"
    files = ["tfidf_lr.pkl"]

    def __init__(self):
        with open(MODEL_DIR / "tfidf_lr.pkl", "rb") as f:
            d = pickle.load(f)
        self.vec, self.clf = d["vec"], d["clf"]

    def proba(self, texts):
        return self.clf.predict_proba(self.vec.transform(texts))


class GloveFFN:
    name, precision = "FFN on avg. GloVe", "fp32"
    files = ["ffn_glove.pt"]

    def __init__(self):
        from train_classical import FFN
        ck = torch.load(MODEL_DIR / "ffn_glove.pt", weights_only=False)
        self.emb = ck["emb"]
        self.model = FFN(100, len(labels()))
        self.model.load_state_dict(ck["state"])
        self.model.eval()

    def proba(self, texts):
        from train_classical import avg_embed
        with torch.no_grad():
            return torch.softmax(self.model(torch.tensor(avg_embed(texts, self.emb, 100))), -1).numpy()


class Bert:
    precision = "fp32"

    def __init__(self, dirname, display):
        self.name = display
        self.files = [f"{dirname}/model.safetensors"]
        self.tok = AutoTokenizer.from_pretrained(MODEL_DIR / dirname)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR / dirname).eval()

    @torch.no_grad()
    def proba(self, texts, bs=128):
        out = []
        for s in range(0, len(texts), bs):
            enc = self.tok(list(texts[s:s + bs]), padding=True, truncation=True, max_length=MAX_LEN,
                           return_tensors="pt")
            out.append(torch.softmax(self.model(**enc).logits, -1))
        return torch.cat(out).numpy()


class QuantBert(Bert):
    precision = "INT8 (dynamic)"

    def __init__(self, dirname, display):
        from quantize import load_quantized
        self.name = display
        self.files = [f"{dirname}-int8.pt"]
        self.tok = AutoTokenizer.from_pretrained(MODEL_DIR / dirname)
        self.model = load_quantized(dirname)


class OnnxQuantBert:
    precision = "INT8 (ONNX Runtime)"

    def __init__(self, dirname, display):
        import onnxruntime as ort
        self.name = display
        self.files = [f"{dirname}-int8.onnx"]
        self.tok = AutoTokenizer.from_pretrained(MODEL_DIR / dirname)
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        self.sess = ort.InferenceSession(str(MODEL_DIR / f"{dirname}-int8.onnx"), so,
                                         providers=["CPUExecutionProvider"])

    def proba(self, texts, bs=128):
        out = []
        for s in range(0, len(texts), bs):
            enc = self.tok(list(texts[s:s + bs]), padding=True, truncation=True, max_length=MAX_LEN,
                           return_tensors="np")
            logits = self.sess.run(None, {k: enc[k] for k in ("input_ids", "attention_mask", "token_type_ids")})[0]
            e = np.exp(logits - logits.max(-1, keepdims=True))
            out.append(e / e.sum(-1, keepdims=True))
        return np.concatenate(out)


APP_MODEL = "BERT-small distilled INT8 (ONNX)"


def all_models(include_extra=True):
    """The model ladder. The last entry is the one the application uses."""
    ms = [TfidfLR(), GloveFFN(), Bert("bert-base-teacher", "BERT-base (teacher)")]
    if include_extra:
        ms += [OnnxQuantBert("bert-base-teacher", "BERT-base INT8 (ONNX)"),
               Bert("bert-small-student", "BERT-small distilled"),
               QuantBert("bert-small-student", "BERT-small distilled INT8 (PyTorch)")]
    ms.append(OnnxQuantBert("bert-small-student", APP_MODEL))
    return ms


def size_mb(model):
    return sum((MODEL_DIR / f).stat().st_size for f in model.files) / 1e6

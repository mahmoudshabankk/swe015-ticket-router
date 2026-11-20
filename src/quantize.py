"""Technique area D: post-training dynamic INT8 quantization (PyTorch and ONNX Runtime).

Linear layers -> INT8 weights with dynamically quantized activations;
embedding tables -> INT8 weight-only (float qparams). Runs on CPU.

    python quantize.py bert-small-student   # the model the app uses
    python quantize.py bert-base-teacher    # extra row for the results table
"""
import sys
import warnings

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification

from config import MODEL_DIR

warnings.filterwarnings("ignore")
from torch.ao.quantization import default_dynamic_qconfig, float_qparams_weight_only_qconfig, quantize_dynamic  # noqa

QSPEC = {nn.Linear: default_dynamic_qconfig, nn.Embedding: float_qparams_weight_only_qconfig}


def quantize(model):
    model.eval()
    return quantize_dynamic(model, QSPEC, dtype=torch.qint8)


def save_quantized(name):
    src = MODEL_DIR / name
    model = AutoModelForSequenceClassification.from_pretrained(src)
    q = quantize(model)
    out = MODEL_DIR / f"{name}-int8.pt"
    torch.save(q.state_dict(), out)
    print("saved", out, f"{out.stat().st_size / 1e6:.1f} MB")


def load_quantized(name):
    """Rebuild the architecture from the fp32 config, quantize the empty shell, then load INT8 weights."""
    cfg = AutoConfig.from_pretrained(MODEL_DIR / name)
    model = AutoModelForSequenceClassification.from_config(cfg)
    q = quantize(model)
    q.load_state_dict(torch.load(MODEL_DIR / f"{name}-int8.pt", weights_only=False))
    return q.eval()




# ----------------------------------------------------------------------------- ONNX Runtime INT8 (used by the app)
def export_onnx_int8(name):
    """Export the fp32 model to ONNX, then ONNX Runtime dynamic quantization (INT8 weights, MatMul/Gemm).

    On this CPU, PyTorch's oneDNN INT8 kernels were slower than fp32, while ONNX Runtime INT8 is about 2x
    faster than ONNX fp32, so the application uses this file."""
    from onnxruntime.quantization import QuantType
    from onnxruntime.quantization import quantize_dynamic as ort_quantize_dynamic
    from transformers import AutoTokenizer
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR / name).eval()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR / name)
    enc = tok(["hello my card was declined"], return_tensors="pt")
    names = ["input_ids", "attention_mask", "token_type_ids"]
    f32, i8 = MODEL_DIR / f"{name}.onnx", MODEL_DIR / f"{name}-int8.onnx"
    torch.onnx.export(model, tuple(enc[k] for k in names), str(f32), input_names=names, output_names=["logits"],
                      dynamic_axes={k: {0: "batch", 1: "seq"} for k in names}, opset_version=17, dynamo=False)
    ort_quantize_dynamic(str(f32), str(i8), weight_type=QuantType.QInt8)
    f32.unlink()
    print("saved", i8, f"{i8.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    save_quantized(sys.argv[1])
    export_onnx_int8(sys.argv[1])

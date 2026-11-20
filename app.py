"""Customer Support Ticket Router - Gradio application.

Run:  python app.py   ->  http://127.0.0.1:7860
Pipeline: regex normalization + model-code extraction -> edit-distance spelling correction
-> ELIZA rules for small talk -> quantized distilled BERT (INT8) intent classifier -> routing.
The other three models of the ladder are shown side by side for comparison.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402

from config import HUMAN_AGENT, INTENT2DOMAIN, OOS, RESULTS_DIR, route  # noqa: E402
import hub  # noqa: E402
from data import labels, load  # noqa: E402

hub.ensure_data()                                 # CLINC150 csv files (needed by the speller)
if not hub.models_present():                      # weights are in a GitHub Release, not in git
    hub.download()
from models import all_models  # noqa: E402
from preprocess import Speller, eliza_reply, normalize  # noqa: E402

LABELS = labels()
MODELS = all_models(include_extra=False)          # tf-idf, FFN, BERT-base fp32, BERT-small INT8
PRIMARY = MODELS[-1]                              # quantized Transformer runs the app
THRESH = json.loads((RESULTS_DIR / "thresholds.json").read_text())
# candidates come from the training vocabulary; the 30k GloVe words count as valid English (never "corrected")
SPELLER = Speller(load("train")["clean"], known_words=MODELS[1].emb.keys())


def predict(model, text):
    t0 = time.perf_counter()
    p = model.proba([text])[0]
    ms = (time.perf_counter() - t0) * 1000
    i = int(p.argmax())
    conf = float(p[i])
    intent = LABELS[i]
    low = conf < THRESH.get(model.name, 0.5)
    team = HUMAN_AGENT if (intent == OOS or low) else route(intent)
    return intent, conf, team, ms, low


def handle(message):
    if not message or not message.strip():
        return "Please type a message.", "", None
    norm = normalize(message)
    corrected, fixes = SPELLER.correct(norm["text"])
    steps = [f"**Normalized:** `{norm['text']}`",
             f"**Model codes extracted:** {', '.join(norm['codes']) or '-'}",
             f"**Spelling fixes (edit distance):** "
             f"{', '.join(f'{a} -> {b}' for a, b in fixes) or '-'}",
             f"**Model input:** `{corrected}`"]

    rows = []
    for m in MODELS:
        intent, conf, team, ms, low = predict(m, corrected)
        rows.append({"model": m.name, "precision": m.precision, "intent": intent,
                     "confidence": round(conf, 3), "routed to": team, "latency (ms)": round(ms, 1)})
    intent, conf, team, ms, low = predict(PRIMARY, corrected)

    rule = eliza_reply(corrected)
    if rule:
        decision = f"### 💬 Small talk (rule-based reply)\n> {rule}"
    elif INTENT2DOMAIN.get(intent) == "small_talk" and not low:
        decision = f"### 💬 Small talk - `{intent}` ({conf:.0%})\n> {eliza_reply(corrected, force=True)}"
    elif team == HUMAN_AGENT:
        why = "out-of-scope" if intent == OOS else f"low confidence ({conf:.0%} for `{intent}`)"
        decision = f"### 🧑 Routed to: **{HUMAN_AGENT}**\nReason: {why}"
    else:
        decision = (f"### ✅ Routed to: **{team}**\nIntent: `{intent}` · confidence **{conf:.0%}** · "
                    f"{PRIMARY.name} · {ms:.1f} ms")
    if norm["codes"]:
        decision += f"\n\nTicket tagged with model code(s): **{', '.join(norm['codes'])}**"
    return decision, "\n\n".join(steps), pd.DataFrame(rows)


EXAMPLES = ["how do i transfer money to my sister's account",
            "my credt card got declnied at the store, model DW-6534",
            "how many moons does jupiter have", "Hellooo!!", "can you set an alarm for 6am",
            "is my flight to paris on time", "my dishwasher DW-6534 is not draining water"]

with gr.Blocks(title="Support Ticket Router") as demo:
    gr.Markdown("# Customer Support Ticket Router\nType any support message. The **quantized distilled "
                "BERT (INT8, ONNX Runtime)** decides the team; all four models of the ladder are compared below.")
    with gr.Row():
        inp = gr.Textbox(label="Customer message (press Enter to route)", lines=1, scale=4)
        btn = gr.Button("Route", variant="primary", scale=1)
    gr.Examples(EXAMPLES, inp)
    with gr.Row():
        out = gr.Markdown()
        pre = gr.Markdown()
    table = gr.Dataframe(label="All models side by side", wrap=True)
    btn.click(handle, inp, [out, pre, table])
    inp.submit(handle, inp, [out, pre, table])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)

"""Builds presentation/slides.html and slides.pdf. Every number is read from results/ (measured runs).

    python presentation/build.py      (edit presentation/info.json first: name, student id, links)
"""
import base64
import json
import subprocess
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
RES = HERE.parent / "results"
info = json.loads((HERE / "info.json").read_text(encoding="utf8"))
df = pd.read_csv(RES / "results.csv")
hw = (RES / "hardware.txt").read_text().strip()
fail = json.loads((RES / "failures.json").read_text(encoding="utf8"))
stats = json.loads((RES / "data_stats.json").read_text())


def img(name):
    return "data:image/png;base64," + base64.b64encode((RES / name).read_bytes()).decode()


def r(model):
    return df[df.model == model].iloc[0]


tfidf, ffn, bert, bq, st, spt, sq = (r(m) for m in [
    "tf-idf + LogReg", "FFN on avg. GloVe", "BERT-base (teacher)", "BERT-base INT8 (ONNX)", "BERT-small distilled",
    "BERT-small distilled INT8 (PyTorch)", "BERT-small distilled INT8 (ONNX)"])
LADDER = {"tf-idf + LogReg": "1 · baseline", "FFN on avg. GloVe": "2 · neural net",
          "BERT-base (teacher)": "3 · Transformer", "BERT-base INT8 (ONNX)": "extra",
          "BERT-small distilled": "extra", "BERT-small distilled INT8 (PyTorch)": "extra",
          "BERT-small distilled INT8 (ONNX)": "4 · quantized (app)"}

rows = "".join(
    f"<tr class='{'hl' if m.model == 'BERT-small distilled INT8 (ONNX)' else ''}'><td>{LADDER[m.model]}</td><td>{m.model}</td>"
    f"<td>{m.precision}</td><td>{m.accuracy * 100:.2f}</td><td>{m.oos_recall * 100:.2f}</td>"
    f"<td>{m.in_scope_acc * 100:.2f}</td><td>{m.routing_acc * 100:.2f}</td><td>{m.size_mb:.1f}</td>"
    f"<td>{m.latency_ms:.2f}</td></tr>" for m in df.itertuples())

fail_rows = "".join(f"<tr><td>&ldquo;{f['text']}&rdquo;</td><td>{f['true']}</td><td>{f['pred']}</td></tr>"
                    for f in fail)
stat_rows = "".join(f"<tr><td>{k}</td><td>{v['n']:,}</td><td>{v['n'] - v['oos']:,}</td><td>{v['oos']:,}</td></tr>"
                    for k, v in stats.items())

speed = bert.latency_ms / sq.latency_ms
shrink = bert.size_mb / sq.size_mb

SLIDES = [
    # 1 title
    f"""<div class="title"><div class="kicker">SWE015 · Introduction to Large Language Models · Homework</div>
    <h1>Customer Support Ticket Router</h1>
    <p class="sub">Use Case 2: route free-text support messages to the right team, with four models from a
    classical baseline to a quantized Transformer</p>
    <table class="meta"><tr><td>Name</td><td>{info['name']}</td></tr><tr><td>Student ID</td><td>{info['student_id']}</td></tr>
    <tr><td>Use case</td><td>2 · Customer Support Ticket Router</td></tr>
    <tr><td>Repository</td><td><a href="{info['repo_url']}">{info['repo_url']}</a></td></tr>
    <tr><td>Demo video</td><td><a href="{info['video_url']}">{info['video_url']}</a></td></tr></table></div>""",
    # 2 intro
    """<div class="sec">Introduction</div><h2>The problem and what success looks like</h2>
    <div class="cols"><div><h3>Problem</h3><p>A support center gets thousands of messages a day. Each one has to reach
    the right team right away. Some messages are out of scope or ambiguous and must go to a human. Greetings and
    small talk should get an instant automatic reply.</p>
    <h3>Who it is for</h3><p>Support operations: fewer messages routed to the wrong team, faster first answers, and
    agents' time spent where it is needed.</p>
    <h3>Use case and domain</h3><p>Use Case 2. The domain is adapted to the 10 service areas of CLINC150
    (banking, cards, travel, smart home, HR, ...). Each area is one team.</p></div>
    <div><h3>What success looks like</h3><ul>
    <li>High <b>intent accuracy</b> on a held-out test set (151 classes incl. out-of-scope)</li>
    <li>High <b>out-of-scope recall</b>: unknown requests go to a human agent, not to a wrong team</li>
    <li>Answers in real time on a <b>CPU-only</b> server (milliseconds per message)</li>
    <li>A small model that ships inside the app (the <b>quantized Transformer</b>)</li></ul>
    <h3>Application</h3><p>Message &rarr; normalize &rarr; intent + confidence &rarr; routed team. Low confidence or
    out-of-scope &rarr; <i>human agent</i>. Small talk &rarr; ELIZA-style reply. All four models are shown side by side.</p></div></div>""",
    # 3 data
    f"""<div class="sec">Development · Data</div><h2>Data: CLINC150 (&ldquo;plus&rdquo; configuration)</h2>
    <div class="cols"><div><h3>Source</h3><p>Larson et al. (2019), <i>An Evaluation Dataset for Intent Classification
    and Out-of-Scope Prediction</i>, EMNLP. Loaded from the Hugging Face Hub as <code>clinc/clinc_oos</code>, config <code>plus</code>.</p>
    <table class="small"><tr><th>Split</th><th>Messages</th><th>In-scope</th><th>Out-of-scope</th></tr>{stat_rows}</table>
    <p>150 intents in 10 domains, plus an <code>oos</code> class, so 151 labels. The <b>same test split</b> is used for
    every model. Validation is used only for model selection and for tuning the confidence threshold.</p></div>
    <div><h3>Preprocessing (same for all models)</h3><ul>
    <li>Regex normalization: lowercase, URLs &rarr; <code>url</code>, e-mails &rarr; <code>email</code>, elongated letters
    (&ldquo;hellooooo&rdquo; &rarr; &ldquo;helloo&rdquo;), punctuation and whitespace</li>
    <li>Regex <b>model-code extraction</b> (e.g. <code>DW-6534</code>); the code is attached to the ticket and removed from the model input</li>
    <li>Spelling correction by <b>minimum edit distance</b> (Damerau-Levenshtein) against the training vocabulary plus product names (used in the app)</li>
    <li>Transformers: WordPiece subword tokenizer, max. 48 tokens</li></ul>
    <h3>Routing</h3><p>intent &rarr; domain (CLINC150 domain map) &rarr; team. Example: <code>card_declined</code> &rarr; credit_cards &rarr; Card Services Team.</p></div></div>""",
    # 4 architecture
    f"""<div class="sec">Development · Architecture</div><h2>Architecture diagram (matches <code>app.py</code> and <code>src/</code>)</h2>
    <img class="full" src="{img('architecture.png')}">""",
    # 5 technique map
    """<div class="sec">Development · Technique map</div><h2>Technique map: which component uses which technique</h2>
    <table><tr><th>Area</th><th>Technique</th><th>Component in the application</th><th>Code</th></tr>
    <tr><td><span class="tag b">B</span> Classical NLP</td><td>Regular expressions: normalization and model-code extraction</td><td>Regex normalizer (first step of every request)</td><td>src/preprocess.py</td></tr>
    <tr><td><span class="tag b">B</span> Classical NLP</td><td>Minimum edit distance spelling correction</td><td>Spelling corrector (misspelled words and product names)</td><td>src/preprocess.py</td></tr>
    <tr><td><span class="tag b">B</span> Classical NLP</td><td>ELIZA-style pattern rules with pronoun reflection</td><td>Small-talk replies (greetings, thanks, &ldquo;I feel ...&rdquo;)</td><td>src/preprocess.py</td></tr>
    <tr><td><span class="tag b">B</span> Classical NLP</td><td>tf-idf (uni+bigrams) + logistic regression</td><td>Model 1: baseline</td><td>src/train_classical.py</td></tr>
    <tr><td><span class="tag a">A</span> Neural networks</td><td>Feedforward net on averaged GloVe embeddings, SGD + backpropagation, dropout</td><td>Model 2: neural network</td><td>src/train_classical.py</td></tr>
    <tr><td><span class="tag c">C</span> Transformers</td><td>Fine-tuning BERT with a classification head; WordPiece tokenizer; attention</td><td>Model 3: BERT-base teacher</td><td>src/train_bert.py</td></tr>
    <tr><td><span class="tag d">D</span> Efficient models</td><td>Knowledge distillation (teacher &rarr; 4-layer student)</td><td>BERT-small student</td><td>src/train_bert.py</td></tr>
    <tr><td><span class="tag d">D</span> Efficient models</td><td>Post-training dynamic INT8 quantization; size and latency benchmark</td><td>Model 4: runs inside the app</td><td>src/quantize.py (ONNX Runtime), src/evaluate.py</td></tr></table>""",
    # 6 models
    """<div class="sec">Development · Models</div><h2>The four models and how each was trained</h2>
    <div class="grid4">
    <div class="card"><div class="n b">1</div><h3>tf-idf + LogReg</h3><p>Uni+bigram tf-idf (sublinear tf), multinomial
    logistic regression, C = 20. scikit-learn. Trains in seconds.</p></div>
    <div class="card"><div class="n a">2</div><h3>FFN on GloVe</h3><p>Average of 100-d GloVe vectors (Wikipedia+Gigaword) &rarr;
    512 &rarr; 512 &rarr; 151, ReLU, dropout 0.3. Trained from scratch with <b>SGD</b> (momentum 0.9, lr 0.05, cosine schedule),
    80 epochs, backpropagation.</p></div>
    <div class="card"><div class="n c">3</div><h3>BERT-base (teacher)</h3><p><code>bert-base-uncased</code> (110M params) + linear
    head, fine-tuned 3 epochs, AdamW lr 4e-5, batch 32, warm-up + linear decay, cross-entropy.</p></div>
    <div class="card"><div class="n d">4</div><h3>Distilled + quantized</h3><p>Student <code>google/bert_uncased_L-4_H-512_A-8</code>
    (4 layers, 29M params) trained 8 epochs on 0.7 &middot; KL(teacher || student, T = 2) + 0.3 &middot; CE. Then <b>dynamic INT8
    quantization</b>: exported to ONNX, then ONNX Runtime <code>quantize_dynamic</code> (INT8 weights, activations quantized at run time).
    PyTorch's dynamic INT8 was also tried (row in the results table) but was <i>slower</i> than fp32 on this CPU.</p></div></div>
    <h3>Out-of-scope handling (same for all models)</h3><p>The <code>oos</code> class is trained like any other intent. On top of
    that, if the top probability is below a threshold &tau;, the message goes to a human agent. &tau; is tuned per model on the
    <b>validation</b> set (grid 0&ndash;0.94) to maximize accuracy. The test set is never used for tuning.</p>""",
    # 7 results
    f"""<div class="sec">Development · Results</div><h2>Results on the same held-out test set (5,500 messages)</h2>
    <table class="res"><tr><th>Ladder</th><th>Model</th><th>Precision</th><th>Accuracy %</th><th>OOS recall %</th>
    <th>In-scope acc. %</th><th>Routing acc. %</th><th>Size MB</th><th>Latency ms</th></tr>{rows}</table>
    <p class="note">Accuracy is over 151 classes incl. out-of-scope. Routing accuracy means the right team (or human agent).
    Size is the weight files on disk. Latency is one message end to end (tokenize + forward), batch 1, the mean of
    300 runs after 20 warm-up runs. Hardware: {hw}.</p>""",
    # 8 chart
    f"""<div class="sec">Development · Results</div><h2>Trade-off: accuracy, size and latency</h2>
    <img class="full" src="{img('results_chart.png')}">
    <p>The model in the app (BERT-small distilled INT8, ONNX Runtime) is <b>{shrink:.0f}&times; smaller</b> and <b>{speed:.1f}&times; faster</b>
    than the BERT-base teacher. Its accuracy is {(sq.accuracy - bert.accuracy) * 100:+.2f} points different
    ({sq.accuracy * 100:.2f}% vs {bert.accuracy * 100:.2f}%).</p>""",
    # 9 conclusion
    f"""<div class="sec">Conclusion</div><h2>What the comparison showed</h2><div class="cols"><div>
    <h3>Where each model wins and loses</h3><ul>
    <li><b>tf-idf + LR</b>: {tfidf.accuracy * 100:.1f}% accuracy at {tfidf.latency_ms:.2f} ms, {tfidf.oos_recall * 100:.1f}% OOS recall. Cheap, but it only sees exact word overlap.</li>
    <li><b>FFN on GloVe</b>: {ffn.accuracy * 100:.1f}% accuracy, {ffn.oos_recall * 100:.1f}% OOS recall. Averaging word vectors loses word order and negation.</li>
    <li><b>BERT-base</b>: {bert.accuracy * 100:.1f}% accuracy, {bert.oos_recall * 100:.1f}% OOS recall. Contextual attention understands paraphrases, but the model is {bert.size_mb:.0f} MB and takes {bert.latency_ms:.1f} ms per message.</li>
    <li><b>Distilled + INT8</b>: {sq.accuracy * 100:.1f}% accuracy at {sq.size_mb:.0f} MB and {sq.latency_ms:.1f} ms. Quantization alone changed accuracy by
    {(sq.accuracy - st.accuracy) * 100:+.2f} points (student) and {(bq.accuracy - bert.accuracy) * 100:+.2f} points (teacher).</li>
    <li><b>PyTorch INT8</b> shrank the model the same way, but took {spt.latency_ms:.1f} ms vs {st.latency_ms:.1f} ms in fp32. Its oneDNN INT8 kernels are slow on this AMD CPU, so ONNX Runtime was chosen.</li></ul></div>
    <div><h3>Take-aways</h3><ul><li>Distillation plus quantization keeps most of the Transformer's quality at a fraction of the size and latency. That is the best fit for a CPU-only router.</li>
    <li>Out-of-scope detection is the hardest part for every model: OOS recall is lower than in-scope accuracy.</li>
    <li>The classical baseline wins clearly on latency and size, so it stays useful as a fallback.</li></ul></div></div>""",
    # 10 limitations
    f"""<div class="sec">Conclusion</div><h2>Limitations, failure examples and next steps</h2>
    <div class="cols"><div><h3>Failure examples (quantized model, real test predictions)</h3>
    <table class="small"><tr><th>Message</th><th>True</th><th>Predicted</th></tr>{fail_rows}</table></div>
    <div><h3>Limitations</h3><ul><li>CLINC150 is English only. The Turkish part of the scenario is not covered.</li>
    <li>The messages are short and clean, while real tickets are long and noisy.</li>
    <li>The threshold on softmax confidence is a crude out-of-scope detector.</li>
    <li>Spelling correction picks the most frequent close word, not the one meant: &ldquo;mony&rdquo; &rarr; &ldquo;many&rdquo; (not &ldquo;money&rdquo;). It first turned &ldquo;draining&rdquo; into &ldquo;raining&rdquo;. Fixed by treating the 30k GloVe words as valid English.</li>
    <li>Appliance repair is not one of the 150 intents, so &ldquo;my dishwasher DW-6534 is not draining water&rdquo; can only go to a human agent or to a wrong team.</li></ul>
    <h3>Next steps</h3><ul><li>Multilingual student (XLM-R / BERTurk) on MASSIVE with Turkish data</li>
    <li>Add real appliance tickets and an active-learning loop with agents</li>
    <li>Static INT8 quantization or pruning of the student for even lower latency</li>
    <li>Better OOS detection (energy score or Mahalanobis distance)</li></ul></div></div>""",
    # 11 AI usage
    """<div class="sec">Conclusion</div><h2>AI usage statement</h2>
    <table><tr><th>Tool</th><th>Used for</th></tr>
    <tr><td>Claude (Anthropic) via Claude Code</td><td>Choosing the use case and the design; writing and debugging the Python code (preprocessing, training, quantization, evaluation, Gradio app); generating the architecture diagram and these slides.</td></tr>
    <tr><td>My own work</td><td>Running all training and evaluation on my machine; checking the code and results; recording and narrating the demo video. I can explain every component in the architecture diagram.</td></tr></table>
    <p>All numbers in this presentation come from my own runs (<code>src/evaluate.py</code> &rarr; <code>results/results.csv</code>). No number was taken from a paper, a model card or an AI answer.</p>
    <h3>Sources</h3><ul><li>CLINC150: Larson et al., EMNLP-IJCNLP 2019 (CC BY 3.0)</li><li>GloVe: Pennington, Socher, Manning, EMNLP 2014 (via gensim-data)</li>
    <li>BERT: Devlin et al., NAACL 2019; small BERT: Turc et al., 2019 (google/bert_uncased_L-4_H-512_A-8)</li><li>Knowledge distillation: Hinton, Vinyals, Dean, 2015</li></ul>""",
]

CSS = """@page{size:13.333in 7.5in;margin:0}*{box-sizing:border-box}body{margin:0;font-family:'Segoe UI',Arial,sans-serif;color:#1f2937}
.slide{width:13.333in;height:7.5in;padding:.55in .7in;page-break-after:always;position:relative;overflow:hidden;background:#fff}
.slide:after{content:attr(data-n);position:absolute;right:.5in;bottom:.3in;color:#9ca3af;font-size:12px}
h1{font-size:46px;margin:.2in 0 .1in;color:#111827}h2{font-size:30px;margin:0 0 .22in;color:#111827}h3{font-size:17px;margin:.12in 0 .05in;color:#4c1d95}
p,li{font-size:15px;line-height:1.42}ul{margin:.04in 0;padding-left:.25in}.sec{font-size:13px;text-transform:uppercase;letter-spacing:.12em;color:#9d174d;font-weight:700;margin-bottom:.06in}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:.45in}table{border-collapse:collapse;width:100%;font-size:14px}th{background:#f3f4f6;text-align:left}
th,td{border-bottom:1px solid #e5e7eb;padding:6px 8px;vertical-align:top}table.small td,table.small th{font-size:12.5px;padding:4px 6px}
table.res td,table.res th{font-size:14px;padding:8px}tr.hl td{background:#fce7f3;font-weight:600}.note{font-size:12.5px;color:#4b5563}
img.full{max-width:100%;max-height:5.4in;display:block;margin:0 auto}code{background:#f3f4f6;padding:1px 4px;border-radius:3px;font-size:.92em}
.title{padding-top:.5in}.kicker{color:#9d174d;font-weight:700;letter-spacing:.08em;font-size:14px}.sub{font-size:20px;color:#4b5563;max-width:10in}
table.meta{width:auto;margin-top:.35in}table.meta td{font-size:17px;border:none;padding:5px 18px 5px 0}table.meta td:first-child{color:#6b7280}
.tag{display:inline-block;width:22px;text-align:center;border-radius:4px;font-weight:700}.a{background:#dcfce7}.b{background:#fef3c7}.c{background:#ede9fe}.d{background:#fce7f3}
.grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:.2in}.card{border:1px solid #e5e7eb;border-radius:10px;padding:.15in}.card p{font-size:13.5px}
.n{display:inline-block;width:30px;height:30px;border-radius:50%;text-align:center;line-height:30px;font-weight:700}a{color:#4c1d95}"""

html = ("<!doctype html><html><head><meta charset='utf-8'><title>Ticket Router slides</title><style>" + CSS
        + "</style></head><body>")
html += "".join(f"<section class='slide' data-n='{i + 1}'>{s}</section>" for i, s in enumerate(SLIDES))
html += "</body></html>"
(HERE / "slides.html").write_text(html, encoding="utf8")

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
subprocess.run([EDGE, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={HERE / 'slides.pdf'}", (HERE / "slides.html").as_uri()], check=True)
print("wrote", HERE / "slides.pdf", f"({len(SLIDES)} slides)")

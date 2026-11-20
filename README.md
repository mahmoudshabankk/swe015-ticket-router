# Customer Support Ticket Router (SWE015 homework, Use Case 2)

You type a free-text support message. The app normalizes it, predicts the intent with a confidence score and
routes it to the right team. Low-confidence and out-of-scope messages go to a **human agent**, and small talk
gets an **ELIZA-style** reply. The app runs a **distilled, INT8-quantized BERT** and shows all four models of
the ladder side by side.

| Ladder | Model | Area |
|---|---|---|
| 1. baseline | tf-idf (uni+bigrams) + logistic regression | B: classical NLP |
| 2. neural net | feedforward network on averaged GloVe embeddings, trained with SGD + backprop | A: neural networks |
| 3. Transformer | `bert-base-uncased` fine-tuned for 151-way intent classification | C: Transformers |
| 4. quantized Transformer | BERT-base distilled into 4-layer BERT-small, then dynamic INT8 quantization | D: efficient models |

Other classical NLP parts (area B) are regex normalization, model-code extraction (`DW-6534`), edit-distance
spelling correction and ELIZA rules (all in `src/preprocess.py`).

## Setup

Python 3.10–3.12, CPU only. No GPU needed.

```bash
git clone https://github.com/mahmoudshabankk/swe015-ticket-router.git
cd swe015-ticket-router
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Run (single command)

```bash
python app.py
```

Then open http://127.0.0.1:7860. On the first run, `app.py` downloads the CLINC150 dataset (Hugging Face
datasets) and the trained weights (about 0.5 GB) from this repository's
[GitHub Release v1.0](https://github.com/mahmoudshabankk/swe015-ticket-router/releases/tag/v1.0):
`models-app.zip`, plus `models-extra.zip` for the extra rows of the results table. The weights are **not**
stored in the git repository.

To re-run the evaluation without training, use `python src/hub.py download-all` and then
`python src/evaluate.py`.

## Reproduce everything from scratch (optional, about 1.5 h on CPU)

```bash
python src/train_all.py
```

This runs data preparation, tf-idf + LR, FFN, BERT-base fine-tuning, distillation, quantization, evaluation,
the diagram and the plots, then writes `results/results.md`. Rebuild the slides with
`python presentation/build.py`.

## Results (same held-out CLINC150 test set, 5,500 messages)

See [results/results.md](results/results.md). Every number is produced by `src/evaluate.py`.

## Layout

```
app.py                   Gradio application (runs the quantized model)
src/config.py            paths, intent -> domain -> team map
src/preprocess.py        regex normalization, model codes, edit distance speller, ELIZA rules
src/data.py              CLINC150 download + splits
src/train_classical.py   tf-idf + LR, FFN on GloVe (SGD)
src/train_bert.py        BERT-base fine-tuning (teacher) and knowledge distillation (student)
src/quantize.py          dynamic INT8 quantization
src/evaluate.py          metrics, size, latency -> results/
src/hub.py               download weights from the GitHub Release
presentation/build.py    slides (HTML -> PDF), numbers read from results/
```

## Data and sources

- CLINC150: Larson et al., *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction*,
  EMNLP-IJCNLP 2019. Hugging Face `clinc/clinc_oos` (config `plus`), CC BY 3.0.
- GloVe 6B 100d: Pennington et al., 2014 (via `gensim-data`).
- `bert-base-uncased` (Devlin et al., 2019) and `google/bert_uncased_L-4_H-512_A-8` (Turc et al., 2019).

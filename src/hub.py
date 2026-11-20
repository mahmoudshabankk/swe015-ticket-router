"""Model weights are NOT stored in the git repository. They are attached to a GitHub Release as zip files.

    python src/hub.py download        # done automatically by app.py when models/ is empty
    python src/hub.py download-all    # also the extra models needed to re-run src/evaluate.py
    python src/hub.py pack            # (author) build the zip files from models/ for uploading
"""
import shutil
import sys
import urllib.request
import zipfile

from config import DATA_DIR, MODEL_DIR

RELEASE = "https://github.com/mahmoudshabankk/swe015-ticket-router/releases/download/v1.0"

# files the application needs (4 ladder models)
APP_FILES = ["tfidf_lr.pkl", "ffn_glove.pt",
             "bert-base-teacher/config.json", "bert-base-teacher/model.safetensors",
             "bert-base-teacher/tokenizer.json", "bert-base-teacher/tokenizer_config.json",
             "bert-small-student/config.json", "bert-small-student/tokenizer.json",
             "bert-small-student/tokenizer_config.json", "bert-small-student-int8.onnx"]
# extra rows of the results table
EXTRA_FILES = ["bert-small-student/model.safetensors", "bert-small-student-int8.pt", "bert-base-teacher-int8.onnx"]
PACKS = {"models-app.zip": APP_FILES, "models-extra.zip": EXTRA_FILES}


def models_present(files=APP_FILES):
    return all((MODEL_DIR / f).exists() for f in files)


def ensure_data():
    if not (DATA_DIR / "train.csv").exists():
        from data import build
        build()


def _fetch(name):
    url = f"{RELEASE}/{name}"
    tmp = MODEL_DIR / name
    print(f"Downloading {url} ...", flush=True)
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    with zipfile.ZipFile(tmp) as z:
        z.extractall(MODEL_DIR)
    tmp.unlink()


def download(extra=False):
    _fetch("models-app.zip")
    if extra:
        _fetch("models-extra.zip")


def pack(out_dir):
    for name, files in PACKS.items():
        with zipfile.ZipFile(f"{out_dir}/{name}", "w", zipfile.ZIP_STORED) as z:
            for f in files:
                z.write(MODEL_DIR / f, f)
        print("packed", name)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "pack":
        pack(sys.argv[2])
    else:
        download(extra=cmd == "download-all")

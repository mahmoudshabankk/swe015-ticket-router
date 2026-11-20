"""Reproduce every model and number from scratch (CPU: roughly 1-1.5 hours).

    python src/train_all.py
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
STEPS = [["data.py"], ["train_classical.py"], ["train_bert.py", "teacher"], ["train_bert.py", "student"],
         ["quantize.py", "bert-small-student"], ["quantize.py", "bert-base-teacher"], ["evaluate.py"],
         ["diagram.py"], ["plots.py"]]

for step in STEPS:
    print(">>>", *step, flush=True)
    subprocess.run([sys.executable, str(HERE / step[0]), *step[1:]], check=True, cwd=HERE)

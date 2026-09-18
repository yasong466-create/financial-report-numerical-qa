#!/usr/bin/env python3
"""Reliable sequential queue for the paper's supplemental experiments."""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def run(*args):
    cmd = [sys.executable, *map(str, args)]
    print("\nQUEUE:", " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

# DeepSeek FinQA-100 is already complete. Continue the remaining providers.
for provider in ("kimi", "qwen", "glm"):
    run(ROOT / "run_all_models.py", "--only", provider, "--max_samples", "100",
        "--run_name", "multimodel_formal_100")

# Rebuild the four-model summary and generate thesis-ready prose.
run(ROOT / "run_all_models.py", "--only", "rebuild_summary", "--max_samples", "100",
    "--run_name", "multimodel_formal_100")
run(ROOT / "generate_paper_report.py", "--run_name", "multimodel_formal_100")

# Run every locally available Chinese case (the current source contains 12).
run(ROOT / "prepare_chinese_cases.py")
run(ROOT / "run_all_models.py", "--data_path", ROOT / "data/chinese_mixue.json",
    "--max_samples", "0", "--run_name", "multimodel_chinese")

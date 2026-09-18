#!/usr/bin/env python3
"""Run the fixed 500-sample FinQA ablation suite with resume support."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
METHODS = [
    "ours",
    "standard_rag_python",
    "ours_no_fusion",
    "ours_no_routing",
    "ours_no_python",
]

for method in METHODS:
    print(f"\n===== ABLATION 500: {method} =====", flush=True)
    command = [
        sys.executable,
        str(ROOT / "run_all_models.py"),
        "--only", "deepseek",
        "--method", method,
        "--max_samples", "500",
        "--data_path", str(ROOT / "data/finqa/test.json"),
        "--run_name", "ablation_500",
    ]
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode != 0:
        print(f"STOP: {method} exited with {result.returncode}", flush=True)
        raise SystemExit(result.returncode)

print("\nAll 500-sample ablation runs completed.", flush=True)

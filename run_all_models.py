#!/usr/bin/env python3
"""Run one fixed FinQA experiment across configured model providers."""
import argparse, csv, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def load_env(path):
    if not path.exists(): return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max_samples", type=int, default=3)
    p.add_argument("--method", default="ours")
    p.add_argument("--only", nargs="*")
    p.add_argument("--data_path", default=str(ROOT / "data/finqa/test.json"))
    p.add_argument("--run_name", default="multimodel")
    a = p.parse_args()
    load_env(ROOT / ".env")
    load_env(ROOT / ".env.models")
    config = json.loads((ROOT / "models.json").read_text(encoding="utf-8"))
    rows = []
    for item in config["models"]:
        name = item["name"]
        if not item.get("enabled", True) or (a.only and name not in a.only): continue
        key = os.getenv(item["api_key_env"], "")
        if not key:
            print(f"SKIP {name}: .env.models 中缺少 {item['api_key_env']}")
            continue
        if not item.get("model") or "请填写" in item["model"]:
            print(f"SKIP {name}: models.json 中尚未填写模型 ID")
            continue
        out = ROOT / "outputs" / a.run_name / name
        out.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.update(OPENAI_API_KEY=key, OPENAI_BASE_URL=item["base_url"], OPENAI_MODEL=item["model"])
        env["LLM_MIN_INTERVAL_SECONDS"] = str(item.get("min_interval_seconds", 0))
        env["LLM_OMIT_TEMPERATURE"] = "1" if item.get("omit_temperature", False) else "0"
        env["LLM_MAX_COMPLETION_TOKENS"] = str(item.get("max_completion_tokens", 0))
        cmd = [sys.executable, str(ROOT / "run_finqa_exp.py"), "--data_path", a.data_path,
               "--method", a.method, "--model", item["model"], "--max_samples", str(a.max_samples),
               "--topk", "5", "--text_topk", "3", "--table_topk", "3", "--output_dir", str(out), "--resume"]
        print(f"\n=== {name}: {item['model']} ===", flush=True)
        code = subprocess.run(cmd, env=env, check=False).returncode
        summary_path = out / f"{a.method}_summary.json"
        if code == 0 and summary_path.exists():
            s = json.loads(summary_path.read_text(encoding="utf-8"))
            rows.append({"provider": name, "model": item["model"], "samples": s.get("num_samples"),
                         "EM": s.get("EM"), "F1": s.get("F1"),
                         "retrieval_hit_rate": s.get("retrieval_hit_rate"),
                         "code_success_rate": s.get("code_success_rate"),
                         "numeric_accuracy": s.get("numeric_accuracy"),
                         "failed_samples": s.get("failed_samples"), "status": "ok"})
        else:
            rows.append({"provider": name, "model": item["model"], "status": "failed"})
    out = ROOT / "outputs" / a.run_name
    out.mkdir(parents=True, exist_ok=True)
    # Preserve successful summaries from earlier provider-specific runs.
    present = {row.get("provider") for row in rows}
    for item in config["models"]:
        name = item["name"]
        if name in present:
            continue
        summary_path = out / name / f"{a.method}_summary.json"
        if not summary_path.exists():
            continue
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append({"provider": name, "model": item["model"], "samples": s.get("num_samples"),
                     "EM": s.get("EM"), "F1": s.get("F1"),
                     "retrieval_hit_rate": s.get("retrieval_hit_rate"),
                     "code_success_rate": s.get("code_success_rate"),
                     "numeric_accuracy": s.get("numeric_accuracy"),
                     "failed_samples": s.get("failed_samples"), "status": "ok"})
    order = {item["name"]: i for i, item in enumerate(config["models"])}
    rows.sort(key=lambda row: order.get(row.get("provider"), 999))
    fields = ["provider", "model", "samples", "EM", "F1", "retrieval_hit_rate",
              "code_success_rate", "numeric_accuracy", "failed_samples", "status"]
    with open(out / "summary.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    print(f"\n汇总结果：{out / 'summary.csv'}")

if __name__ == "__main__": main()

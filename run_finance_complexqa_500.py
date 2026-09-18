#!/usr/bin/env python3
"""FinanceComplexQA 中文公开子集 500 条、可断点续跑的补充实验。"""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / ".python_deps_clean"))

from run_cfqa_public_exp import BM25, char_f1, exact_match, load_env, numerical_equal
from run_finqa_exp import call_llm, generate_python_code, safe_execute_python

DATA_DIR = ROOT / "data/public/finance_complexqa"
QA_ZIP = DATA_DIR / "FinComplexQA-Pro.zip"
DOC_ZIP = DATA_DIR / "Reference_documents.zip"
CALC_RE = re.compile(
    r"多少|百分比|比例|占比|增长率|收益率|利润率|金额|总额|合计|之和|差额|"
    r"几倍|计算|下降了|增加了|减少了|同比|环比|净利率|毛利率"
)


def gold_text(row):
    value = row.get("gold", "")
    return "\n".join(map(str, value)) if isinstance(value, list) else str(value)


def load_cn_rows():
    rows = {}
    with ZipFile(QA_ZIP) as archive:
        for name in archive.namelist():
            if not (name.startswith("CN/scene_categories/") and name.endswith(".jsonl")):
                continue
            for line in archive.read(name).decode("utf-8-sig").splitlines():
                if line.strip():
                    row = json.loads(line)
                    rows[row["finqa_id"]] = row
    return list(rows.values())


def select_balanced(rows, limit):
    """Round-robin across scene and task; calculations are interleaved first."""
    for row in rows:
        row["is_calculation"] = bool(CALC_RE.search(row["question"]))
    rows.sort(key=lambda x: (not x["is_calculation"], x["finqa_id"]))
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["doc_type"], row["task_type"])].append(row)
    selected = []
    keys = sorted(buckets)
    while len(selected) < min(limit, len(rows)):
        changed = False
        for key in keys:
            if buckets[key] and len(selected) < limit:
                selected.append(buckets[key].pop(0)); changed = True
        if not changed:
            break
    return selected


def latex_to_text(raw):
    text = raw.decode("utf-8-sig", errors="replace")
    text = re.sub(r"%.*", " ", text)
    text = re.sub(r"\\(?:begin|end)\{[^}]+\}", "\n", text)
    text = re.sub(r"\\(?:section|subsection|subsubsection|caption)\*?\{([^{}]*)\}", r"\n\1\n", text)
    text = re.sub(r"\\(?:textbf|textit|emph)\{([^{}]*)\}", r"\1", text)
    text = text.replace("&", " | ").replace("\\\\", "\n")
    text = re.sub(r"\\[a-zA-Z]+(?:\[[^]]*\])?", " ", text)
    text = re.sub(r"[{}$]", " ", text)
    return re.sub(r"[ \t]+", " ", text)


def chunk_text(text, size=1600, overlap=250):
    paragraphs = [p.strip() for p in re.split(r"\n+", text) if len(p.strip()) >= 20]
    chunks, current = [], ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) + 1 > size:
            chunks.append(current)
            current = current[-overlap:] + "\n" + paragraph
        else:
            current = (current + "\n" + paragraph).strip()
    if current:
        chunks.append(current)
    return chunks


def load_corpus(rows):
    requested = {}
    for row in rows:
        zip_name = f"CN/{row['doc_type']}/{row['Reference_documents']}"
        requested[zip_name] = row["finqa_id"]
    docs, missing = [], []
    with ZipFile(DOC_ZIP) as archive:
        available = set(archive.namelist())
        for source in sorted(requested):
            if source not in available:
                missing.append(source); continue
            for index, chunk in enumerate(chunk_text(latex_to_text(archive.read(source)))):
                docs.append({"content": chunk, "source": source, "chunk": index})
    if missing:
        print(f"警告：{len(missing)} 个参考文档未找到", file=sys.stderr)
    return docs


def format_evidence(rows, max_chars=12000):
    blocks, used = [], 0
    for i, row in enumerate(rows, 1):
        block = f"[证据{i}｜{row['source']}｜片段{row['chunk']}]\n{row['content']}"
        block = block[:max(0, max_chars - used)]
        if block:
            blocks.append(block); used += len(block)
        if used >= max_chars:
            break
    return "\n\n".join(blocks)


def answer(sample, retrieved, model, method):
    evidence = format_evidence(retrieved)
    code_success = None
    if method == "ours" and sample["is_calculation"]:
        code = generate_python_code(sample["question"], evidence, model=model)
        result = safe_execute_python(code)
        code_success = bool(result["success"])
        if result["success"] and result["answer"] is not None:
            return str(result["answer"]), code_success
    prompt = f"""请仅依据给定参考证据回答金融问题。
要求：答案准确、简洁；保留必要的数值、单位和关键依据；不得使用证据之外的信息。

问题：{sample['question']}

参考证据：
{evidence}
"""
    return call_llm(prompt, model=model), code_success


def write_rows(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--only", nargs="*", default=["deepseek"])
    parser.add_argument("--methods", nargs="*", choices=["standard_rag", "ours"], default=["standard_rag", "ours"])
    parser.add_argument("--run_name", default="finance_complexqa_500")
    args = parser.parse_args()

    load_env(ROOT / ".env"); load_env(ROOT / ".env.models")
    configs = json.loads((ROOT / "models.json").read_text(encoding="utf-8"))["models"]
    samples = select_balanced(load_cn_rows(), args.max_samples)
    corpus = load_corpus(samples)
    retriever = BM25(corpus)
    retrievals = {row["finqa_id"]: retriever.topk(row["question"], args.topk) for row in samples}
    output_root = ROOT / "outputs" / args.run_name
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "sample_manifest.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    summaries = []

    for config in configs:
        provider = config["name"]
        if provider not in args.only:
            continue
        key = os.getenv(config["api_key_env"], "")
        if not key:
            print(f"跳过 {provider}：缺少 {config['api_key_env']}"); continue
        os.environ.update(OPENAI_API_KEY=key, OPENAI_BASE_URL=config["base_url"], OPENAI_MODEL=config["model"],
                          LLM_MIN_INTERVAL_SECONDS=str(config.get("min_interval_seconds", 0)),
                          LLM_OMIT_TEMPERATURE="1" if config.get("omit_temperature", False) else "0",
                          LLM_MAX_COMPLETION_TOKENS=str(config.get("max_completion_tokens", 0)))
        for method in args.methods:
            target = output_root / provider / method
            target.mkdir(parents=True, exist_ok=True)
            pred_file = target / "predictions.csv"
            saved = {}
            if pred_file.exists():
                with pred_file.open(encoding="utf-8-sig") as stream:
                    saved = {r["id"]: r for r in csv.DictReader(stream) if not r.get("error", "").strip()}
            for index, sample in enumerate(samples, 1):
                sid = sample["finqa_id"]
                if sid in saved:
                    continue
                print(f"[{provider}/{method} {index}/{len(samples)}] {sample['question'][:36]}", flush=True)
                retrieved = retrievals[sid]
                try:
                    pred, code_ok = answer(sample, retrieved, config["model"], method); error = ""
                except Exception as exc:
                    pred, code_ok, error = "", None, str(exc)
                gold = gold_text(sample)
                gold_source = f"CN/{sample['doc_type']}/{sample['Reference_documents']}"
                saved[sid] = {
                    "id": sid, "method": method, "scene": sample["doc_type"], "task": sample["task_type"],
                    "is_calculation": int(sample["is_calculation"]), "question": sample["question"],
                    "gold": gold, "pred": pred, "em": exact_match(pred, gold), "f1": char_f1(pred, gold),
                    "retrieval_hit": int(any(r["source"] == gold_source for r in retrieved)),
                    "code_success": code_ok,
                    "numeric_correct": int(numerical_equal(pred, gold)) if sample["is_calculation"] else "",
                    "retrieved_json": json.dumps([{"source": r["source"], "chunk": r["chunk"], "score": r["score"]} for r in retrieved], ensure_ascii=False),
                    "error": error,
                }
                write_rows(pred_file, [saved[x["finqa_id"]] for x in samples if x["finqa_id"] in saved])
            records = [saved[x["finqa_id"]] for x in samples if x["finqa_id"] in saved and not saved[x["finqa_id"]].get("error")]
            calc = [r for r in records if str(r["is_calculation"]) == "1"]
            code_rows = [r for r in calc if str(r["code_success"]) not in {"", "None"}]
            summary = {
                "provider": provider, "model": config["model"], "method": method, "samples": len(samples),
                "completed": len(records), "calculation_samples": len(calc),
                "EM": sum(float(r["em"]) for r in records) / max(len(records), 1),
                "F1": sum(float(r["f1"]) for r in records) / max(len(records), 1),
                "Recall@5": sum(float(r["retrieval_hit"]) for r in records) / max(len(records), 1),
                "code_success_rate": sum(str(r["code_success"]) == "True" for r in code_rows) / max(len(code_rows), 1),
                "numeric_accuracy": sum(float(r["numeric_correct"] or 0) for r in calc) / max(len(calc), 1),
            }
            (target / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            summaries.append(summary)
    write_rows(output_root / "summary.csv", summaries)
    print(f"结果目录：{output_root}")


if __name__ == "__main__":
    main()

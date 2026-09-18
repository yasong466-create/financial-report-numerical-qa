#!/usr/bin/env python3
"""Run the fixed public-CFQA experiment across configured domestic LLMs."""

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / ".python_deps_clean"))

from run_finqa_exp import call_llm, generate_python_code, safe_execute_python


def load_env(path):
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def tokenize(text):
    text = str(text).lower()
    latin = re.findall(r"[a-z0-9_.%+-]+", text)
    # PDF extraction often inserts spaces between Chinese characters in tables.
    chinese_text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    chinese_runs = re.findall(r"[\u4e00-\u9fff]+", chinese_text)
    chinese = []
    for run in chinese_runs:
        chinese.extend(run)
        chinese.extend(run[i:i + 2] for i in range(len(run) - 1))
    return latin + chinese


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.tokens = [tokenize(row["content"]) for row in docs]
        self.lengths = [len(row) for row in self.tokens]
        self.avgdl = sum(self.lengths) / max(len(self.lengths), 1)
        self.df = defaultdict(int)
        for row in self.tokens:
            for token in set(row):
                self.df[token] += 1
        self.n = len(docs)

    def topk(self, query, k):
        query_tokens = tokenize(query)
        scored = []
        for index, tokens in enumerate(self.tokens):
            tf = Counter(tokens)
            score = 0.0
            for token in query_tokens:
                freq = tf.get(token, 0)
                if not freq:
                    continue
                df = self.df[token]
                idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
                denom = freq + self.k1 * (1 - self.b + self.b * len(tokens) / max(self.avgdl, 1))
                score += idf * freq * (self.k1 + 1) / denom
            scored.append((score, index))
        result = []
        for score, index in sorted(scored, reverse=True)[:k]:
            row = dict(self.docs[index])
            row["score"] = score
            result.append(row)
        return result


def normalize(text):
    text = str(text or "").lower().replace(",", "").replace("，", "")
    return re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)


def numbers(text):
    return [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", str(text or "").replace(",", ""))]


def numerical_equal(pred, gold):
    predicted, expected = numbers(pred), numbers(gold)
    if not predicted or not expected:
        return False
    for p in predicted:
        for g in expected:
            for candidate in (p, p / 100, p * 100):
                if abs(candidate - g) <= max(1e-3, abs(g) * 1e-3):
                    return True
    return False


def exact_match(pred, gold):
    p, g = normalize(pred), normalize(gold)
    return int(p == g or (len(g) >= 4 and g in p) or numerical_equal(pred, gold))


def char_f1(pred, gold):
    if exact_match(pred, gold):
        return 1.0
    p, g = list(normalize(pred)), list(normalize(gold))
    if not p or not g:
        return 0.0
    common = sum((Counter(p) & Counter(g)).values())
    if not common:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def format_evidence(rows, max_chars=12000):
    blocks, used = [], 0
    for index, row in enumerate(rows, 1):
        meta = row["meta"]
        block = f"[证据{index}｜{meta['filename']}｜PDF第{meta['pdf_page_number']}页]\n{row['content']}"
        if used + len(block) > max_chars:
            block = block[:max(0, max_chars - used)]
        if block:
            blocks.append(block)
            used += len(block)
        if used >= max_chars:
            break
    return "\n\n".join(blocks)


def retrieval_query(question, question_type):
    query = str(question)
    query = re.sub(r"(请问|是多少|为多少|有多少|什么是|分别|总额|亿元|万元|人民币|？|\?)", "", query)
    if question_type == "事实提取":
        query = re.sub(r"20\d{2}(?:年|年度)?", "", query)
        query = query.replace("年度", "").replace("公司", "")
    return query.strip() or str(question)


def build_document_retrievers(corpus):
    grouped = defaultdict(list)
    for row in corpus:
        grouped[row["meta"]["filename"]].append(row)
    return {filename: BM25(rows) for filename, rows in grouped.items()}


def retrieve(question_row, global_retriever, document_retrievers, k):
    question = question_row["question"]
    years = set(re.findall(r"20\d{2}", question))
    candidates = []
    for filename in document_retrievers:
        match = re.search(r"\.SH-([^-]+)-", filename)
        company = match.group(1) if match else ""
        if company and company in question and (not years or any(year in filename for year in years)):
            candidates.append(filename)
    query = retrieval_query(question, question_row["type"])
    if not candidates:
        return global_retriever.topk(query, k)
    rows = []
    for filename in candidates:
        rows.extend(document_retrievers[filename].topk(query, k))
    return sorted(rows, key=lambda row: row["score"], reverse=True)[:k]


def answer_question(sample, retrieved, model):
    evidence = format_evidence(retrieved)
    code_success = None
    if sample["type"] == "比较计算":
        code = generate_python_code(sample["question"], evidence, model=model)
        execution = safe_execute_python(code)
        code_success = bool(execution["success"])
        if execution["success"] and execution["answer"] is not None:
            return str(execution["answer"]), code_success
    prompt = f"""请仅根据给定年报证据回答金融问题。
要求：直接给出简洁最终答案；保留必要单位；不要虚构证据中没有的信息。

问题：{sample['question']}

证据：
{evidence}
"""
    return call_llm(prompt, model=model), code_success


def select_questions(questions, available, max_samples):
    covered = [row for row in questions if row["filename"] in available]
    if max_samples <= 0 or max_samples >= len(covered):
        return covered
    buckets = defaultdict(list)
    for row in covered:
        buckets[row["type"]].append(row)
    selected = []
    while len(selected) < max_samples:
        changed = False
        for question_type in sorted(buckets):
            if buckets[question_type] and len(selected) < max_samples:
                selected.append(buckets[question_type].pop(0))
                changed = True
        if not changed:
            break
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=129)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--run_name", default="cfqa_public_129")
    args = parser.parse_args()

    load_env(ROOT / ".env")
    load_env(ROOT / ".env.models")
    models = json.loads((ROOT / "models.json").read_text(encoding="utf-8"))["models"]
    corpus = json.loads((ROOT / "data/cfqa_zhu2026/page_corpus.json").read_text(encoding="utf-8"))
    questions = json.loads((ROOT / "data/cfqa_zhu2026/gold_answer_subset.json").read_text(encoding="utf-8"))
    available = {row["meta"]["filename"] for row in corpus}
    questions = select_questions(questions, available, args.max_samples)
    retriever = BM25(corpus)
    document_retrievers = build_document_retrievers(corpus)
    retrieval_cache = [retrieve(row, retriever, document_retrievers, args.topk) for row in questions]
    output_root = ROOT / "outputs" / args.run_name
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []

    for config in models:
        provider = config["name"]
        if not config.get("enabled", True) or (args.only and provider not in args.only):
            continue
        key = os.getenv(config["api_key_env"], "")
        if not key:
            print(f"SKIP {provider}: 缺少 {config['api_key_env']}")
            continue
        os.environ.update(
            OPENAI_API_KEY=key,
            OPENAI_BASE_URL=config["base_url"],
            OPENAI_MODEL=config["model"],
            LLM_MIN_INTERVAL_SECONDS=str(config.get("min_interval_seconds", 0)),
            LLM_OMIT_TEMPERATURE="1" if config.get("omit_temperature", False) else "0",
            LLM_MAX_COMPLETION_TOKENS=str(config.get("max_completion_tokens", 0)),
        )
        provider_dir = output_root / provider
        provider_dir.mkdir(parents=True, exist_ok=True)
        prediction_file = provider_dir / "predictions.csv"
        existing = {}
        saved_rows = {}
        if prediction_file.exists():
            with prediction_file.open(encoding="utf-8-sig") as stream:
                saved_rows = {
                    row["id"]: row
                    for row in csv.DictReader(stream)
                }
                # Resume only successful rows. Failed API rows must be retried
                # instead of being treated as completed samples.
                existing = {
                    sample_id: row
                    for sample_id, row in saved_rows.items()
                    if not str(row.get("error", "")).strip()
                }
        records = []
        for index, (sample, retrieved) in enumerate(zip(questions, retrieval_cache), 1):
            sample_id = f"{sample['filename']}::{sample['page']}::{sample['question']}"
            if sample_id in existing:
                records.append(existing[sample_id])
                continue
            print(f"[{provider} {index}/{len(questions)}] {sample['question'][:40]}")
            try:
                pred, code_success = answer_question(sample, retrieved, config["model"])
                error = ""
            except Exception as exc:
                pred, code_success, error = "", None, str(exc)
            hit = int(any(
                row["meta"]["filename"] == sample["filename"]
                # CFQA's page field follows the printed page number; the PDF
                # file has one unnumbered front page, hence the +1 offset.
                and row["meta"]["page_index"] == int(sample["page"]) + 1
                for row in retrieved
            ))
            result_row = {
                "id": sample_id, "type": sample["type"], "question": sample["question"],
                "gold": sample["answer"], "pred": pred,
                "em": exact_match(pred, sample["answer"]), "f1": char_f1(pred, sample["answer"]),
                "retrieval_hit": hit, "code_success": code_success,
                "numeric_correct": int(numerical_equal(pred, sample["answer"])) if sample["type"] == "比较计算" else "",
                "gold_filename": sample["filename"], "gold_page": sample["page"],
                "retrieved_json": json.dumps([row["meta"] for row in retrieved], ensure_ascii=False),
                "error": error,
            }
            records.append(result_row)
            saved_rows[sample_id] = result_row
            # Always rewrite the union of all previously saved rows and the
            # current result. This keeps later successful rows intact even if
            # the process is interrupted while retrying an earlier failure.
            ordered_saved_rows = []
            for saved_sample in questions:
                saved_id = f"{saved_sample['filename']}::{saved_sample['page']}::{saved_sample['question']}"
                if saved_id in saved_rows:
                    ordered_saved_rows.append(saved_rows[saved_id])
            with prediction_file.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(ordered_saved_rows[0]))
                writer.writeheader(); writer.writerows(ordered_saved_rows)

        completed = [row for row in records if not row.get("error")]
        calc = [row for row in completed if row["type"] == "比较计算"]
        code_rows = [row for row in calc if str(row["code_success"]) not in {"", "None"}]
        summary = {
            "provider": provider, "model": config["model"], "samples": len(records),
            "completed": len(completed),
            "EM": sum(float(row["em"]) for row in completed) / max(len(completed), 1),
            "F1": sum(float(row["f1"]) for row in completed) / max(len(completed), 1),
            "retrieval_hit_rate": sum(float(row["retrieval_hit"]) for row in completed) / max(len(completed), 1),
            "code_success_rate": sum(str(row["code_success"]) == "True" for row in code_rows) / max(len(code_rows), 1),
            "numeric_accuracy": sum(float(row["numeric_correct"] or 0) for row in calc) / max(len(calc), 1),
        }
        (provider_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summaries.append(summary)

    if summaries:
        with (output_root / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader(); writer.writerows(summaries)
    print(f"结果目录：{output_root}")


if __name__ == "__main__":
    main()

import argparse
import csv
import json
import math
import os
import re
import sys
import time
import random
from collections import Counter, defaultdict
from pathlib import Path

LOCAL_DEPS = Path(__file__).resolve().parent / ".python_deps_clean"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

ENV_FILE = Path(__file__).resolve().parent / ".env"
LAST_LLM_CALL_AT = 0.0


def load_dotenv_file(path=ENV_FILE):
    if not path.exists():
        return

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value


load_dotenv_file()

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# =========================
# 1. 基础工具
# =========================

def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"数据文件不存在：{path}\n"
            "请先把 FinQA 数据集文件放到这个位置，或用 --data_path 指向正确的 json 文件。"
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path, records):
    fieldnames = [
        "id",
        "question",
        "gold",
        "pred",
        "em",
        "f1",
        "code_success",
        "numeric_correct",
        "retrieval_hit",
        "model",
        "retrieved_json",
        "method",
        "error",
    ]

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def normalize_answer(s):
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = s.replace(",", "")
    s = s.replace("$", "")
    s = s.replace("%", "")
    s = re.sub(r"\s+", " ", s)
    return s


def extract_number(s):
    if s is None:
        return None
    s = str(s).replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums:
        return None
    try:
        return float(nums[0])
    except Exception:
        return None


def exact_match(pred, gold):
    p = normalize_answer(pred)
    g = normalize_answer(gold)

    if p == g:
        return 1

    pn = extract_number(pred)
    gn = extract_number(gold)

    if pn is not None and gn is not None:
        if abs(pn - gn) <= 1e-3:
            return 1
        if abs(gn) > 1e-8 and abs((pn - gn) / gn) <= 1e-3:
            return 1
        # FinQA answers often store rates as decimals while models answer in
        # percentage points, e.g. gold=0.14464 and pred=14.464.
        for scaled_pn in (pn / 100, pn * 100):
            if abs(scaled_pn - gn) <= 1e-3:
                return 1
            if abs(gn) > 1e-8 and abs((scaled_pn - gn) / gn) <= 1e-3:
                return 1

    return 0


def token_f1(pred, gold):
    # For numerical QA, formatting variants such as 94 and 94.0 (or a
    # decimal/percentage pair) are semantically identical. Keep F1 aligned
    # with the shared numerical normalization used by EM.
    if exact_match(pred, gold):
        return 1.0

    p_tokens = normalize_answer(pred).split()
    g_tokens = normalize_answer(gold).split()

    if len(p_tokens) == 0 and len(g_tokens) == 0:
        return 1.0
    if len(p_tokens) == 0 or len(g_tokens) == 0:
        return 0.0

    common = Counter(p_tokens) & Counter(g_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(p_tokens)
    recall = num_same / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


def get_gold_answer(sample):
    qa = sample.get("qa", {})
    return qa.get("exe_ans") or qa.get("answer") or ""


def get_question(sample):
    return sample.get("qa", {}).get("question", "")


def get_gold_evidence_texts(sample):
    gold_inds = sample.get("qa", {}).get("gold_inds", {}) or {}
    return [normalize_answer(v) for v in gold_inds.values() if str(v).strip()]


def retrieval_hit(sample, retrieved):
    """Return 1 when Top-K contains any annotated gold evidence."""
    gold_texts = get_gold_evidence_texts(sample)
    if not gold_texts:
        return None
    retrieved_texts = [normalize_answer(x.get("content", "")) for x in retrieved]
    return int(any(
        gold in text or text in gold
        for gold in gold_texts for text in retrieved_texts if text
    ))


# =========================
# 2. 简单 BM25 检索
# =========================

def simple_tokenize(text):
    text = str(text).lower()
    return re.findall(r"[a-zA-Z0-9_.%-]+", text)


class SimpleBM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.doc_tokens = [simple_tokenize(d["content"]) for d in docs]
        self.doc_len = [len(x) for x in self.doc_tokens]
        self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)

        self.df = defaultdict(int)
        for tokens in self.doc_tokens:
            for t in set(tokens):
                self.df[t] += 1

        self.N = len(docs)
        self.idf = {}
        for t, df in self.df.items():
            self.idf[t] = math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def score(self, query):
        q_tokens = simple_tokenize(query)
        scores = []

        for idx, tokens in enumerate(self.doc_tokens):
            tf = Counter(tokens)
            dl = self.doc_len[idx]
            score = 0.0

            for t in q_tokens:
                if t not in tf:
                    continue
                idf = self.idf.get(t, 0.0)
                freq = tf[t]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-8))
                score += idf * freq * (self.k1 + 1) / denom

            scores.append(score)

        return scores

    def topk(self, query, k=5):
        scores = self.score(query)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in ranked[:k]:
            item = dict(self.docs[idx])
            item["score"] = score
            results.append(item)
        return results


# =========================
# 3. FinQA 样本转证据库
# =========================

def build_text_docs(sample):
    docs = []

    for i, text in enumerate(sample.get("pre_text", [])):
        if str(text).strip():
            docs.append({
                "type": "text",
                "content": str(text),
                "meta": {"part": "pre_text", "idx": i}
            })

    for i, text in enumerate(sample.get("post_text", [])):
        if str(text).strip():
            docs.append({
                "type": "text",
                "content": str(text),
                "meta": {"part": "post_text", "idx": i}
            })

    return docs


def build_table_docs_flat(sample):
    """
    Standard RAG 用：把表格行拉平成普通文本。
    """
    table = sample.get("table", [])
    docs = []

    if not table:
        return docs

    header = table[0]

    for r_idx, row in enumerate(table[1:], start=1):
        pairs = []
        for c_idx, cell in enumerate(row):
            col_name = header[c_idx] if c_idx < len(header) else f"col_{c_idx}"
            pairs.append(f"{col_name}: {cell}")

        content = " | ".join(pairs)

        docs.append({
            "type": "table_flat",
            "content": content,
            "meta": {"row_idx": r_idx}
        })

    return docs


def build_table_docs_structured(sample):
    """
    Ours 用：保留表头、行名、列名、单元格对应关系。
    """
    table = sample.get("table", [])
    docs = []

    if not table:
        return docs

    header = table[0]

    for r_idx, row in enumerate(table[1:], start=1):
        row_name = row[0] if row else ""

        cells = []
        for c_idx, cell in enumerate(row):
            col_name = header[c_idx] if c_idx < len(header) else f"col_{c_idx}"
            cells.append({
                "column": col_name,
                "value": cell
            })

        content = (
            f"row_indicator: {row_name}\n"
            f"columns_and_values: "
            + " | ".join([f"{x['column']} = {x['value']}" for x in cells])
        )

        docs.append({
            "type": "table_structured",
            "content": content,
            "meta": {
                "row_idx": r_idx,
                "row_indicator": row_name,
                "cells": cells
            }
        })

    return docs


def format_evidence(evidence_list, max_chars=3500):
    texts = []
    total = 0

    for i, e in enumerate(evidence_list, start=1):
        block = f"[Evidence {i} | {e['type']}]\n{e['content']}"
        if total + len(block) > max_chars:
            break
        texts.append(block)
        total += len(block)

    return "\n\n".join(texts)


# =========================
# 4. 大模型调用
# =========================

def call_llm(prompt, model=None):
    """
    默认使用 DeepSeek 的 OpenAI-compatible 接口。
    需要环境变量：
    OPENAI_API_KEY
    OPENAI_BASE_URL 可选，默认 https://api.deepseek.com
    OPENAI_MODEL 可选，默认 deepseek-chat
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("请先安装 openai：pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
    model = model or os.getenv("OPENAI_MODEL", "deepseek-chat")

    if not api_key:
        raise RuntimeError(
            "请先设置 OPENAI_API_KEY 环境变量，或在项目根目录创建 .env 并写入 OPENAI_API_KEY=你的DeepSeek_API_KEY"
        )

    global LAST_LLM_CALL_AT
    client = OpenAI(api_key=api_key, base_url=base_url if base_url else None)
    max_retries = int(os.getenv("LLM_MAX_RETRIES", "6"))
    timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "120"))
    min_interval = float(os.getenv("LLM_MIN_INTERVAL_SECONDS", "0"))
    last_error = None
    for attempt in range(max_retries):
        try:
            remaining = min_interval - (time.monotonic() - LAST_LLM_CALL_AT)
            if remaining > 0:
                time.sleep(remaining)
            LAST_LLM_CALL_AT = time.monotonic()
            request_args = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a financial question answering assistant. Answer accurately and concisely."},
                    {"role": "user", "content": prompt}
                ],
                "timeout": timeout,
            }
            if os.getenv("LLM_OMIT_TEMPERATURE", "0") != "1":
                request_args["temperature"] = 0
            max_tokens = int(os.getenv("LLM_MAX_COMPLETION_TOKENS", "0"))
            if max_tokens > 0:
                request_args["max_completion_tokens"] = max_tokens
            resp = client.chat.completions.create(**request_args)
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            last_error = exc
            if attempt + 1 >= max_retries:
                break
            delay = max(min_interval, min(60.0, 2 ** attempt)) + random.random()
            status = getattr(exc, "status_code", "unknown")
            print(f"API call failed ({attempt + 1}/{max_retries}, status={status}); retry in {delay:.1f}s", file=sys.stderr)
            time.sleep(delay)
    status = getattr(last_error, "status_code", "unknown")
    raise RuntimeError(
        f"API call failed after {max_retries} attempts (status={status}): {last_error}"
    )


def answer_with_llm(question, evidence=None, model=None):
    if evidence:
        prompt = f"""
You need to answer the financial question based only on the evidence.

Question:
{question}

Evidence:
{evidence}

Return only the final answer. Do not add unnecessary explanation.
"""
    else:
        prompt = f"""
Answer the financial question directly.

Question:
{question}

Return only the final answer. Do not add unnecessary explanation.
"""

    return call_llm(prompt, model=model)


def generate_python_code(question, evidence, model=None):
    prompt = f"""
You are given a financial question and retrieved evidence.

Task:
Generate Python code to compute the answer.

Rules:
1. Use only numbers from the evidence.
2. Assign the final result to a variable named answer.
3. Do not print anything.
4. Do not import external packages.
5. Return only Python code, no markdown, no explanation.

Question:
{question}

Evidence:
{evidence}
"""
    code = call_llm(prompt, model=model)
    code = code.replace("```python", "").replace("```", "").strip()
    return code


def safe_execute_python(code):
    allowed_builtins = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "float": float,
        "int": int,
        "str": str,
    }

    env = {
        "__builtins__": allowed_builtins,
        "math": math,
    }

    local_vars = {}

    try:
        exec(code, env, local_vars)
        return {
            "success": True,
            "answer": local_vars.get("answer", None),
            "error": ""
        }
    except Exception as e:
        return {
            "success": False,
            "answer": None,
            "error": str(e)
        }


# =========================
# 5. 问题类型判断
# =========================

def is_calculation_question(question):
    q = question.lower()
    keywords = [
        "increase", "decrease", "growth", "grew", "decline",
        "percentage", "percent", "ratio", "rate",
        "difference", "change", "compared", "from", "to",
        "average", "sum", "total", "how much more", "how much less",
        "增长率", "增长了多少", "增加了多少", "下降了多少", "减少了多少",
        "占比", "比例", "同比", "环比", "差额", "变化幅度", "相差",
        "平均", "合计", "总和", "毛利率", "费用率", "资产负债率"
    ]
    return any(k in q for k in keywords)


# =========================
# 6. 六种实验方法
# =========================

def run_llm_only(sample, args):
    question = get_question(sample)
    pred = answer_with_llm(question, evidence=None, model=args.model)

    return {
        "pred": pred,
        "code_success": None,
        "retrieved": []
    }


def run_standard_rag(sample, args, use_python=False):
    question = get_question(sample)

    docs = build_text_docs(sample) + build_table_docs_flat(sample)
    bm25 = SimpleBM25(docs)
    retrieved = bm25.topk(question, k=args.topk)
    evidence = format_evidence(retrieved)

    code_success = None

    if use_python and is_calculation_question(question):
        code = generate_python_code(question, evidence, model=args.model)
        result = safe_execute_python(code)
        code_success = result["success"]

        if result["success"] and result["answer"] is not None:
            pred = str(result["answer"])
        else:
            pred = answer_with_llm(question, evidence=evidence, model=args.model)
    else:
        pred = answer_with_llm(question, evidence=evidence, model=args.model)

    return {
        "pred": pred,
        "code_success": code_success,
        "retrieved": retrieved
    }


def run_text_only(sample, args):
    question = get_question(sample)

    docs = build_text_docs(sample)
    if not docs:
        return {"pred": "", "code_success": None, "retrieved": []}

    bm25 = SimpleBM25(docs)
    retrieved = bm25.topk(question, k=args.topk)
    evidence = format_evidence(retrieved)

    pred = answer_with_llm(question, evidence=evidence, model=args.model)

    return {
        "pred": pred,
        "code_success": None,
        "retrieved": retrieved
    }


def run_table_only(sample, args):
    question = get_question(sample)

    docs = build_table_docs_structured(sample)
    if not docs:
        return {"pred": "", "code_success": None, "retrieved": []}

    bm25 = SimpleBM25(docs)
    retrieved = bm25.topk(question, k=args.topk)
    evidence = format_evidence(retrieved)

    pred = answer_with_llm(question, evidence=evidence, model=args.model)

    return {
        "pred": pred,
        "code_success": None,
        "retrieved": retrieved
    }


def run_ours(sample, args):
    question = get_question(sample)

    text_docs = build_text_docs(sample)
    table_docs = build_table_docs_structured(sample)

    retrieved = []

    if text_docs:
        text_bm25 = SimpleBM25(text_docs)
        text_ret = text_bm25.topk(question, k=args.text_topk)
        retrieved.extend(text_ret)

    if table_docs:
        table_bm25 = SimpleBM25(table_docs)
        table_ret = table_bm25.topk(question, k=args.table_topk)
        retrieved.extend(table_ret)

    # 简单融合排序：按 BM25 分数排序
    retrieved = sorted(retrieved, key=lambda x: x.get("score", 0), reverse=True)
    retrieved = retrieved[:args.topk]

    evidence = format_evidence(retrieved)

    code_success = None

    if is_calculation_question(question):
        code = generate_python_code(question, evidence, model=args.model)
        result = safe_execute_python(code)
        code_success = result["success"]

        if result["success"] and result["answer"] is not None:
            pred = str(result["answer"])
        else:
            pred = answer_with_llm(question, evidence=evidence, model=args.model)
    else:
        pred = answer_with_llm(question, evidence=evidence, model=args.model)

    return {
        "pred": pred,
        "code_success": code_success,
        "retrieved": retrieved
    }


def retrieve_ours_evidence(sample, args, use_fusion=True):
    question = get_question(sample)

    text_docs = build_text_docs(sample)
    table_docs = build_table_docs_structured(sample)

    text_ret = []
    table_ret = []

    if text_docs:
        text_bm25 = SimpleBM25(text_docs)
        text_ret = text_bm25.topk(question, k=args.text_topk)

    if table_docs:
        table_bm25 = SimpleBM25(table_docs)
        table_ret = table_bm25.topk(question, k=args.table_topk)

    retrieved = text_ret + table_ret

    if use_fusion:
        retrieved = sorted(retrieved, key=lambda x: x.get("score", 0), reverse=True)

    return retrieved[:args.topk]


def run_ours_no_fusion(sample, args):
    question = get_question(sample)
    retrieved = retrieve_ours_evidence(sample, args, use_fusion=False)
    evidence = format_evidence(retrieved)

    code_success = None

    if is_calculation_question(question):
        code = generate_python_code(question, evidence, model=args.model)
        result = safe_execute_python(code)
        code_success = result["success"]

        if result["success"] and result["answer"] is not None:
            pred = str(result["answer"])
        else:
            pred = answer_with_llm(question, evidence=evidence, model=args.model)
    else:
        pred = answer_with_llm(question, evidence=evidence, model=args.model)

    return {
        "pred": pred,
        "code_success": code_success,
        "retrieved": retrieved
    }


def run_ours_no_python(sample, args):
    question = get_question(sample)
    retrieved = retrieve_ours_evidence(sample, args, use_fusion=True)
    evidence = format_evidence(retrieved)

    pred = answer_with_llm(question, evidence=evidence, model=args.model)

    return {
        "pred": pred,
        "code_success": None,
        "retrieved": retrieved
    }


def run_ours_no_routing(sample, args):
    question = get_question(sample)
    retrieved = retrieve_ours_evidence(sample, args, use_fusion=True)
    evidence = format_evidence(retrieved)

    code = generate_python_code(question, evidence, model=args.model)
    result = safe_execute_python(code)
    code_success = result["success"]

    if result["success"] and result["answer"] is not None:
        pred = str(result["answer"])
    else:
        pred = answer_with_llm(question, evidence=evidence, model=args.model)

    return {
        "pred": pred,
        "code_success": code_success,
        "retrieved": retrieved
    }


# =========================
# 7. 主实验入口
# =========================

def run_one(sample, args):
    if args.method == "llm_only":
        return run_llm_only(sample, args)

    if args.method == "standard_rag":
        return run_standard_rag(sample, args, use_python=False)

    if args.method == "text_only":
        return run_text_only(sample, args)

    if args.method == "table_only":
        return run_table_only(sample, args)

    if args.method == "standard_rag_python":
        return run_standard_rag(sample, args, use_python=True)

    if args.method == "ours":
        return run_ours(sample, args)

    if args.method == "ours_no_fusion":
        return run_ours_no_fusion(sample, args)

    if args.method == "ours_no_python":
        return run_ours_no_python(sample, args)

    if args.method == "ours_no_routing":
        return run_ours_no_routing(sample, args)

    raise ValueError(f"Unknown method: {args.method}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--method", type=str, required=True,
                        choices=[
                            "llm_only",
                            "standard_rag",
                            "text_only",
                            "table_only",
                            "standard_rag_python",
                            "ours",
                            "ours_no_fusion",
                            "ours_no_python",
                            "ours_no_routing"
                        ])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--text_topk", type=int, default=3)
    parser.add_argument("--table_topk", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    try:
        data = load_json(args.data_path)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        parser.error(str(e))

    if not isinstance(data, list):
        parser.error(f"数据文件格式不正确：期望 JSON list，实际是 {type(data).__name__}")

    if args.max_samples > 0:
        data = data[:args.max_samples]

    out_csv = Path(args.output_dir) / f"{args.method}_predictions.csv"
    out_json = Path(args.output_dir) / f"{args.method}_summary.json"
    records = []
    completed_ids = set()
    if args.resume and out_csv.exists():
        with open(out_csv, "r", encoding="utf-8-sig", newline="") as f:
            records = list(csv.DictReader(f))
        completed_ids = {str(r.get("id", "")) for r in records if str(r.get("id", ""))}
    em_list = []
    f1_list = []
    code_success_list = []

    for sample in tqdm(data, desc=f"Running {args.method}"):
        sample_id = str(sample.get("id", ""))
        if sample_id and sample_id in completed_ids:
            continue
        question = get_question(sample)
        gold = get_gold_answer(sample)

        try:
            result = run_one(sample, args)
            pred = result["pred"]
            code_success = result["code_success"]
            error = ""
        except Exception as e:
            pred = ""
            code_success = None
            result = {"retrieved": [], "error": str(e)}
            error = str(e)

        em = exact_match(pred, gold)
        f1 = token_f1(pred, gold)
        numeric_correct = em if extract_number(gold) is not None else None
        hit = retrieval_hit(sample, result.get("retrieved", []))

        em_list.append(em)
        f1_list.append(f1)

        if code_success is not None:
            code_success_list.append(1 if code_success else 0)

        records.append({
            "id": sample.get("id", ""),
            "question": question,
            "gold": gold,
            "pred": pred,
            "em": em,
            "f1": f1,
            "code_success": code_success,
            "numeric_correct": numeric_correct,
            "retrieval_hit": hit,
            "model": args.model or os.getenv("OPENAI_MODEL", "deepseek-chat"),
            "retrieved_json": json.dumps(result.get("retrieved", []), ensure_ascii=False),
            "method": args.method,
            "error": error
        })

        # Paid runs are checkpointed after every sample for safe resume.
        write_csv(out_csv, records)

    em_values = [float(r["em"]) for r in records if str(r.get("em", "")) != ""]
    f1_values = [float(r["f1"]) for r in records if str(r.get("f1", "")) != ""]
    code_values = [1.0 if str(r.get("code_success", "")).lower() == "true" else 0.0
                   for r in records if str(r.get("code_success", "")).lower() in {"true", "false"}]
    hit_values = [float(r["retrieval_hit"]) for r in records
                  if str(r.get("retrieval_hit", "")) not in {"", "None"}]
    numeric_values = [float(r["numeric_correct"]) for r in records
                      if str(r.get("numeric_correct", "")) not in {"", "None"}]

    summary = {
        "method": args.method,
        "num_samples": len(records),
        "model": args.model or os.getenv("OPENAI_MODEL", "deepseek-chat"),
        "EM": sum(em_values) / max(len(em_values), 1),
        "F1": sum(f1_values) / max(len(f1_values), 1),
        "code_success_rate": (
            sum(code_values) / len(code_values) if code_values else None
        ),
        "retrieval_hit_rate": sum(hit_values) / len(hit_values) if hit_values else None,
        "numeric_accuracy": sum(numeric_values) / len(numeric_values) if numeric_values else None,
        "failed_samples": sum(1 for r in records if str(r.get("error", "")).strip()),
    }

    write_csv(out_csv, records)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved predictions to: {out_csv}")
    print(f"Saved summary to: {out_json}")


if __name__ == "__main__":
    main()

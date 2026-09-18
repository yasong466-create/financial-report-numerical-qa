#!/usr/bin/env python3
"""Audit and prepare the public Zhu et al. (2026) CFQA release."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/public/cfqa_zhu2026/datas/test_advanced_500.json"
OUT_DIR = ROOT / "data/cfqa_zhu2026"
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = json.loads(SOURCE.read_text(encoding="utf-8"))
gold = [x for x in rows if str(x.get("answer", "")).strip()]
blank = [x for x in rows if not str(x.get("answer", "")).strip()]

(OUT_DIR / "all_500_questions.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
)
(OUT_DIR / "gold_answer_subset.json").write_text(
    json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8"
)
(OUT_DIR / "blank_answer_subset.json").write_text(
    json.dumps(blank, ensure_ascii=False, indent=2), encoding="utf-8"
)

type_all = Counter(x.get("type", "未标注") for x in rows)
type_gold = Counter(x.get("type", "未标注") for x in gold)
files_all = {x.get("filename", "") for x in rows if x.get("filename")}
files_gold = {x.get("filename", "") for x in gold if x.get("filename")}

audit = {
    "dataset": "CFQA: A Chinese Financial Question Answering Benchmark From Corporate Annual Reports",
    "authors": "Zhu, Liu, and Kurfali (2026)",
    "license": "Apache-2.0 for the GitHub project; Hugging Face dataset card states CC-BY-4.0",
    "source_file": str(SOURCE.relative_to(ROOT)),
    "total_questions": len(rows),
    "nonempty_gold_answers": len(gold),
    "blank_answers": len(blank),
    "all_report_filenames": len(files_all),
    "gold_subset_report_filenames": len(files_gold),
    "all_type_distribution": dict(type_all),
    "gold_type_distribution": dict(type_gold),
    "rag_ready": False,
    "rag_blocker": "The public repository does not bundle annual-report PDFs or extracted page chunks.",
    "safe_usage": {
        "all_500": "question-type and coverage statistics only",
        "gold_subset": "answer evaluation after the corresponding public annual-report PDFs are supplied",
        "blank_subset": "must not be used for EM/F1 or described as human-annotated gold answers"
    }
}
(OUT_DIR / "dataset_audit.json").write_text(
    json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
)

print(json.dumps(audit, ensure_ascii=False, indent=2))

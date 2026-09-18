#!/usr/bin/env python3
"""Extract page-level text from downloaded CFQA annual-report PDFs."""

import json
from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError as exc:
    raise SystemExit("缺少 pypdf。请使用 VS Code 任务运行，或执行：python3 -m pip install pypdf") from exc


ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "data/cfqa_zhu2026/reports"
OUT_FILE = ROOT / "data/cfqa_zhu2026/page_corpus.json"
STATS_FILE = ROOT / "data/cfqa_zhu2026/corpus_stats.json"


def main():
    chunks = []
    files = []
    failures = []
    for pdf_path in sorted(REPORT_DIR.glob("*.pdf")):
        try:
            reader = PdfReader(pdf_path)
            text_pages = 0
            for page_index, page in enumerate(reader.pages):
                content = (page.extract_text() or "").strip()
                if not content:
                    continue
                text_pages += 1
                chunks.append(
                    {
                        "id": f"{pdf_path.stem}_page_{page_index}",
                        "type": "annual_report_page",
                        "content": content,
                        "meta": {
                            "filename": pdf_path.name,
                            "page_index": page_index,
                            "pdf_page_number": page_index + 1,
                        },
                    }
                )
            files.append(
                {
                    "filename": pdf_path.name,
                    "pdf_pages": len(reader.pages),
                    "text_pages": text_pages,
                    "bytes": pdf_path.stat().st_size,
                }
            )
            print(f"OK {pdf_path.name}: {text_pages}/{len(reader.pages)} text pages")
        except Exception as exc:
            failures.append({"filename": pdf_path.name, "error": str(exc)})
            print(f"FAIL {pdf_path.name}: {exc}")

    OUT_FILE.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    questions = json.loads((ROOT / "data/cfqa_zhu2026/gold_answer_subset.json").read_text(encoding="utf-8"))
    available = {row["filename"] for row in files}
    covered = [row for row in questions if row["filename"] in available]
    stats = {
        "downloaded_reports": len(files),
        "page_chunks": len(chunks),
        "gold_questions_total": len(questions),
        "gold_questions_covered": len(covered),
        "gold_questions_not_covered": len(questions) - len(covered),
        "files": files,
        "failures": failures,
    }
    STATS_FILE.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k not in {"files", "failures"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

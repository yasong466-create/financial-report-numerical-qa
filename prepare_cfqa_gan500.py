#!/usr/bin/env python3
"""Prepare a deterministic 500-item subset of the official Gan et al. CFQA test set."""

import json
import re
from collections import Counter, defaultdict, deque
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/public/cfqa_gan2025/dataset/split_by_company/split_by_company_test.json"
OUT_DIR = ROOT / "data/cfqa_gan500"
TARGET = 500
MAX_PER_COMPANY = 25
TARGET_COMPANIES = 20


def extract_year(question):
    years = re.findall(r"(20\d{2})年", str(question))
    return int(years[0]) if years else None


def main():
    rows = json.loads(SOURCE.read_text(encoding="utf-8"))
    eligible = []
    for row in rows:
        if not str(row.get("答案", "")).strip():
            continue
        year = extract_year(row.get("问题", ""))
        if year is None:
            continue
        item = dict(row)
        item["报告年份"] = year
        item["样本唯一标识"] = f"{row['股票代码']}::{year}::{row['id']}"
        eligible.append(item)

    by_company_year = defaultdict(lambda: defaultdict(list))
    for row in eligible:
        key = (str(row["股票代码"]), str(row["公司"]))
        by_company_year[key][row["报告年份"]].append(row)

    ranked_companies = sorted(
        by_company_year,
        key=lambda company: (-sum(len(items) for items in by_company_year[company].values()), company),
    )[:TARGET_COMPANIES]
    company_queues = {}
    for company in sorted(ranked_companies):
        yearly = by_company_year[company]
        queue = []
        year_queues = {year: deque(sorted(items, key=lambda x: int(x["id"]))) for year, items in yearly.items()}
        while any(year_queues.values()) and len(queue) < MAX_PER_COMPANY:
            for year in sorted(year_queues):
                if year_queues[year] and len(queue) < MAX_PER_COMPANY:
                    queue.append(year_queues[year].popleft())
        if queue:
            company_queues[company] = deque(queue)

    selected = []
    while len(selected) < TARGET and any(company_queues.values()):
        for company in sorted(company_queues):
            if company_queues[company] and len(selected) < TARGET:
                selected.append(company_queues[company].popleft())

    if len(selected) != TARGET:
        raise RuntimeError(f"Expected {TARGET} selected samples, found {len(selected)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "questions_500.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_pairs = sorted({(str(x["股票代码"]), str(x["公司"]), x["报告年份"]) for x in selected})
    manifest = [
        {"股票代码": code, "公司": company, "报告年份": year, "status": "missing"}
        for code, company, year in report_pairs
    ]
    (OUT_DIR / "report_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    audit = {
        "source": str(SOURCE.relative_to(ROOT)),
        "selected_samples": len(selected),
        "companies": len({(x["股票代码"], x["公司"]) for x in selected}),
        "annual_reports_required": len(report_pairs),
        "year_distribution": dict(sorted(Counter(x["报告年份"] for x in selected).items())),
        "max_samples_per_company": MAX_PER_COMPANY,
        "target_companies": TARGET_COMPANIES,
        "selection": "top-coverage companies, company-round-robin, year-balanced, deterministic by official sample id",
    }
    (OUT_DIR / "selection_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

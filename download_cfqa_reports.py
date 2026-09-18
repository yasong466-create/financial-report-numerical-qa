#!/usr/bin/env python3
"""Download the 23 CFQA gold-subset annual reports from the official SSE site."""

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from urllib.parse import urlencode


ROOT = Path(__file__).resolve().parent
QUESTIONS = ROOT / "data/cfqa_zhu2026/gold_answer_subset.json"
OUT_DIR = ROOT / "data/cfqa_zhu2026/reports"
MANIFEST = ROOT / "data/cfqa_zhu2026/report_manifest.json"
SSE_QUERY = "https://query.sse.com.cn/security/stock/queryCompanyBulletin.do"
SSE_HOST = "https://static.sse.com.cn"
CNINFO_OVERRIDES = {
    ("601319", 2022): "https://static.cninfo.com.cn/finalpage/2023-03-25/1216221031.PDF",
    ("601088", 2023): "https://static.cninfo.com.cn/finalpage/2024-03-23/1219390021.PDF",
    ("600030", 2023): "https://static.cninfo.com.cn/finalpage/2024-03-27/1219411922.PDF",
    ("600970", 2023): "https://static.cninfo.com.cn/finalpage/2024-03-27/1219411131.PDF",
    ("601398", 2023): "https://static.cninfo.com.cn/finalpage/2024-03-28/1219429144.PDF",
    ("601888", 2023): "https://static.cninfo.com.cn/finalpage/2024-03-28/1219426653.PDF",
    ("600916", 2023): "https://static.cninfo.com.cn/finalpage/2024-04-29/1219890590.PDF",
    ("601088", 2024): "https://static.cninfo.com.cn/finalpage/2025-03-22/1222870380.PDF",
    ("601398", 2024): "https://static.cninfo.com.cn/finalpage/2025-03-29/1222948914.PDF",
    ("601126", 2024): "https://static.cninfo.com.cn/finalpage/2025-03-31/1222962147.PDF",
    ("600916", 2024): "https://static.cninfo.com.cn/finalpage/2025-04-30/1223421224.PDF",
}
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Referer": "https://www.sse.com.cn/",
}


def get_json(url, params):
    completed = subprocess.run(
        [
            "curl", "--fail", "--silent", "--show-error", "--location",
            "--max-time", "60", "-H", f"User-Agent: {HEADERS['User-Agent']}",
            "-H", f"Referer: {HEADERS['Referer']}", f"{url}?{urlencode(params)}",
        ],
        check=True,
        capture_output=True,
    )
    return json.loads(completed.stdout.decode("utf-8"))


def download(url, destination):
    temp = destination.with_suffix(".pdf.part")
    subprocess.run(
        [
            "curl", "--fail", "--silent", "--show-error", "--location",
            "--max-time", "300", "--retry", "2",
            "-H", f"User-Agent: {HEADERS['User-Agent']}",
            "-H", f"Referer: {HEADERS['Referer']}",
            "--output", str(temp), url,
        ],
        check=True,
    )
    if temp.stat().st_size < 10_000 or temp.read_bytes()[:4] != b"%PDF":
        temp.unlink(missing_ok=True)
        raise ValueError("downloaded content is not a valid PDF")
    temp.replace(destination)


def parse_target(filename):
    code_match = re.search(r"-(6\d{5})\.SH-", filename)
    year_match = re.search(r"(20\d{2})(?:年年度|年度)报告", filename)
    if not code_match or not year_match:
        raise ValueError(f"cannot parse stock code/report year: {filename}")
    return code_match.group(1), int(year_match.group(1))


def query_reports(stock_code):
    payload = get_json(
        SSE_QUERY,
        {
            "isPagination": "true",
            "productId": stock_code,
            "keyWord": "",
            "securityType": "0101,120100,020100,020200,120200",
            "reportType2": "DQBG",
            "reportType": "YEARLY",
            "pageHelp.pageSize": "50",
            "pageHelp.pageNo": "1",
            "pageHelp.beginPage": "1",
            "pageHelp.cacheSize": "1",
        },
    )
    return payload.get("pageHelp", {}).get("data", [])


def choose_report(rows, report_year):
    targets = (f"{report_year}年年度报告", f"{report_year}年度报告")
    candidates = []
    for row in rows:
        title = str(row.get("TITLE", "")).replace(" ", "")
        bulletin_type = str(row.get("BULLETIN_TYPE", ""))
        if any(target in title for target in targets) and "摘要" not in title and bulletin_type != "年报摘要":
            candidates.append(row)
    if not candidates:
        return None
    candidates.sort(key=lambda row: (row.get("SSEDATE", ""), row.get("URL", "")))
    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="redownload existing PDFs")
    args = parser.parse_args()

    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    filenames = sorted({row["filename"] for row in questions})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    query_cache = {}
    manifest = []

    for index, filename in enumerate(filenames, 1):
        code, year = parse_target(filename)
        destination = OUT_DIR / filename
        record = {
            "filename": filename,
            "stock_code": code,
            "report_year": year,
            "local_path": str(destination.relative_to(ROOT)),
        }
        try:
            if code not in query_cache:
                query_cache[code] = query_reports(code)
                time.sleep(0.3)
            selected = choose_report(query_cache[code], year)
            override_url = CNINFO_OVERRIDES.get((code, year))
            if not selected and not override_url:
                raise LookupError("matching full annual report not found in SSE response")
            source_url = override_url or SSE_HOST + selected["URL"]
            record.update(
                source_url=source_url,
                sse_title=selected.get("TITLE") if selected else None,
                disclosure_date=selected.get("SSEDATE") if selected else filename[:10],
                download_source="CNINFO" if (code, year) in CNINFO_OVERRIDES else "SSE",
            )
            if args.force or not destination.exists():
                download(source_url, destination)
                status = "downloaded"
            else:
                status = "existing"
            record.update(status=status, bytes=destination.stat().st_size)
            print(f"[{index:02d}/{len(filenames)}] OK {code} {year}: {status}")
        except Exception as exc:
            record.update(status="failed", error=str(exc))
            print(f"[{index:02d}/{len(filenames)}] FAIL {code} {year}: {exc}")
        manifest.append(record)

    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = sum(row["status"] in {"downloaded", "existing"} for row in manifest)
    print(f"\nDownloaded/verified: {ok}/{len(manifest)}")
    print(f"Manifest: {MANIFEST}")
    raise SystemExit(0 if ok == len(manifest) else 1)


if __name__ == "__main__":
    main()

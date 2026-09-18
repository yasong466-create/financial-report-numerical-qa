#!/usr/bin/env python3
"""Build a fair 128-sample comparison shared by all four model providers."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs" / "cfqa_common_128"
SOURCES = {
    "deepseek": ROOT / "outputs/cfqa_public_129/deepseek",
    "qwen": ROOT / "outputs/cfqa_public_129/qwen",
    "kimi": ROOT / "outputs/cfqa_public_129/kimi",
    "glm": ROOT / "outputs/cfqa_glm4flash_129/glm",
}
EXCLUDED_QUESTION = "中国神华2023年度是否实施现金分红？如实施，分红金额和分红率各是多少？"


def read_rows(path):
    with path.open(encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(provider, model, rows):
    completed = [row for row in rows if not str(row.get("error", "")).strip()]
    calc = [row for row in completed if row["type"] == "比较计算"]
    code_rows = [row for row in calc if str(row["code_success"]) not in {"", "None"}]
    return {
        "provider": provider,
        "model": model,
        "samples": len(rows),
        "completed": len(completed),
        "EM": sum(float(row["em"]) for row in completed) / max(len(completed), 1),
        "F1": sum(float(row["f1"]) for row in completed) / max(len(completed), 1),
        "retrieval_hit_rate": sum(float(row["retrieval_hit"]) for row in completed) / max(len(completed), 1),
        "code_success_rate": sum(str(row["code_success"]) == "True" for row in code_rows) / max(len(code_rows), 1),
        "numeric_accuracy": sum(float(row["numeric_correct"] or 0) for row in calc) / max(len(calc), 1),
    }


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = []
    excluded_ids = set()

    for source in SOURCES.values():
        for row in read_rows(source / "predictions.csv"):
            if row["question"] == EXCLUDED_QUESTION:
                excluded_ids.add(row["id"])

    if len(excluded_ids) != 1:
        raise RuntimeError(f"Expected one shared excluded sample ID, found {len(excluded_ids)}")
    excluded_id = next(iter(excluded_ids))

    for provider, source in SOURCES.items():
        rows = [row for row in read_rows(source / "predictions.csv") if row["id"] != excluded_id]
        if len(rows) != 128:
            raise RuntimeError(f"{provider}: expected 128 rows after exclusion, found {len(rows)}")
        failed = [row for row in rows if str(row.get("error", "")).strip()]
        if failed:
            raise RuntimeError(f"{provider}: {len(failed)} failed rows remain in common subset")

        source_summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
        summary = summarize(provider, source_summary["model"], rows)
        provider_dir = OUTPUT_ROOT / provider
        write_rows(provider_dir / "predictions.csv", rows)
        (provider_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summaries.append(summary)

    with (OUTPUT_ROOT / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    manifest = {
        "comparison_samples": 128,
        "excluded_id": excluded_id,
        "excluded_question": EXCLUDED_QUESTION,
        "reason": "GLM-4-Flash API content filter rejected this sample (HTTP 400, code 1301).",
        "policy": "The same sample is excluded from every provider before recomputing metrics.",
    }
    (OUTPUT_ROOT / "exclusion_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

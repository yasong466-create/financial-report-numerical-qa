#!/usr/bin/env python3
"""Convert the existing Mixue Chinese case file to the unified experiment schema."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "outputs/chinese_mixue_validation/mixue_chinese_qa_samples.json"
TARGET = ROOT / "data/chinese_mixue.json"

raw = json.loads(SOURCE.read_text(encoding="utf-8"))
texts = {x["id"]: x for x in raw.get("texts", [])}
tables = {x["id"]: x for x in raw.get("tables", [])}
samples = []

for qa in raw.get("qa", []):
    table_obj = tables.get(qa.get("gold_table"))
    table = []
    gold_inds = {}
    if table_obj and table_obj.get("rows"):
        columns = []
        for row in table_obj["rows"]:
            for key in row:
                if key not in columns:
                    columns.append(key)
        table = [columns] + [[str(row.get(c, "")) for c in columns] for row in table_obj["rows"]]
        indicator = qa.get("gold_indicator")
        matching = [row for row in table_obj["rows"] if not indicator or row.get("指标") == indicator]
        for i, row in enumerate(matching):
            gold_inds[f"table_{i}"] = " | ".join(f"{k}: {v}" for k, v in row.items())
    for text_id in qa.get("gold_text") or []:
        if text_id in texts:
            gold_inds[text_id] = texts[text_id]["content"]
    samples.append({
        "id": qa["id"],
        "pre_text": [x["content"] for x in raw.get("texts", [])],
        "post_text": [],
        "table": table,
        "qa": {
            "question": qa["question"],
            "answer": qa["answer"],
            "exe_ans": qa["answer"],
            "gold_inds": gold_inds,
            "need_calc": qa.get("need_calc", False),
            "type": qa.get("type", ""),
        },
    })

TARGET.parent.mkdir(parents=True, exist_ok=True)
TARGET.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {len(samples)} samples to {TARGET}")

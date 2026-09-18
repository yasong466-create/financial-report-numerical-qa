#!/usr/bin/env python3
"""Generate a thesis-ready Markdown section from a multimodel summary CSV."""
import argparse, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def pct(value):
    if value in (None, "", "None"):
        return "—"
    return f"{float(value) * 100:.2f}"

p = argparse.ArgumentParser()
p.add_argument("--run_name", default="multimodel_formal_100")
a = p.parse_args()
folder = ROOT / "outputs" / a.run_name
rows = list(csv.DictReader(open(folder / "summary.csv", encoding="utf-8-sig")))
ok = [r for r in rows if r.get("status") == "ok"]
sample_count = max((int(r.get("samples") or 0) for r in ok), default=0)

lines = [
    "# 不同大语言模型底座下的稳健性分析",
    "",
    f"为降低单一大语言模型底座对实验结论的影响，本文选取 {len(ok)} 种国内大语言模型，在 FinQA 测试集的统一样本上开展模型底座稳健性实验。实验固定文本与表格预处理流程、BM25 检索方法、Top-K=5、提示词模板、任务路由规则、Python 符号计算流程和答案归一化规则，仅替换答案生成与代码生成阶段的大语言模型。各模型使用相同的前 {sample_count} 条测试样本，以保证实验结果具有可比性。",
    "",
    "| 模型底座 | 样本数 | EM/% | F1/% | Recall@5/% | 代码执行成功率/% | 数值计算正确率/% | 失败数 |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for r in ok:
    lines.append(
        f"| {r['model']} | {r.get('samples','')} | {pct(r.get('EM'))} | {pct(r.get('F1'))} | "
        f"{pct(r.get('retrieval_hit_rate'))} | {pct(r.get('code_success_rate'))} | "
        f"{pct(r.get('numeric_accuracy'))} | {r.get('failed_samples','')} |"
    )
lines += [
    "",
    "不同模型的 EM、F1 和数值计算正确率存在差异，表明底座模型的金融语义理解、公式选择和代码生成能力会影响最终问答效果。由于各组实验使用完全相同的检索流程，Recall@5 原则上应保持一致；若出现差异，应优先检查样本是否一致或是否存在失败记录。代码执行成功率仅表示程序能够运行，并不等同于公式选择和最终计算结果正确，因此需要结合数值计算正确率共同分析。",
    "",
    "本实验主要用于考察模型底座变化对本文方法的影响。若使用的是测试集子集，相关结果用于稳健性趋势分析，完整性能结论仍应以全测试集实验为准。",
    "",
    "## 盲审修改说明",
    "",
    "针对专家提出的‘对比模型类型较少、模型代表性不足’问题，本文新增不同大语言模型底座下的稳健性分析。在保持测试样本、检索流程、Top-K、提示词、符号计算流程及评价规则一致的前提下，仅替换生成阶段的模型底座，并从 EM、F1、证据命中率、代码执行成功率和数值计算正确率等维度进行比较，以更完整地分析模型底座差异对金融报告数值问答效果的影响。",
]
(folder / "论文实验说明.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(folder / "论文实验说明.md")

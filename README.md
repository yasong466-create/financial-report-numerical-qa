# Financial Report Numerical Question Answering

本仓库为硕士学位论文《面向上市公司信息披露的金融报告数值问答技术研究》的实验代码与结果归档。

研究面向金融报告中的文本—表格混合问答，主要包含：

- 文本证据与财务表格证据的差异化组织；
- 基于 BM25 的双通道证据检索；
- 计算型问题识别与 Python 符号执行；
- FinQA 主实验、500 条样本消融实验和多模型实验；
- FinanceComplexQA 中文公开数据补充实验。

## 仓库结构

```text
.
├── run_finqa_exp.py                 # FinQA 主实验入口
├── run_all_models.py                # DeepSeek、Qwen、GLM、Kimi 多模型实验
├── run_finance_complexqa_500.py     # FinanceComplexQA 500 条样本实验
├── run_ablation_500_queue.py        # 500 条样本消融实验
├── prepare_*.py / build_*.py        # 数据准备脚本
├── models.example.json              # 模型配置示例（不含密钥）
├── results/                          # 论文所用汇总结果
└── docs/                             # 实验说明
```

## 环境安装

建议使用 Python 3.10 或以上版本：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## API 密钥配置

复制环境变量模板并填写自己的密钥：

```bash
cp .env.example .env
```

`.env` 已被 Git 忽略，不应提交到公开仓库。模型名称和接口地址可参考 `models.example.json`。

## 数据集

本仓库不重复分发第三方数据集原文件。请根据各数据集的官方许可下载，并放置到对应目录：

```text
data/finqa/test.json
data/finance_complexqa/
```

- FinQA：<https://github.com/czyssrs/FinQA>
- FinanceComplexQA：请以论文中采用版本的官方数据页面及许可为准。

## 运行示例

FinQA 单模型实验：

```bash
python run_finqa_exp.py \
  --data_path data/finqa/test.json \
  --method ours \
  --max_samples 100 \
  --text_topk 3 \
  --table_topk 3 \
  --topk 5
```

多模型补充实验：

```bash
python run_all_models.py --max_samples 100 --method ours
```

FinanceComplexQA 补充实验：

```bash
python run_finance_complexqa_500.py \
  --max_samples 500 \
  --topk 5 \
  --methods standard_rag ours
```

## 主要实验参数

| 参数 | 设置 |
|---|---|
| 文本检索候选数 | 3 |
| 表格检索候选数 | 3 |
| 最终证据数 | 5 |
| 检索方法 | BM25 |
| 生成温度 | 0 |
| 数值误差阈值 | 1e-3 |

## 复现说明

不同模型服务可能随时间更新，接口限流、模型版本和输出随机性也可能造成结果波动。复现实验时应记录模型名称、运行日期、提示词、样本范围和评价脚本版本。

## 数据与安全说明

- 仓库不包含任何 API 密钥；
- 不上传第三方数据集的大体积或受许可限制原文件；
- 自建案例与实验输出仅用于学术研究，请核对原始披露材料后使用；
- 代码和实验结果不构成投资建议。

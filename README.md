# KnowMe Memory Gate model

> 面向 Agent 长期记忆检索的轻量级小模型后训练项目。

KnowMe Memory Gate model 基于 **Qwen3-1.7B**，通过 **LoRA SFT + GRPO** 训练一个本地小模型，用于完成两个任务：

1. **判断当前用户输入是否需要检索长期记忆**
2. **在需要检索时，生成适合 SQLite FTS5 / BM25 的检索 query**

目标是用一个轻量、本地可部署的小模型，替代每轮都调用远程 API 模型进行记忆检索判断的方案，同时保持稳定的检索效果。

模型输出固定格式 JSON：

```json
{
  "retrieve": true,
  "query": "会议 时间 偏好",
  "reason": "需要查询历史记忆"
}
```

其中：

- `retrieve`：是否需要检索长期记忆
- `query`：真正发送给检索器的查询词
- `reason`：供人阅读的简短解释

> **本仓库不会读取或复制真实用户记忆。**
> 训练与评测数据均来自公开学术数据集，检索环境使用与线上行为一致但完全隔离的 SQLite fixture。

---

## 项目特点

- **轻量本地模型**：基于 Qwen3-1.7B，适合本地部署
- **不仅做二分类**：同时学习“该不该检索”和“该检索什么”
- **完整后训练流程**：支持 LoRA SFT 与 GRPO
- **真实检索评测**：query 会真正送入 SQLite FTS5 / BM25，而不是只做字符串匹配
- **干扰增强检索评测**：加入 distractor memories，测试模型在更复杂记忆池中的鲁棒性
- **与 KnowMe 对齐**：训练和评测使用与 KnowMe 主项目一致的检索协议

---

## 结果

所有模型均在同一 held-out test set 上评测，并使用一致的输出协议与检索后端。完整实验指标说明见：[docs/metrics.md](docs/metrics.md)

### Standard Test

| Model                   | Macro F1 ↑ | Precision ↑ |   Recall ↑ | JSON Valid ↑ | Format Valid ↑ |
| ----------------------- | ---------: | ----------: | ---------: | -----------: | -------------: |
| DeepSeek V4.1 Flash     |     0.7746 |      0.7786 |     0.7676 |       1.0000 |         0.9824 |
| MiMo-V2.6-Flash         |     0.7676 |      0.7676 |     0.7676 |       0.9965 |         0.9648 |
| Qwen3-1.7B (Base)       |     0.4960 |      0.5263 |     0.8451 |       1.0000 |         0.8134 |
| Qwen3-1.7B + SFT        |     0.9577 |      0.9452 |     0.9718 |       1.0000 |         0.9930 |
| Qwen3-1.7B + SFT + GRPO | **0.9613** |  **0.9456** | **0.9789** |       1.0000 |     **1.0000** |


### Distractor-Augmented Retrieval

在候选记忆池中加入额外干扰记忆，用于评估 query 在更困难检索环境中的排序与命中能力。

| Model                   | Hit Rate ↑ |    Hit@1 ↑ |    Hit@3 ↑ |    Hit@5 ↑ |      MRR ↑ | Conditional MRR ↑ |
| ----------------------- | ---------: | ---------: | ---------: | ---------: | ---------: | ----------------: |
| DeepSeek V4.1 Flash     |     0.6690 |     0.4859 |     0.6690 |     0.6690 |     0.5669 |            0.7594 |
| MiMo-V2.6-Flash         |     0.6268 |     0.5141 |     0.6268 |     0.6268 |     0.5634 |            0.7692 |
| Qwen3-1.7B (Base)       |     0.4366 |     0.3380 |     0.4366 |     0.4366 |     0.3826 |            0.5906 |
| Qwen3-1.7B + SFT        |     0.8310 |     0.6620 |     0.8239 |     0.8310 |     0.7330 |            0.7597 |
| Qwen3-1.7B + SFT + GRPO | **0.8873** | **0.6761** | **0.8592** | **0.8873** | **0.7582** |        **0.7746** |

> Checkpoint 只在 dev set 上选择，test set 仅用于最终结果汇报。



---


## 整体流程

```text
用户输入
   │
   ▼
Qwen3-1.7B Memory Gate
   │
   ├───────────────┐
   │               │
retrieve=false   retrieve=true
   │               │
   │               ▼
   │          生成检索 query
   │               │
   │               ▼
   │      SQLite FTS5 / BM25
   │               │
   └─────────┬─────┘
             ▼
          Agent 后续处理
```

模型实际需要学习两个问题：

```text
1. 我现在需不需要查长期记忆？
2. 如果要查，我应该搜什么？
```

这也是本项目区别于普通二分类门控模型的地方。

---

## 项目结构

```text
.
├── configs/            # 模型输出格式 schema
├── data/               # 训练与评测数据
│   ├── train_split/                 # gate_{train,dev,test}.jsonl 数据划分
│   ├── gate_final_2855.full.jsonl   # 全量样本：prompt + 记忆池 + 证据标注
│   └── grpo_hard/                   # GRPO 用的干扰记忆池，见下方说明
├── docs/               # 详细文档
├── eval/               # 评测实现与历史结果
│   ├── run_gate_eval.py             # 全部指标的实现
│   └── results/                     # 每个 run 的 metrics.json（predictions 不入库）
├── scripts/            # 训练、评测、日志脚本
├── src/
│   ├── dataset/        # 数据构造工具链
│   └── retrieval/      # SQLite FTS5/BM25 检索 fixture
├── tests/              # 奖励函数与评测一致性测试
├── pyproject.toml      # 依赖声明与 extras（train / data / dev）
└── uv.lock
```

> `data/grpo_hard/` 体积较大且可完全离线重建，因此不随仓库发布。克隆后需要先跑一次：

```bash
uv run python -m src.dataset.build_grpo_hard_haystack --include-test
```

详细文档：

| 文档                         | 内容                                   |
| ---------------------------- | -------------------------------------- |
| `docs/data.md`               | 数据来源、schema、构造流程与数据划分   |
| `docs/training.md`           | 环境、SFT / GRPO 配置、checkpoint 管理 |
| `docs/metrics.md`            | 评测指标定义与读法                     |
| `docs/reward.md`             | GRPO 奖励函数设计                      |
| `docs/knowme_integration.md` | 与 KnowMe 主项目的分工与数据隔离       |

---


# 环境安装

推荐环境：

- Python ≥ 3.11
- 默认训练配置建议使用 ≥24 GB 显存
- Qwen3-1.7B
- ms-swift 4.5.3

克隆仓库并安装依赖：

```bash
git clone https://github.com/Longlong418/KnowMe-memory_gate_model_post_train.git
cd KnowMe-memory_gate_model_post_train

uv venv --python 3.12
uv pip install -e ".[train,data,dev]"
```

部分依赖版本已在 `pyproject.toml` 中固定。

详细环境注意建议阅读：[docs/training.md](docs/training.md)

---

# 下载预训练模型

模型权重不随仓库发布。

```bash
uv run hf download Qwen/Qwen3-1.7B \
    --local-dir model/Qwen3-1.7B
```

也可以手动指定：

```bash
export MODEL_PATH=/path/to/Qwen3-1.7B
```

训练脚本会自动搜索常见模型目录。

---



# 训练

## Stage 1 — SFT

首先使用 LoRA 对 Qwen3-1.7B 进行监督微调：

```bash
export SWANLAB_API_KEY='...'

bash scripts/train_sft_qwen3_1_7b.sh
```

默认训练约 3 个 epoch，并产出：

```text
best_gate_checkpoint
```

SFT 主要学习：

```text
用户输入
   │
   ▼
是否需要记忆？
 ┌─┴─┐
 │   │
否   是
 │   │
 │   ▼
 │ 生成 query
 │   │
 └─┬─┘
   ▼
JSON 输出
```

---

## Stage 2 — GRPO

GRPO 数据需要重新构建：

```bash
uv run python -m src.dataset.build_grpo_hard_haystack \
    --include-test
```

生成结果位于：

```text
data/grpo_hard/
```
GRPO 从 SFT checkpoint 继续训练：

```bash
bash scripts/train_grpo_qwen3_1_7b.sh
```

GRPO 的奖励不只看最终文本，而是直接结合真实检索行为，包括：

- 门控判断是否正确
- 输出格式是否合法
- query 是否有效
- 是否成功检索到目标记忆
- 目标记忆的排序质量
- 是否产生不必要检索



完整奖励设计见：[docs/reward.md](docs/reward.md)

训练参数与 batch 约束见：[docs/training.md](docs/training.md)

---

# 评测

## 本地模型

评测 dev：

```bash
bash scripts/eval_gate_checkpoint.sh <checkpoint> dev
```

最终评测 test：

```bash
bash scripts/eval_gate_checkpoint.sh <checkpoint> test
```

评测原始 Qwen3-1.7B：

```bash
bash scripts/eval_base_qwen3_1_7b.sh dev
```

评测结果默认写入：

```text
eval/results/<run>/
```

每个 run 会保存 predictions 与 machine-readable metrics。

---

## API 模型

远程 API 模型使用同一套 task protocol 与 evaluator。

冒烟测试：

```bash
uv run python scripts/eval_api_model.py \
    --split test \
    --limit 8
```

完整 test：

```bash
uv run python scripts/eval_api_model.py \
    --split test
```

---

# 与 KnowMe 主项目的关系

详细请看：[docs/knowme_integration.md](docs/knowme_integration.md)

# 部署到本地

coming soon

# 数据来源与数据构造

训练与评测数据共 **2,855 条**，来自四个公开 memory benchmark：

| 数据集        |      数量 | 记忆类型 |
| ------------- | --------: | -------- |
| LoCoMo        |       980 | Fact     |
| LongMemEval   |       996 | Episode  |
| PersonaMem-v2 |       631 | Fact     |
| RHELM         |       248 | Fact     |
| **总计**      | **2,855** |          |

数据划分：

| Split |  数量 |
| ----- | ----: |
| Train | 2,287 |
| Dev   |   284 |
| Test  |   284 |

数据按 **memory world** 分组划分，而不是随机拆分单条问题，以降低同一记忆上下文跨 train / dev / test 泄漏的风险。

完整的数据构造流程、schema 和已知数据问题见：

[docs/data.md](docs/data.md)

---


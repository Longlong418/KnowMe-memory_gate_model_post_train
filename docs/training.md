# 环境与训练

本文档介绍 SFT 和 GRPO 的环境配置、训练方式与常用参数。

---

## 1. 环境

推荐环境：

- Python ≥ 3.11（实测 3.12）
- 单卡 ≥ 24 GB 显存
- `ms-swift==4.5.3`

使用 `uv` 创建环境：

```bash
uv venv --python 3.12
uv pip install -e ".[train,data,dev]"
```

项目在 `pyproject.toml` 中固定了几个容易出问题的依赖：

| 包         | 版本     | 说明                                              |
| ---------- | -------- | ------------------------------------------------- |
| `ms-swift` | `4.5.3`  | SFT / GRPO 训练框架                               |
| `swanlab`  | `<0.8`   | 新版本与当前 transformers 回调存在兼容问题        |
| `msgspec`  | `>=0.18` | ms-swift 训练器运行时需要，但其依赖声明中没有包含 |

实测环境：

```text
torch          2.10.0+cu128
transformers   5.16.1
trl            0.29.1
peft           0.20.0
accelerate     1.15.0
swanlab        0.7.20
msgspec        0.22.0
```

安装后可以简单检查：

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
uv run python -c "import msgspec, swift.rlhf_trainers.utils; print('ok')"
uv run python -c "import swanlab; print(swanlab.__version__)"
```

### 常见环境问题

如果训练启动时报：

```text
ModuleNotFoundError: msgspec
```

补装：

```bash
uv pip install msgspec
```

如果出现：

```text
RuntimeError: operator torchvision::nms does not exist
```

说明当前 `torchvision` 和 `torch` 不匹配。本项目是纯文本训练，不需要 torchvision，可以直接：

```bash
uv pip uninstall torchvision
```

另外请保持：

```text
swanlab < 0.8
```

当前 ms-swift / transformers 的 SwanLab 回调与 0.8+ 存在兼容问题。

---

## 2. 模型

基础模型使用：

```text
Qwen3-1.7B
```

脚本会按下面顺序自动查找：

```text
$REPO_DIR/model/Qwen3-1.7B
/root/autodl-tmp/Qwen3-1.7B
$HOME/models/Qwen3-1.7B
$DATA_ROOT/Qwen3-1.7B
```

也可以手动指定：

```bash
export MODEL_PATH=/path/to/Qwen3-1.7B
```

---

## 3. 路径

所有训练脚本都会加载：

```text
scripts/_paths.sh
```

它负责统一解析模型、数据和运行目录。

常用变量：

| 变量         | 默认值                             |
| ------------ | ---------------------------------- |
| `REPO_DIR`   | 当前仓库根目录                     |
| `PYTHON`     | `$REPO_DIR/.venv/bin/python`       |
| `SWIFT_BIN`  | `$REPO_DIR/.venv/bin/swift`        |
| `DATA_ROOT`  | 优先使用 `/root/autodl-tmp/<repo>` |
| `RUNS_ROOT`  | `$DATA_ROOT/runs`                  |
| `MODEL_PATH` | 按上一节的顺序自动查找             |

一般情况下直接运行训练脚本即可，不需要手动配置路径：

```bash
bash scripts/train_sft_qwen3_1_7b.sh
```

如果 `/root/autodl-tmp` 可写，checkpoint 会自动保存到数据盘，避免占用系统盘。

---

## 4. SFT

启动训练：

```bash
export SWANLAB_API_KEY='...'

bash scripts/train_sft_qwen3_1_7b.sh
```

默认配置：

| 参数                   | 值                          |
| ---------------------- | --------------------------- |
| Train                  | `gate_train.jsonl`，2287 条 |
| Dev                    | `gate_dev.jsonl`，284 条    |
| LoRA rank              | 16                          |
| LoRA alpha             | 32                          |
| LoRA dropout           | 0.05                        |
| Target modules         | `all-linear`                |
| Epoch                  | 3                           |
| Learning rate          | `1e-4`                      |
| Warmup ratio           | `0.05`                      |
| Train batch size       | 16                          |
| Eval batch size        | 8                           |
| Gradient accumulation  | 1                           |
| Max length             | 1024                        |
| Precision              | bfloat16                    |
| Gradient checkpointing | 开启                        |

### 最优 checkpoint

每个 epoch 保存 checkpoint 后，会自动在 dev set 上进行完整评测。

默认使用：

```text
retrieval.branch.conditional_mrr
```

选择最佳 checkpoint。

最终结果会保存为：

```text
<run>/best_gate_checkpoint
```

训练结束后还会自动对最佳 checkpoint 跑一次 test。

如果需要修改选择指标：

```bash
GATE_BEST_METRIC=end_to_end.memory_recall \
bash scripts/train_sft_qwen3_1_7b.sh
```

调整 batch：

```bash
TRAIN_BATCH_SIZE=4 \
EVAL_BATCH_SIZE=2 \
GRAD_ACCUM_STEPS=4 \
bash scripts/train_sft_qwen3_1_7b.sh
```

> checkpoint 只使用 dev set 选择，test set 不参与模型选择。

---

## 5. GRPO

启动：

```bash
export SWANLAB_API_KEY='...'

bash scripts/train_grpo_qwen3_1_7b.sh
```

GRPO 默认从最新的：

```text
best_gate_checkpoint
```

继续训练。

也可以手动指定：

```bash
SFT_ADAPTER=/path/to/best_gate_checkpoint \
bash scripts/train_grpo_qwen3_1_7b.sh
```

训练时：

```text
--adapters
--ref_adapters
```

都指向 SFT checkpoint，因此 KL reference 是 SFT 模型。

默认配置：

| 参数                    | 值                               |
| ----------------------- | -------------------------------- |
| Train                   | `gate_grpo_train.jsonl`，2287 条 |
| Dev                     | `gate_grpo_dev.jsonl`，284 条    |
| Reward                  | `gate_reward_v2`                 |
| `NUM_GENERATIONS`       | 16                               |
| `TRAIN_BATCH_SIZE`      | 16                               |
| `EVAL_BATCH_SIZE`       | 16                               |
| `GENERATION_BATCH_SIZE` | 16                               |
| Epoch                   | 3                                |
| Learning rate           | `5e-6`                           |
| Temperature             | 0.7                              |
| Top-p                   | 0.9                              |
| Max completion length   | 128                              |
| Thinking                | off                              |
| vLLM                    | off                              |

奖励函数见：

```text
docs/reward.md
```

---

### 5.1 GRPO 的 batch 约束

这里有一个很重要的规则：

```text
TRAIN_BATCH_SIZE % NUM_GENERATIONS == 0
```

同时建议保证：

```text
GENERATION_BATCH_SIZE % NUM_GENERATIONS == 0
EVAL_BATCH_SIZE       % NUM_GENERATIONS == 0
```

原因很简单：

> GRPO 的优势是在同一个 prompt 的多个 generation 之间比较和归一化的，因此一个训练 batch 最好包含完整的 GRPO group。

例如：

```text
NUM_GENERATIONS = 16
TRAIN_BATCH_SIZE = 16
```

表示：

```text
一个 prompt
   ↓
生成 16 条 completion
   ↓
16 条作为一个完整 GRPO group
   ↓
完成一次 optimizer update
```

不要使用类似：

```text
TRAIN_BATCH_SIZE=4
NUM_GENERATIONS=8
```

这种配置。

它会把一个 GRPO group 拆到多个 optimizer step 中，训练行为会变得不符合预期。

训练脚本已经加入检查，不满足条件时会直接退出。

---

### 5.2 Checkpoint

GRPO checkpoint 会按照 transformers Trainer 的：

```text
save_total_limit
```

自动轮转删除旧 checkpoint。

默认：

```text
SAVE_TOTAL_LIMIT=12
```

可以修改：

```bash
SAVE_TOTAL_LIMIT=20 \
bash scripts/train_grpo_qwen3_1_7b.sh
```

如果希望全部保留：

```bash
SAVE_TOTAL_LIMIT= \
bash scripts/train_grpo_qwen3_1_7b.sh
```

如果要选择最佳 GRPO checkpoint，建议训练结束后分别在 dev 上评测：

```bash
bash scripts/eval_gate_checkpoint.sh <checkpoint> dev
```

然后根据 dev 指标选择。

**不要使用 test set 选择 checkpoint。**

---

## 6. SwanLab

SFT 和 GRPO 都使用 SwanLab 记录训练过程。

运行前需要：

```bash
export SWANLAB_API_KEY='...'
```

不要把 API Key 写进仓库。

常用配置：

| 变量               | 作用             |
| ------------------ | ---------------- |
| `SWANLAB_PROJECT`  | 项目名           |
| `SWANLAB_EXP_NAME` | 实验名           |
| `SWANLAB_API_KEY`  | API Key          |
| `SWANLAB_RUN_ID`   | Run ID           |
| `SWANLAB_RESUME`   | 是否续写已有 run |

默认项目：

```text
knowme-memory-gate-local
```

训练过程会记录：

```text
train loss
GRPO reward
dev metrics
test metrics
```

最终 test 指标会写回同一个 SwanLab run。

GRPO 的主要奖励曲线：

```text
rewards/GateRewardV2/mean
```

---

## 7. 硬件与耗时

默认配置面向：

```text
单卡 24 GB+
```

实测环境：

```text
RTX 5090 32 GB
```

只使用一张卡即可训练。

默认 GRPO 配置：

```text
NUM_GENERATIONS=16
TRAIN_BATCH_SIZE=16
GENERATION_BATCH_SIZE=16
```

每个 prompt 会生成 16 条 rollout，并完成一次 optimizer step。

训练集有 2287 条，因此：

```text
1 epoch ≈ 2287 steps
3 epochs ≈ 6861 steps
```



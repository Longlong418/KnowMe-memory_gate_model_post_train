# GRPO 奖励函数

实现文件：

```text
scripts/grpo_gate_reward.py
```

注册名：

```text
gate_reward_v2
```

通过 ms-swift 的 `--external_plugins` 加载。

---

## 1. 奖励输入

奖励函数主要使用：

- 模型输出
- `gate_label`
- `language`
- `memory_rows_json`
- `evidence_ids`

模型输出格式：

```json
{
  "retrieve": true,
  "query": "会议 时间 偏好",
  "reason": "需要查询历史记忆"
}
```

---

## 2. 奖励规则

| 情况                             |  Reward |
| -------------------------------- | ------: |
| 输出无法解析或格式非法           |    `-1` |
| 负样本，`retrieve=false`         |    `+1` |
| 负样本，`retrieve=true`          |    `-1` |
| 正样本，`retrieve=false`         |    `-1` |
| 正样本，检索命中，目标排名为 `r` | `1 / r` |
| 正样本，执行检索但未命中         |  `-0.5` |

命中奖励：

\[
R=\frac{1}{r}
\]

其中 \(r\) 为目标记忆的排名。

| Rank | Reward |
| ---: | -----: |
|    1 |    1.0 |
|    2 |    0.5 |
|    3 |  0.333 |
|    4 |   0.25 |

奖励大小关系：

```text
-1 < -0.5 < 0.25 < 0.333 < 0.5 < 1.0
```

---

## 3. Rank

检索分为两个分支：

```text
facts
episodes
```

rank 使用目标证据所在分支的 **branch-local rank**。

例如：

```text
episodes
├── rank 1: target
├── rank 2: distractor
└── rank 3: distractor
```

则：

```text
rank = 1
reward = 1.0
```

不会把 facts 和 episodes 拼接后重新计算排名。

---

## 4. 与 MRR 的关系

命中奖励使用：

\[
R=\frac{1}{r}
\]

与 Reciprocal Rank 定义一致。

MRR：

\[
\mathrm{MRR}
=
\frac{1}{N}
\sum_{i=1}^{N}
\frac{1}{r_i}
\]

未命中样本记为 0。

因此 GRPO 的检索奖励直接优化目标记忆的排序位置。

---

## 5. 格式检查

奖励函数直接复用评测器中的：

```text
output_is_valid()
```

格式要求包括：

- `retrieve` 必须是布尔值
- `reason` 必须是字符串
- `retrieve=false` 时 `query=""`
- `retrieve=true` 时 query 必须满足格式要求
- 中文多词 query 使用半角空格分隔

完整定义见：

```text
docs/evaluation.md
```

---

## 6. 检索实现

奖励函数使用和评测器相同的 SQLite FTS5 / BM25 检索逻辑。

流程：

```text
query
  ↓
SQLite FTS5 / BM25
  ↓
facts / episodes
  ↓
计算 evidence rank
  ↓
生成 reward
```

不使用金标 query。

---

## 7. ms-swift 配置

```bash
--external_plugins "$REPO_DIR/scripts/grpo_gate_reward.py" \
--reward_funcs gate_reward_v2 \
--reward_weights 1.0 \
--remove_unused_columns false
```

其中：

```text
--remove_unused_columns false
```

必须保留，否则奖励函数需要的数据列会被 Trainer 删除。

---

## 8. SwanLab 指标

奖励函数类名：

```text
GateRewardV2
```

SwanLab 中对应曲线：

```text
rewards/GateRewardV2/mean
```

训练时主要查看：

```text
train/reward
rewards/GateRewardV2/mean
```

---

## 9. 离线验证

正式训练前运行：

```bash
uv run python scripts/replay_gate_reward.py
```

检查：

- 数据列是否正确
- evidence 是否正确
- retrieval 是否正常
- reward 与 evaluation 是否一致

验证失败时脚本直接退出。

---

## 10. 奖励总结

```text
格式错误              → -1
判断错误              → -1
检索但未命中          → -0.5
命中 rank 4           → 0.25
命中 rank 3           → 0.333
命中 rank 2           → 0.5
命中 rank 1           → 1.0
正确拒绝负样本         → 1.0
```
# 数据构造流程

本文档记录从公开数据集到 SFT、GRPO 数据的完整构造流程。

## 1. 总体流程

```text
公开数据集
  -> 抽样
  -> 提取有 evidence 的正样本
  -> API 生成 3-5 个 query 候选
  -> SQLite FTS5/BM25 验证和选择 query
  -> 构造负样本
  -> 中英文翻译
  -> API/规则过滤和人工清洗
  -> 按 source group 划分 train/dev/test
  -> 构造 GRPO hidden hard-haystack
  -> SFT / GRPO 训练和评估
```

SFT 数据学习 retrieve 判断、query 格式和基本 query 生成能力；GRPO 数据在相同 prompt 后面加入隐藏干扰记忆，用真实 SQLite FTS5/BM25 结果计算 reward。

## 2. 公开数据集抽样

当前使用：

```text
LoCoMo
LongMemEval
RHELM
MemoryAgentBench
PersonaMem-v2
```

抽样命令：

```powershell
python -m src.dataset.sample_public --count 500 --seed 42
```

曾使用的规模约为：

```text
LoCoMo：500
LongMemEval：500
RHELM：158
MemoryAgentBench：300
PersonaMem-v2：500
```

抽样结果位于 `data/samples/`。

## 3. 提取 evidence-only 正样本

正样本必须有明确的正确记忆，否则无法验证 query 是否正确。每条正样本至少包含：

```json
{
  "id": "locomo_0189",
  "user_message": "What game is John hooked on playing?",
  "memory_rows": [
    {
      "kind": "fact",
      "id": "memory-17",
      "subject": "John",
      "content": "John is hooked on FIFA 23."
    }
  ],
  "evidence_ids": ["memory-17"]
}
```

`memory_rows` 是可写入 SQLite 的记忆，`evidence_ids` 是回答当前问题所需的正确记忆 ID。没有 evidence 指向的样本不能直接作为 query 检索正样本。

## 4. API 生成 query 候选

每个正样本生成 3-5 个候选 query。约束：

- 使用 2-6 个高信号词；
- 中文重要检索词使用 ASCII 空格；
- 不输出数组、解释或 Markdown；
- query 能够在 evidence memory 中检索到；
- 不机械复制整句用户问题。

候选结果缓存于：

```text
data/derived/query_candidate_cache.json
```

缓存可以避免 API 中断后重复付费调用。

## 5. SQLite FTS5/BM25 验证

验证复用：

```text
src/retrieval/store.py
```

对每个候选 query：

1. 把该样本记忆写入临时 SQLite；
2. 使用 FTS5/BM25 执行检索；
3. 获取返回的记忆 ID；
4. 判断 `retrieved_ids` 是否与 `evidence_ids` 有交集；
5. 过滤未命中的 query；
6. 从命中的候选中选择最终 query。

命中条件：

```text
retrieved_ids ∩ evidence_ids != empty
```

当前检索顺序是 facts 内部 BM25 顺序，再接 episodes 内部 BM25 顺序；facts 和 episodes 目前不是统一的全局 BM25 排名。

## 6. 构造 SFT 正负样本

### 6.1 正样本

```json
{
  "retrieve": true,
  "query": "John FIFA game",
  "reason": "需要查找相关历史记忆"
}
```

要求 query 能命中 evidence，符合 2-6 词格式，中文使用 ASCII 空格。

### 6.2 负样本

负样本用于训练模型判断当前问题不需要历史记忆：

```json
{
  "retrieve": false,
  "query": "",
  "reason": "当前问题不需要历史记忆"
}
```

负样本可来自普通问题、API 生成的 hard-negative 用户问题，以及人工确认不应触发检索的问题。必须过滤仍然依赖原始 evidence、API 空响应、JSON 错误和内容过滤失败的样本。

## 7. 翻译、过滤和人工清洗

中英文样本需要同步处理：

```text
user_message
memory_rows
gate_target.query
gate_target.reason
```

翻译后重新检查：

- 中文 query 是否使用 ASCII 空格；
- query 是否还能命中翻译后的 memory；
- `evidence_ids` 是否保持不变；
- 翻译是否改变了问题答案；
- `retrieve` 标签是否仍然正确。

API 失败、空响应、格式错误和内容过滤器拒绝的样本不能直接进入最终数据。最终文件：

```text
data/gate_final_2855.full.jsonl
data/gate_final_2855.sft.jsonl
```

`full.jsonl` 保留评估所需的 memory/evidence 字段，`sft.jsonl` 用于 SFT。

## 8. train/dev/test 划分

必须先按 source group 划分，再构造 GRPO haystack，避免同一对话、问题组或 persona 泄漏：

```text
data/train_split/gate_train.jsonl：2287
data/train_split/gate_dev.jsonl：284
data/train_split/gate_test.jsonl：284
```

train 用于训练，dev 用于验证和 checkpoint 选择，test 只在训练和调参结束后评估一次。

## 9. GRPO 数据格式

模型只看到用户侧 `messages`，以下字段只供 reward 使用：

```json
{
  "id": "locomo_0189",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "gate_label": true,
  "memory_rows_json": "[...]",
  "evidence_ids_json": "[\"memory-17\"]",
  "hard_negative_ids_json": "[...]",
  "language": "en",
  "source_dataset": "locomo",
  "reward_version": "gate_reward_v2"
}
```

以下字段不能拼进 prompt：

```text
gate_label
gold query
evidence_ids_json
memory_rows_json
hard_negative_ids_json
```

## 10. 构造 hard-haystack

运行：

```powershell
scripts/build_grpo_hard_haystack.ps1 -IncludeTest
```

输出：

```text
data/grpo_hard/gate_grpo_train.jsonl
data/grpo_hard/gate_grpo_dev.jsonl
data/grpo_hard/gate_grpo_test.jsonl
data/grpo_hard/*.audit.jsonl
data/grpo_hard/manifest.json
```

### 10.1 正样本

```text
正确 evidence memory + 6 条干扰记忆
```

### 10.2 负样本

```text
约 7 条干扰记忆
```

这些记忆只在 reward 内部使用，模型不会看到。

### 10.3 干扰记忆选择顺序

1. 使用当前用户问题跑 BM25；
2. 排除当前样本自己的记忆；
3. 排除正确 `evidence_ids`；
4. 排除同一个 source group；
5. 优先相同数据集；
6. 优先相同语言；
7. 优先 BM25 排名靠前；
8. 不足时使用随机记忆补齐。

每个正样本都会再次使用生产环境 FTS5/BM25 验证 gold query，确保加入干扰后 evidence 仍然可命中。

当前构造结果：

```text
train：1145/1145 正样本命中
dev：142/142 正样本命中
test：142/142 正样本命中
```

`*.audit.jsonl` 只用于数据质量检查，不能作为训练集。

## 11. Reward 使用的数据

reward 读取：

```text
gate_label
memory_rows_json
evidence_ids_json
```

reward 不读取 gold query，而是让模型自己生成 query，再使用 SQLite FTS5/BM25 判断是否命中 evidence。

基础 reward：

```text
正样本 + retrieve=true + 命中 evidence：+1
正样本 + retrieve=false：-1
正样本 + retrieve=true + 未命中：0 或负惩罚
负样本 + retrieve=false：+1
负样本 + retrieve=true：-1
格式错误：-1
```

加入 hard-haystack 后，可以进一步按 evidence 排名给分，例如排名第 1 得 1.0，排名第 2-3 得 0.7，排名第 4-5 得 0.4。



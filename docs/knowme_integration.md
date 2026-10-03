# 与 KnowMe 主项目的关系

本项目是 [KnowMe](https://github.com/Longlong418/KnowMe) 的一个独立后训练项目。

**KnowMe** 是一个面向个人使用场景的 Agent 项目，包含 Agent Loop、上下文管理、长期记忆、工具调用等核心能力。本项目关注其中的 **长期记忆检索环节**，尝试使用一个本地小模型完成记忆检索前的判断与 query 生成。

在 KnowMe 中，一次长期记忆检索大致经过：

```text
用户输入
   ↓
Memory Gate
   ├── 不需要记忆 → 进入 Agent Loop
   │
   └── 需要记忆
          ↓
      生成检索 Query
          ↓
      SQLite FTS5 / BM25
          ↓
      返回相关记忆
          ↓
      加入上下文
          ↓
       Agent Loop
```

其中 **Memory Gate** 主要负责两个任务：

1. 判断当前用户输入是否需要检索长期记忆；
2. 如果需要，生成适合检索器使用的 Query。

如果每轮对话都通过远程 API 模型完成这一步，会产生额外的推理延迟和调用成本。因此，本项目将这一模块单独拆出，使用 **Qwen3-1.7B** 进行后训练：

```text
Qwen3-1.7B
    ↓
   SFT
    ↓
  GRPO
    ↓
Memory Gate Model
    ↓
  KnowMe
```

目标是得到一个能够本地部署的小模型，在较低推理成本下完成 KnowMe 中的 Memory Gate 工作。

## 两个项目的分工

| KnowMe             | 本项目                     |
| ------------------ | -------------------------- |
| Agent Loop         | Memory Gate 模型训练       |
| Context Management | 训练数据构造               |
| Long-term Memory   | SFT                        |
| Memory Retrieval   | GRPO                       |
| Tools              | Retrieval-aware Reward     |
| Applications       | 模型评测与 Checkpoint 选择 |

两个项目共享的是 **Memory Gate 的输入输出协议和检索逻辑**。

本项目的评测不会只判断 `retrieve=true/false` 是否正确，而是会将模型生成的 Query 真正送入与 KnowMe 对齐的 SQLite FTS5 / BM25 检索流程中，评估完整链路：

```text
是否应该检索
      ↓
生成什么 Query
      ↓
执行真实检索
      ↓
能否找到目标记忆
      ↓
目标记忆排在什么位置
```

因此，本项目优化的不只是一个二分类任务，而是 **Memory Gate + Query Generation + Retrieval** 的完整检索过程。

## 数据隔离

训练仓库与 KnowMe 中的真实用户数据完全隔离。

本项目不会读取：

```text
.knowme/state.db
```

也不会复制真实用户记忆用于训练或评测。

训练与评测数据均由公开数据集构造，检索过程使用独立的 SQLite fixture 完成。
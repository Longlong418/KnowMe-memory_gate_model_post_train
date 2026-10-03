# Metrics

本文档说明 README 中使用的评测指标。除特殊说明外，指标均为 **越高越好**。

---

## 1. 门控判断

设：

- `TP`：需要检索，模型也判断需要检索
- `FP`：不需要检索，但模型判断需要检索
- `FN`：需要检索，但模型判断不需要检索
- `TN`：不需要检索，模型也判断不需要检索

### Precision

模型判断“需要检索”的样本中，真正需要检索的比例：

\[
\mathrm{Precision}=\frac{TP}{TP+FP}
\]

### Recall

所有真正需要检索的样本中，被模型成功放行的比例：

\[
\mathrm{Recall}=\frac{TP}{TP+FN}
\]

### Macro F1

分别计算“需要检索”和“不需要检索”两类的 F1，再取平均：

\[
F1_c=\frac{2P_cR_c}{P_c+R_c}
\]

\[
\mathrm{Macro\ F1}=\frac{F1_{\text{positive}}+F1_{\text{negative}}}{2}
\]

相比 Accuracy，Macro F1 对两类样本赋予相同权重，更适合衡量整体门控能力。

---

## 2. 输出合法性

设测试集样本数为 \(N\)。

### JSON Valid Rate

能够被正确解析为合法 JSON 的输出比例：

\[
\mathrm{JSON\ Valid\ Rate}
=
\frac{N_{\text{json-valid}}}{N}
\]

### Format Valid Rate

不仅 JSON 合法，而且满足完整输出协议的比例，例如字段类型、`query` 长度以及 `retrieve=false` 时 query 必须为空：

\[
\mathrm{Format\ Valid\ Rate}
=
\frac{N_{\text{format-valid}}}{N}
\]

因此：

\[
\mathrm{Format\ Valid\ Rate}
\le
\mathrm{JSON\ Valid\ Rate}
\]

---

## 3. 检索成功率

设真正需要检索的样本集合为 \(P\)，其数量为 \(|P|\)。

### Hit Rate

所有应检索样本中，最终成功找到目标记忆的比例：

\[
\mathrm{Hit\ Rate}
=
\frac{\sum_{i\in P}\mathbf{1}[\mathrm{hit}_i]}{|P|}
\]

如果模型没有触发检索、query 不合法或没有检索到目标记忆，该样本均记为未命中。

---

## 4. 检索排序质量

设目标记忆在样本 \(i\) 的检索结果中的排名为 \(r_i\)。若未命中，则该样本的 reciprocal rank 记为 0。

### Hit@K

目标记忆出现在前 \(K\) 个结果中的比例：

\[
\mathrm{Hit@K}
=
\frac{1}{|P|}
\sum_{i\in P}
\mathbf{1}[r_i\le K]
\]

本项目报告：

\[
\mathrm{Hit@1},\quad
\mathrm{Hit@3},\quad
\mathrm{Hit@5}
\]

### MRR

Mean Reciprocal Rank 衡量目标记忆整体排得有多靠前：

\[
\mathrm{MRR}
=
\frac{1}{|P|}
\sum_{i\in P}
\frac{1}{r_i}
\]

未触发检索、query 无效或未命中的样本贡献为 0，因此 MRR 同时受到门控、query 质量和排序位置影响。

### Conditional MRR

只在模型**实际触发检索并生成有效 query** 的样本上计算 MRR。

设这部分样本集合为 \(Q\)：

\[
\mathrm{Conditional\ MRR}
=
\frac{1}{|Q|}
\sum_{i\in Q}
\frac{1}{r_i}
\]

它更侧重衡量：

> **当模型已经决定检索后，生成的 query 排序质量如何。**

Conditional MRR 不会惩罚那些“根本没有触发检索”的正样本，因此应与 **Recall** 和 **Hit Rate** 一起查看。

---

## 5. 指标对应关系

| 目标                          | 指标                                |
| ----------------------------- | ----------------------------------- |
| 是否知道什么时候该检索        | Macro F1 / Precision / Recall       |
| 输出是否符合协议              | JSON Valid Rate / Format Valid Rate |
| 最终有没有找到目标记忆        | Hit Rate                            |
| 正确记忆排得是否靠前          | Hit@1 / Hit@3 / Hit@5 / MRR         |
| 已触发检索时 query 的排序质量 | Conditional MRR                     |

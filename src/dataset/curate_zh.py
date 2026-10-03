"""Create bilingual gate examples focused on Chinese personal-memory behavior.

The examples are intentionally hand-authored and normalized.  They provide
stable evidence IDs for the local FTS5/BM25 validator, while the negative set
contains personal-looking messages that are already self-contained.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def fact(memory_id: str, subject: str, content: str) -> dict[str, str]:
    return {"id": memory_id, "kind": "fact", "subject": subject, "content": content}


def episode(memory_id: str, happened_at: str, summary: str) -> dict[str, str]:
    return {"id": memory_id, "kind": "episode", "happened_at": happened_at, "summary": summary}


def positive(
    sample_id: str,
    question: str,
    query: str,
    memories: list[dict[str, str]],
    evidence: list[str],
    *,
    language: str = "zh",
) -> dict[str, Any]:
    return {
        "id": sample_id,
        "language": language,
        "source": {"dataset": "curated_gate", "label_type": "positive_memory_need"},
        "user_message": question,
        "memory_rows": memories,
        "evidence_ids": evidence,
        "gate_target": {
            "retrieve": True,
            "query": query,
            "reason": "需要查询历史记忆",
        },
    }


def negative(sample_id: str, question: str, *, language: str) -> dict[str, Any]:
    return {
        "id": sample_id,
        "language": language,
        "source": {"dataset": "curated_gate", "label_type": "hard_negative_self_contained"},
        "user_message": question,
        "memory_rows": [],
        "evidence_ids": [],
        "gate_target": {
            "retrieve": False,
            "query": "",
            "reason": "当前信息已足够",
        },
    }


def build() -> list[dict[str, Any]]:
    shared = [
        fact("bg_editor", "工作", "我平时使用 VS Code 编辑器。"),
        fact("bg_city", "城市", "我曾经去过成都旅行。"),
        fact("bg_food", "饮食", "我喜欢吃清淡的食物。"),
    ]
    rows: list[dict[str, Any]] = [
        positive(
            "zh_positive_0001",
            "我之前决定把论文投到哪个会议？",
            "论文 投稿 会议",
            shared + [fact("zh_mem_0001", "投稿", "我在 2025 年决定把这篇论文投稿到 EMNLP。")],
            ["zh_mem_0001"],
        ),
        positive(
            "zh_positive_0002",
            "我以前更喜欢 PostgreSQL 还是 MySQL？",
            "喜欢 PostgreSQL MySQL",
            shared + [fact("zh_mem_0002", "数据库偏好", "我以前更喜欢 PostgreSQL，而不是 MySQL。")],
            ["zh_mem_0002"],
        ),
        positive(
            "zh_positive_0003",
            "我上次说的个人知识库项目叫什么？",
            "个人 知识库 项目",
            shared + [fact("zh_mem_0003", "项目", "我的个人知识库项目叫 KnowMe。")],
            ["zh_mem_0003"],
        ),
        positive(
            "zh_positive_0004",
            "我之前计划什么时候去上海？",
            "计划 上海 时间",
            shared + [episode("zh_mem_0004", "2025-06-18", "我计划在 2025 年 6 月 18 日去上海参加开发者大会。")],
            ["zh_mem_0004"],
        ),
        positive(
            "zh_positive_0005",
            "我说过自己的开发机现在更倾向于 Windows 还是 Mac 吗？",
            "开发机 Windows Mac",
            shared + [fact("zh_mem_0005", "设备偏好", "我最近更倾向于把 Mac 作为开发机，之前主要使用 Windows。")],
            ["zh_mem_0005"],
        ),
        positive(
            "zh_positive_0006",
            "我之前提到过谁负责帮我看论文初稿？",
            "论文 初稿 负责",
            shared + [fact("zh_mem_0006", "论文协作", "林娜答应帮我看论文初稿。")],
            ["zh_mem_0006"],
        ),
        positive(
            "zh_positive_0007",
            "我去年参加的那个机器学习会议是哪一个？",
            "去年 机器学习 会议",
            shared + [episode("zh_mem_0007", "2024-10-20", "我去年参加了 NeurIPS 机器学习会议。")],
            ["zh_mem_0007"],
        ),
        positive(
            "zh_positive_0008",
            "我之前说过最想学习哪种强化学习算法？",
            "学习 强化学习 算法",
            shared + [fact("zh_mem_0008", "学习计划", "我之前最想系统学习 PPO 强化学习算法。")],
            ["zh_mem_0008"],
        ),
        positive(
            "zh_positive_0009",
            "我上次提到的家庭活动是什么？",
            "家人 散步 上次",
            shared + [episode("zh_mem_0009", "2025-05-06", "5 月 6 日我和家人一起去郊外散步。")],
            ["zh_mem_0009"],
        ),
        positive(
            "zh_positive_0010",
            "我最近改过论文投稿计划吗？",
            "论文 投稿 计划",
            shared + [
                episode("zh_mem_0010a", "2025-01-10", "我最初计划把论文投稿到 ACL。"),
                episode("zh_mem_0010b", "2025-02-02", "后来我把投稿计划改成 EMNLP。"),
            ],
            ["zh_mem_0010a", "zh_mem_0010b"],
        ),
        positive(
            "zh_positive_0011",
            "我以前记录过每周跑步的目标是多少？",
            "每周 跑步 目标",
            shared + [fact("zh_mem_0011", "运动目标", "我以前给自己定的目标是每周跑步三次。")],
            ["zh_mem_0011"],
        ),
        positive(
            "zh_positive_0012",
            "我说过最喜欢哪种编程语言？",
            "喜欢 编程语言",
            shared + [fact("zh_mem_0012", "编程偏好", "我说过自己最喜欢使用 Python 编程。")],
            ["zh_mem_0012"],
        ),
        negative("zh_negative_0001", "我现在喜欢 PostgreSQL，请比较它和 MySQL 的优缺点。", language="zh"),
        negative("zh_negative_0002", "我明天准备去上海，帮我规划一个两天行程。", language="zh"),
        negative("zh_negative_0003", "我已经决定周五提交报告，请帮我检查这句话的语法。", language="zh"),
        negative("zh_negative_0004", "我刚买了一台 MacBook，帮我列一个开发环境安装清单。", language="zh"),
        negative("zh_negative_0005", "我今天想学习 PPO，请给我安排一个两小时的学习计划。", language="zh"),
        negative("zh_negative_0006", "我已经告诉你会议在周五，请帮我写一条提醒。", language="zh"),
        negative("zh_negative_0007", "我的项目叫 KnowMe，请帮我写一段项目介绍。", language="zh"),
        negative("zh_negative_0008", "我正在使用 Windows，帮我解释如何安装 Python。", language="zh"),
        negative("en_negative_0001", "I am currently using PostgreSQL. Compare it with MySQL.", language="en"),
        negative("en_negative_0002", "I plan to visit Shanghai tomorrow. Make a two-day itinerary.", language="en"),
        negative("en_negative_0003", "I already chose EMNLP for this paper. Check the grammar of this sentence.", language="en"),
        negative("en_negative_0004", "I just bought a MacBook. Give me a development setup checklist.", language="en"),
        negative("en_negative_0005", "I want to study PPO today. Plan a two-hour study session.", language="en"),
    ]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=ROOT / "data/samples/zh_gate.jsonl")
    args = parser.parse_args()
    rows = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"total": len(rows), "positive": sum(r["gate_target"]["retrieve"] for r in rows), "negative": sum(not r["gate_target"]["retrieve"] for r in rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

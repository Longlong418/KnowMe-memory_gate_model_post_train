"""Build a balanced gate dataset from accepted public samples and negatives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.dataset.builder import validate_sample


ROOT = Path(__file__).resolve().parents[2]

SYNTHETIC_NEGATIVES = [
    "2+2 等于多少？",
    "Explain what an HTTP 404 error means.",
    "Write a Python function that reverses a string.",
    "What is the capital of France?",
    "帮我把这句话翻译成英文：今天天气很好。",
    "Summarize the advantages of SQLite FTS5.",
    "What is the difference between TCP and UDP?",
    "给我写一个简单的番茄钟计划。",
    "How do I format a JSON file?",
    "我明天准备去上海，帮我规划一个两天行程。",
    "我现在喜欢 PostgreSQL，请比较它和 MySQL 的优缺点。",
    "我已经决定周五提交报告，请帮我检查这句话的语法。",
    "我刚买了一台 MacBook，帮我列一个开发环境安装清单。",
    "我今天想学习 PPO，请给我安排一个两小时的学习计划。",
    "I am currently using PostgreSQL. Compare it with MySQL for this project.",
    "I already told you the meeting is on Friday; draft a short reminder.",
    "I plan to visit Shanghai tomorrow. Make a two-day itinerary from these details.",
    "I prefer dark mode in this chat. Suggest a matching color palette.",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def build(derived_dir: Path, output: Path) -> dict[str, int]:
    positives: list[dict[str, Any]] = []
    source_negatives: list[dict[str, Any]] = []
    for name in ("locomo", "longmemeval", "rhelm", "zh_gate"):
        path = derived_dir / f"{name}.sft.jsonl"
        if not path.exists():
            continue
        for row in read_jsonl(path):
            if row.get("gate_target", {}).get("retrieve") is True:
                positives.append(row)
            else:
                source_negatives.append(row)

    if not positives:
        raise RuntimeError("no accepted positive samples found")

    # Use compact real memory rows as a realistic background bank for negatives;
    # the gate itself still only sees user_message at inference time.
    background = positives[0].get("memory_rows", [])[:4]
    negatives: list[dict[str, Any]] = source_negatives
    # Accurate Retrieval is a retrieval benchmark, not a gate-negative source.
    # Do not relabel its questions as false merely because they are general QA.

    for index, question in enumerate(SYNTHETIC_NEGATIVES, start=1):
        negatives.append({
            "id": f"negative_synthetic_{index:04d}",
            "language": "zh" if any("\u4e00" <= char <= "\u9fff" for char in question) else "en",
            "source": {"dataset": "synthetic", "label_type": "negative_self_contained"},
            "user_message": question,
            "memory_rows": background,
            "evidence_ids": [],
            "gate_target": {"retrieve": False, "query": "", "reason": "当前信息已足够"},
        })

    negatives = negatives[: len(positives)]
    mixed = positives + negatives
    validated = [validate_sample(row) for row in mixed]
    write_jsonl(output, validated)
    return {"positive": len(positives), "negative": len(negatives), "total": len(validated)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--derived-dir", type=Path, default=ROOT / "data/derived/final")
    args = parser.parse_args()
    print(json.dumps(build(args.derived_dir, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()

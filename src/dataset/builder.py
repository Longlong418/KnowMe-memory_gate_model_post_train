"""Convert normalized memory examples into validated gate-training JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.retrieval import EpisodeStore, FactStore, connect


GATE_PROMPT = """\
You are a retrieval gate for a personal assistant's long-term memory.

Given the user's CURRENT message, decide whether answering it well requires
information from the user's stored long-term memory.

Long-term memory may contain:
- past events and conversations
- personal preferences
- people and relationships
- previous decisions and plans
- ongoing projects
- personal constraints or relevant historical context

Reply with ONLY this JSON, nothing else:
{{"retrieve": true/false, "query": "<search keywords if true, else empty>", "reason": "<一句中文，给用户看，不超过15字>"}}

Rules:

1. Retrieve when important information required to answer the request is
   missing from the current message but may exist in long-term memory.

2. Retrieve when relevant stored preferences, constraints, previous decisions,
   or personal history would materially improve the answer.

3. Do NOT retrieve merely because the message mentions the user, their life,
   a person, a project, or a personal topic.

4. If the current message already contains all personal information needed to
   answer well, do NOT retrieve.

5. General knowledge, math, coding questions, casual conversation, translation,
   rewriting, and other self-contained requests usually do NOT require memory.

6. When retrieve=true, generate a concise search query using 3-8 high-signal
   keywords derived ONLY from the current message. Do not invent facts that
   are not present in the message.

7. For Chinese queries, separate important search terms with ASCII spaces.

8. When retrieve=false, query must be "".

9. Never return an array.

Examples:

User: What's 2+2?
{{"retrieve":false,"query":"","reason":"当前问题无需历史记忆"}}

User: When am I meeting Alex?
{{"retrieve":true,"query":"Alex meeting time schedule","reason":"需要查看之前的安排"}}

User: I'm meeting Alex tomorrow at 3pm. Draft a reminder.
{{"retrieve":false,"query":"","reason":"当前信息已经足够"}}

User: Which database did we decide to use for the project?
{{"retrieve":true,"query":"project database decision","reason":"需要查看之前的决定"}}

User: We decided to use PostgreSQL. Help me plan the migration.
{{"retrieve":false,"query":"","reason":"当前信息已经足够"}}

User message: {message}"""


def _required_text(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be a non-empty string")
    return text


def _load_memory(sample: dict) -> tuple[FactStore, EpisodeStore, dict[tuple[str, int], str]]:
    database = connect()
    facts = FactStore(database)
    episodes = EpisodeStore(database)
    database_ids: dict[tuple[str, int], str] = {}
    stable_ids: set[str] = set()

    for index, row in enumerate(sample.get("memory_rows") or []):
        stable_id = _required_text(row.get("id"), f"memory_rows[{index}].id")
        if stable_id in stable_ids:
            raise ValueError(f"duplicate memory id: {stable_id}")
        stable_ids.add(stable_id)

        kind = _required_text(row.get("kind"), f"memory_rows[{index}].kind")
        if kind == "fact":
            database_id = facts.add(
                _required_text(row.get("subject"), f"memory_rows[{index}].subject"),
                _required_text(row.get("content"), f"memory_rows[{index}].content"),
            )
        elif kind == "episode":
            database_id = episodes.add(
                _required_text(row.get("happened_at"), f"memory_rows[{index}].happened_at"),
                _required_text(row.get("summary"), f"memory_rows[{index}].summary"),
            )
        else:
            raise ValueError(f"memory_rows[{index}].kind must be fact or episode")
        database_ids[(kind, database_id)] = stable_id

    return facts, episodes, database_ids


def _retrieved_ids(
    facts: FactStore,
    episodes: EpisodeStore,
    database_ids: dict[tuple[str, int], str],
    query: str,
) -> list[str]:
    retrieved: list[str] = []
    for row in facts.search(query, top_k=4):
        retrieved.append(database_ids[("fact", row["id"])])
    for row in episodes.search(query, top_k=3):
        retrieved.append(database_ids[("episode", row["id"])])
    return retrieved


def validate_sample(sample: dict, min_recall: float = 0.0) -> dict:
    """Validate one normalized sample using the production retrieval behavior."""
    sample_id = _required_text(sample.get("id"), "id")
    language = _required_text(sample.get("language"), "language")
    user_message = _required_text(sample.get("user_message"), "user_message")
    translation = sample.get("translation")
    if isinstance(translation, dict) and translation.get("fallback"):
        raise ValueError("translation fallback samples must be reviewed before training")
    target = sample.get("gate_target")
    if not isinstance(target, dict):
        raise ValueError("gate_target must be an object")

    retrieve = target.get("retrieve")
    if not isinstance(retrieve, bool):
        raise ValueError("gate_target.retrieve must be a boolean")
    query = str(target.get("query") or "").strip()
    reason = str(target.get("reason") or "").strip()
    evidence_ids = [str(value) for value in sample.get("evidence_ids") or []]

    facts, episodes, database_ids = _load_memory(sample)
    known_ids = set(database_ids.values())
    unknown_evidence = sorted(set(evidence_ids) - known_ids)
    if unknown_evidence:
        raise ValueError(f"unknown evidence ids: {', '.join(unknown_evidence)}")

    if retrieve:
        if not query:
            raise ValueError("a retrieve=true sample must have a query")
        query_terms = query.split()
        if not 2 <= len(query_terms) <= 6:
            raise ValueError("a retrieve=true query must contain 2-6 space-separated terms")
        if not evidence_ids:
            raise ValueError("a retrieve=true sample must have evidence_ids")
        retrieved_ids = _retrieved_ids(facts, episodes, database_ids, query)
        hit_ids = [evidence_id for evidence_id in evidence_ids if evidence_id in retrieved_ids]
        recall = len(hit_ids) / len(evidence_ids)
        # A query is useful when it retrieves at least one annotated correct
        # memory. Requiring every evidence row would reject multi-hop samples
        # merely because production top-k is smaller than their evidence set.
        passed = bool(hit_ids) and recall >= min_recall
    else:
        if query:
            raise ValueError("a retrieve=false sample must have an empty query")
        if evidence_ids:
            raise ValueError("a retrieve=false sample must not have evidence_ids")
        retrieved_ids = []
        hit_ids = []
        recall = 1.0
        passed = True

    expected_reason = "需要查询历史记忆" if retrieve else "当前信息已足够"
    if reason != expected_reason:
        raise ValueError(f"reason must be exactly: {expected_reason}")

    assistant = json.dumps(
        {"retrieve": retrieve, "query": query, "reason": reason},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return {
        "id": sample_id,
        "language": language,
        "source": sample.get("source") or {},
        "user_message": user_message,
        "memory_rows": sample.get("memory_rows") or [],
        "evidence_ids": evidence_ids,
        "gate_target": {"retrieve": retrieve, "query": query, "reason": reason},
        "messages": [
            {"role": "user", "content": GATE_PROMPT.format(message=user_message)},
            {"role": "assistant", "content": assistant},
        ],
        "validation": {
            "passed": passed,
            "min_recall": min_recall,
            "recall": recall,
            "retrieved_ids": retrieved_ids,
            "hit_evidence_ids": hit_ids,
        },
    }


def _read_jsonl(path: Path) -> list[dict]:
    samples: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: each line must be an object")
        samples.append(value)
    return samples


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def build_dataset(
    input_path: Path,
    output_path: Path,
    rejected_path: Path,
    min_recall: float = 1.0,
) -> dict:
    accepted: list[dict] = []
    rejected: list[dict] = []

    for sample in _read_jsonl(input_path):
        try:
            built = validate_sample(sample, min_recall=min_recall)
            if built["validation"]["passed"]:
                accepted.append(built)
            else:
                rejected.append(built)
        except ValueError as error:
            rejected.append({"sample": sample, "validation": {"passed": False, "error": str(error)}})

    _write_jsonl(output_path, accepted)
    _write_jsonl(rejected_path, rejected)
    return {"total": len(accepted) + len(rejected), "accepted": len(accepted), "rejected": len(rejected)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="normalized source JSONL")
    parser.add_argument("output", type=Path, help="validated SFT JSONL")
    parser.add_argument(
        "--rejected",
        type=Path,
        default=Path("data/derived/rejected.jsonl"),
        help="samples that fail schema or evidence recall",
    )
    parser.add_argument("--min-recall", type=float, default=0.0)
    arguments = parser.parse_args()
    if not 0.0 <= arguments.min_recall <= 1.0:
        parser.error("--min-recall must be between 0 and 1")

    summary = build_dataset(
        arguments.input,
        arguments.output,
        arguments.rejected,
        min_recall=arguments.min_recall,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

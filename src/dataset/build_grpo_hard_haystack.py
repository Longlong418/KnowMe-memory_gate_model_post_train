"""Build GRPO rows with BM25 hard-negative memory haystacks.

Only the user-side messages are exposed to the model.  The expanded memory
haystack is stored in a JSON string for the reward function.  Candidate
memories are selected within each split and, where possible, from the same
dataset/language but a different source conversation/persona.  Every positive
haystack is checked with the gold query so adding distractors never makes the
gold evidence unreachable under the production top-k retrieval behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval.run_gate_eval import _retrieved_ids
from src.dataset.build_grpo_data import _messages_without_target
from src.retrieval import EpisodeStore, FactStore, connect


@dataclass(frozen=True)
class MemoryCandidate:
    owner_id: str
    source_group: str
    dataset: str
    language: str
    memory: dict[str, Any]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not row.get("id"):
                raise ValueError(f"row without id in {path}:{line_number}")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _source_info(row: dict[str, Any]) -> tuple[str, str]:
    source = row.get("source") or {}
    if not isinstance(source, dict):
        source = {}
    dataset = str(source.get("dataset") or "unknown")
    source_value = next(
        (
            str(source[key])
            for key in ("sample_id", "question_id", "persona_id")
            if source.get(key) is not None
        ),
        str(row["id"]),
    )
    return dataset, f"{dataset}:{source_value}"


def _candidate_pool(rows: list[dict[str, Any]]) -> list[MemoryCandidate]:
    pool: list[MemoryCandidate] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if not (row.get("gate_target") or {}).get("retrieve"):
            continue
        dataset, source_group = _source_info(row)
        language = str(row.get("language") or "")
        for memory in row.get("memory_rows") or []:
            if not isinstance(memory, dict) or memory.get("kind") not in {"fact", "episode"}:
                continue
            key = (str(memory.get("kind")), str(memory.get("id")))
            if key in seen:
                continue
            seen.add(key)
            pool.append(MemoryCandidate(row["id"], source_group, dataset, language, memory))
    return pool


class RetrievalIndex:
    def __init__(self, candidates: list[MemoryCandidate]):
        self.database = connect()
        self.facts = FactStore(self.database)
        self.episodes = EpisodeStore(self.database)
        self.mapping: dict[tuple[str, int], MemoryCandidate] = {}
        for candidate in candidates:
            memory = candidate.memory
            if memory["kind"] == "fact":
                database_id = self.facts.add(memory["subject"], memory["content"])
                self.mapping[("fact", database_id)] = candidate
            else:
                database_id = self.episodes.add(memory["happened_at"], memory["summary"])
                self.mapping[("episode", database_id)] = candidate

    def ordered(self, query: str, top_k: int = 40) -> list[MemoryCandidate]:
        ordered: list[MemoryCandidate] = []
        for item in self.facts.search(query, top_k=top_k):
            ordered.append(self.mapping[("fact", item["id"])])
        for item in self.episodes.search(query, top_k=top_k):
            ordered.append(self.mapping[("episode", item["id"])])
        unique: list[MemoryCandidate] = []
        seen: set[tuple[str, str]] = set()
        for candidate in ordered:
            key = (candidate.owner_id, str(candidate.memory.get("id")))
            if key not in seen:
                seen.add(key)
                unique.append(candidate)
        return unique

    def close(self) -> None:
        self.database.close()


def _eligible(
    candidate: MemoryCandidate,
    target: dict[str, Any],
    evidence_ids: set[str],
    *,
    same_dataset: bool,
    same_language: bool,
) -> bool:
    if candidate.owner_id == target["id"]:
        return False
    if str(candidate.memory.get("id")) in evidence_ids:
        return False
    dataset, source_group = _source_info(target)
    if candidate.source_group == source_group:
        return False
    if same_dataset and candidate.dataset != dataset:
        return False
    if same_language and candidate.language != str(target.get("language") or ""):
        return False
    return True


def _gold_hit(target: dict[str, Any], memories: list[dict[str, Any]]) -> bool:
    query = str((target.get("gate_target") or {}).get("query") or "")
    if not query:
        return False
    retrieved = _retrieved_ids(memories, query)
    evidence = {str(value) for value in target.get("evidence_ids") or []}
    return bool(evidence.intersection(retrieved))


def _select_for_positive(
    target: dict[str, Any],
    index: RetrievalIndex,
    pool: list[MemoryCandidate],
    rng: random.Random,
    count: int,
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    gold = list(target.get("memory_rows") or [])
    evidence_ids = {str(value) for value in target.get("evidence_ids") or []}
    ordered = index.ordered(str(target.get("user_message") or ""))
    selected: list[MemoryCandidate] = []
    selected_keys: set[tuple[str, str]] = set()

    def try_add(candidate: MemoryCandidate) -> bool:
        key = (str(candidate.memory.get("kind")), str(candidate.memory.get("id")))
        if key in selected_keys or len(selected) >= count:
            return False
        memories = gold + [item.memory for item in selected] + [candidate.memory]
        if not _gold_hit(target, memories):
            return False
        selected.append(candidate)
        selected_keys.add(key)
        return True

    for strict in ((True, True), (True, False), (False, True), (False, False)):
        for candidate in ordered:
            if len(selected) >= count:
                break
            if _eligible(candidate, target, evidence_ids, same_dataset=strict[0], same_language=strict[1]):
                try_add(candidate)
        if len(selected) >= count:
            break

    fallback = list(pool)
    rng.shuffle(fallback)
    for candidate in fallback:
        if len(selected) >= count:
            break
        if _eligible(candidate, target, evidence_ids, same_dataset=False, same_language=False):
            try_add(candidate)

    selected_memories = [item.memory for item in selected]
    memories = gold + selected_memories
    for _ in range(20):
        shuffled = list(memories)
        rng.shuffle(shuffled)
        if _gold_hit(target, shuffled):
            memories = shuffled
            break
    else:
        # Keep gold first as a safe fallback when BM25 ties make random
        # insertion order push evidence beyond the production top-k limit.
        memories = gold + selected_memories
    return memories, [str(item.memory["id"]) for item in selected], selected_memories


def _select_for_negative(
    target: dict[str, Any],
    index: RetrievalIndex,
    pool: list[MemoryCandidate],
    rng: random.Random,
    count: int,
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    ordered = index.ordered(str(target.get("user_message") or ""))
    candidates = ordered + pool
    selected: list[MemoryCandidate] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        if len(selected) >= count:
            break
        if not _eligible(candidate, target, set(), same_dataset=True, same_language=True):
            continue
        key = (str(candidate.memory.get("kind")), str(candidate.memory.get("id")))
        if key in seen:
            continue
        seen.add(key)
        selected.append(candidate)
    if len(selected) < count:
        fallback = list(pool)
        rng.shuffle(fallback)
        for candidate in fallback:
            if len(selected) >= count:
                break
            if not _eligible(candidate, target, set(), same_dataset=False, same_language=False):
                continue
            key = (str(candidate.memory.get("kind")), str(candidate.memory.get("id")))
            if key in seen:
                continue
            seen.add(key)
            selected.append(candidate)
    memories = [item.memory for item in selected]
    rng.shuffle(memories)
    return memories, [str(item.memory["id"]) for item in selected], [item.memory for item in selected]


def _make_output_row(
    split_row: dict[str, Any],
    full_row: dict[str, Any],
    memories: list[dict[str, Any]],
    hard_negative_ids: list[str],
) -> dict[str, Any]:
    target = full_row.get("gate_target") or {}
    source, source_group = _source_info(full_row)
    return {
        "id": split_row["id"],
        "messages": _messages_without_target(split_row),
        "gate_label": bool(target.get("retrieve")),
        "memory_rows_json": _json_text(memories),
        "evidence_ids_json": _json_text(full_row.get("evidence_ids") or []),
        "hard_negative_ids_json": _json_text(hard_negative_ids),
        "language": split_row.get("language", full_row.get("language", "")),
        "source_dataset": source,
        "source_group": source_group,
        "source_id": (full_row.get("source") or {}).get("sample_id", ""),
        "reward_version": "gate_reward_v2",
    }


def build(
    full_path: Path,
    split_dir: Path,
    output_dir: Path,
    distractors: int,
    seed: int,
    include_test: bool = False,
) -> dict[str, Any]:
    full_rows = _read_jsonl(full_path)
    full_by_id = {row["id"]: row for row in full_rows}
    split_names = ["train", "dev"] + (["test"] if include_test else [])
    summary: dict[str, Any] = {"full_rows": len(full_rows), "distractors_requested": distractors}
    for split in split_names:
        split_rows = _read_jsonl(split_dir / f"gate_{split}.jsonl")
        split_full = [full_by_id[row["id"]] for row in split_rows]
        pool = _candidate_pool(split_full)
        index = RetrievalIndex(pool)
        rng = random.Random(seed + split_names.index(split))
        output_rows: list[dict[str, Any]] = []
        audit_rows: list[dict[str, Any]] = []
        try:
            for split_row, full_row in zip(split_rows, split_full):
                if (full_row.get("gate_target") or {}).get("retrieve"):
                    memories, negative_ids, _ = _select_for_positive(
                        full_row, index, pool, rng, distractors
                    )
                    gold_hit = _gold_hit(full_row, memories)
                else:
                    memories, negative_ids, _ = _select_for_negative(
                        full_row, index, pool, rng, distractors + 1
                    )
                    gold_hit = None
                output_rows.append(_make_output_row(split_row, full_row, memories, negative_ids))
                audit_rows.append({
                    "id": full_row["id"],
                    "gate_label": bool((full_row.get("gate_target") or {}).get("retrieve")),
                    "evidence_ids": full_row.get("evidence_ids") or [],
                    "hard_negative_ids": negative_ids,
                    "memory_count": len(memories),
                    "gold_query_hit": gold_hit,
                })
        finally:
            index.close()
        _write_jsonl(output_dir / f"gate_grpo_{split}.jsonl", output_rows)
        _write_jsonl(output_dir / f"gate_grpo_{split}.audit.jsonl", audit_rows)
        positives = [row for row in audit_rows if row["gate_label"]]
        summary[split] = {
            "rows": len(output_rows),
            "positive": len(positives),
            "negative": len(output_rows) - len(positives),
            "positive_gold_hit": sum(bool(row["gold_query_hit"]) for row in positives),
            "positive_gold_hit_rate": (
                round(sum(bool(row["gold_query_hit"]) for row in positives) / len(positives), 6)
                if positives else None
            ),
            "avg_memory_count": round(sum(row["memory_count"] for row in audit_rows) / len(audit_rows), 3),
        }
    manifest = {
        "full_path": str(full_path),
        "split_dir": str(split_dir),
        "output_dir": str(output_dir),
        "summary": summary,
        "prompt_fields": ["messages"],
        "reward_fields": [
            "gate_label",
            "memory_rows_json",
            "evidence_ids_json",
            "hard_negative_ids_json",
        ],
        "audit_files_are_not_training_data": True,
        "test_is_included": include_test,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, default=Path("data/gate_final_2855.full.jsonl"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/train_split"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/grpo_hard"))
    parser.add_argument("--distractors", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-test", action="store_true")
    args = parser.parse_args()
    if args.distractors < 1:
        raise SystemExit("--distractors must be positive")
    print(json.dumps(build(
        args.full, args.split_dir, args.output_dir, args.distractors, args.seed, args.include_test
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

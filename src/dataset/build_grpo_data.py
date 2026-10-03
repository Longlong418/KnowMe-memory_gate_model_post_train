"""Build GRPO-ready rows without leaking the assistant target into the prompt.

The model receives only the user-side ``messages``.  Reward code can read the
additional JSON-string columns to evaluate the generated gate decision against
the evidence memory.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
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


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _messages_without_target(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"{row['id']}: messages must be a list")
    prompt_messages = [
        message
        for message in messages
        if isinstance(message, dict) and message.get("role") != "assistant"
    ]
    if not prompt_messages or prompt_messages[-1].get("role") != "user":
        raise ValueError(f"{row['id']}: expected a final user message")
    return prompt_messages


def _make_row(split_row: dict[str, Any], full_row: dict[str, Any]) -> dict[str, Any]:
    target = full_row.get("gate_target") or {}
    if not isinstance(target, dict) or "retrieve" not in target:
        raise ValueError(f"{full_row['id']}: missing gate_target.retrieve")
    memory_rows = full_row.get("memory_rows") or []
    evidence_ids = full_row.get("evidence_ids") or []
    if not isinstance(memory_rows, list) or not isinstance(evidence_ids, list):
        raise ValueError(f"{full_row['id']}: memory_rows/evidence_ids must be lists")
    source = full_row.get("source") or split_row.get("source") or {}
    if not isinstance(source, dict):
        source = {"value": str(source)}
    return {
        "id": split_row["id"],
        "messages": _messages_without_target(split_row),
        # Hidden reward columns: they are not referenced by messages.
        "gate_label": bool(target["retrieve"]),
        "memory_rows_json": _json_text(memory_rows),
        "evidence_ids_json": _json_text(evidence_ids),
        "language": split_row.get("language", full_row.get("language", "")),
        "source_dataset": source.get("dataset", ""),
        "source_id": source.get("sample_id") or source.get("question_id") or "",
        "reward_version": "gate_reward_v1",
    }


def build(
    full_path: Path,
    split_dir: Path,
    output_dir: Path,
    include_test: bool = False,
) -> dict[str, Any]:
    full_rows = _read_jsonl(full_path)
    full_by_id = {row["id"]: row for row in full_rows}
    if len(full_by_id) != len(full_rows):
        raise ValueError("duplicate ids in full dataset")

    split_names = ["train", "dev"] + (["test"] if include_test else [])
    summary: dict[str, Any] = {"full_rows": len(full_rows), "include_test": include_test}
    for split in split_names:
        split_path = split_dir / f"gate_{split}.jsonl"
        split_rows = _read_jsonl(split_path)
        output_rows: list[dict[str, Any]] = []
        missing: list[str] = []
        for split_row in split_rows:
            full_row = full_by_id.get(split_row["id"])
            if full_row is None:
                missing.append(split_row["id"])
                continue
            output_rows.append(_make_row(split_row, full_row))
        if missing:
            raise ValueError(f"{split}: {len(missing)} ids missing from full dataset; first={missing[:3]}")
        output_path = output_dir / f"gate_grpo_{split}.jsonl"
        _write_jsonl(output_path, output_rows)
        summary[split] = {
            "rows": len(output_rows),
            "positive": sum(1 for row in output_rows if row["gate_label"]),
            "negative": sum(1 for row in output_rows if not row["gate_label"]),
            "path": str(output_path),
        }

    manifest = {
        "full_path": str(full_path),
        "split_dir": str(split_dir),
        "output_dir": str(output_dir),
        "splits": split_names,
        "summary": summary,
        "prompt_fields": ["messages"],
        "reward_fields": [
            "gate_label",
            "memory_rows_json",
            "evidence_ids_json",
        ],
        "test_is_included": include_test,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        type=Path,
        default=Path("data/gate_final_2855.full.jsonl"),
        help="full dataset containing memory_rows and evidence_ids",
    )
    parser.add_argument("--split-dir", type=Path, default=Path("data/train_split"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/grpo"))
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="also write a test GRPO file; do not use it for GRPO training",
    )
    args = parser.parse_args()
    summary = build(args.full, args.split_dir, args.output_dir, args.include_test)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

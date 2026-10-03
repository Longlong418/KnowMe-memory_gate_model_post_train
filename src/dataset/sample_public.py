"""Extract small, inspectable normalized samples from public memory datasets.

This module intentionally stops before model-generated translation or query
rewriting.  The output is the normalized input expected by ``dataset.builder``;
the production-shaped SQLite validator remains the source of truth for recall.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
SAMPLES = ROOT / "data" / "samples"

STOPWORDS = {
    "what", "when", "where", "which", "who", "whom", "why", "how",
    "did", "does", "do", "was", "were", "is", "are", "the", "a", "an",
    "i", "my", "me", "we", "our", "and", "or", "of", "to", "for", "in",
    "on", "at", "from", "with", "about", "after", "before", "this", "that",
    "have", "had", "has", "can", "could", "would", "should", "during",
}


def query_from_question(question: str) -> str:
    """Create a conservative single-string BM25 query from a user question."""
    tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9'-]{2,}\b|\b\d{4}(?:-\d{2})?\b|[\u4e00-\u9fff]{2,}", question)
    selected: list[str] = []
    for token in tokens:
        if token.lower() in STOPWORDS or token.lower() in {x.lower() for x in selected}:
            continue
        selected.append(token)
        if len(selected) == 6:
            break
    return " ".join(selected)


def target(question: str, source: str, *, retrieve: bool = True) -> dict[str, Any]:
    return {
        "retrieve": retrieve,
        "query": query_from_question(question) if retrieve else "",
        "reason": "需要查询历史记忆" if retrieve else "当前信息已足够",
    }


def fact(memory_id: str, subject: str, content: str) -> dict[str, str]:
    return {"id": memory_id, "kind": "fact", "subject": subject, "content": content}


def episode(memory_id: str, happened_at: str, summary: str) -> dict[str, str]:
    return {"id": memory_id, "kind": "episode", "happened_at": happened_at, "summary": summary}


def one_line(text: str) -> str:
    return re.sub(r"[\r\n\u0085\u2028\u2029]+", " ", text).strip()


def locomo(count: int, rng: random.Random) -> list[dict[str, Any]]:
    rows = json.loads((RAW / "locomo_repo/data/locomo10.json").read_text(encoding="utf-8"))
    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for conversation in rows:
        for qa in conversation["qa"]:
            candidates.append((conversation, qa))
    rng.shuffle(candidates)
    output = []
    for index, (source, qa) in enumerate(candidates[:count], start=1):
        memory_rows: list[dict[str, str]] = []
        for key, turns in source["conversation"].items():
            if not key.startswith("session_") or not isinstance(turns, list):
                continue
            for turn in turns:
                text = turn.get("text", "").strip()
                if not text:
                    continue
                memory_rows.append(
                    fact(
                        f"dialogue_{turn['dia_id']}",
                        str(turn.get("speaker", "dialogue")),
                        f"{turn.get('speaker', 'speaker')}: {text}",
                    )
                )
        # LoCoMo normally stores one evidence turn per item, but some
        # multi-hop rows encode several turn IDs in a single whitespace-
        # separated string. Normalize both forms before compacting evidence.
        evidence_values: list[str] = []
        for value in qa.get("evidence", []) or []:
            evidence_values.extend(str(value).split())
        evidence_ids = [f"dialogue_{value}" for value in evidence_values]
        question = str(qa["question"])
        output.append({
            "id": f"locomo_{index:04d}",
            "language": "en",
            "source": {"dataset": "locomo", "sample_id": source.get("sample_id"), "category": qa.get("category")},
            "user_message": question,
            "memory_rows": memory_rows,
            "evidence_ids": evidence_ids,
            "gate_target": target(question, "LoCoMo"),
            "raw": {"question": question, "answer": qa.get("answer"), "evidence": qa.get("evidence"), "category": qa.get("category")},
        })
    return output


def longmemeval(count: int, rng: random.Random) -> list[dict[str, Any]]:
    rows = json.loads((RAW / "longmemeval/longmemeval_oracle.json").read_text(encoding="utf-8"))
    rng.shuffle(rows)
    output = []
    for index, row in enumerate(rows[:count], start=1):
        memory_rows = []
        for session_id, session in zip(row["haystack_session_ids"], row["haystack_sessions"]):
            summary = "\n".join(
                f"{turn.get('role', 'speaker')}: {turn.get('content', '')}" for turn in session
            ).strip()
            memory_rows.append(episode(f"session_{session_id}", row.get("question_date", ""), summary[:12000]))
        evidence_ids = [f"session_{value}" for value in row.get("answer_session_ids", [])]
        question = str(row["question"])
        output.append({
            "id": f"longmemeval_{index:04d}",
            "language": "en",
            "source": {"dataset": "longmemeval-cleaned", "question_id": row.get("question_id"), "question_type": row.get("question_type")},
            "user_message": question,
            "memory_rows": memory_rows,
            "evidence_ids": evidence_ids,
            "gate_target": target(question, "LongMemEval"),
            "raw": {key: row.get(key) for key in ("question_id", "question_type", "question", "answer", "question_date", "haystack_session_ids", "answer_session_ids")},
        })
    return output


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rhelm_evidence(token: str, conversation_dir: Path, attachment_dir: Path, email_dir: Path) -> str:
    token = token.strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}:\d+$", token):
        date, turn_number = token.split(":")
        path = conversation_dir / f"conversation_{date.replace('-', '_')}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            for turn in data.get("conversation", []):
                if str(turn.get("turn")) == turn_number:
                    return f"{turn.get('user', '')}\n{turn.get('assistant', '')}".strip()
    if token.startswith("Emails_"):
        date = re.search(r"\d{4}-\d{2}-\d{2}", token)
        if date:
            files = sorted(email_dir.glob(f"*{date.group().replace('-', '_')}*.txt"))
            if files:
                return files[0].read_text(encoding="utf-8", errors="replace")[:8000]
    filename, _, heading = token.partition(":")
    path = attachment_dir / filename
    if path.exists():
        text = path.read_text(encoding="utf-8", errors="replace")
        if heading and heading in text:
            start = text.index(heading)
            return text[start : start + 8000]
        return text[:8000]
    return token


def rhelm(count: int, rng: random.Random) -> list[dict[str, Any]]:
    qa_path = next((RAW / "rhelm").glob("QA_final/*.jsonl"), None)
    if qa_path is None:
        qa_path = next((RAW / "rhelm").glob("*.jsonl"))
    rows = _read_jsonl(qa_path)
    rng.shuffle(rows)
    conversation_dir = RAW / "rhelm/conversations"
    attachment_dir = RAW / "rhelm/attachments"
    email_dir = RAW / "rhelm/emails"
    output = []
    for index, row in enumerate(rows[:count], start=1):
        evidence_tokens = [part.strip() for raw in row.get("supporting_evidence", []) for part in raw.split(",")]
        memory_rows = []
        evidence_ids = []
        for evidence_index, token in enumerate(evidence_tokens, start=1):
            memory_id = f"evidence_{index}_{evidence_index}"
            content = _rhelm_evidence(token, conversation_dir, attachment_dir, email_dir)
            memory_rows.append(fact(memory_id, row.get("question_type", "evidence"), content))
            evidence_ids.append(memory_id)
        question = str(row["question"])
        output.append({
            "id": f"rhelm_{index:04d}",
            "language": "en",
            "source": {"dataset": "RHELM", "source_file": qa_path.name, "question_type": row.get("question_type"), "characteristics": row.get("characteristics")},
            "user_message": question,
            "memory_rows": memory_rows,
            "evidence_ids": evidence_ids,
            "gate_target": target(question, "RHELM"),
            "raw": {key: row.get(key) for key in ("id", "question", "answer", "question_date", "question_type", "supporting_evidence", "characteristics")},
        })
    return output


def _parse_json_cell(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def memoryagentbench(count: int, rng: random.Random) -> list[dict[str, Any]]:
    split_files = {"Accurate_Retrieval": RAW / "memoryagentbench/Accurate_Retrieval_rows.json"}
    expanded: list[tuple[str, int, str, str, dict[str, Any]]] = []
    for split, path in split_files.items():
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        for item in payload.get("rows", []):
            row = item.get("row", {})
            questions = _parse_json_cell(row.get("questions", []))
            answers = _parse_json_cell(row.get("answers", []))
            metadata = _parse_json_cell(row.get("metadata", {}))
            for qa_index, question in enumerate(questions or []):
                answer = answers[qa_index] if isinstance(answers, list) and qa_index < len(answers) else None
                expanded.append((split, item.get("row_idx", 0), str(question), json.dumps(answer, ensure_ascii=False), {"metadata": metadata, "context": row.get("context", "")}))
    rng.shuffle(expanded)
    output = []
    for index, (split, row_index, question, answer, extra) in enumerate(expanded[:count], start=1):
        memory_id = f"context_{index:04d}"
        context = str(extra["context"])
        output.append({
            "id": f"memoryagentbench_{index:04d}",
            "language": "en",
            "source": {"dataset": "MemoryAgentBench", "split": split, "row_idx": row_index},
            "user_message": question,
            "memory_rows": [episode(memory_id, split, one_line(context)[:16000])],
            "evidence_ids": [memory_id],
            "gate_target": target(question, "MemoryAgentBench"),
            "raw": {"split": split, "row_idx": row_index, "question": question, "answer": answer, "metadata": extra["metadata"], "context_chars": len(context)},
        })
    return output


def personamem(count: int, rng: random.Random) -> list[dict[str, Any]]:
    path = RAW / "personamem_v2/benchmark.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rng.shuffle(rows)
    output = []
    for index, row in enumerate(rows[:count], start=1):
        preference = row.get("preference", "").strip() or row.get("related_conversation_snippet", "").strip()
        memory_id = f"preference_{index:04d}"
        question_value: Any = row.get("user_query", "").strip()
        if question_value.startswith("{"):
            try:
                question_value = ast.literal_eval(question_value).get("content", question_value)
            except (ValueError, SyntaxError):
                pass
        question = str(question_value).strip()
        output.append({
            "id": f"personamem_v2_{index:04d}",
            "language": "en",
            "source": {"dataset": "PersonaMem-v2", "persona_id": row.get("persona_id"), "pref_type": row.get("pref_type"), "updated": row.get("updated")},
            "user_message": question,
            "memory_rows": [fact(memory_id, row.get("who", "preference"), preference)],
            "evidence_ids": [memory_id],
            "gate_target": target(question, "PersonaMem-v2"),
            "raw": {key: row.get(key) for key in ("persona_id", "user_query", "correct_answer", "preference", "related_conversation_snippet", "who", "updated", "prev_pref", "pref_type", "conversation_scenario", "chat_history_32k_link")},
        })
    return output


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for row in rows:
        encoded = json.dumps(row, ensure_ascii=False)
        encoded = encoded.replace("\u0085", "\\u0085").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
        lines.append(encoded + "\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10, help="samples per source")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    datasets = {
        "locomo": locomo(args.count, rng),
        "longmemeval": longmemeval(args.count, rng),
        "rhelm": rhelm(args.count, rng),
        "memoryagentbench": memoryagentbench(args.count, rng),
        "personamem_v2": personamem(args.count, rng),
    }
    for name, rows in datasets.items():
        write_jsonl(SAMPLES / f"{name}_{len(rows)}.jsonl", rows)
        print(json.dumps({"dataset": name, "samples": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

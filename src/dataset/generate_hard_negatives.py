"""Generate self-contained hard negatives paired with positive memory samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from src.dataset.openrouter import CACHE_MODEL, MODEL, chat, get_api_key


ROOT = Path(__file__).resolve().parents[2]

SYSTEM_PROMPT = """You create hard negative examples for a personal-memory retrieval gate.
Given a memory-dependent user question and the supporting memory text, rewrite
the request into a natural, self-contained user message that does NOT require
retrieving long-term memory. Put every personal fact needed for the answer in
the new message. Keep the user's intent and language (English) similar.
Do not mention memory, retrieval, previous conversations, or 'as I said'.
Return ONLY JSON: {\"user_message\": \"...\"}.
"""


def _extract_json(text: str) -> dict[str, Any]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("model response did not contain JSON")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model response was not an object")
    return value


def _call(api_key: str, question: str, memory_text: str, session: str) -> str:
    prompt = json.dumps(
        {"user_message": question, "supporting_memory": memory_text[:12000]},
        ensure_ascii=False,
    )
    result = chat(
        api_key,
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1200,
        session=session,
        retries=1,
    )
    choice = result["choices"][0]
    content = choice["message"].get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"empty model content (finish_reason={choice.get('finish_reason')})")
    value = _extract_json(content)
    message = str(value.get("user_message") or "").strip()
    if len(message) < 8 or message == question.strip():
        raise ValueError("hard negative was empty or unchanged")
    if re.search(
        r"\b(?:previously|earlier you said|as I (?:said|mentioned)|from (?:your|my) memory|"
        r"long[- ]term memory|retriev(?:e|al|ing)|look(?:ed)? up)\b",
        message,
        re.IGNORECASE,
    ):
        raise ValueError("hard negative still refers to historical memory")
    return message


def _cache_key(row: dict[str, Any]) -> str:
    payload = json.dumps(
        {"id": row.get("id"), "question": row.get("user_message"), "model": CACHE_MODEL},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _save_cache(path: Path, cache: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    for _ in range(5):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            time.sleep(0.25)
    path.write_text(temporary.read_text(encoding="utf-8"), encoding="utf-8")
    temporary.unlink(missing_ok=True)


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _memory_text(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for memory in row.get("memory_rows") or []:
        text = memory.get("content") or memory.get("summary") or ""
        if text:
            parts.append(str(text))
    return "\n".join(parts)


def generate(input_path: Path, output_path: Path, cache_path: Path, workers: int = 1) -> dict[str, int]:
    api_key = get_api_key()
    rows = _read(input_path)
    cache = _load_cache(cache_path)

    def process(index: int, row: dict[str, Any]) -> tuple[int, dict[str, Any], str | None, str | None, str | None]:
        key = _cache_key(row)
        if key in cache:
            message = cache[key]
        else:
            last_error: Exception | None = None
            message = ""
            for attempt in range(2):
                try:
                    message = _call(
                        api_key,
                        str(row["user_message"]),
                        _memory_text(row),
                        f"hard-negative-{input_path.stem}-{index}-attempt-{attempt + 1}",
                    )
                    break
                except Exception as error:
                    last_error = error
                    if attempt == 0:
                        time.sleep(0.5)
            if not message:
                error = last_error or RuntimeError("hard-negative generation failed")
                return index, row, None, key, f"{type(error).__name__}: {error}"
        negative = {
            "id": f"{row['id']}__negative",
            "language": "en",
            "source": {**(row.get("source") or {}), "label_type": "hard_negative_self_contained", "paired_positive_id": row["id"]},
            "user_message": message,
            "memory_rows": [],
            "evidence_ids": [],
            "gate_target": {"retrieve": False, "query": "", "reason": "当前信息已足够"},
        }
        return index, negative, key, message, None

    results: list[tuple[int, dict[str, Any], str | None, str | None, str | None]] = []

    def record(completed: int, result: tuple[int, dict[str, Any], str | None, str | None, str | None]) -> None:
        results.append(result)
        _, _, key, message, error = result
        if key and message:
            cache[key] = message
            _save_cache(cache_path, cache)
        percent = completed / max(len(rows), 1) * 100
        suffix = f" failed={error[:80]}" if error else ""
        sys.stdout.write(f"\r[hard-negative] {completed}/{len(rows)} ({percent:5.1f}%){suffix}")
        sys.stdout.flush()

    if workers <= 1:
        for completed, row in enumerate(rows, start=1):
            index = completed - 1
            started = time.monotonic()
            sys.stdout.write(f"\r[hard-negative] {completed}/{len(rows)} starting {row.get('id')} ...")
            sys.stdout.flush()
            record(completed, process(index, row))
            elapsed = time.monotonic() - started
            sys.stdout.write(f"\r[hard-negative] {completed}/{len(rows)} ({completed / max(len(rows), 1) * 100:5.1f}%) {elapsed:.1f}s")
            sys.stdout.flush()
        if rows:
            print()
    else:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = [executor.submit(process, index, row) for index, row in enumerate(rows)]
            for completed, future in enumerate(as_completed(futures), start=1):
                record(completed, future.result())
        if rows:
            print()
    results.sort(key=lambda value: value[0])
    negatives = [value[1] for value in results if value[2] is not None and value[3] is not None]
    failed = len(results) - len(negatives)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in negatives), encoding="utf-8")
    return {"positive_input": len(rows), "negative_generated": len(negatives), "failed": failed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cache", type=Path, default=ROOT / "data/derived/hard_negative_cache.json")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(generate(args.input, args.output, args.cache, args.workers), ensure_ascii=False))


if __name__ == "__main__":
    main()

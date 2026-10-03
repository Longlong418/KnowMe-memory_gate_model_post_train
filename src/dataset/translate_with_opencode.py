"""Translate normalized samples while preserving retrieval supervision.

This step translates the user message and every searchable memory field. IDs,
evidence links, timestamps, and gate labels are never translated. Query labels
are cleared for positive rows and should be regenerated from the translated
user message by ``annotate_with_opencode``; the translator never sees evidence
when generating a query.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
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

from src.dataset.openrouter import MODEL, chat, get_api_key


ROOT = Path(__file__).resolve().parents[2]
TARGET_LANGUAGE = "Chinese"
MAX_CHUNK_CHARS = 12000

TEXT_SYSTEM = """Translate English text into natural Simplified Chinese.
Return ONLY a JSON object: {\"text\": \"...\"}.
Preserve names, product names, programming identifiers, dates, numbers, and
technical terms. Do not add explanations or facts that are not in the input.
"""

ROWS_SYSTEM = """Translate the text field of each memory row into natural Simplified Chinese.
Return ONLY a JSON object with this exact shape:
{\"rows\":[{\"id\":\"same id\",\"text\":\"translated text\"}]}
Return one item for every input row, preserve IDs exactly and preserve order.
Do not translate IDs, dates, numbers, names, product names, or technical terms.
Do not add facts that are not in the source text.
"""


def _extract_json(text: str) -> dict[str, Any]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("model response did not contain a JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model response was not an object")
    return value


def _looks_garbled(text: str) -> bool:
    if "�" in text or "����" in text:
        return True
    question_marks = text.count("?") + text.count("？")
    return bool(text) and question_marks / max(len(text), 1) > 0.25


def _call(api_key: str, system: str, prompt: str, session: str, max_tokens: int) -> dict[str, Any]:
    result = chat(
        api_key,
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=max_tokens,
        session=session,
    )
    content = result["choices"][0]["message"].get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("translation response had no content")
    return _extract_json(content)


def _cache_key(kind: str, text: str) -> str:
    return f"{kind}:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def _translate_text(api_key: str, text: str, session: str, cache: dict[str, str]) -> str:
    key = _cache_key("text", text)
    if key in cache:
        return cache[key]
    result = _call(api_key, TEXT_SYSTEM, json.dumps({"text": text}, ensure_ascii=False), session, 500)
    translated = str(result.get("text") or "").strip()
    if not translated or _looks_garbled(translated):
        raise ValueError("translation response was empty or garbled")
    cache[key] = translated
    return translated


def _row_text(row: dict[str, Any]) -> str:
    if row.get("kind") == "fact":
        return str(row.get("content") or "")
    return str(row.get("summary") or "")


def _translate_rows(api_key: str, rows: list[dict[str, Any]], session: str, cache: dict[str, str]) -> list[str]:
    output: list[str] = []
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0
    for row in rows:
        row_chars = len(_row_text(row)) + 80
        if current and current_chars + row_chars > MAX_CHUNK_CHARS:
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(row)
        current_chars += row_chars
    if current:
        chunks.append(current)

    for chunk_index, chunk in enumerate(chunks):
        pending = [{"id": row.get("id"), "text": _row_text(row)} for row in chunk]
        cache_keys = [_cache_key("row", str(item["text"])) for item in pending]
        if all(key in cache for key in cache_keys):
            output.extend(cache[key] for key in cache_keys)
            continue
        result = _call(api_key, ROWS_SYSTEM, json.dumps({"rows": pending}, ensure_ascii=False), f"{session}-rows-{chunk_index}", 5000)
        translated_rows = result.get("rows")
        if not isinstance(translated_rows, list) or len(translated_rows) != len(pending):
            raise ValueError("translated row count did not match input")
        by_id = {str(item.get("id")): str(item.get("text") or "").strip() for item in translated_rows}
        for item, key in zip(pending, cache_keys):
            translated = by_id.get(str(item["id"]), "")
            if not translated or _looks_garbled(translated):
                raise ValueError("translated memory row was empty or garbled")
            cache[key] = translated
            output.append(translated)
    return output


def _translate_row(api_key: str, index: int, row: dict[str, Any], cache: dict[str, str], id_suffix: str) -> tuple[int, dict[str, Any], bool]:
    translated = json.loads(json.dumps(row, ensure_ascii=False))
    try:
        id_mapping = {
            str(memory.get("id")): f"{memory.get('id')}{id_suffix}"
            for memory in translated.get("memory_rows") or []
        }
        for memory in translated.get("memory_rows") or []:
            memory["id"] = id_mapping.get(str(memory.get("id")), memory.get("id"))
        translated["id"] = f"{translated.get('id')}{id_suffix}"
        translated["evidence_ids"] = [
            id_mapping.get(str(value), f"{value}{id_suffix}")
            for value in translated.get("evidence_ids") or []
        ]
        translated["user_message"] = _translate_text(api_key, str(row["user_message"]), f"translate-{index}-question", cache)
        memory_rows = translated.get("memory_rows") or []
        translated_texts = _translate_rows(api_key, memory_rows, f"translate-{index}", cache)
        for memory, text in zip(memory_rows, translated_texts):
            if memory.get("kind") == "fact":
                memory["content"] = text
            else:
                memory["summary"] = text
        target = translated.get("gate_target") or {}
        target["query"] = "" if target.get("retrieve") else ""
        target["reason"] = "需要查询历史记忆" if target.get("retrieve") else "当前信息已足够"
        translated["gate_target"] = target
        translated["language"] = "zh"
        translated["translation"] = {"provider": "openrouter", "model": MODEL, "fallback": False}
        return index, translated, False
    except (OSError, ValueError, KeyError, IndexError, http.client.HTTPException):
        translated["translation"] = {"provider": "openrouter", "model": MODEL, "fallback": True}
        return index, translated, True


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


def translate(input_path: Path, output_path: Path, cache_path: Path, id_suffix: str = "") -> dict[str, int]:
    api_key = get_api_key()
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    results: list[tuple[int, dict[str, Any], bool]] = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_translate_row, api_key, index, row, cache, id_suffix) for index, row in enumerate(rows)]
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if result[2]:
                index, _, _ = result
                result = _translate_row(
                    api_key,
                    index,
                    rows[index],
                    cache,
                    id_suffix,
                )
            results.append(result)
            _save_cache(cache_path, dict(cache))
            percent = completed / max(len(futures), 1) * 100
            sys.stdout.write(f"\r[translation] {completed}/{len(futures)} ({percent:5.1f}%)")
            sys.stdout.flush()
    if futures:
        print()
    results.sort(key=lambda value: value[0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for _, row, _ in results), encoding="utf-8")
    _save_cache(cache_path, cache)
    fallbacks = sum(fallback for _, _, fallback in results)
    return {"total": len(rows), "translated": len(rows) - fallbacks, "fallback": fallbacks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cache", type=Path, default=ROOT / "data/derived/translation_cache.json")
    parser.add_argument("--id-suffix", default="")
    args = parser.parse_args()
    print(json.dumps(translate(args.input, args.output, args.cache, args.id_suffix), ensure_ascii=False))


if __name__ == "__main__":
    main()

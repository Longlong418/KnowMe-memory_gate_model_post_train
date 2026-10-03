"""Use an OpenCode Go model to rewrite query/reason labels.

The benchmark-derived retrieve/evidence labels remain authoritative. The remote
model only proposes a single query string and a short display reason; every
proposal is still checked by the local FTS5/BM25 validator afterwards.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from src.dataset.openrouter import MODEL, chat, get_api_key


ROOT = Path(__file__).resolve().parents[2]

SYSTEM_PROMPT = """You create labels for a memory retrieval gate.
Return ONLY one valid JSON object with exactly these keys:
{"retrieve": true or false, "query": "...", "reason": "..."}
The retrieve value is fixed by the caller; do not change it.
If retrieve is true, query must be one string, never an array, with 2-6 high-signal keywords.
Use ASCII spaces between Chinese search terms. Do not copy the full user message.
If retrieve is false, query must be an empty string.
Reason must be one short Chinese sentence, no more than 15 Chinese characters.
"""


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("model response did not contain a JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model response was not an object")
    return value


def call_model(api_key: str, user_message: str, retrieve: bool, session: str) -> dict[str, Any]:
    prompt = f"Gold retrieve label: {str(retrieve).lower()}\nUser message: {user_message}"
    result = chat(
        api_key,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=180,
        session=session,
    )
    content = result["choices"][0]["message"]["content"]
    annotation = _extract_json(content)
    if annotation.get("retrieve") is not retrieve:
        annotation["retrieve"] = retrieve
    query = annotation.get("query", "")
    if isinstance(query, list):
        query = " ".join(str(item) for item in query)
    annotation["query"] = str(query or "").strip() if retrieve else ""
    # The Go proxy is reliable for English query terms but can return garbled
    # Chinese text. Keep the user-visible reason deterministic and Chinese.
    annotation["reason"] = "需要查询历史记忆" if retrieve else "当前信息已足够"
    return annotation


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def annotate(input_path: Path, output_path: Path, *, sleep_seconds: float = 0.0) -> dict[str, int]:
    api_key = get_api_key()
    rows = _read_jsonl(input_path)
    def process(index: int, original: dict[str, Any]) -> tuple[int, dict[str, Any], bool]:
        row = dict(original)
        target = row.get("gate_target") or {}
        retrieve = bool(target.get("retrieve"))
        try:
            generated = call_model(api_key, str(row["user_message"]), retrieve, f"knowme-{input_path.stem}-{index}")
        except (OSError, ValueError, KeyError, urllib.error.HTTPError) as error:
            generated = {"retrieve": retrieve, "query": str(target.get("query") or "") if retrieve else "", "reason": "需要查询历史记忆" if retrieve else "当前信息已足够"}
            row["annotation_error"] = type(error).__name__
            return index, row, True
        row["gate_target"] = generated
        row["annotation"] = {"provider": "openrouter", "model": MODEL, "fallback": False}
        return index, row, False

    results: list[tuple[int, dict[str, Any], bool]] = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process, index, row) for index, row in enumerate(rows)]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda value: value[0])
    annotated = [value[1] for value in results]
    fallbacks = sum(value[2] for value in results)
    _write_jsonl(output_path, annotated)
    return {"total": len(rows), "annotated": len(rows) - fallbacks, "fallback": fallbacks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()
    print(json.dumps(annotate(args.input, args.output, sleep_seconds=args.sleep), ensure_ascii=False))


if __name__ == "__main__":
    main()

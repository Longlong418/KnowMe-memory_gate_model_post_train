"""Ask a teacher model for multiple query candidates without exposing memory."""

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

SYSTEM_PROMPT = """You create search query candidates for a personal-memory retrieval gate.
The user message is the only source of information. Never assume or invent
facts from memory. Return ONLY JSON with this shape:
{\"queries\":[\"candidate 1\",\"candidate 2\",\"candidate 3\"]}
For a retrieve=false example return {\"queries\":[]}.
For retrieve=true return 3-5 different single-string queries. Each query must
contain 2-6 high-signal keywords separated by ASCII spaces. For Chinese, use
ASCII spaces between search terms. Never return arrays as a query value.
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


def call_model(api_key: str, user_message: str, retrieve: bool, session: str) -> list[str]:
    prompt = f"Gold retrieve label: {str(retrieve).lower()}\nUser message: {user_message}"
    result = chat(
        api_key,
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=320,
        session=session,
    )
    value = _extract_json(result["choices"][0]["message"]["content"])
    queries = value.get("queries", [])
    if not isinstance(queries, list):
        raise ValueError("queries was not a list")
    return list(dict.fromkeys(str(query).strip() for query in queries if str(query).strip()))[:5]


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _cache_key(row: dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "id": row.get("id"),
            "user_message": row.get("user_message"),
            "retrieve": bool((row.get("gate_target") or {}).get("retrieve")),
            "model": CACHE_MODEL,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _save_cache(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    last_error: PermissionError | None = None
    for _ in range(5):
        try:
            temporary.replace(path)
            return
        except PermissionError as error:
            last_error = error
            time.sleep(0.25)

    # Some Windows editors/antivirus tools keep the target open briefly. Fall
    # back to a direct write after the atomic replace retries are exhausted.
    try:
        path.write_text(temporary.read_text(encoding="utf-8"), encoding="utf-8")
        temporary.unlink(missing_ok=True)
    except PermissionError:
        raise last_error or PermissionError(f"cannot write cache: {path}")


def generate(input_path: Path, output_path: Path, cache_path: Path) -> dict[str, int]:
    api_key = get_api_key()
    rows = _read(input_path)
    cache = _load_cache(cache_path)

    def process(index: int, original: dict[str, Any]) -> tuple[int, dict[str, Any], bool, str, dict[str, Any] | None]:
        row = dict(original)
        target = row.get("gate_target") or {}
        retrieve = bool(target.get("retrieve"))
        key = _cache_key(row)
        cached = cache.get(key)
        if cached is not None:
            row["query_candidates"] = cached.get("queries", [])
            row["candidate_generation"] = {
                "provider": "openrouter",
                "model": MODEL,
                "fallback": False,
                "cached": True,
            }
            return index, row, False, key, None
        queries: list[str] = []
        fallback = True
        for attempt in range(2):
            try:
                queries = call_model(
                    api_key,
                    str(row["user_message"]),
                    retrieve,
                    f"query-candidates-{input_path.stem}-{index}-attempt-{attempt + 1}",
                )
                fallback = False
                break
            except Exception:
                if attempt == 0:
                    time.sleep(0.5)
        if fallback:
            queries = [str(target.get("query") or "")] if retrieve else []
        row["query_candidates"] = queries
        row["candidate_generation"] = {"provider": "openrouter", "model": MODEL, "fallback": fallback}
        cache_update = {"queries": queries} if not fallback else None
        return index, row, fallback, key, cache_update

    results: list[tuple[int, dict[str, Any], bool]] = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process, index, row) for index, row in enumerate(rows)]
        for completed, future in enumerate(as_completed(futures), start=1):
            index, row, fallback, key, cache_update = future.result()
            results.append((index, row, fallback))
            if cache_update is not None:
                cache[key] = cache_update
                _save_cache(cache_path, cache)
            percent = completed / max(len(futures), 1) * 100
            sys.stdout.write(f"\r[query-candidates] {completed}/{len(futures)} ({percent:5.1f}%)")
            sys.stdout.flush()
    if futures:
        print()
    results.sort(key=lambda value: value[0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for _, row, _ in results), encoding="utf-8")
    fallbacks = sum(value[2] for value in results)
    return {"total": len(rows), "generated": len(rows) - fallbacks, "fallback": fallbacks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cache", type=Path, default=ROOT / "data/derived/query_candidate_cache.json")
    args = parser.parse_args()
    print(json.dumps(generate(args.input, args.output, args.cache), ensure_ascii=False))


if __name__ == "__main__":
    main()

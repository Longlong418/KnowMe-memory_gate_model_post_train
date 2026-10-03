"""Audit retrieve=false samples with an independent teacher model."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from src.dataset.openrouter import MODEL, chat, get_api_key


ROOT = Path(__file__).resolve().parents[2]

SYSTEM_PROMPT = """You audit a training example for a long-term memory retrieval gate.
You receive only one candidate user message. Decide whether the message is
self-contained and answerable from its own text plus ordinary general knowledge.

KEEP when every fact needed to answer is present in the message, or the task is
ordinary general knowledge, arithmetic, explanation, or advice.
REJECT when answering requires an unstated private fact, the user's location,
preferences, account, past conversation, an unresolved pronoun/reference, or
an exact fact that must be retrieved from the user's history.
Examples of REJECT: "restaurants within 15 minutes of my apartment" without an
address; "what did I tell you about..."; "which city was in that photo?" when
the photo is not supplied.

Return ONLY JSON:
{"verdict":"keep" or "reject", "confidence":0.0, "reason":"short reason"}
"""


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("audit response did not contain JSON")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("audit response was not an object")
    return value


def _cache_key(row: dict[str, Any]) -> str:
    value = json.dumps(
        {"id": row.get("id"), "user_message": row.get("user_message"), "model": MODEL},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _call(key: str, message: str, session: str) -> dict[str, Any]:
    result = chat(
        key,
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": message}],
        temperature=0.0,
        max_tokens=300,
        session=session,
        retries=2,
    )
    content = result["choices"][0]["message"].get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("audit response had no content")
    value = _extract_json(content)
    verdict = str(value.get("verdict") or "").strip().lower()
    confidence = float(value.get("confidence", 0.0))
    if verdict not in {"keep", "reject"}:
        raise ValueError("audit verdict must be keep or reject")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("audit confidence must be between 0 and 1")
    return {"verdict": verdict, "confidence": confidence, "reason": str(value.get("reason") or "").strip()}


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def audit(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    cache_path: Path,
    workers: int = 4,
    limit: int | None = None,
    min_confidence: float = 0.8,
) -> dict[str, int]:
    key = get_api_key()
    rows = _read(input_path)
    positives = [row for row in rows if bool((row.get("gate_target") or {}).get("retrieve"))]
    negatives = [row for row in rows if not bool((row.get("gate_target") or {}).get("retrieve"))]
    if limit is not None and limit > 0:
        negatives = negatives[:limit]
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    pending = [row for row in negatives if _cache_key(row) not in cache]

    def process(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        cache_key = _cache_key(row)
        if cache_key in cache:
            decision = cache[cache_key]
        else:
            try:
                decision = _call(key, str(row.get("user_message") or ""), f"negative-audit-{row.get('id')}")
            except Exception as error:
                decision = {"verdict": "uncertain", "confidence": 0.0, "reason": f"{type(error).__name__}: {error}"}
        return cache_key, decision

    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(process, row) for row in pending]
        for future in as_completed(futures):
            cache_key, decision = future.result()
            cache[cache_key] = decision
            completed += 1
            if completed % 10 == 0 or completed == len(futures):
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
                sys.stdout.write(f"\r[audit] {completed}/{len(futures)}")
                sys.stdout.flush()
    if pending:
        print()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    audited: list[dict[str, Any]] = []
    report: list[dict[str, Any]] = []
    for row in negatives:
        decision = cache[_cache_key(row)]
        keep = decision.get("verdict") == "keep" and float(decision.get("confidence", 0.0)) >= min_confidence
        report.append({"id": row.get("id"), "user_message": row.get("user_message"), **decision, "accepted": keep})
        if keep:
            audited.append(row)

    final_rows = positives + audited
    _write(output_path, final_rows)
    _write(report_path, report)
    return {
        "input_rows": len(rows),
        "positive_rows": len(positives),
        "negative_rows_audited": len(negatives),
        "negative_kept": len(audited),
        "negative_rejected_or_uncertain": len(negatives) - len(audited),
        "api_requests": len(pending),
        "output_rows": len(final_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=ROOT / "data/derived/negative_audit_cache.json")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-confidence", type=float, default=0.8)
    args = parser.parse_args()
    if not 0.0 <= args.min_confidence <= 1.0:
        parser.error("--min-confidence must be between 0 and 1")
    print(json.dumps(audit(args.input, args.output, args.report, args.cache, args.workers, args.limit or None, args.min_confidence), ensure_ascii=False))


if __name__ == "__main__":
    main()

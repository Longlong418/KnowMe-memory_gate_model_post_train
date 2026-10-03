"""Evaluate an OpenAI-compatible cloud model on the gate dev/test split.

The gate is compared against hosted models, which cannot go through
``swift infer``.  This calls the endpoint directly, writes the same
``predictions.jsonl`` shape the local path produces, and then hands the result
to ``eval.run_gate_eval`` -- so these numbers come out of exactly the same code
as every other row in ``docs/results.md``.

Three wire-format details are not guessable and each cost a debugging round:

* the endpoint requires an ``x-opencode-session`` header, otherwise the request
  comes back ``{"type":"MissingSessionID"}``;
* it must be sent an explicit ``User-Agent``.  Requests carrying Python's
  default ``Python-urllib/3.x`` are rejected at the edge with HTTP 403 and
  ``error code: 1010``, while curl to the same URL succeeds;
* ``enable_thinking: false`` is *accepted and silently ignored* -- the budget is
  spent on reasoning and ``content`` returns empty.  ``reasoning_effort:
  "none"`` is what actually switches thinking off, and it is what makes the run
  comparable to the local ``--enable_thinking false``.

Scopes: the memory pool is not part of the prompt, so a model's answer does not
depend on it and one API call can be scored under both.  ``hard_haystack`` is
what ``scripts/eval_gate_checkpoint.sh`` defaults to; ``full_memory`` is what
the legacy test numbers in ``docs/results.md`` section 3 were produced under.
Both metrics files are written, so the user picks the comparison deliberately
rather than by accident.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from eval.run_gate_eval import _extract_json, evaluate, read_jsonl  # noqa: E402

DEFAULT_ENDPOINT = "https://opencode.ai/zen/go/v1/chat/completions"
DEFAULT_MODELS = ("mimo-v2.6-flash", "deepseek-v4.1-flash")
# `.env` in this repo holds `opencode_api_key`; the upper-case spelling is
# accepted so an exported variable works too.
API_KEY_NAMES = ("OPENCODE_API_KEY", "opencode_api_key")
# Any explicit User-Agent gets past the edge filter; this one also shows up in
# the gateway's logs as us rather than as "some scraper".
USER_AGENT = "knowme-memory-gate-eval/0.1"
SESSION = "knowme-memory-gate-api-eval"
SCOPES = ("hard_haystack", "full_memory")


def _load_dotenv() -> None:
    """Read the repo-root ``.env`` into the environment, without overwriting."""
    path = REPO_DIR / ".env"
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip().strip("\"'")
        if name and value and name not in os.environ:
            os.environ[name] = value


def _api_key(explicit: str | None) -> str:
    """The key itself is never printed or written to an artifact."""
    if explicit:
        return explicit
    _load_dotenv()
    for name in API_KEY_NAMES:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    raise SystemExit(
        f"No API key. Put one of {', '.join(API_KEY_NAMES)} in {REPO_DIR / '.env'}, "
        "or export it, or pass --api-key-env NAME."
    )


def _prompt_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    """The conversation to send.

    The split files keep the gold assistant turn because SFT trains on it, so it
    has to be stripped here -- sending it would hand the model the answer and
    score a meaningless 100%.  This mirrors what ``swift infer`` does with the
    same dataset, which generates from the last non-assistant turn.
    """
    messages = [m for m in (row.get("messages") or []) if isinstance(m, dict)]
    while messages and messages[-1].get("role") == "assistant":
        messages.pop()
    if not messages:
        raise ValueError(f"row {row.get('id')!r} has no prompt to send")
    return [
        {"role": str(m.get("role") or "user"), "content": str(m.get("content") or "")}
        for m in messages
    ]


def _post(
    endpoint: str,
    key: str,
    payload: dict[str, Any],
    *,
    timeout: float,
    attempts: int,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
        "x-opencode-session": SESSION,
    }
    last_error: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read()[:300].decode("utf-8", "replace")
            # A 4xx other than 429 means the request itself is wrong (bad model
            # id, bad parameter, missing header) -- retrying cannot help and
            # would just burn the plan's quota.
            if error.code != 429 and error.code < 500:
                raise RuntimeError(f"HTTP {error.code}: {detail}") from error
            last_error = RuntimeError(f"HTTP {error.code}: {detail}")
        except (OSError, http.client.HTTPException) as error:
            last_error = error
        if attempt + 1 < attempts:
            time.sleep(min(2.0**attempt, 20.0))
    raise last_error or RuntimeError("request failed")


def _call_model(
    row: dict[str, Any], model: str, args: argparse.Namespace, key: str
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stream": False,
        "messages": _prompt_messages(row),
    }
    if args.thinking == "off":
        payload["reasoning_effort"] = "none"

    data = _post(
        args.endpoint, key, payload, timeout=args.timeout, attempts=args.attempts
    )
    choices = data.get("choices") or [{}]
    message = choices[0].get("message") or {}
    content = message.get("content")
    return {
        "id": row["id"],
        "raw_output": content if isinstance(content, str) else "",
        "finish_reason": choices[0].get("finish_reason"),
        "usage": data.get("usage") or {},
    }


def _read_cache(path: Path) -> dict[str, dict[str, Any]]:
    """Completed rows from an earlier run.

    These calls are billed, so a re-run must not pay for them again: whatever is
    already on disk is reused, and only missing or failed rows are requested.
    """
    if not path.exists():
        return {}
    done: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(path):
        if record.get("id") and not record.get("error"):
            done[str(record["id"])] = record
    return done


def _collect(
    rows: list[dict[str, Any]], model: str, args: argparse.Namespace, key: str, cache: Path
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    done = _read_cache(cache)
    pending = [row for row in rows if str(row["id"]) not in done]
    if done:
        print(f"  {model}: {len(done)} rows already cached, {len(pending)} to request")

    failures: list[dict[str, str]] = []
    completed: list[dict[str, Any]] = list(done.values())

    def run(row: dict[str, Any]) -> dict[str, Any]:
        try:
            return _call_model(row, model, args, key)
        except Exception as error:  # noqa: BLE001 -- recorded per row, reported below
            return {"id": row["id"], "error": f"{type(error).__name__}: {error}"}

    if pending:
        # cache is appended as results arrive, so an interrupted run keeps them
        cache.parent.mkdir(parents=True, exist_ok=True)
        with (
            cache.open("a", encoding="utf-8") as sink,
            concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool,
        ):
            for index, record in enumerate(pool.map(run, pending), start=1):
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                sink.flush()
                if record.get("error"):
                    failures.append({"id": str(record["id"]), "error": str(record["error"])})
                else:
                    completed.append(record)
                if index % 25 == 0 or index == len(pending):
                    print(f"  {model}: {index}/{len(pending)}", flush=True)

    by_id = {str(record["id"]): record for record in completed}
    ordered = [by_id[str(row["id"])] for row in rows if str(row["id"]) in by_id]
    return ordered, failures


def _usage_total(records: list[dict[str, Any]]) -> dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for record in records:
        usage = record.get("usage") or {}
        for field in totals:
            value = usage.get(field)
            if isinstance(value, int):
                totals[field] += value
    return totals


def _slug(model: str) -> str:
    return model.replace("/", "_")


def _summary_line(label: str, metrics: dict[str, Any]) -> str:
    classification = metrics.get("classification") or {}
    output = metrics.get("output") or {}
    end_to_end = metrics.get("end_to_end") or {}
    branch = (metrics.get("retrieval") or {}).get("branch") or {}

    def fmt(value: Any) -> str:
        return f"{value:.4f}" if isinstance(value, (int, float)) else "--"

    return (
        f"| {label} | {fmt(classification.get('macro_f1'))} "
        f"| {fmt(classification.get('positive_recall'))} "
        f"| {fmt(classification.get('unnecessary_retrieval_rate'))} "
        f"| {fmt(output.get('format_valid_rate'))} "
        f"| {fmt(branch.get('conditional_mrr'))} "
        f"| {fmt(end_to_end.get('memory_recall'))} |"
    )


def _haystack_path(split: str, requested: str) -> Path | None:
    if requested.lower() in ("", "none", "off"):
        return None
    if requested != "auto":
        return Path(requested).resolve()
    return REPO_DIR / f"data/grpo_hard/gate_grpo_{split}.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", default="test", choices=("dev", "test"))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--endpoint", default=os.environ.get("OPENCODE_ENDPOINT", DEFAULT_ENDPOINT))
    parser.add_argument(
        "--haystack",
        default="auto",
        help="hard-haystack file, 'auto' for the conventional per-split path, "
        "'none' to evaluate on full memory only",
    )
    parser.add_argument("--result-root", type=Path, default=REPO_DIR / "eval/results")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="matches the local evaluator's --max_new_tokens 128; ample with thinking off",
    )
    parser.add_argument(
        "--thinking",
        choices=("off", "default"),
        default="off",
        help="'off' sends reasoning_effort=none, matching the local --enable_thinking false",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--limit", type=int, help="only the first N rows, for a cheap smoke test")
    parser.add_argument("--api-key", help="prefer the .env file; this exists for CI")
    parser.add_argument("--api-key-env", help="name of the environment variable holding the key")
    args = parser.parse_args()

    split_path = REPO_DIR / f"data/train_split/gate_{args.split}.jsonl"
    full_path = REPO_DIR / "data/gate_final_2855.full.jsonl"
    if not split_path.is_file() or not full_path.is_file():
        raise SystemExit(f"Evaluation data not found: {split_path} / {full_path}")

    haystack = _haystack_path(args.split, args.haystack)
    if haystack is not None and not haystack.is_file():
        print(f"Warning: haystack not found ({haystack}); full memory only.", file=sys.stderr)
        haystack = None

    key = _api_key(args.api_key or (os.environ.get(args.api_key_env) if args.api_key_env else None))
    rows = read_jsonl(split_path)
    if args.limit:
        rows = rows[: args.limit]
    if haystack is not None:
        known = {str(row["id"]) for row in read_jsonl(haystack)}
        missing = [str(row["id"]) for row in rows if str(row["id"]) not in known]
        if missing:
            raise SystemExit(f"haystack is missing {len(missing)} split rows, e.g. {missing[0]}")

    # hard_haystack is the primary scope because it is what the local eval
    # scripts default to; full_memory is kept alongside for the legacy table.
    primary = "hard_haystack" if haystack is not None else "full_memory"
    print(
        f"Evaluating {len(rows)} {args.split} rows on {len(args.models)} model(s)\n"
        f"  endpoint : {args.endpoint}\n"
        f"  scope    : {primary} (primary)"
        + ("" if haystack is None else f"  <- {haystack.name}")
    )

    report: dict[str, Any] = {
        "endpoint": args.endpoint,
        "split": args.split,
        "rows": len(rows),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "thinking": args.thinking,
        "primary_scope": primary,
        "haystack": str(haystack) if haystack else None,
        "models": {},
    }
    lines: list[str] = []

    for model in args.models:
        print(f"\n== {model}")
        # A limited run is a smoke test, not a result: it is scored against the
        # *whole* split (missing rows count as format failures), so it must never
        # land under the name a real run would use.
        suffix = f"_limit{args.limit}" if args.limit else ""
        result_dir = args.result_root / f"api_{_slug(model)}_{args.split}{suffix}"
        result_dir.mkdir(parents=True, exist_ok=True)
        cache = result_dir / "raw.jsonl"

        records, failures = _collect(rows, model, args, key, cache)
        predictions = result_dir / "predictions.jsonl"
        with predictions.open("w", encoding="utf-8") as sink:
            for record in records:
                content = record.get("raw_output") or ""
                sink.write(
                    json.dumps(
                        {
                            "id": record["id"],
                            "raw_output": content,
                            "prediction": _extract_json(content),
                            "finish_reason": record.get("finish_reason"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        scopes: dict[str, Any] = {}
        for scope in SCOPES:
            if scope == "hard_haystack" and haystack is None:
                continue
            metrics = evaluate(
                split_path, full_path, predictions, haystack if scope == "hard_haystack" else None
            )
            # One file per scope, and never a scope-ambiguous "metrics.json": a
            # later run with a different --haystack must not silently replace the
            # other scope's numbers.  hard_haystack additionally gets the plain
            # name, which is what eval_gate_checkpoint.sh writes -- so metrics.json
            # means the same thing in every directory under eval/results/.
            payload = json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
            (result_dir / f"metrics.{scope}.json").write_text(payload, encoding="utf-8")
            if scope == "hard_haystack":
                (result_dir / "metrics.json").write_text(payload, encoding="utf-8")
            scopes[scope] = metrics
            lines.append(_summary_line(f"{model} ({scope})", metrics))

        tokens = _usage_total(records)
        report["models"][model] = {
            "predictions": len(records),
            "failures": failures,
            "tokens": tokens,
            "result_dir": str(result_dir),
            "finish_reasons": {
                reason: sum(1 for r in records if r.get("finish_reason") == reason)
                for reason in sorted({r.get("finish_reason") for r in records})
            },
            "metrics": {scope: str(result_dir / f"metrics.{scope}.json") for scope in scopes},
        }
        print(f"  predictions -> {predictions}")
        print(f"  tokens      -> {tokens['total_tokens']} total")
        for scope, metrics in scopes.items():
            print("  " + _summary_line(f"{scope}", metrics).strip("| ").replace(" | ", "  "))

    suffix = "_limit" + str(args.limit) if args.limit else ""
    (args.result_root / f"api_models_{args.split}{suffix}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if args.limit:
        print(
            f"\nNOTE: --limit {args.limit} only requested {args.limit} rows, but the metrics "
            "above were computed against all rows in the split (the rest score as format "
            "failures). Smoke test only -- do not quote these numbers."
        )

    print(
        "\n| model (scope) | macro_f1 | positive_recall | unnecessary_retrieval | "
        "format_valid | branch cMRR | memory_recall |"
    )
    print("|---|---|---|---|---|---|---|")
    for line in lines:
        print(line)

    failed = {
        model: entry["failures"]
        for model, entry in report["models"].items()
        if entry["failures"]
    }
    if failed:
        print("\nSome rows failed; artifacts are written and the cache keeps what succeeded.")
        for model, failures in failed.items():
            print(f"  {model}: {len(failures)} failed, e.g. {failures[0]['error'][:160]}")
        print("  Re-run the same command to retry only those rows.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

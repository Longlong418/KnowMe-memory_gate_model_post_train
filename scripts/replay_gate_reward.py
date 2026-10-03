"""Replay the GRPO gate reward against a saved eval run, offline.

Before spending GPU time on a GRPO run, this checks that the reward and the
evaluator agree on the *same* completions. It joins a predictions file to the
hard-haystack split exactly the way ``eval.run_gate_eval.evaluate`` does
(predictions carry no ``id``, so the join is positional), scores every row with
``gate_reward_v2``, and compares the resulting reward histogram against the
``retrieval.branch`` numbers already published in that run's ``metrics.json``.

The reward and the evaluator share ``_evidence_rank``, so agreement on the rank
itself is by construction. What this actually catches is the *wiring*: the wrong
column, the wrong join order, the wrong ``language`` (which decides the zh
format rule), and the ``INVALID`` / ``DECLINED`` / ``MISS`` boundaries. Those are
the ways a reward silently diverges from the metrics it is supposed to optimize.

    .venv/bin/python scripts/replay_gate_reward.py
    .venv/bin/python scripts/replay_gate_reward.py --predictions <run>/predictions.jsonl \
        --metrics <run>/metrics.json --haystack data/grpo_hard/gate_grpo_dev.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from eval.run_gate_eval import (  # noqa: E402
    _evidence_rank,
    _extract_json,
    _retrieved_branches,
    output_is_valid,
    read_jsonl,
)
from scripts.grpo_gate_reward import DECLINED, MISS, GateRewardV2  # noqa: E402

DEFAULT_PREDICTIONS = REPO_DIR / "eval/results/gate_sft_best_checkpoint_dev/predictions.jsonl"
DEFAULT_METRICS = REPO_DIR / "eval/results/gate_sft_best_checkpoint_dev/metrics.json"
DEFAULT_SPLIT = REPO_DIR / "data/train_split/gate_dev.jsonl"
DEFAULT_HAYSTACK = REPO_DIR / "data/grpo_hard/gate_grpo_dev.jsonl"


def _completion(prediction: dict[str, Any]) -> str:
    response = prediction.get("response")
    if isinstance(response, str):
        return response
    for message in reversed(prediction.get("messages") or []):
        if message.get("role") == "assistant":
            return message.get("content") or ""
    return ""


def join_rows(
    split_path: Path, haystack_path: Path, predictions_path: Path
) -> list[dict[str, Any]]:
    """Reproduce ``evaluate()``'s positional join, against the hard haystack."""
    split_rows = read_jsonl(split_path)
    haystack_by_id = {row["id"]: row for row in read_jsonl(haystack_path)}
    prediction_rows = read_jsonl(predictions_path)
    if len(prediction_rows) != len(split_rows):
        raise SystemExit(
            f"predictions ({len(prediction_rows)}) != split ({len(split_rows)}): "
            "a positional join would be wrong"
        )
    joined = []
    for index, split_row in enumerate(split_rows):
        haystack_row = haystack_by_id.get(split_row["id"])
        if haystack_row is None:
            raise SystemExit(f"haystack id missing for split row: {split_row['id']}")
        if split_row.get("language") != haystack_row.get("language"):
            raise SystemExit(
                f"language disagrees for {split_row['id']}: "
                f"{split_row.get('language')!r} vs {haystack_row.get('language')!r}"
            )
        joined.append(
            {
                "id": split_row["id"],
                "language": split_row["language"],
                "gate_label": bool(haystack_row["gate_label"]),
                "memory_rows_json": haystack_row["memory_rows_json"],
                "evidence_ids_json": haystack_row["evidence_ids_json"],
                "completion": _completion(prediction_rows[index]),
            }
        )
    return joined


def score(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reward = GateRewardV2()
    values = reward(
        [row["completion"] for row in rows],
        gate_label=[row["gate_label"] for row in rows],
        memory_rows_json=[row["memory_rows_json"] for row in rows],
        evidence_ids_json=[row["evidence_ids_json"] for row in rows],
        language=[row["language"] for row in rows],
        trainer_state=None,  # injected by Swift; must be absorbed
    )
    if len(values) != len(rows):
        raise SystemExit(f"reward returned {len(values)} values for {len(rows)} rows")

    scored = []
    for row, value in zip(rows, values):
        memories = json.loads(row["memory_rows_json"])
        evidence = {str(v) for v in json.loads(row["evidence_ids_json"])}
        prediction = _extract_json(row["completion"])
        rank = None
        if (
            row["gate_label"]
            and isinstance(prediction, dict)
            and prediction.get("retrieve") is True
            and output_is_valid(prediction, row["language"])
        ):
            fact_ids, episode_ids = _retrieved_branches(memories, prediction["query"])
            rank = _evidence_rank(fact_ids, episode_ids, evidence)
        scored.append({**row, "reward": value, "rank": rank})
    return scored


def expected_from_metrics(metrics: dict[str, Any]) -> dict[float, int]:
    """Reward histogram implied by the published evaluator numbers."""
    branch = metrics["retrieval"]["branch"]
    total = branch["positive_total"]
    at = {k: round(branch[f"hit_at_{k}"] * total) for k in (1, 2, 3, 4)}
    confusion = metrics["confusion"]
    # Every "-1" row, by the reward's own decision order (format check first):
    #   * positive rows that did not retrieve          -> fn
    #   * positive rows that retrieved but are format-invalid
    #     -> tp minus the rows the evaluator counted as candidates
    #   * negative rows that retrieved (valid or not)  -> fp
    # The invalid count is deliberately not used on its own: a format-invalid
    # *negative* row that retrieved is already inside fp, and adding the whole
    # invalid count would double-count it.
    penalised = confusion["fn"] + (confusion["tp"] - branch["query_candidates"])
    penalised += confusion["fp"]
    return {
        1.0: at[1] + confusion["tn"],
        0.5: at[2] - at[1],
        round(1 / 3, 6): at[3] - at[2],
        0.25: at[4] - at[3],  # reachable: facts have a rank-4 cutoff
        MISS: branch["query_candidates"] - branch["hits"],
        DECLINED: penalised,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--haystack", type=Path, default=DEFAULT_HAYSTACK)
    args = parser.parse_args()

    rows = join_rows(args.split, args.haystack, args.predictions)
    scored = score(rows)
    measured = Counter(round(row["reward"], 6) for row in scored)
    expected = Counter({round(k, 6): v for k, v in expected_from_metrics(
        json.loads(args.metrics.read_text(encoding="utf-8"))
    ).items()})

    # Independent per-row check: the reward must equal 1/rank for every row the
    # evaluator could rank.
    mismatched = [
        row
        for row in scored
        if row["rank"] is not None
        and row["gate_label"]
        and abs(row["reward"] - 1.0 / row["rank"]) > 1e-9
    ]

    print(f"{'reward':>8}  {'measured':>8}  {'expected':>8}")
    ok = True
    for value in sorted(set(measured) | set(expected), reverse=True):
        got, want = measured.get(value, 0), expected.get(value, 0)
        flag = "" if got == want else "   <-- MISMATCH"
        ok = ok and got == want
        print(f"{value:>8}  {got:>8}  {want:>8}{flag}")
    total = sum(measured.values())
    mean = sum(row["reward"] for row in scored) / total
    print(f"{'total':>8}  {total:>8}  {sum(expected.values()):>8}")
    print(f"\nmean reward: {mean:+.4f}")
    print(f"rank-path rows disagreeing with 1/rank: {len(mismatched)}")
    for row in mismatched[:5]:
        print(f"  {row['id']}: rank={row['rank']} reward={row['reward']}")

    if not ok or mismatched:
        raise SystemExit("reward and evaluator disagree -- do not launch GRPO yet")
    print("\nOK: reward matches the published metrics.")


if __name__ == "__main__":
    main()

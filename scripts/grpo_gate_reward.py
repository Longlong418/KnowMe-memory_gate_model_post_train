"""Custom ms-swift GRPO reward for the memory retrieval gate (rank-aware v2).

The prompt contains only ``messages``.  Swift forwards every other dataset column
to this reward as a keyword argument whose value is a list aligned with
``completions`` (``swift/rl_core/grpo_algorithm.py``), so this file reads
``gate_label``, ``memory_rows_json``, ``evidence_ids_json`` and ``language``.

Reward scale:

* malformed, or failing the evaluator's format contract: ``-1``
* negative row, ``retrieve=false`` with an empty query:        ``+1``
* negative row, ``retrieve=true``:                              ``-1``
* positive row, ``retrieve=false``:                             ``-1``
* positive row, retrieved, evidence at branch-internal rank r:  ``1/r``
* positive row, retrieved, evidence absent from the top-k:      ``-0.5``

The scale is totally ordered, and that ordering is the design::

    retrieve=false (-1)  <  miss (-0.5)  <  1/4  <  1/3  <  1/2  <  1

``miss`` deliberately sits strictly between "declined to retrieve" and the worst
possible hit.  If a miss cost as much as declining, the cheapest way to raise
reward would be to stop retrieving at all -- which inflates the ``conditional_*``
metrics (unattempted rows drop out of the denominator) while destroying
``positive_recall`` and end-to-end memory recall.

Rank is branch-internal (``_evidence_rank``), matching ``retrieval.branch`` and
therefore the checkpoint-selection metric.  Concatenating the branches first would
score an episode's rank-1 hit as rank 5 whenever the same row also holds facts.
The gold query is never loaded, so the policy is rewarded purely for writing a
query that finds the evidence under the production FTS5/BM25 logic.

The format contract is the evaluator's own ``output_is_valid``, imported rather
than reimplemented, so the reward and the reported metrics cannot drift apart.

``hard_negative_ids_json`` is accepted but unused: it is exactly
``memory_rows - evidence_ids`` (verified over the whole hard split), so every
retrieved non-evidence row is already a hard negative and ``1/r`` already accounts
for how many slots the evidence sat behind.  Scoring them again would double-count.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

try:
    from eval.run_gate_eval import (
        _evidence_rank,
        _extract_json,
        _retrieved_branches,
        output_is_valid,
    )
except ModuleNotFoundError:  # Allow running straight from a checkout.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from eval.run_gate_eval import (
        _evidence_rank,
        _extract_json,
        _retrieved_branches,
        output_is_valid,
    )

try:
    from swift.rewards import ORM, orms
except ImportError:  # Allows local smoke tests without installing ms-swift.
    class ORM:  # type: ignore[no-redef]
        pass

    orms: dict[str, type] = {}


# The ladder, named so the ordering invariant is auditable in one place.
DECLINED = -1.0  # positive row, gate said "no retrieval" -- the costly miss
INVALID = -1.0  # malformed, or failing output_is_valid
NEGATIVE_RETRIEVE = -1.0  # negative row, gate said "retrieve" -- wasted work
MISS = -0.5  # retrieved, but the evidence never came back
NEGATIVE_IDLE = 1.0  # negative row, correctly left alone


def _as_list(value: Any, size: int) -> list[Any]:
    if isinstance(value, (list, tuple)):
        values = list(value)
        if len(values) == size:
            return values
    return [value] * size


def _text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        if isinstance(completion.get("content"), str):
            return completion["content"]
        if isinstance(completion.get("text"), str):
            return completion["text"]
    if isinstance(completion, list):
        for item in reversed(completion):
            if isinstance(item, dict) and isinstance(item.get("content"), str):
                return item["content"]
    return str(completion)


def _parse_json_column(value: Any, default: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value if value is not None else default


class GateRewardV2(ORM):
    def __call__(
        self,
        completions: Iterable[Any],
        gate_label: Any = None,
        memory_rows_json: Any = None,
        evidence_ids_json: Any = None,
        language: Any = None,
        hard_negative_ids_json: Any = None,
        **kwargs: Any,
    ) -> list[float]:
        completions = list(completions)
        size = len(completions)
        labels = _as_list(gate_label, size)
        memories = _as_list(memory_rows_json, size)
        evidence = _as_list(evidence_ids_json, size)
        languages = _as_list(language, size)

        # Every index must yield a float: Swift turns None into NaN.
        return [
            self._score(completion, label, memory_value, evidence_value, lang)
            for completion, label, memory_value, evidence_value, lang in zip(
                completions, labels, memories, evidence, languages
            )
        ]

    @staticmethod
    def _score(
        completion: Any,
        label: Any,
        memory_value: Any,
        evidence_value: Any,
        language: Any,
    ) -> float:
        prediction = _extract_json(_text(completion))
        if not output_is_valid(prediction, language if isinstance(language, str) else ""):
            return INVALID

        if not bool(label):
            return NEGATIVE_RETRIEVE if prediction["retrieve"] else NEGATIVE_IDLE
        if not prediction["retrieve"]:
            return DECLINED

        memory_rows = _parse_json_column(memory_value, [])
        evidence_ids = _parse_json_column(evidence_value, [])
        if not isinstance(memory_rows, list) or not isinstance(evidence_ids, list):
            # Data-integrity path; the hard split always carries both columns.
            # A constant reward here yields zero advantage, so GRPO simply
            # ignores the row rather than learning from a broken target.
            return INVALID

        fact_ids, episode_ids = _retrieved_branches(memory_rows, prediction["query"])
        rank = _evidence_rank(fact_ids, episode_ids, {str(value) for value in evidence_ids})
        return MISS if rank is None else 1.0 / rank


orms["gate_reward_v2"] = GateRewardV2


if __name__ == "__main__":
    reward = GateRewardV2()
    examples = [
        # Negative row, left alone.
        ('{"retrieve":false,"query":"","reason":"当前信息已足够"}', False, "[]", "[]", "en"),
        # Positive row, retrieved, one memory holding the evidence -> rank 1.
        (
            '{"retrieve":true,"query":"game John hooked","reason":"需要查询历史记忆"}',
            True,
            json.dumps([{"id": 1, "kind": "fact", "subject": "game", "content": "John is hooked on chess"}]),
            "[1]",
            "en",
        ),
    ]
    print(
        reward(
            [item[0] for item in examples],
            [item[1] for item in examples],
            [item[2] for item in examples],
            [item[3] for item in examples],
            [item[4] for item in examples],
        )
    )

"""Tests for the rank-aware gate reward in ``scripts/grpo_gate_reward.py``.

Runs under pytest, and also standalone because this repo's venv does not have
pytest installed::

    .venv/bin/python tests/test_grpo_gate_reward.py

The central property under test is that the reward agrees with the evaluator:
same format contract, same branch-internal rank, and a totally ordered reward
ladder that never makes "decline to retrieve" cheaper than "try and miss".
"""

from __future__ import annotations

import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

DEV_HAYSTACK = REPO_DIR / "data" / "grpo_hard" / "gate_grpo_dev.jsonl"


def _load_reward_module():
    spec = importlib.util.spec_from_file_location(
        "grpo_gate_reward", REPO_DIR / "scripts" / "grpo_gate_reward.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reward_module = _load_reward_module()
GateRewardV2 = reward_module.GateRewardV2


# ---------------------------------------------------------------- helpers


def _rows() -> dict[str, dict]:
    rows = {}
    for line in DEV_HAYSTACK.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


def _memory_text(memory: dict) -> str:
    return memory.get("summary") or memory.get("content") or ""


def _evidence_query(row: dict) -> str:
    """The first few words of the gold memory — stands in for a good query."""
    memories = json.loads(row["memory_rows_json"])
    evidence = {str(value) for value in json.loads(row["evidence_ids_json"])}
    gold = next(memory for memory in memories if str(memory["id"]) in evidence)
    return " ".join(_memory_text(gold).split()[:4])


def _pick(rows: dict[str, dict], kind: str) -> dict:
    """First positive row whose evidence sits in ``kind`` and is retrievable."""
    from eval.run_gate_eval import _evidence_kinds, _evidence_rank, _retrieved_branches

    for row in rows.values():
        if not row["gate_label"]:
            continue
        memories = json.loads(row["memory_rows_json"])
        evidence = {str(value) for value in json.loads(row["evidence_ids_json"])}
        kinds = _evidence_kinds(memories)
        if {kinds.get(value) for value in evidence} != {kind}:
            continue
        query = _evidence_query(row)
        if len(query.split()) < 2:
            continue
        fact_ids, episode_ids = _retrieved_branches(memories, query)
        if _evidence_rank(fact_ids, episode_ids, evidence) is not None:
            return row
    raise AssertionError(f"no retrievable {kind}-evidence positive row in {DEV_HAYSTACK}")


def _reward(
    completion: str,
    *,
    label: bool,
    memory_rows: list[dict] | None = None,
    evidence_ids: list | None = None,
    language: str = "en",
) -> float:
    values = GateRewardV2()(
        [completion],
        gate_label=[label],
        memory_rows_json=[json.dumps(memory_rows or [])],
        evidence_ids_json=[json.dumps(evidence_ids or [])],
        language=[language],
    )
    assert len(values) == 1
    return values[0]


def _gate(retrieve: bool, query: str, reason: str = "需要查询历史记忆") -> str:
    return json.dumps({"retrieve": retrieve, "query": query, "reason": reason})


def _valid_retrieve(query: str = "alpha beta", reason: str = "需要查询历史记忆") -> str:
    return _gate(True, query, reason)


@contextmanager
def _patched_rank(value):
    original = reward_module._evidence_rank
    reward_module._evidence_rank = lambda *args, **kwargs: value
    try:
        yield
    finally:
        reward_module._evidence_rank = original


# ------------------------------------------------------------------ tests


def test_reward_ladder_is_totally_ordered():
    """The ladder is the design: declining must be strictly worse than missing."""
    assert reward_module.DECLINED < reward_module.MISS < 1 / 4
    assert 1 / 4 < 1 / 3 < 1 / 2 < 1.0
    assert reward_module.INVALID <= reward_module.DECLINED
    assert reward_module.NEGATIVE_RETRIEVE <= reward_module.DECLINED


def test_rank_shaping_is_one_over_rank():
    rows = _rows()
    row = _pick(rows, "fact")
    query = _evidence_query(row)
    memories = json.loads(row["memory_rows_json"])
    evidence = json.loads(row["evidence_ids_json"])

    for rank in (1, 2, 3, 4):
        with _patched_rank(rank):
            value = _reward(
                _valid_retrieve(query), label=True, memory_rows=memories, evidence_ids=evidence
            )
        assert abs(value - 1.0 / rank) < 1e-9, f"rank {rank} scored {value}"


def test_miss_scores_between_declined_and_worst_hit():
    rows = _rows()
    row = _pick(rows, "fact")
    memories = json.loads(row["memory_rows_json"])
    evidence = json.loads(row["evidence_ids_json"])

    with _patched_rank(None):
        missed = _reward(
            _valid_retrieve(_evidence_query(row)),
            label=True,
            memory_rows=memories,
            evidence_ids=evidence,
        )
    declined = _reward(_gate(False, ""), label=True, memory_rows=memories, evidence_ids=evidence)

    assert missed == reward_module.MISS
    assert declined == reward_module.DECLINED
    assert declined < missed < 1 / 4


def test_negative_rows():
    idle = _reward(_gate(False, "", "当前信息已足够"), label=False)
    retrieved = _reward(_valid_retrieve(), label=False)
    assert idle == reward_module.NEGATIVE_IDLE == 1.0
    assert retrieved == reward_module.NEGATIVE_RETRIEVE == -1.0


def test_positive_declined_scores_worst():
    rows = _rows()
    row = _pick(rows, "episode")
    value = _reward(
        _gate(False, "", "当前信息已足够"),
        label=True,
        memory_rows=json.loads(row["memory_rows_json"]),
        evidence_ids=json.loads(row["evidence_ids_json"]),
    )
    assert value == reward_module.DECLINED == -1.0


def test_format_contract_matches_the_evaluator():
    """Every rejected output must be one the evaluator also calls invalid.

    The reward imports ``output_is_valid`` rather than reimplementing it, so this
    pins that decision: if someone re-adds a private format check, the zh case
    below is the one that silently drifts.
    """
    from eval.run_gate_eval import output_is_valid

    cases = [
        # malformed JSON
        ("not json at all", "en"),
        # missing ``reason`` — v1's _valid_gate never looked at it
        ('{"retrieve": true, "query": "alpha beta"}', "en"),
        # a bare single term is below the 2-term floor
        (_gate(True, "alpha"), "en"),
        # retrieve=false must carry an empty query
        (_gate(False, "alpha beta"), "en"),
        # zh multi-term query without an ASCII space: splits on U+3000, so the
        # evaluator rejects it while v1's _valid_gate accepted it
        (_gate(True, "北京　天气"), "zh"),
    ]
    for completion, language in cases:
        parsed = reward_module._extract_json(completion)
        assert not output_is_valid(parsed, language), f"expected invalid: {completion!r}"
        assert _reward(completion, label=True, language=language) == reward_module.INVALID

    # Positive control: a well-formed output is not rejected.
    assert output_is_valid(reward_module._extract_json(_valid_retrieve()), "en")
    assert reward_module._extract_json(_valid_retrieve()) is not None


def test_zh_fullwidth_space_regression():
    """v1 paid +1 for this; the evaluator counts it format-invalid."""
    completion = _gate(True, "北京　天气")
    assert _reward(completion, label=True, language="zh") == -1.0
    # The same query in a language without the rule is still acceptable.
    assert _reward(completion, label=True, language="en") != -1.0


def test_episode_evidence_is_scored_by_branch_rank():
    """Injecting facts must not push episode evidence down the reward scale.

    Real rows cannot exercise this (episode-evidence rows never carry a fact
    pool), so the divergence is constructed: the same episode, with four facts
    added that match the query.  Concatenating branches first would score the
    rank-1 episode as rank 5.
    """
    from eval.run_gate_eval import _evidence_rank, _first_rank, _retrieved_branches

    rows = _rows()
    row = _pick(rows, "episode")
    query = _evidence_query(row)
    memories = json.loads(row["memory_rows_json"])
    evidence = {str(value) for value in json.loads(row["evidence_ids_json"])}

    baseline = _reward(
        _valid_retrieve(query), label=True, memory_rows=memories, evidence_ids=sorted(evidence),
        language=row["language"],
    )

    injected = memories + [
        {"id": 9000 + index, "kind": "fact", "subject": f"probe{index}", "content": f"{query} 补充{index}"}
        for index in range(4)
    ]
    fact_ids, episode_ids = _retrieved_branches(injected, query)
    concat_rank = _first_rank(fact_ids + episode_ids, evidence)

    # The premise: the injection really does bury the episode in concat order.
    assert len(fact_ids) == 4, f"injected facts were not retrieved: {len(fact_ids)}"
    assert concat_rank == 5, f"expected concat rank 5, got {concat_rank}"
    assert _evidence_rank(fact_ids, episode_ids, evidence) == 1

    with_facts = _reward(
        _valid_retrieve(query), label=True, memory_rows=injected, evidence_ids=sorted(evidence),
        language=row["language"],
    )
    assert with_facts == baseline == 1.0
    assert with_facts > 1.0 / concat_rank


def test_returns_one_float_per_completion():
    """Swift coerces None to NaN, so every index must carry a float."""
    rows = _rows()
    row = _pick(rows, "episode")
    memories = json.loads(row["memory_rows_json"])
    evidence = json.loads(row["evidence_ids_json"])

    completions = [
        _valid_retrieve(_evidence_query(row)),
        _gate(False, "", "当前信息已足够"),
        "garbage",
    ]
    values = GateRewardV2()(
        completions,
        gate_label=[True, False, True],
        memory_rows_json=[json.dumps(memories)] * 3,
        evidence_ids_json=[json.dumps(evidence)] * 3,
        language=["en"] * 3,
        trainer_state=None,  # injected by Swift; must be absorbed
    )
    assert len(values) == len(completions)
    assert all(isinstance(value, float) for value in values)
    assert all(value == value for value in values), "NaN in the reward"


def test_scalar_columns_are_broadcast():
    """Defensive: the documented Swift path sends lists, but scalars are tolerated."""
    values = GateRewardV2()(
        [_gate(False, "", "当前信息已足够")] * 2,
        gate_label=False,
        memory_rows_json="[]",
        evidence_ids_json="[]",
        language="en",
    )
    assert values == [1.0, 1.0]


def test_real_rows_agree_with_the_rank_function():
    """Integration: reward == 1/rank for every retrievable dev positive."""
    from eval.run_gate_eval import _evidence_rank, _retrieved_branches

    rows = _rows()
    checked = 0
    for row in rows.values():
        if not row["gate_label"]:
            continue
        query = _evidence_query(row)
        if len(query.split()) < 2:
            continue
        memories = json.loads(row["memory_rows_json"])
        evidence = {str(value) for value in json.loads(row["evidence_ids_json"])}
        fact_ids, episode_ids = _retrieved_branches(memories, query)
        rank = _evidence_rank(fact_ids, episode_ids, evidence)
        if rank is None:
            continue
        value = _reward(
            _valid_retrieve(query),
            label=True,
            memory_rows=memories,
            evidence_ids=sorted(evidence),
            language=row["language"],
        )
        assert abs(value - 1.0 / rank) < 1e-9, f"{row['id']}: rank {rank} scored {value}"
        checked += 1
    assert checked >= 100, f"only {checked} rows exercised the rank path"


if __name__ == "__main__":
    import traceback

    tests = sorted(
        (name, value)
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    )
    failures = 0
    for name, function in tests:
        try:
            function()
        except Exception:  # noqa: BLE001 - a test runner reports everything
            failures += 1
            print(f"FAIL {name}")
            traceback.print_exc()
        else:
            print(f"ok   {name}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)

"""Tests for the Test-metrics SwanLab reporter in ``scripts/log_gate_metrics_to_swanlab.py``.

Runs under pytest, and also standalone because this repo's venv does not have
pytest installed::

    .venv/bin/python tests/test_log_gate_metrics.py

Two properties are under test. First, that ``metrics.json`` is split losslessly
into chartable numbers and descriptive config: every numeric leaf is logged,
string leaves become config, and ``None`` leaves -- which a slice produces when it
has no samples of a kind, e.g. zh/fact -- are dropped rather than charted as a gap.
Second, that the run-target decision asks swanlab for the training run when an id
is available (and always with a resume mode, since swanlab rejects an id without
one) and otherwise opens its own grouped test run.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

REPORTER = REPO_DIR / "scripts" / "log_gate_metrics_to_swanlab.py"
REAL_METRICS = REPO_DIR / "eval" / "results" / "baselines_qwen3_1.7b_dev" / "metrics.json"


def _load_reporter_module():
    spec = importlib.util.spec_from_file_location("log_gate_metrics_to_swanlab", REPORTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reporter = _load_reporter_module()


def _flatten(payload):
    metrics: dict[str, float] = {}
    config: dict[str, str] = {}
    reporter.flatten("test/", payload, metrics, config)
    return metrics, config


def test_flat_numbers_are_logged_with_the_prefix():
    metrics, config = _flatten({"rows": 284, "accuracy": 0.5})
    assert metrics == {"test/rows": 284.0, "test/accuracy": 0.5}
    assert config == {}


def test_nested_keys_keep_their_path():
    metrics, _ = _flatten({"retrieval": {"branch": {"conditional_mrr": 0.7231}}})
    assert metrics == {"test/retrieval.branch.conditional_mrr": 0.7231}


def test_strings_become_config_and_none_is_dropped():
    metrics, config = _flatten(
        {"retrieval_scope": "hard_haystack", "hits": 108, "fact": {"conditional_mrr": None}}
    )
    # None is a slice with no such samples; logging it would chart a hole.
    assert metrics == {"test/hits": 108.0}
    assert config == {"test/retrieval_scope": "hard_haystack"}


def test_booleans_are_logged_as_numbers_not_strings():
    # bool is a subclass of int, so the ordering of the isinstance branches matters.
    metrics, config = _flatten({"flag": True})
    assert metrics == {"test/flag": 1.0}
    assert config == {}


def test_integers_are_widened_so_swanlab_does_not_mix_types():
    metrics, _ = _flatten({"a": 1, "b": 1.0})
    assert metrics == {"test/a": 1.0, "test/b": 1.0}
    assert all(isinstance(value, float) for value in metrics.values())


def test_run_id_appends_to_the_training_run():
    args, target = reporter.build_init_args(
        {"SWANLAB_PROJECT": "p", "SWANLAB_RUN_ID": "run-1"},
        split="test",
        experiment_name=None,
        job_type="test",
        group=None,
    )
    assert args == {"project": "p", "id": "run-1", "resume": "allow"}
    # The training run already fixed these; repeating them could only conflict.
    assert "experiment_name" not in args and "job_type" not in args
    assert "run-1" in target


def test_run_id_without_resume_still_sends_a_resume_mode():
    # swanlab raises "You can't pass id when resume=never", so the reporter must
    # never hand over an id on its own.
    args, _ = reporter.build_init_args(
        {"SWANLAB_RUN_ID": "run-1"},
        split="test",
        experiment_name=None,
        job_type="test",
        group=None,
    )
    assert args["resume"] == "allow"


def test_explicit_resume_mode_wins():
    args, _ = reporter.build_init_args(
        {"SWANLAB_RUN_ID": "run-1", "SWANLAB_RESUME": "must"},
        split="test",
        experiment_name=None,
        job_type="test",
        group=None,
    )
    assert args["resume"] == "must"


def test_without_a_run_id_it_opens_a_grouped_test_run():
    args, target = reporter.build_init_args(
        {"SWANLAB_PROJECT": "p", "SWANLAB_EXP_NAME": "exp"},
        split="test",
        experiment_name=None,
        job_type="test",
        group=None,
    )
    assert args == {
        "project": "p",
        "experiment_name": "exp-test",
        "job_type": "test",
        "group": "exp",
    }
    assert "id" not in args


def test_the_real_dev_metrics_file_splits_cleanly():
    import json

    assert REAL_METRICS.is_file(), f"fixture missing: {REAL_METRICS}"
    metrics, config = _flatten(json.loads(REAL_METRICS.read_text(encoding="utf-8")))

    # Headline metrics the dashboard is supposed to show.
    for key in (
        "test/classification.macro_f1",
        "test/retrieval.branch.conditional_mrr",
        "test/end_to_end.memory_recall",
    ):
        assert key in metrics, f"{key} was not logged"
        assert 0.0 <= metrics[key] <= 1.0

    # zh has no fact samples: the *rates* are null in the file and must not
    # survive, while the counts are a real 0 and must.
    zh_fact = "test/slices.language.zh.retrieval.fact."
    assert zh_fact + "conditional_mrr" not in metrics
    assert zh_fact + "hit_rate" not in metrics
    assert metrics[zh_fact + "hits"] == 0.0
    assert metrics[zh_fact + "positive_total"] == 0.0
    assert config["test/retrieval_scope"] == "hard_haystack"
    assert metrics["test/rows"] == 284.0


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

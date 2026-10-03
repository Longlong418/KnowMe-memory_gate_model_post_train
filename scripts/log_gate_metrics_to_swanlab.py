"""Push a finished gate-eval ``metrics.json`` to SwanLab.

``swift infer`` is not a Trainer, so there is no ``--report_to`` to switch on and
no callback that could log for it: the evaluation process writes ``metrics.json``
and exits. This script is the reporter that step was missing.

The training scripts call it after the final Test evaluation, which is why Test
metrics now show up at all. It is equally usable by hand after any evaluation:

    SWANLAB_PROJECT=knowme-memory-gate-local SWANLAB_EXP_NAME=qwen3-1.7b-grpo \\
    .venv/bin/python scripts/log_gate_metrics_to_swanlab.py \\
        --metrics eval/results/<run>_test/metrics.json

Two modes, chosen by whether ``SWANLAB_RUN_ID`` is set:

* **append to the training run** (what the training scripts do) — ``SWANLAB_RUN_ID``
  and ``SWANLAB_RESUME=allow`` are exported before training, so the run is
  addressable by id and this script re-inits with the same id. One experiment then
  holds loss, reward, Dev and Test. The id must be original to the *training*
  process: transformers' ``SwanLabCallback`` is what reads it, not swift, so
  setting it only here would create a run that never had a training curve.

  ``--step`` matters in this mode: without it swanlab appends at its own counter,
  which may not line up with the training curve's x-axis. The training scripts
  pass the final global step (from ``gate_best.json`` / ``checkpoint-<N>``).

  Two ways this mode fails loudly rather than silently:
  swanlab only supports ``resume`` in **cloud** mode (offline/local downgrade it to
  ``never`` and drop the id), and passing an id with ``resume='never'`` is an
  error, so ``SWANLAB_RUN_ID`` without ``SWANLAB_RESUME`` crashes training at
  ``on_train_begin``. Export both or neither.

* **separate run** (fallback: this script used on its own, no ``SWANLAB_RUN_ID``) —
  a new experiment named ``<SWANLAB_EXP_NAME>-test`` with ``job_type="test"``,
  grouped under ``SWANLAB_EXP_NAME``.

Every key is prefixed with the split name (``test/classification.macro_f1``) so
that -- in the resume mode especially -- Test numbers cannot overwrite the
training run's own ``eval/...`` entries. Numeric leaves are logged as metrics;
string leaves (``retrieval_scope``, ``haystack``) go to the run config; ``None``
leaves -- a slice with no such samples, e.g. zh/fact -- are dropped, because
swanlab would chart them as a gap.

Exit status is 0 when there is nothing to do (no ``SWANLAB_API_KEY``: a plain
evaluation should not fail just because nobody wants a dashboard) and 1 when
logging was attempted and failed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))


def flatten(prefix: str, value: Any, metrics: dict[str, float], config: dict[str, str]) -> None:
    """Split a nested metrics dict into chartable numbers and descriptive strings."""
    if isinstance(value, dict):
        for key, item in value.items():
            flatten(f"{prefix}{key}.", item, metrics, config)
    elif isinstance(value, bool):
        metrics[prefix[:-1]] = float(value)
    elif isinstance(value, (int, float)):
        metrics[prefix[:-1]] = float(value)
    elif isinstance(value, str):
        config[prefix[:-1]] = value
    # None: a metric that is undefined for this slice (e.g. zh/fact). Dropped --
    # swanlab would plot it as a hole in the curve rather than an absence.


def build_init_args(
    env: Mapping[str, str],
    *,
    split: str,
    experiment_name: str | None,
    job_type: str,
    group: str | None,
) -> tuple[dict[str, Any], str]:
    """Decide what to pass to ``swanlab.init``, and describe the target.

    ``SWANLAB_RUN_ID`` present means "append to the training run". Its experiment
    name, job type and group were fixed when training started, so they are
    deliberately not passed again -- repeating them could only conflict.
    """
    project = env.get("SWANLAB_PROJECT") or None
    run_id = env.get("SWANLAB_RUN_ID") or None
    if run_id:
        # `resume` is not optional alongside `id`: swanlab raises
        # "You can't pass id when resume=never" when the id arrives without it.
        return (
            {"project": project, "id": run_id, "resume": env.get("SWANLAB_RESUME") or "allow"},
            f"existing run {run_id}",
        )

    exp_name = experiment_name or env.get("SWANLAB_EXP_NAME")
    group = group or env.get("SWANLAB_EXP_NAME")
    init_args: dict[str, Any] = {
        "project": project,
        "experiment_name": f"{exp_name}-{split}" if exp_name else split,
        "job_type": job_type,
    }
    if group:
        init_args["group"] = group
    return init_args, f"new run {init_args['experiment_name']}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True, help="path to metrics.json")
    parser.add_argument("--split", default="test", help="split name, used in the run name")
    parser.add_argument(
        "--prefix",
        default=None,
        help="metric key prefix (default: the split name, e.g. 'test/...')",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="run name (default: $SWANLAB_EXP_NAME + '-' + split)",
    )
    parser.add_argument("--job-type", default="test")
    parser.add_argument("--group", default=None, help="default: $SWANLAB_EXP_NAME")
    parser.add_argument(
        "--step",
        default=None,
        help=(
            "log at this step (default: let swanlab use its own counter). An empty "
            "value means the same thing, so a caller can pass it unconditionally."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would be logged and exit without contacting SwanLab",
    )
    args = parser.parse_args()

    if not args.metrics.is_file():
        print(f"metrics file not found: {args.metrics}", file=sys.stderr)
        return 1
    payload = json.loads(args.metrics.read_text(encoding="utf-8"))

    prefix = args.prefix if args.prefix is not None else f"{args.split}/"
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    metrics: dict[str, float] = {}
    config: dict[str, str] = {}
    flatten(prefix, payload, metrics, config)

    if args.dry_run:
        print(f"{len(metrics)} metrics, {len(config)} config entries, prefix {prefix!r}")
        for key in sorted(metrics)[:10]:
            print(f"  {key} = {metrics[key]}")
        if len(metrics) > 10:
            print(f"  ... and {len(metrics) - 10} more")
        return 0

    if not os.environ.get("SWANLAB_API_KEY"):
        print(
            "SWANLAB_API_KEY is not set; skipping SwanLab upload "
            f"(metrics remain in {args.metrics}).",
            file=sys.stderr,
        )
        return 0

    try:
        import swanlab
    except ImportError as exc:  # pragma: no cover - depends on the environment
        print(f"swanlab is not importable: {exc}", file=sys.stderr)
        return 1

    init_args, target = build_init_args(
        os.environ,
        split=args.split,
        experiment_name=args.experiment_name,
        job_type=args.job_type,
        group=args.group,
    )
    project = init_args.get("project")

    try:
        swanlab.init(**init_args)
        if config:
            swanlab.config.update(config)
        step = int(args.step) if args.step not in (None, "") else None
        if step is None:
            swanlab.log(metrics)
        else:
            swanlab.log(metrics, step=step)
        swanlab.finish()
    except Exception as exc:  # noqa: BLE001 - report the cause, do not traceback-spam
        print(f"SwanLab upload failed ({target}): {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"Logged {len(metrics)} test metrics to SwanLab ({target}, project {project}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

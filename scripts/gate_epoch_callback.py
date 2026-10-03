"""ms-swift callback for generated Dev evaluation and checkpoint selection."""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

from swift.callbacks import callbacks_map
from swift.callbacks.base import TrainerCallback


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _nested_value(metrics: dict[str, Any], path: str) -> float | None:
    value: Any = metrics
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return _number(value)


def _flatten_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    if isinstance(value, dict):
        flattened: dict[str, float] = {}
        for key, child in value.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            flattened.update(_flatten_numeric(child, child_prefix))
        return flattened
    numeric = _number(value)
    return {prefix: numeric} if numeric is not None and prefix else {}


class GateEpochEvalCallback(TrainerCallback):
    """Evaluate each saved epoch checkpoint and retain the best gate metric."""

    def __init__(self, args, trainer):
        super().__init__(args, trainer)
        self.repo_dir = Path(os.environ["REPO_DIR"]).resolve()
        self.python = Path(os.environ.get("PYTHON", f"{self.repo_dir}/.venv/bin/python"))
        self.swift_bin = Path(os.environ.get("SWIFT_BIN", f"{self.repo_dir}/.venv/bin/swift"))
        self.model = Path(os.environ["MODEL_PATH"]).resolve()
        self.dev_data = Path(os.environ.get("DEV_DATA", f"{self.repo_dir}/data/train_split/gate_dev.jsonl"))
        self.full_data = self.repo_dir / "data/gate_final_2855.full.jsonl"
        self.output_dir = Path(args.output_dir).resolve()
        # Rank metrics only exist under the hard-haystack scope, so the default
        # selection metric is the branch-internal MRR conditioned on the samples
        # where the model actually emitted a query.
        self.best_metric = os.environ.get(
            "GATE_BEST_METRIC", "retrieval.branch.conditional_mrr"
        )
        self.haystack = self._resolve_haystack()
        self.history_path = self.output_dir / "gate_dev_history.jsonl"
        self.best_path = self.output_dir / "gate_best.json"
        self.strict = os.environ.get("GATE_EVAL_STRICT", "1") != "0"
        if self.haystack is not None and not self.haystack.is_file():
            if self.best_metric.startswith("retrieval."):
                raise RuntimeError(
                    f"GATE_BEST_METRIC={self.best_metric!r} needs the hidden haystack, but "
                    f"{self.haystack} does not exist. Point GATE_HAYSTACK at the file, or "
                    "select a metric that does not start with 'retrieval.'."
                )
            print(
                f"Warning: hard-haystack file not found ({self.haystack}); evaluating on "
                "the evidence-only scope, so rank metrics will be trivial.",
                flush=True,
            )
            self.haystack = None

    def _resolve_haystack(self) -> Path | None:
        """Locate the hidden haystack for the dev split, or ``None`` to disable it."""
        configured = os.environ.get("GATE_HAYSTACK")
        if configured is not None:
            return Path(configured).resolve() if configured else None
        split_name = self.dev_data.stem.removeprefix("gate_")
        return self.repo_dir / "data" / "grpo_hard" / f"gate_grpo_{split_name}.jsonl"

    def _evaluate(self, checkpoint: Path, step: int) -> dict[str, Any]:
        result_dir = self.output_dir / "gate_eval" / f"checkpoint-{step}"
        command = [
            str(self.python),
            "-m",
            "eval.evaluate_checkpoint",
            "--swift-bin",
            str(self.swift_bin),
            "--model",
            str(self.model),
            "--adapters",
            str(checkpoint),
            "--split",
            str(self.dev_data),
            "--full",
            str(self.full_data),
            "--output-dir",
            str(result_dir),
            "--repo-dir",
            str(self.repo_dir),
        ]
        if self.haystack is not None:
            command += ["--haystack", str(self.haystack)]
        completed = subprocess.run(
            command,
            cwd=self.repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        metrics_path = result_dir / "metrics.json"
        if not metrics_path.exists():
            raise RuntimeError(f"evaluation did not produce {metrics_path}: {completed.stdout[-1000:]}")
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    def _write_history(self, record: dict[str, Any]) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _select(self, checkpoint: Path, step: int, metrics: dict[str, Any]) -> None:
        score = _nested_value(metrics, self.best_metric)
        if score is None:
            raise KeyError(f"Best metric {self.best_metric!r} is missing or non-numeric")
        record = {
            "epoch": self.trainer.state.epoch,
            "step": step,
            "checkpoint": str(checkpoint),
            "best_metric": self.best_metric,
            "score": score,
            "metrics": metrics,
        }
        self._write_history(record)
        previous = None
        if self.best_path.exists():
            previous = json.loads(self.best_path.read_text(encoding="utf-8"))
        is_better = previous is None or score > float(previous["score"])
        numeric = _flatten_numeric(metrics, "dev")
        numeric["dev/selection_score"] = score
        numeric["dev/is_best"] = 1.0 if is_better else 0.0
        numeric["dev/epoch"] = float(self.trainer.state.epoch or 0)
        self.trainer.log(numeric)

        if is_better:
            self.best_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def on_save(self, args, state, control, **kwargs):
        if not getattr(state, "is_world_process_zero", True):
            return control
        checkpoint = getattr(state, "last_model_checkpoint", None)
        if not checkpoint:
            checkpoint = str(self.output_dir / f"checkpoint-{state.global_step}")
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_dir():
            return control
        try:
            metrics = self._evaluate(checkpoint_path, int(state.global_step))
            self._select(checkpoint_path, int(state.global_step), metrics)
        except Exception:
            if self.strict:
                raise
            print(f"Gate Dev evaluation failed for {checkpoint_path}", flush=True)
        return control


callbacks_map["gate_epoch_eval"] = GateEpochEvalCallback

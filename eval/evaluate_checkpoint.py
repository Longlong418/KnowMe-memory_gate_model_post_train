"""Run gate inference and metrics for one LoRA checkpoint."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from eval.run_gate_eval import evaluate


def run(
    *,
    swift_bin: Path,
    model: Path,
    adapters: Path,
    split: Path,
    full: Path,
    output_dir: Path,
    repo_dir: Path,
    haystack: Path | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    predictions.unlink(missing_ok=True)
    metrics_path.unlink(missing_ok=True)
    command = [
        str(swift_bin),
        "infer",
        "--model",
        str(model),
        "--adapters",
        str(adapters),
        "--val_dataset",
        str(split),
        "--result_path",
        str(predictions),
        "--infer_backend",
        "transformers",
        "--max_new_tokens",
        "128",
        "--temperature",
        "0",
        "--enable_thinking",
        "false",
        # Do not replay the checkpoint's args.json: it pins the training machine's
        # absolute paths (external_plugins, callbacks) and eval sets everything
        # it needs on this command line.
        "--load_args",
        "false",
    ]
    subprocess.run(command, cwd=repo_dir, check=True)
    result = evaluate(split, full, predictions, haystack)
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--swift-bin", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapters", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument(
        "--haystack",
        type=Path,
        help=(
            "optional GRPO hard-haystack file; enables the retrieval.* rank metrics "
            "instead of the trivial evidence-only scope"
        ),
    )
    args = parser.parse_args()
    result = run(
        swift_bin=args.swift_bin,
        model=args.model,
        adapters=args.adapters,
        split=args.split,
        full=args.full,
        output_dir=args.output_dir,
        repo_dir=args.repo_dir,
        haystack=args.haystack,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

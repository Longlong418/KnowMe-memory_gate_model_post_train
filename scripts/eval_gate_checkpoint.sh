#!/usr/bin/env bash
set -euo pipefail

# Locates REPO_DIR / PYTHON / SWIFT_BIN / MODEL_PATH and the run directories;
# see that file for the search order and for the overrides.
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_paths.sh"

ADAPTER_PATH="${1:?Usage: $0 /path/to/checkpoint [dev|test]}"
SPLIT_NAME="${2:-dev}"
SPLIT_PATH="$REPO_DIR/data/train_split/gate_${SPLIT_NAME}.jsonl"
FULL_PATH="$REPO_DIR/data/gate_final_2855.full.jsonl"
# Hidden memory haystack for this split; enables the retrieval.* rank metrics.
# Set HAYSTACK_PATH="" to fall back to the trivial evidence-only scope.
HAYSTACK_PATH="${HAYSTACK_PATH-$REPO_DIR/data/grpo_hard/gate_grpo_${SPLIT_NAME}.jsonl}"
RESULT_DIR="${RESULT_DIR:-$REPO_DIR/eval/results/$(basename "$ADAPTER_PATH")_$SPLIT_NAME}"
PREDICTIONS="$RESULT_DIR/predictions.jsonl"
METRICS="$RESULT_DIR/metrics.json"

if [[ "$SPLIT_NAME" != "dev" && "$SPLIT_NAME" != "test" ]]; then
  echo "Split must be dev or test." >&2
  exit 1
fi
HAYSTACK_ARGS=()
if [[ -n "$HAYSTACK_PATH" ]]; then
  if [[ -f "$HAYSTACK_PATH" ]]; then
    HAYSTACK_ARGS=(--haystack "$HAYSTACK_PATH")
  else
    echo "Warning: hard-haystack file not found: $HAYSTACK_PATH" >&2
    echo "         Falling back to the evidence-only scope; rank metrics will be trivial." >&2
  fi
fi

mkdir -p "$RESULT_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
rm -f "$PREDICTIONS" "$METRICS"

# --load_args false: an adapter directory carries the args.json of its training
# run, which would otherwise be replayed here and import that machine's absolute
# paths (external_plugins, callbacks).  Evaluation passes everything it needs
# explicitly, so it should not inherit training arguments.
"$SWIFT_BIN" infer \
  --model "$MODEL_PATH" \
  --adapters "$ADAPTER_PATH" \
  --val_dataset "$SPLIT_PATH" \
  --result_path "$PREDICTIONS" \
  --infer_backend transformers \
  --max_new_tokens 128 \
  --temperature 0 \
  --enable_thinking false \
  --load_args false

"$PYTHON" -m eval.run_gate_eval \
  --split "$SPLIT_PATH" \
  --full "$FULL_PATH" \
  --predictions "$PREDICTIONS" \
  "${HAYSTACK_ARGS[@]}" \
  --output "$METRICS"

echo "Metrics written to $METRICS"

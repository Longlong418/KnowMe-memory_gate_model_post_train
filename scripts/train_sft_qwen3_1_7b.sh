#!/usr/bin/env bash
set -euo pipefail

# Locates REPO_DIR / PYTHON / SWIFT_BIN / MODEL_PATH and the run directories;
# see that file for the search order and for the overrides.
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_paths.sh"

OUTPUT_DIR="${OUTPUT_DIR:-$RUNS_ROOT/sft_qwen3_1.7b}"
TRAIN_DATA="${TRAIN_DATA:-$REPO_DIR/data/train_split/gate_train.jsonl}"
DEV_DATA="${DEV_DATA:-$REPO_DIR/data/train_split/gate_dev.jsonl}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python environment not found: $PYTHON" >&2
  exit 1
fi
if [[ ! -x "$SWIFT_BIN" ]]; then
  echo "ms-swift executable not found: $SWIFT_BIN" >&2
  exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Model directory not found: $MODEL_PATH" >&2
  exit 1
fi
if [[ -z "${SWANLAB_API_KEY:-}" ]]; then
  echo "Set SWANLAB_API_KEY before training." >&2
  exit 1
fi
# On the HF path the reporter is transformers' SwanLabCallback, not swift's: it
# reads the project from the SWANLAB_PROJECT environment variable and the
# experiment name from --run_name. swift's own --swanlab_project /
# --swanlab_exp_name are only consumed by the Megatron callback, so passing them
# alone leaves the run in a project named after the working directory.
# "-local" is not about this machine: the original `knowme-memory-gate` was
# migrated server-side to SwanLab's new project scheme, and mounting it with any
# SDK below 0.9.0 now fails with
#   422 {"message":"新版本项目请使用新版 SDK（版本号 >= 0.9.0）"}
# A project name the server has not migrated still works old-style. Do not rename
# this back without also upgrading swanlab past 0.9, which breaks transformers'
# callback: at >=0.8 swanlab.get_run() raises instead of returning None, and
# SwanLabCallback.setup() does `if swanlab.get_run() is None: init()`.
# Both machines use this same name so their runs land in one project.
SWANLAB_PROJECT="${SWANLAB_PROJECT:-knowme-memory-gate-local}"
SWANLAB_EXP_NAME="${SWANLAB_EXP_NAME:-qwen3-1.7b-sft}"
# The final Test metrics are appended to this same run, so the run must be
# addressable by id: transformers' SwanLabCallback reads SWANLAB_RUN_ID and
# SWANLAB_RESUME, and scripts/log_gate_metrics_to_swanlab.py re-inits with the
# same id once evaluation is done. Export both or neither -- an id without a
# resume mode makes swanlab raise "You can't pass id when resume=never" at
# on_train_begin. The timestamp is what stops a second training run from
# resuming (and so merging into) the first; to continue a run deliberately, set
# SWANLAB_RUN_ID yourself.
# SWANLAB_RUN_ID="" is a deliberate opt-out -- Test metrics then get a run of
# their own -- so only default it when the variable is genuinely unset.
if [[ -z "${SWANLAB_RUN_ID+set}" ]]; then
  SWANLAB_RUN_ID="$SWANLAB_EXP_NAME-$(date +%Y%m%d-%H%M%S)"
fi
SWANLAB_RESUME="${SWANLAB_RESUME:-allow}"
export SWANLAB_PROJECT SWANLAB_EXP_NAME SWANLAB_RUN_ID SWANLAB_RESUME

export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export REPO_DIR PYTHON SWIFT_BIN MODEL_PATH TRAIN_DATA DEV_DATA OUTPUT_DIR
export RUNS_ROOT MODELSCOPE_CACHE
# Checkpoint selection (scripts/gate_epoch_callback.py) evaluates on the hidden
# hard haystack so the retrieval.* rank metrics are non-trivial. It derives the
# haystack from DEV_DATA, e.g. gate_dev.jsonl -> data/grpo_hard/gate_grpo_dev.jsonl.
# Override with GATE_HAYSTACK=/path (empty string disables it) and
# GATE_BEST_METRIC=<dotted.metric.path>; see docs/evaluation.md and docs/results.md.
mkdir -p "$OUTPUT_DIR" "$MODELSCOPE_CACHE"

"$SWIFT_BIN" sft \
  --model "$MODEL_PATH" \
  --dataset "$TRAIN_DATA" \
  --val_dataset "$DEV_DATA" \
  --tuner_type lora \
  --torch_dtype bfloat16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size "$TRAIN_BATCH_SIZE" \
  --per_device_eval_batch_size "$EVAL_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
  --gradient_checkpointing true \
  --learning_rate 1e-4 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --target_modules all-linear \
  --loss_scale last_round \
  --max_length 1024 \
  --dataset_num_proc "$DATASET_NUM_PROC" \
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
  --eval_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 100 \
  --save_only_model true \
  --external_plugins "$REPO_DIR/scripts/gate_epoch_callback.py" \
  --callbacks gate_epoch_eval \
  --logging_steps 10 \
  --warmup_ratio 0.05 \
  --report_to swanlab \
  --run_name "$SWANLAB_EXP_NAME" \
  --swanlab_project "$SWANLAB_PROJECT" \
  --swanlab_exp_name "$SWANLAB_EXP_NAME" \
  --output_dir "$OUTPUT_DIR"

BEST_RECORD="$($PYTHON - "$OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
records = sorted(root.rglob("gate_best.json"), key=lambda path: path.stat().st_mtime, reverse=True)
for best_path in records:
    try:
        record = json.loads(best_path.read_text(encoding="utf-8"))
        checkpoint = Path(record["checkpoint"])
    except (OSError, json.JSONDecodeError, KeyError):
        continue
    if checkpoint.is_dir():
        print(best_path)
        break
PY
)"

if [[ -z "$BEST_RECORD" ]]; then
  echo "Unable to locate best checkpoint under $OUTPUT_DIR" >&2
  exit 1
fi

RUN_OUTPUT_DIR="$(dirname "$BEST_RECORD")"
BEST_SOURCE_CHECKPOINT="$($PYTHON - "$BEST_RECORD" <<'PY'
import json
import sys
from pathlib import Path

record = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
checkpoint = Path(record["checkpoint"])
if checkpoint.is_dir():
    print(checkpoint)
PY
)"
if [[ -z "$BEST_SOURCE_CHECKPOINT" ]]; then
  echo "Best checkpoint recorded in $BEST_RECORD is missing" >&2
  exit 1
fi
BEST_CHECKPOINT="$RUN_OUTPUT_DIR/best_gate_checkpoint"
rm -rf "$BEST_CHECKPOINT"
cp -a "$BEST_SOURCE_CHECKPOINT" "$BEST_CHECKPOINT"

for checkpoint in "$RUN_OUTPUT_DIR"/checkpoint-*; do
  if [[ -d "$checkpoint" ]]; then
    rm -rf "$checkpoint"
  fi
done
rm -f "$RUN_OUTPUT_DIR/last-checkpoint"
ln -s "best_gate_checkpoint" "$RUN_OUTPUT_DIR/last-checkpoint"

BEST_CHECKPOINT="$BEST_CHECKPOINT" "$PYTHON" - "$BEST_RECORD" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
record = json.loads(path.read_text(encoding="utf-8"))
record["source_checkpoint"] = record["checkpoint"]
record["checkpoint"] = os.environ["BEST_CHECKPOINT"]
path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
RUN_NAME="$(basename "$RUN_OUTPUT_DIR")"
echo "Best Dev checkpoint: $BEST_CHECKPOINT"
TEST_RESULT_DIR="${RESULT_DIR:-$REPO_DIR/eval/results/${RUN_NAME}_test}"
RESULT_DIR="$TEST_RESULT_DIR" \
  bash "$REPO_DIR/scripts/eval_gate_checkpoint.sh" "$BEST_CHECKPOINT" test

# The global step the best checkpoint was saved at, so the Test point lands at
# the end of the training curve rather than wherever swanlab's counter happens
# to be. Empty means "let swanlab decide".
TEST_STEP="$($PYTHON - "$BEST_RECORD" <<'PY'
import json
import sys
from pathlib import Path

step = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get("step")
print(step if isinstance(step, int) else "")
PY
)"

# swift infer has no reporter of its own, so the Test numbers are pushed to
# SwanLab here, into the training run. See scripts/log_gate_metrics_to_swanlab.py.
"$PYTHON" "$REPO_DIR/scripts/log_gate_metrics_to_swanlab.py" \
  --metrics "$TEST_RESULT_DIR/metrics.json" --split test --step "$TEST_STEP" \
  || echo "Warning: Test metrics did not reach SwanLab; they are in $TEST_RESULT_DIR/metrics.json" >&2

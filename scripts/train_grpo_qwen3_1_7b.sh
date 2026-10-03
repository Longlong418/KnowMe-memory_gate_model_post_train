#!/usr/bin/env bash
set -euo pipefail

# Locates REPO_DIR / PYTHON / SWIFT_BIN / MODEL_PATH and the run directories;
# see that file for the search order and for the overrides.
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_paths.sh"

SFT_RUN_DIR="${SFT_RUN_DIR:-$RUNS_ROOT/sft_qwen3_1.7b}"
# Empty by default: the adapter is then discovered below, because the SFT run
# directory differs per machine (the cloud checkpoints live on the data disk, a
# local run's are under outputs/ or runs/). Set it to skip the search.
SFT_ADAPTER="${SFT_ADAPTER:-}"
# grpo_hard, not grpo: the rank-aware reward is 1/r, which carries no signal
# unless the memory pool contains distractors for the evidence to be ranked
# against. data/grpo has pool == evidence, so every hit would be rank 1.
TRAIN_DATA="${TRAIN_DATA:-$REPO_DIR/data/grpo_hard/gate_grpo_train.jsonl}"
DEV_DATA="${DEV_DATA:-$REPO_DIR/data/grpo_hard/gate_grpo_dev.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUNS_ROOT/grpo_qwen3_1.7b}"
# One optimizer step has to hold whole GRPO groups. GRPO normalizes each advantage
# inside a group of NUM_GENERATIONS (swift/rlhf_trainers/grpo_trainer.py:424), and swift
# derives
#   steps_per_generation = generation_batch_size // (per_device_train_batch_size * world_size)
# (swift/rlhf_trainers/args_mixin.py:210-214), so a step backprops on exactly
# TRAIN_BATCH_SIZE completions. When that is not a multiple of NUM_GENERATIONS, the two
# halves of one prompt are applied as two successive updates instead of one.
# At TRAIN_BATCH_SIZE=4 with NUM_GENERATIONS=8 this happened silently:
# steps_per_generation became 2, so run v4-20261003-160127 spent max_steps=4574 on a
# 2287-row epoch and every update saw half a group. Neither swift nor this script
# checked it, hence the guard below.
# Like the two checks under it, this ignores world_size: that only ever rejects a config
# swift would accept, never the reverse.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
# Must be a multiple of NUM_GENERATIONS: trl checks
# (per_device_eval_batch_size * num_processes) % num_generations == 0 and raises
# at startup otherwise. With the old default of 1 and NUM_GENERATIONS=4 the run
# died immediately with "The global eval batch size (1 * 1) must be divisible by
# the number of generations used for evaluation (4)". Raise this along with
# NUM_GENERATIONS (e.g. NUM_GENERATIONS=8 -> EVAL_BATCH_SIZE=8).
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
NUM_GENERATIONS="${NUM_GENERATIONS:-16}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-$NUM_GENERATIONS}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-4}"
EVAL_STEPS="${EVAL_STEPS:-500}"
SAVE_STEPS="${SAVE_STEPS:-500}"
# Checkpoints rotate, so a checkpoint is gone roughly SAVE_TOTAL_LIMIT save
# intervals after it is written -- in run v4-20261003-160127 checkpoint-200 was
# written at 16:09 and deleted at ~16:23. Two things follow:
#   * evaluating a checkpoint in place only works while it is still inside that
#     window, so DEV selection on a long run needs the limit raised first;
#   * the run's own final Test evaluation only ever looks at the last complete
#     checkpoint, so it cannot pick a better earlier one for you.
# SAVE_TOTAL_LIMIT="" is a deliberate opt-out meaning "keep every checkpoint":
# transformers treats save_total_limit=None as unlimited. Only default it when
# the variable is genuinely unset (same idiom as SWANLAB_RUN_ID below).
if [[ -z "${SAVE_TOTAL_LIMIT+set}" ]]; then
  SAVE_TOTAL_LIMIT=12
fi
if [[ -n "$SAVE_TOTAL_LIMIT" && ! "$SAVE_TOTAL_LIMIT" =~ ^[1-9][0-9]*$ ]]; then
  echo "SAVE_TOTAL_LIMIT must be a positive integer, or empty to keep every checkpoint (got '$SAVE_TOTAL_LIMIT')." >&2
  exit 1
fi
SAVE_TOTAL_LIMIT_ARGS=()
[[ -n "$SAVE_TOTAL_LIMIT" ]] && SAVE_TOTAL_LIMIT_ARGS=(--save_total_limit "$SAVE_TOTAL_LIMIT")

if ! [[ "$TRAIN_BATCH_SIZE" =~ ^[1-9][0-9]*$ && "$NUM_GENERATIONS" =~ ^[1-9][0-9]*$ && "$GENERATION_BATCH_SIZE" =~ ^[1-9][0-9]*$ && "$EVAL_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRAIN_BATCH_SIZE, NUM_GENERATIONS, GENERATION_BATCH_SIZE and EVAL_BATCH_SIZE must be positive integers." >&2
  exit 1
fi
if (( TRAIN_BATCH_SIZE % NUM_GENERATIONS != 0 )); then
  echo "TRAIN_BATCH_SIZE ($TRAIN_BATCH_SIZE) must be a multiple of NUM_GENERATIONS ($NUM_GENERATIONS);" >&2
  echo "otherwise one optimizer step backprops on part of a group, not a whole one: swift" >&2
  echo "sets steps_per_generation = generation_batch_size / TRAIN_BATCH_SIZE > 1, which" >&2
  echo "halves the effective batch and doubles max_steps. Use a multiple of NUM_GENERATIONS:" >&2
  echo "TRAIN_BATCH_SIZE=8 or 16 with NUM_GENERATIONS=8, 16 or 32 with NUM_GENERATIONS=16." >&2
  exit 1
fi
if (( GENERATION_BATCH_SIZE % NUM_GENERATIONS != 0 )); then
  echo "GENERATION_BATCH_SIZE ($GENERATION_BATCH_SIZE) must be divisible by NUM_GENERATIONS ($NUM_GENERATIONS)." >&2
  exit 1
fi
if (( EVAL_BATCH_SIZE % NUM_GENERATIONS != 0 )); then
  echo "EVAL_BATCH_SIZE ($EVAL_BATCH_SIZE) must be divisible by NUM_GENERATIONS ($NUM_GENERATIONS);" >&2
  echo "trl rejects the config at startup otherwise." >&2
  exit 1
fi

if [[ ! -x "$PYTHON" || ! -x "$SWIFT_BIN" ]]; then
  echo "Python or ms-swift executable not found under $REPO_DIR/.venv" >&2
  exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Model directory not found: $MODEL_PATH" >&2
  exit 1
fi
if [[ ! -f "$TRAIN_DATA" || ! -f "$DEV_DATA" ]]; then
  echo "GRPO hard-haystack data not found under $REPO_DIR/data/grpo_hard." >&2
  echo "Rebuild it with: $PYTHON -m src.dataset.build_grpo_hard_haystack --include-test" >&2
  exit 1
fi
if [[ -z "${SWANLAB_API_KEY:-}" ]]; then
  echo "SWANLAB_API_KEY is not set; export it before GRPO training." >&2
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
# A project name the server has not migrated still works old-style (verified end
# to end, including the id+resume re-attach below). Do not rename this back
# without also upgrading swanlab past 0.9, which breaks transformers' callback:
# at >=0.8 swanlab.get_run() raises instead of returning None, and
# SwanLabCallback.setup() does `if swanlab.get_run() is None: init()`.
# Both machines use this same name so their runs land in one project.
SWANLAB_PROJECT="${SWANLAB_PROJECT:-knowme-memory-gate-local}"
SWANLAB_EXP_NAME="${SWANLAB_EXP_NAME:-qwen3-1.7b-grpo}"
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

if [[ -z "$SFT_ADAPTER" ]]; then
  SFT_ADAPTER="$($PYTHON - "$SFT_RUN_DIR" "$REPO_DIR/outputs" "$RUNS_ROOT" <<'PY'
import sys
from pathlib import Path

# The SFT script copies the best epoch to <run>/best_gate_checkpoint, so this
# finds a finished SFT run wherever it was written. Newest wins: the point is to
# continue from the most recent SFT, and a GRPO run directory holds no such
# directory, so it cannot be picked up by mistake.
candidates = []
for arg in sys.argv[1:]:
    root = Path(arg)
    if root.is_dir():
        candidates.extend(p for p in root.rglob('best_gate_checkpoint') if p.is_dir())
if not candidates:
    raise SystemExit(
        'No best_gate_checkpoint found under ' + ', '.join(sys.argv[1:]) +
        '; set SFT_ADAPTER explicitly.'
    )
print(max(candidates, key=lambda p: p.stat().st_mtime))
PY
)"
fi
if [[ ! -d "$SFT_ADAPTER" ]]; then
  echo "SFT adapter not found: $SFT_ADAPTER" >&2
  exit 1
fi

export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export RUNS_ROOT MODELSCOPE_CACHE
mkdir -p "$OUTPUT_DIR" "$MODELSCOPE_CACHE"

echo "Base model: $MODEL_PATH"
echo "Starting SFT adapter: $SFT_ADAPTER"
echo "GRPO output: $OUTPUT_DIR"

TRAIN_STARTED_AT="$("$PYTHON" -c 'import time; print(time.time())')"

"$SWIFT_BIN" rlhf \
  --rlhf_type grpo \
  --model "$MODEL_PATH" \
  --adapters "$SFT_ADAPTER" \
  --ref_adapters "$SFT_ADAPTER" \
  --dataset "$TRAIN_DATA" \
  --val_dataset "$DEV_DATA" \
  --tuner_type lora \
  --torch_dtype bfloat16 \
  --num_train_epochs "$NUM_EPOCHS" \
  --per_device_train_batch_size "$TRAIN_BATCH_SIZE" \
  --per_device_eval_batch_size "$EVAL_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
  --gradient_checkpointing true \
  --learning_rate "$LEARNING_RATE" \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --target_modules all-linear \
  --max_length 1024 \
  --max_completion_length 128 \
  --num_generations "$NUM_GENERATIONS" \
  --generation_batch_size "$GENERATION_BATCH_SIZE" \
  --temperature 0.7 \
  --top_p 0.9 \
  --enable_thinking false \
  --use_vllm false \
  --dataset_num_proc "$DATASET_NUM_PROC" \
  --remove_unused_columns false \
  --external_plugins "$REPO_DIR/scripts/grpo_gate_reward.py" \
  --reward_funcs gate_reward_v2 \
  --reward_weights 1.0 \
  --logging_steps 1 \
  --log_completions true \
  --eval_strategy steps \
  --eval_steps "$EVAL_STEPS" \
  --save_strategy steps \
  --save_steps "$SAVE_STEPS" \
  ${SAVE_TOTAL_LIMIT_ARGS[@]+"${SAVE_TOTAL_LIMIT_ARGS[@]}"} \
  --report_to swanlab \
  --run_name "$SWANLAB_EXP_NAME" \
  --swanlab_token "$SWANLAB_API_KEY" \
  --swanlab_project "$SWANLAB_PROJECT" \
  --swanlab_exp_name "$SWANLAB_EXP_NAME" \
  --output_dir "$OUTPUT_DIR"

FINAL_CHECKPOINT="$("$PYTHON" - "$OUTPUT_DIR" "$TRAIN_STARTED_AT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
started_at = float(sys.argv[2])
candidates = []
for state_path in root.rglob('checkpoint-*/trainer_state.json'):
    if state_path.stat().st_mtime < started_at:
        continue
    checkpoint = state_path.parent
    if not (checkpoint / 'adapter_config.json').is_file():
        continue
    if not any((checkpoint / name).is_file() for name in (
        'adapter_model.safetensors', 'adapter_model.bin'
    )):
        continue
    state = json.loads(state_path.read_text(encoding='utf-8'))
    candidates.append((state_path.stat().st_mtime, state['global_step'], checkpoint))
if not candidates:
    raise SystemExit('No complete checkpoint from this GRPO run found; Test evaluation aborted.')
latest_run = max(candidates)[2].parent
print(max((item for item in candidates if item[2].parent == latest_run),
          key=lambda item: item[1])[2])
PY
)"

TEST_RESULT_DIR="${RESULT_DIR:-$(dirname "$FINAL_CHECKPOINT")/gate_eval/test}"
echo "GRPO training completed. Evaluating final checkpoint: $FINAL_CHECKPOINT"
(
  cd "$REPO_DIR"
  REPO_DIR="$REPO_DIR" PYTHON="$PYTHON" SWIFT_BIN="$SWIFT_BIN" \
    MODEL_PATH="$MODEL_PATH" RESULT_DIR="$TEST_RESULT_DIR" \
    bash "$REPO_DIR/scripts/eval_gate_checkpoint.sh" "$FINAL_CHECKPOINT" test
)
echo "Test predictions: $TEST_RESULT_DIR/predictions.jsonl"
echo "Test metrics: $TEST_RESULT_DIR/metrics.json"

# The final checkpoint is named checkpoint-<step>, which is the global step to
# log the Test metrics at, so the point lands at the end of the training curve.
TEST_STEP="${FINAL_CHECKPOINT##*-}"

# swift infer has no reporter of its own, so the Test numbers are pushed to
# SwanLab here, into the training run. See scripts/log_gate_metrics_to_swanlab.py.
"$PYTHON" "$REPO_DIR/scripts/log_gate_metrics_to_swanlab.py" \
  --metrics "$TEST_RESULT_DIR/metrics.json" --split test --step "$TEST_STEP" \
  || echo "Warning: Test metrics did not reach SwanLab; they are in $TEST_RESULT_DIR/metrics.json" >&2

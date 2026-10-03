# Shared path resolution, sourced by the training and evaluation scripts.
# Sourced, never executed: it only sets variables, and every one of them can be
# overridden by exporting it before the script runs.
#
# The cloud box keeps the repo at /root/knowme-memory_gate_post_train, the model
# on the data disk at /root/autodl-tmp/Qwen3-1.7B, and checkpoints under
# /root/autodl-tmp/knowme-memory_gate_post_train/runs. None of those paths exist
# on a local checkout, so nothing here is hardcoded to /root: the repo is located
# from this file's own position, the data disk is used only when it is actually
# writable, and the model directory is searched for.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(dirname -- "$SCRIPT_DIR")}"
PYTHON="${PYTHON:-$REPO_DIR/.venv/bin/python}"
SWIFT_BIN="${SWIFT_BIN:-$REPO_DIR/.venv/bin/swift}"

# Checkpoints and the ModelScope cache are large, so on the cloud box they belong
# on the data disk while the repo sits on the small system disk. Elsewhere
# /root/autodl-tmp is absent or unwritable and they stay inside the repo.
# DATA_ROOT carries the repo name so the cloud run directories keep the exact
# paths (e.g. .../autodl-tmp/knowme-memory_gate_post_train/runs/...) that the
# existing checkpoints and docs use.
if [[ -z "${DATA_ROOT:-}" ]]; then
  if [[ -w /root/autodl-tmp ]]; then
    DATA_ROOT="/root/autodl-tmp/$(basename -- "$REPO_DIR")"
  else
    DATA_ROOT="$REPO_DIR"
  fi
fi
RUNS_ROOT="$DATA_ROOT/runs"
MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$DATA_ROOT/modelscope}"

# The model is downloaded by hand, so its location varies between machines. The
# first candidate that exists wins; when none does, MODEL_PATH names the
# repo-local location and the callers' "Model directory not found" error points
# at where to put it.
if [[ -z "${MODEL_PATH:-}" ]]; then
  for candidate in \
    "$REPO_DIR/model/Qwen3-1.7B" \
    "/root/autodl-tmp/Qwen3-1.7B" \
    "$HOME/models/Qwen3-1.7B" \
    "$DATA_ROOT/Qwen3-1.7B"
  do
    if [[ -d "$candidate" ]]; then
      MODEL_PATH="$candidate"
      break
    fi
  done
  MODEL_PATH="${MODEL_PATH:-$REPO_DIR/model/Qwen3-1.7B}"
fi

export DATA_ROOT RUNS_ROOT MODELSCOPE_CACHE

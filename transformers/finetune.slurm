#!/bin/bash
#SBATCH --job-name=bloom-finetune     	       # Name of the job shown in SLURM queue
#SBATCH --ntasks=1                             # Number of tasks
#SBATCH --tasks-per-node=1                     # One task per node
#SBATCH --cpus-per-task=4                      # Number of CPUs allocated per task
#SBATCH --gpus=1                               # Request 1 GPU
#SBATCH --gpus-per-node=1                      # GPUs per node
#SBATCH --mem=32G                              # Request 32 GB of system memory
#SBATCH --constraint=v100                      # Request specific GPU type (V100)
#SBATCH --time=02:00:00                        # Maximum runtime (HH:MM:SS)
#SBATCH --output=log/%x-%j.out                 # Redirect SLURM .out log to log/ directory


# Fail fast + clean up any background processes when exiting
set -Eeuo pipefail
cleanup() {
  status=$?
  trap - EXIT INT TERM
  kill -- -$$ 2>/dev/null || true   # kill whole process group (e.g., mlflow ui, loggers)
  exit "$status"
}
trap cleanup EXIT INT TERM

# ------------------------------
# Resolve directory paths
# ------------------------------
SCRIPT_DIR=$(pwd)

JOB_ID=${SLURM_JOB_ID:-manual}
EXPERIMENT_DIR="$SCRIPT_DIR"

# Log directories (all adjacent to SLURM script)
LOG_OUT_DIR="$EXPERIMENT_DIR/log"
GPU_LOG_DIR="$EXPERIMENT_DIR/gpu_memory/$JOB_ID"
CPU_LOG_DIR="$EXPERIMENT_DIR/cpu_memory/$JOB_ID"

mkdir -p "$LOG_OUT_DIR" "$GPU_LOG_DIR" "$CPU_LOG_DIR"

# ------------------------------
# Environment Setup
# ------------------------------
source /ibex/user/$USER/miniforge/etc/profile.d/conda.sh
conda activate deepspeed-finetune
module load cuda/12.4.1


# === Where mlflow-server writes the URI ===
PUBLISH_DIR="${PUBLISH_DIR:-$SLURM_SUBMIT_DIR/.mlflow}"   # adjust if server uses a different path
URI_FILE="$PUBLISH_DIR/current_uri.txt"

# Wait until the server has written the file
for i in {1..60}; do
  if [[ -f "$URI_FILE" ]]; then
    break
  fi
  echo "Waiting for MLflow URI in $URI_FILE ..."
  sleep 2
done

if [[ ! -f "$URI_FILE" ]]; then
  echo "ERROR: Could not find MLflow URI file ($URI_FILE)"
  exit 1
fi

MLFLOW_URI="$(cat "$URI_FILE")"

echo "Using MLflow Tracking URI: $MLFLOW_URI"

# Set defaults for experiment + run group if not passed
: "${MLFLOW_EXPERIMENT_NAME:=bloom-finetune-deepspeed}"
: "${MLFLOW_RUN_GROUP:=${SLURM_JOB_NAME}-${SLURM_JOB_ID}}"


# ------------------------------
# GPU Memory Logging
# ------------------------------
nvidia-smi \
  --query-gpu=timestamp,index,name,memory.used,memory.total \
  --format=csv,nounits -l 5 > "$GPU_LOG_DIR/gpu_memory_log.csv" &
GPU_LOG_PID=$!


# ------------------------------
# Launch Training
# ------------------------------
python scripts/train.py \
  --mlflow_uri "$MLFLOW_URI" \
  --mlflow_experiment "$MLFLOW_EXPERIMENT_NAME"  \
  --mlflow_run_group "$MLFLOW_RUN_GROUP"

TRAIN_PID=$!

# ------------------------------
# CPU Memory Logging
# ------------------------------
psrecord $TRAIN_PID --include-children --interval 5 \
  --log "$CPU_LOG_DIR/cpu_memory_log.txt" &
CPU_LOG_PID=$!

# ------------------------------
# Wait & Cleanup
# ------------------------------
#wait $TRAIN_PID
kill $GPU_LOG_PID
kill $CPU_LOG_PID

echo "Analyzing memory logs for job ${SLURM_JOB_ID:-manual}"
python scripts/analyze_memory.py ${SLURM_JOB_ID:-manual} --path "$SCRIPT_DIR"

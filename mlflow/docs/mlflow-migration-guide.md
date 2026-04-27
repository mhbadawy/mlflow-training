# MLflow integration for training jobs

This guide describes how to run an MLflow tracking server on IBEX (Slurm), wire your training script to it, and log Hugging Face Transformers runs—including optional final model artifacts.

## Prerequisites

- A conda environment with **MLflow** and **Transformers** (for example `mlflow-pytorch-transformer` as in the Slurm example below).
- **`nvidia-ml-py`** in that same environment if you want GPU-related system metrics in MLflow. If `pynvml` fails to import despite `pip` reporting satisfaction, force a clean install into the active environment:

```bash
python -m pip install --upgrade --ignore-installed nvidia-ml-py mlflow
```

## 1. Run the MLflow UI on a compute node (Slurm)

Save the following as a job script (for example `mlflow_ui.sbatch`), adjust paths or resource directives if your site policy requires it, then submit with `sbatch mlflow_ui.sbatch`.

The script:

- Starts `mlflow ui` bound to `0.0.0.0` so other nodes can reach it.
- Uses a **file** backend under `mlruns` (override with `RUN_DIR`).
- Writes the tracking URI and metadata under `.mlflow` (override with `PUBLISH_DIR`) so training jobs can `source` or read a stable location.

```bash
#!/bin/bash
#SBATCH --job-name=mlflow-ui
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/mlflow-ui-%j.out

source /ibex/user/$USER/miniforge/etc/profile.d/conda.sh

conda activate mlflow-pytorch-transformer

set -euo pipefail

# === Config ===
PORT="${PORT:-5000}"
RUN_DIR="${RUN_DIR:-$SLURM_SUBMIT_DIR/mlruns}"           # MLflow FileStore
PUBLISH_DIR="${PUBLISH_DIR:-$SLURM_SUBMIT_DIR/.mlflow}"  # Published URI for clients
mkdir -p "$RUN_DIR" "$PUBLISH_DIR" logs

# === Node identity ===
IP="$(hostname -I | awk '{print $1}')"
HOST="$(hostname)"

# === MLflow UI ===
srun mlflow ui \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --backend-store-uri "file:${RUN_DIR}" &

MLFLOW_PID=$!

# === Publish tracking URI for training jobs ===
URI="http://$IP:$PORT"
echo "$URI" > "$PUBLISH_DIR/current_uri.txt"
cat > "$PUBLISH_DIR/current_uri.env" <<EOF
export MLFLOW_TRACKING_URI="$URI"
EOF
cat > "$PUBLISH_DIR/current_uri.json" <<EOF
{"tracking_uri":"$URI","host":"$HOST","ip":"$IP","port":$PORT,"job_id":"$SLURM_JOB_ID"}
EOF

ln -sf "$PUBLISH_DIR/current_uri.txt" "$PUBLISH_DIR/LATEST"

echo ""
echo "MLflow UI running on $HOST ($IP):$PORT"
echo "Tracking URI: $URI"
echo ""
echo "To view the UI from your laptop, open a tunnel:"
echo "   ssh -N -L 5000:${IP}:${PORT} ${USER}@glogin.ibex.kaust.edu.sa"
echo ""

wait "$MLFLOW_PID"
```

**Using the published URI.** From the same filesystem tree (or wherever `current_uri.env` is written), training jobs can load the URI before launching Python:

```bash
source .mlflow/current_uri.env
# Optional: echo $MLFLOW_TRACKING_URI
```

## 2. Python: imports and CLI arguments

```python
import argparse
import mlflow

parser = argparse.ArgumentParser()

parser.add_argument(
    "--mlflow_uri",
    type=str,
    required=True,
    help="MLflow tracking server URI (e.g. http://<node-ip>:5000)",
)
parser.add_argument(
    "--mlflow_experiment",
    type=str,
    required=True,
    help="MLflow experiment name",
)
parser.add_argument(
    "--mlflow_run_group",
    type=str,
    required=True,
    help="Logical run name / group label for this training job",
)

args = parser.parse_args()
```

## 3. Configure MLflow and Hugging Face autologging

Call this after parsing arguments and before constructing the trainer (order may matter for autolog hooks):

```python
mlflow.set_tracking_uri(args.mlflow_uri)
mlflow.set_experiment(args.mlflow_experiment)
mlflow.transformers.autolog(log_models=False)
```

## 4. Send training metrics to MLflow

In your `TrainingArguments`, set:

```python
report_to="mlflow",
```

so the Hugging Face Trainer streams scalars and related logs to the active MLflow run.

## 5. Wrap training in an MLflow run and log the final model

Place `trainer.train()` inside `mlflow.start_run`. Log the full Transformers bundle only from the global main process to avoid duplicate uploads in distributed training:

```python
with mlflow.start_run(run_name=args.mlflow_run_group, log_system_metrics=True):
    trainer.train()
    if trainer.is_world_process_zero():
        print("Uploading final model to MLflow server...")
        mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            artifact_path="final_model",
        )
```

## 6. Launch training with the new arguments

Set the environment variables (for example from `current_uri.env` and your own names), then invoke your entrypoint:

```bash
python scripts/train.py \
  --mlflow_uri "$MLFLOW_TRACKING_URI" \
  --mlflow_experiment "$MLFLOW_EXPERIMENT_NAME" \
  --mlflow_run_group "$MLFLOW_RUN_GROUP"
```

## 7. Adding custom model metrics (Optional)

```py
import mlflow

with mlflow.start_run():
    # ... model training ...
    accuracy = 0.95
    f1_score = 0.92
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metrics({"f1_score": f1_score, "precision": 0.93})

```



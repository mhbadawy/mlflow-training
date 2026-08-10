#!/bin/bash
#SBATCH --job-name=mlflow-ui
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/mlflow-ui-%j.out

set -euo pipefail

# === Config ===
PORT="${PORT:-5000}"
RUN_DIR="${RUN_DIR:-$SLURM_SUBMIT_DIR/mlruns}"         # where mlflow stores runs
PUBLISH_DIR="${PUBLISH_DIR:-$SLURM_SUBMIT_DIR/.mlflow}" # publish URI for clients
mkdir -p "$RUN_DIR" "$PUBLISH_DIR" logs

# === Discover node IP ===
IP="$(hostname -I | awk '{print $1}')"
HOST="$(hostname)"

# === Start mlflow UI ===
# --backend-store-uri points at the local FileStore (mlruns) dir.
srun mlflow ui \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --backend-store-uri "file:${RUN_DIR}" &

MLFLOW_PID=$!

# === Publish the URI for training jobs ===
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


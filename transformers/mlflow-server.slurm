#!/bin/bash
#SBATCH --job-name=mlflow-server
#SBATCH --output=logs/%x-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=08:00:00        # adjust as needed

set -euo pipefail

source /ibex/user/$USER/miniforge/etc/profile.d/conda.sh
conda activate mlflow+pytorch


# === Config ===
PORT="${PORT:-5000}"
RUN_DIR="${RUN_DIR:-$PWD/../finetuning/mlruns}"   # where "mlruns" will live
mkdir -p "$RUN_DIR" logs

HOSTNAME=$(hostname)
IP=$(hostname -I | awk '{print $1}')

echo "Node: $HOSTNAME (${IP})"
echo "UI  : http://${IP}:${PORT}"
echo "Runs dir: ${RUN_DIR}/mlruns"
echo "Log: logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"

# Helpful for clients: write a little connection hint
cat > "logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.conn" <<EOF
To view the UI from your laptop:
1) SSH to the login node and create a tunnel:
   ssh -N -L 5000:${IP}:${PORT} ${USER}@glogin.ibex.kaust.edu.sa

2) Open http://localhost:5000 in your browser.
EOF

# === Start mlflow UI ===
# --backend-store-uri points at the local FileStore (mlruns) dir.
# Binding to 0.0.0.0 lets you tunnel in; some clusters require 127.0.0.1 + ProxyJump.
srun mlflow ui \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --backend-store-uri "file:${RUN_DIR}"

# When the process exits, SLURM will finish the job.


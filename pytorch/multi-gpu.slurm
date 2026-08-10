#!/bin/bash

#SBATCH --job-name=resnet-multi-gpu
#SBATCH --gpus=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --time=02:00:00
#SBATCH --constraint=v100

scontrol show job $SLURM_JOBID 

source /ibex/user/$USER/miniforge/etc/profile.d/conda.sh
conda activate mlflow+pytorch


# === MLflow wiring ===
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
: "${MLFLOW_EXPERIMENT_NAME:=imagenet-ddp}"
: "${MLFLOW_RUN_GROUP:=${SLURM_JOB_NAME}-${SLURM_JOB_ID}}"



export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

export DATA_DIR=/ibex/ai/reference/CV/tinyimagenet

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
echo "Node IDs of participating nodes ${nodes_array[*]}"

# Get the IP address and set port for MASTER node
head_node="${nodes_array[0]}"
echo "Getting the IP address of the head node ${head_node}"
master_ip=$(srun -n 1 -N 1 --gpus=1 -w ${head_node} /bin/hostname -I | cut -d " " -f 2)
master_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
echo "head node is ${master_ip}:${master_port}"

sleep 5


export OMP_NUM_THREADS=1
for (( i=0; i< ${SLURM_NNODES}; i++ ))
do
     srun -n 1 -N 1 -c ${SLURM_CPUS_PER_TASK} -w ${nodes_array[i]} --gpus=${SLURM_GPUS_PER_NODE}  \
      python -m torch.distributed.launch --use_env \
     --nproc_per_node=${SLURM_GPUS_PER_NODE} --nnodes=${SLURM_NNODES} --node_rank=${i} \
     --master_addr=${master_ip} --master_port=${master_port} \
     scripts/train.py --epochs=5 --lr=0.001 --num-workers=${SLURM_CPUS_PER_TASK} --batch-size=64 \
      --mlflow_uri "$MLFLOW_URI" \
      --mlflow_experiment "$MLFLOW_EXPERIMENT_NAME" --mlflow_run_group "$MLFLOW_RUN_GROUP" &
     
done
wait


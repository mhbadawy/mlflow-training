## 1. Introduction

This workshop provides a practical, hands-on introduction to **MLflow** for experiment tracking on HPC clusters such as
**Ibex**.  
Participants will learn how to launch an MLflow tracking server through **Slurm**, log experiments from distributed
training jobs, organize metrics and artifacts, and visualize results using the MLflow UI.

MLflow is becoming a standard component in reproducible machine learning workflows. By the end of this workshop, you
will understand how to integrate it into both small-scale and large-scale training pipelines, including jobs that run
with **DeepSpeed**, **FSDP**, or **NeMo**.

---

## 2. What You Will Learn

By completing this workshop, you will learn how to:

- Understand the core components of MLflow: **Tracking**, **Artifacts**, **Runs**, **Experiments**, and **Models**
- Start an **MLflow UI server** on a Slurm node and expose it safely for remote access
- Log metrics, parameters, configuration files, and model checkpoints from Python training scripts
- Organize your ML experiments using consistent naming, directory structure, and tagging
- Connect large distributed training jobs (multi-GPU, multi-node) to a single MLflow server
- Use the MLflow UI to compare runs, validate improvements, and debug training behavior
- Store results in portable FileStore directories (`mlruns/`) and publish the tracking URI for other jobs
- Follow HPC best practices for ports, job isolation, and data cleanup

---

## 3. Understanding MLflow

Before diving into the hands-on exercises, it’s important to understand the core ideas behind MLflow and how they fit
into a reproducible machine learning workflow—especially in an HPC environment like Ibex.

### 3.1 Key Concepts

MLflow is organized around a few fundamental components:

- **Runs**  
  A single execution of your training script.  
  Each run stores parameters, metrics, logs, and artifacts.

- **Experiments**  
  A collection of related runs (e.g., “baseline”, “zero-offload”, “8-GPU training”).  
  Experiments help group and compare results.

- **Parameters (params)**  
  Values you want to track: learning rate, batch size, weight decay, model name, etc.

- **Metrics**  
  Time-series values such as loss, accuracy, throughput, memory usage, etc.

- **Artifacts**  
  Files produced during training: model checkpoints, plots, logs, configs, and more.

- **Models**  
  MLflow provides a structured way to version, package, and deploy models, though this workshop focuses mainly on the
  tracking side.

---

### 3.2 Tracking vs. Registry vs. Deployment

MLflow consists of several modules:

- **Tracking** → logs runs, metrics, params, artifacts
- **Model Registry** → stores and version-controls production-ready models
- **Deployment** → serves models to external systems
- **Projects** → defines reproducible environments

In this workshop, we focus on **MLflow Tracking**, because it integrates cleanly with HPC workflows and distributed
training.

---

### 3.3 Why Use MLflow on HPC (Ibex)?

HPC training jobs generate large volumes of logs and metrics. Without a tracking tool, it becomes difficult to:

- Compare multiple runs reliably
- Track configuration changes over time
- Understand how hyperparameters affect final performance
- Debug long-running multi-GPU or multi-node jobs
- Share results across team members or across training scripts

MLflow helps solve these issues by providing:

- A centralized **tracking server**
- A structured FileStore for metrics and artifacts
- A web UI to compare runs
- A consistent interface across all frameworks (DeepSpeed, FSDP, NeMo, ColossalAI, etc.)

This makes MLflow especially valuable in large-scale distributed training environments.

---

## 4. Workshop Directory Structure

The repository is structured to cleanly separate:

- The **MLflow tracking server**
- Multiple **client-side training frameworks**
- All **training scripts**
- A shared **environment definition**

This makes it easy for participants to launch a single MLflow server and run any number of independent training jobs (
PyTorch or HuggingFace), all logging into the same UI.

```text
mlflow/
├── environment
│   └── environment.yml
├── experiments
│   ├── client
│   │   ├── pytorch
│   │   │   └── finetune.slurm
│   │   └── transformer
│   │       └── finetune.slurm
│   └── server
│       └── mlflow-server.slurm
├── README.md
└── scripts
    ├── pytorch
    │   └── train.py
    └── transformer
        ├── config.py
        ├── data_loader.py
        ├── model.py
        └── train.py
```

### What this structure provides

- **One shared MLflow server**  
  A single `mlflow-server.slurm` job in `experiments/server/` that publishes the tracking URI used by all client jobs.

- **Multiple client frameworks**  
  Each located in `experiments/client/`, where every framework has its own `finetune.slurm`:
    - `client/pytorch/` → PyTorch DDP + MLflow
    - `client/transformer/` → HuggingFace Trainer + DeepSpeed + MLflow

- **Clean separation of responsibilities**
    - *Server jobs* located under `experiments/server/`
    - *Client jobs* located under `experiments/client/<framework>/`
    - *Training code* located under `scripts/<framework>/`

- **Reproducibility**  
  A shared Conda environment (`environment/environment.yml`) ensures consistent dependencies for both server and client
  jobs across users and compute nodes.

This structure cleanly supports multi-user workshops where many client jobs run simultaneously and all log into the same
MLflow tracking server.

---

## 5. Environment Setup

We'll use Conda to manage packages and dependencies

Make sure you're at the [`mlflow`](.) directory, the run these lines:

```bash
cd environment
conda env create -f environment.yml
```

---

## 6. Running the MLflow Tracking Server on Slurm

The MLflow Tracking Server is the central component of the workflow.
It provides:

- A **UI** for browsing experiments

- A **Tracking URI** that training jobs write logs to

- A **FileStore** (`mlruns/`) that stores metrics, artifacts, and models

In this workshop, participants launch the MLflow UI as a Slurm job, not manually.
This ensures:

- MLflow runs on a compute node with enough memory

- Every participant receives **their own isolated MLflow server**

- Training jobs can dynamically discover the correct URI

Example:

```commandline
sbatch mlflow-server.slurm
```

This job starts an MLflow UI instance on a single Slurm node and publishes a tracking URI that all training jobs will
use.

### 6.1 What the MLflow Server Slurm Script Does

1. **It sets up the environment:**

    ```bash
    source /ibex/user/$USER/miniforge/etc/profile.d/conda.sh
    conda activate mlflow-pytorch-transformer
    ```
   This ensures:
    - mlflow CLI exists

    - Python dependencies for MLflow UI are present

    - Training jobs and servers share the same environment


2. **Configures working directories:**

   Training jobs do NOT hardcode any URI.
   They discover the MLflow server through these published files.
   This is the key principle that makes the workshop scalable and reliable.

    ```bash
    PORT="${PORT:-5000}"
    RUN_DIR="${RUN_DIR:-$SLURM_SUBMIT_DIR/mlruns}"
    PUBLISH_DIR="${PUBLISH_DIR:-$SLURM_SUBMIT_DIR/.mlflow}"
    mkdir -p "$RUN_DIR" "$PUBLISH_DIR" logs
    ```

    - `RUN_DIR`: **FileStore** that holds all MLflow experiments (`mlruns/`)
    - `PUBLISH_DIR`:    Location where the server publishes **tracking URIs**


3. **Finds the node IP:**

   To allow remote access through SSH tunneling, the script determines the node IP:

    ```bash
    IP="$(hostname -I | awk '{print $1}')"
    HOST="$(hostname)"
    ```

4. **Starts MLflow UI:**

    ```bash
    srun mlflow ui \
      --host 0.0.0.0 \
      --port "${PORT}" \
      --backend-store-uri "file:${RUN_DIR}" &
    ```
   Why --host 0.0.0.0?

    - Allows external traffic (tunnel) to connect

    - Otherwise UI would be local-only


5. **Captures its PID:**

    ```bash
    MLFLOW_PID=$!
    ```

### 6.2 Publishing the MLflow Tracking URI

1. **The server publishes the full MLflow tracking URI (e.g., `http://10.0.0.5:5000`) as:**

    ```bash
    URI="http://$IP:$PORT"
    echo "$URI" > "$PUBLISH_DIR/current_uri.txt"
    ```

   Training jobs rely on **this** exact file to know which MLflow server to log into.


2. **Also exports helper files:**

    ```bash
    cat > "$PUBLISH_DIR/current_uri.env" <<EOF
    export MLFLOW_TRACKING_URI="$URI"
    EOF
    
    cat > "$PUBLISH_DIR/current_uri.json" <<EOF
    {"tracking_uri":"$URI","host":"$HOST","ip":"$IP","port":$PORT,"job_id":"$SLURM_JOB_ID"}
    EOF
    ```
   Why multiple formats?

    - `current_uri.txt`:    Training scripts read this **directly**
    - `current_uri.env`:    Users can source: `source .mlflow/current_uri.env`
    - `current_uri.json`:    Tools can parse it **programmatically**


3. **Creates a convenience symlink:**

    ```bash
    ln -sf "$PUBLISH_DIR/current_uri.txt" "$PUBLISH_DIR/LATEST"
    ```

### 6.3 Viewing MLflow UI from Your Laptop

1. **The script prints a tunnel command:**

    ```bash
    ssh -N -L 5000:<IP>:<PORT> ${USER}@glogin.ibex.kaust.edu.sa
    ```

2. **Once the tunnel is active, users open:**

    ```bash
    http://localhost:5000
    ```
   This brings up the MLflow experiment UI.

The script ends with:

```bash
wait "$MLFLOW_PID"
```

which **keeps the job alive** until the UI is terminated.

---

## 7. Training Jobs and How They Connect to MLflow

The second half of the pipeline is training jobs automatically attaching to the MLflow server created in Section 6.

**This requires:**

- Discovering the MLflow URI

- Passing the URI + experiment name to the trainer

- Initializing MLflow from inside the training script

- Logging metrics + artifacts

### 7.1 The Training Script Discovers the MLflow Server URI

Every experiment directory contains a training script:

```bash 
experiments/<framework>/finetune.slurm 
```

Inside, we define where to find the MLflow URI:

```bash
PUBLISH_DIR="${PUBLISH_DIR:-$SLURM_SUBMIT_DIR/.mlflow}"
URI_FILE="$PUBLISH_DIR/current_uri.txt"
```

Training script waits until the server writes the file:

```bash
for i in {1..60}; do
  if [[ -f "$URI_FILE" ]]; then
    break
  fi
  echo "Waiting for MLflow URI in $URI_FILE ..."
  sleep 2
done
```

Error handling:

```bash
if [[ ! -f "$URI_FILE" ]]; then
  echo "ERROR: Could not find MLflow URI file ($URI_FILE)"
  exit 1
fi
```

Read the URI:

```bash
MLFLOW_URI="$(cat "$URI_FILE")"
echo "Using MLflow Tracking URI: $MLFLOW_URI"
```

**Why this matters**

- Training jobs become 100% dynamic
- No need to hardcode ports
- Multiple users can run separate servers
- Multiple servers can run simultaneously
- This is robust HPC-friendly service discovery.

### 7.2 Passing MLflow Parameters Into Python Training Scripts

The Slurm training script launches distributed PyTorch using:

```bash
python -m torch.distributed.launch --use_env \
  --nproc_per_node=${SLURM_GPUS_PER_NODE} \
  --nnodes=${SLURM_NNODES} --node_rank=${i} \
  --master_addr=${master_ip} --master_port=${master_port} \
  "$PYTORCH_SCRIPT_DIR/train.py" \
      --epochs=5 \
      --lr=0.001 \
      --num-workers=${SLURM_CPUS_PER_TASK} \
      --batch-size=64 \
      --mlflow_uri "$MLFLOW_URI" \
      --mlflow_experiment "$MLFLOW_EXPERIMENT_NAME" \
      --mlflow_run_group "$MLFLOW_RUN_GROUP"
```

We pass MLflow settings as arguments:

- `--mlflow_uri`: Where MLflow logs should go
- `--mlflow_experiment`: Which experiment group to store runs under
- `--mlflow_run_group`:    Grouping for multi-node jobs

This ensures consistent tracking across all GPUs and all ranks.

### 7.3 Initializing MLflow Inside the Training Script

#### (PyTorch DDP Trainer vs. Hugging Face Accelerate)

Connecting a training script to MLflow requires two steps:

1. Tell the script **where** the MLflow server is

2. Log **parameters and metrics** during training

Both the PyTorch DDP trainer and the Hugging Face Accelerate trainer do this,
but the amount of code required is **very different.**

Below we first show **the PyTorch DDP approach** (manual setup)
and then the **Accelerate approach** (high-level and minimal).

### 7.3.1 MLflow Initialization in the PyTorch DDP Trainer

PyTorch alone does not provide built-in experiment tracking.
So the DDP trainer must explicitly:

- Configure MLflow URI

- Set the experiment

- Start a run

- Log tags and parameters

- Manage rank-0-only logging

- Close the run manually

This gives full control but requires more boilerplate.

**MLflow setup in train.py (PyTorch):**

```python
args = parse_args()

# 1. Point MLflow to the server started by the Slurm script
mlflow.set_tracking_uri(args.mlflow_uri)

# 2. Create (or reuse) an experiment
mlflow.set_experiment(args.mlflow_experiment)

# 3. Create a parent MLflow run for the entire DDP job
parent_run = None
if is_rank0():
    parent_run = mlflow.start_run(
        run_name=args.mlflow_run_group,
        log_system_metrics=True
    )
```

**Adding metadata & tags (rank 0 only)**

```python
if is_rank0():
    mlflow.set_tags({
        "slurm.job_id": os.environ.get("SLURM_JOBID"),
        "slurm.job_name": os.environ.get("SLURM_JOB_NAME"),
        "slurm.nodelist": os.environ.get("SLURM_JOB_NODELIST"),
        "host": socket.gethostname(),
        "world_size": os.environ.get("WORLD_SIZE"),
        "ddp": "true",
    })

```

**Logging parameters (rank 0 only)**

```python
mlflow.log_params({
    "batch_size_per_gpu": args.batch_size,
    "epochs": args.epochs,
    "lr": args.lr,
    "momentum": args.momentum,
    "weight_decay": args.weight_decay,
    "arch": "resnet50",
    "amp": "true"
})

```

**Logging metrics every epoch**

```python
mlflow.log_metrics({
    "loss/train": float(global_train_loss),
    "acc/train": float(global_train_acc),
    "loss/val": float(global_val_loss),
    "acc/val": float(global_val_acc),
}, step=epoch)
```

**Closing the run**

```python
if is_rank0() and mlflow.active_run() is not None:
    mlflow.end_run()
```

**Why PyTorch requires more work**

Because:

- DDP gives no built-in concept of “experiment” or “metrics”

- You must manually ensure only rank-0 logs

- You must manually set up the tracking URI

- You must manually handle context, parameters, tags, and artifacts

- This gives maximum flexibility, but requires manual engineering.

### 7.3.2 Hugging Face Accelerate: MLflow Logging Made Easy

While the PyTorch DDP trainer requires **manual MLflow setup** (tracking URI, experiment, tags, parent run, rank-0
logic, manual metric logging), the Hugging Face Trainer makes MLflow integration **dramatically simpler.**

The Hugging Face version does not require:

- Manually managing DDP ranks

- Creating a parent MLflow run on rank 0

- Logging parameters manually

- Logging metrics every epoch manually

- Explicitly closing the run

Instead, the workflow becomes:

- Configure MLflow with two lines

- Wrap trainer.train() inside a single mlflow.start_run() block

- Optionally log any summary metrics or artifacts

Done — MLflow handles the rest

**MLflow Setup (Hugging Face Trainer)**

The script receives the same MLflow arguments as the PyTorch version:

- `--mlflow_uri`

- `--mlflow_experiment`

- `--mlflow_run_group`

But the code required to initialize MLflow is far smaller:

```python
mlflow.set_tracking_uri(args.mlflow_uri)
mlflow.set_experiment(args.mlflow_experiment)
```

**Starting the MLflow Run**

Here, it’s just:

```python
with mlflow.start_run(
        run_name=args.mlflow_run_group,
        log_system_metrics=True,
):
    trainer.train()
```

This does three things automatically:

1. Creates a run named after the job’s run-group

2. Logs system metrics (CPU%, memory, etc.)

3. Closes the run automatically when the block ends

No need for rank-0 checks — the HF Trainer already executes its training loop in a way that only the right process
performs logging.

---

## 8. Experimenting on Ibex (Framework-Specific Workflows)

This section explains exactly how to run each experiment on Ibex and where to view the results in the MLflow UI started
by the tracking server.

In this layout, you start **one MLflow server** under `experiments/server/` and run any number of **client jobs** (
PyTorch / Transformer) that log to it.

Directory recap:

```text
mlflow/
├── experiments/
│   ├── client/
│   │   ├── pytorch/
│   │   │   └── finetune.slurm
│   │   └── transformer/
│   │       └── finetune.slurm
│   └── server/
│       └── mlflow-server.slurm

```

### 8.1 Start the MLflow Server (once per session)

```commandline
cd experiments/server
sbatch mlflow-server.slurm
```

Check it’s running:

```commandline
squeue -u $USER
```

Check allocated Port and IP from logs:

```commandline
cd logs
cat mlflow-ui-<JOB_ID>.out
```

Look for:

```text
To view the UI from your laptop, open a tunnel:
   ssh -N -L < PORT>:<IP>:<PORT> <USER>@glogin.ibex.kaust.edu.sa
```

Open the UI from your laptop (using the IP/port printed in the server log):

```commandline
ssh -N -L < PORT>:<IP>:<PORT> <USER>@glogin.ibex.kaust.edu.sa
```

Then in your browser:

```text
http://localhost:5000
```

Leave this running while you submit clients.

### 8.2 PyTorch Client (DDP ResNet)

Make sure you're at the [`mlflow`](.) directory

```commandline
cd experiments/client/pytorch
sbatch finetune.slurm
```

This job:

- Reads the server URI (via `.mlflow/current_uri.txt` from the server path you configured in the Slurm script).

- Runs the DDP ResNet training.

- Logs into the shared MLflow server.

Watch runs appear under the configured experiment in:

```text
http://localhost:5000
```

### 8.3 Transformer Client (HF Trainer + DeepSpeed)

Make sure you're at the [`mlflow`](.) directory

```commandline
cd experiments/client/transformer
sbatch finetune.slurm
```

This job:

- Reads the same server URI.

- Runs BLOOM fine-tuning with Trainer.

- Logs into the same MLflow server.

Again, refresh:

```text
http://localhost:5000
```

You should see both PyTorch and Transformer runs side-by-side.

### 8.4 Multiple Clients

You can have several client jobs (even mixed frameworks) attached to the same server:

```commandline
# Example
cd experiments/client/pytorch
sbatch finetune.slurm

cd experiments/client/transformer
sbatch finetune.slurm
```

All will write into the same `mlruns/` owned by:

```commandline
experiments/server/
```

### 8.5 Stopping the Server

When done:

```commandline
squeue -u $USER          # find the mlflow-ui job
scancel <MLFLOW_JOBID>
```

Runs remain in:

```commandline
mlflow/experiments/server/mlruns/
```

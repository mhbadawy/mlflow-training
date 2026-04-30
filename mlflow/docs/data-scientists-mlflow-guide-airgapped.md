# MLflow for data scientists on air-gapped systems (e.g. Shaheen 3)

This guide consolidates our MLflow documentation for **data scientists** who run experiments on **isolated or air-gapped** infrastructure. It focuses on **experiments, parameters, metrics, artifacts, and reproducibility**. It does **not** cover model deployment, shipping, or production operations.

---
## Table of Contents: Key Questions Answered

1. [Why is MLflow the right choice for Shaheen 3 and air-gapped HPC systems?](#part-1--why-mlflow-and-why-mlflow-instead-of-weights--biases)
2. [How does MLflow compare to Weights & Biases for experiment tracking?](#12-similarities-between-mlflow-and-weights--biases)
3. [How does experiment tracking work in MLflow? (Parameters, metrics, artifacts, & more)](#11-why-mlflow-in-general)
4. [How to use MLFlow in pytoch training script](#24---vanilla-pytorch-with-mlflow)
5. [How to use MLFlow in pytoch-lightning training script](#25---vanilla-pytorch-with-mlflow)
6. [How to use MLFlow with Hugging-face Trainer](#26-configure-mlflow-and-hugging-face-autologging)
7. [How do I view and analyze MLflow experiment metrics without a UI?](#410-how-this-integrates-with-your-system)
8. [How do I plot live metrics and monitor runs in the terminal? What is YouPlot?](#42-what-youplot-is)
9. [What are the most common CLI commands and patterns I should know?](#411-quick-reference-table)
10. [Why can’t I just use the MLflow UI? Is the CLI a good alternative?](#41-why-we-need-a-cli-plotting-tool)
11. [How do I find my run IDs and metric files?](#411-quick-reference-table)

---

## Part 1 — Why MLflow, and why MLflow instead of Weights & Biases

**Audience and reality check:** If you work on **Shaheen 3** (or similarly **air-gapped** HPC), **Weights & Biases is not an option** for normal operation: compute nodes do not reach the public internet, and exporting experiment metadata to a vendor SaaS conflicts with site policy and security review. The comparison below shows why W&B’s default model breaks on this class of system, and why standardizing on MLflow is the rational sustainable path for production experiment tracking in your environment.

### 1.1 Why MLflow in general

[MLflow](https://mlflow.org/) is an open-source platform organized around **tracking**, **projects**, **models**, and **model registry** (the exact feature set evolves with releases). For daily science work, the important idea is that MLflow is designed to be **embedded in your environment**: a **tracking URI** on a lab server, a job on a cluster, or a service inside your account. Experiments are stored in backends **you** choose (local files, SQL), which makes it straightforward to align with **data residency** and **batch/offline** training on clusters **without** requiring persistent internet from compute nodes.

At a high level you get:

- **Runs** that record **parameters**, **metrics**, **tags**, and **artifacts** (plots, checkpoints, `requirements.txt`, training logs saved as files, etc.).
- **Experiments** as named buckets of runs (one problem or project line of work).
- An **API-first** workflow: `start_run`, log param/metric/artifact, query later from Python or the CLI.

### 1.2 Similarities between MLflow and Weights & Biases

Both platforms address the same core problem—making training runs **inspectable and comparable**—and overlap in several practical ways:

- **Experiment tracking:** Each run can record **hyperparameters**, **scalar metrics**, **tags**, and **timestamps**, so you can filter and sort runs without rereading log files by hand.
- **Artifacts and rich outputs:** You can attach **plots, checkpoints, tables, and other files** to a run for later review or handoff, rather than scattering outputs only on local disk.
- **Integrations:** Widely used training stacks (e.g. **PyTorch Lightning**, **Hugging Face `Trainer`**) expose first-party or well-supported loggers for both tools, so you rarely need a bespoke logging layer for standard training loops.
- **Comparative views:** Both provide **UIs or APIs** to compare runs side by side (metrics curves, config diffs), which matters when you are choosing between model variants or debugging regressions.
- **System and hardware telemetry:** Both can surface **host- and device-level stats** (CPU/GPU utilization, memory, and related signals where the integration allows), which helps separate “slow code” from “saturated hardware.”

Those similarities explain why teams *outside* an air gap often treat them as interchangeable for day-to-day logging. **Your** constraint is not feature overlap but **where the data is allowed to live**.

### 1.3 MLflow vs Weights & Biases — executive summary

On a **closed** network, the question is not “which has the prettier default dashboard” but **which tool can legally and technically run at all**. **MLflow wins that question decisively.** The table still lists W&B for context so you know what you are *not* missing in practice on Shaheen 3.


| Dimension            | MLflow (your fit)                                                                                          | Weights & Biases (misaligned here)                                                        |
| -------------------- | ---------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Primary model        | Open source; **you** operate tracking; no vendor in the data path                                          | SaaS-first; **default path requires their cloud**                                         |
| Hosting              | **Self-hosted** backends on **your** LAN or cluster filesystem                                             | Cloud by default;                                                                         |
| Experiment UI        | **Metrics, params, artifacts** — what you need for science; CLI + exports close the loop without a browser | Polished team UI — **irrelevant if you cannot ship metrics out**                          |
| Hugging Face         | Native `report_to="mlflow"` and MLflow Transformers integration — **first-class** for your training code   | `report_to="wandb"` assumes **outbound connectivity** from the job                        |
| Sweeps / HPO         | Pair with **Optuna, Ray Tune, Slurm arrays** — matches how HPC jobs are already launched                   | Built-in sweeps are a **nice-to-have** you cannot rely on if the logger cannot phone home |
| Compliance / air gap | **Designed for** on-prem, no external telemetry **by default**                                             | Not possible on an air-gapped system (like shaheen-3)                                     |


**Bottom line for this guide:** If you cannot use W&B’s cloud, **debating W&B’s UI polish is a distraction**. MLflow gives you **working experiment tracking** without fighting your network boundary — migrate so new runs land where your policies already allow them.


### 1.4 Detailed comparison (what matters on a cluster or air-gapped site)

#### Deployment and networking

**MLflow.** You run the tracking server (or use a managed offering **you** control). On HPC, the standard pattern is a job or service on the cluster network, artifacts on **shared filesystem** or object storage **inside** the boundary. **No third party ever sees your metrics** unless you explicitly bridge out — which you will not on Shaheen 3.

**W&B.** The **default** is logging to **W&B’s cloud**. Training nodes need **outbound HTTPS** — which **air-gapped compute does not provide**.


#### Data governance and security

**MLflow.** Governance is **entirely yours**: artifact permissions, who can hit the tracking server, audit logs on **your** metal. Air-gapped and **zero external telemetry** are **normal** deployment modes, not special SKUs.


#### Integrations and the Hugging Face `Trainer`

Both support `report_to`, not a capability gap: `report_to="mlflow"` is officially supported and maps the same training metrics into runs you own. On Shaheen 3, **MLflow is the integration that actually runs end-to-end**. Prefer MLflow in new code

#### Integration and the lightning 

Both logging utilites are supported by lightning using:

```py
from lightning.pytorch.loggers import WandbLogger
```
or
```py
from lightning.pytorch.loggers import MLFlowLogger
```


### 1.5 Why you should standardize on MLflow on Shaheen 3

On an air-gapped supercomputer, **MLflow is the practical default** for experiment tracking:

1. **Policy and network reality.** Compute nodes do **not** call external experiment APIs; metadata and artifacts must stay on **institution-controlled** storage.
2. **HPC-native workflow.** Training is **Slurm** (or equivalent); a **self-hosted** tracking URI and file/DB backend match how jobs are actually scheduled and how scratch and project filesystems work.
3. **Reproducibility is the mission.** You need **durable, queryable runs** — parameters, metrics, artifacts, git tags — auditable inside the fence. MLflow delivers that **without** an external observability dependency.

### 1.6 What about Weights & Biases hype?

Stories about W&B’s **pretty dashboards** and **Sweeps** assume **internet-connected** GPUs and procurement that allows **routine export** of experiment streams to a vendor. **That is not your Shaheen 3 operating mode.** 

<!-- ### 1.7 Conclusion — migrate to MLflow and commit

**Weights & Biases is the wrong abstraction for routine experiment tracking on an air-gapped Shaheen 3 class system:** it assumes connectivity and vendor infrastructure your environment does **not** grant. **MLflow is the right abstraction**: open source, on-prem, `MLFLOW_TRACKING_URI` beside your jobs**, runs and artifacts on **your** storage, CLI and notebooks for inspection without an outbound pipe. -->

---

## Part 2 — Migrating from Weights & Biases to MLflow


### 2.1 Pre-built MLflow Image

You do **not** need to manually install MLflow or related packages. A pre-built environment image will be provided, which comes with **MLflow** (and related deps) already installed and configured.

### 2.2 Optional cluster pattern: publishing the tracking URI (Slurm / IBEX-style)

The following illustrates how an MLflow **HTTP** frontend can run on a compute node while publishing a `MLFLOW_TRACKING_URI` file that training jobs can `source`. This matches **HPC-first** workflows: training scripts read a stable path on shared filesystem rather than guessing hostnames.

Save as a job script (for example `mlflow_ui.sbatch`), adjust paths or resource directives per site policy, then submit with `sbatch mlflow_ui.sbatch`.

The script:

- Starts `mlflow ui` bound to `0.0.0.0` so other nodes can reach it.
- Uses a **file** backend under `mlruns` (override with `RUN_DIR`).
- Writes the tracking URI and metadata under `.mlflow` (override with `PUBLISH_DIR`) so training jobs can `source` or read a stable location.
- You can overide PUBLISH_DIR path and RUN_DIR path as needed when executing slurm script as following:
`
sbatch --export=PUBLISH_DIR='/your/custom/path/.mlflow',RUN_DIR='/your/other/custom/path/mlruns' mlflow-server.slurm
`

```bash
#!/bin/bash
#SBATCH --job-name=mlflow-ui
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/mlflow-ui-%j.out


# The following commented lines are the conda environment setup on ibex for this example, replace them with the pre-installed mlflow images that have been provided 
# ====================================================
# source /ibex/user/$USER/miniforge/etc/profile.d/conda.sh
# conda activate mlflow-pytorch-transformer
# ====================================================


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


**Note:** you can use `$MLFLOW_TRACKING_URI` as an environment variable directly from the training script or you can pass it as an argument for the training script, both are valid. The following steps explains the argument passing approach. 

### 2.3 Python: CLI arguments for MLflow

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
### 2.4 - Vanilla PyTorch with MLflow
#### Auto logging:
you don't need anything but these 2 line:

```py
import mlflow
mlflow.pytorch.autolog()
```

**Example:**

```py
import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Enable autologging
mlflow.pytorch.autolog()

# Create synthetic data
X = torch.randn(1000, 784)
y = torch.randint(0, 10, (1000,))
train_loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

# Your existing PyTorch code works unchanged
model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop - metrics, parameters, and models logged automatically
for epoch in range(10):
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

#### Manual logging:
You can explicitly type your mlflow logging if you need further customization:
using mlflow functions such as: `mlflow.start_run()` , `mlflow.log_params()` , `mlflow.log_metrics()` and `mlflow.pytorch.log_model()`.
The bellow example shows how to use them:
```py
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)


# Training parameters
params = {
    "epochs": 5,
    "learning_rate": 1e-3,
    "batch_size": 64,
}

# Training with MLflow logging
with mlflow.start_run():
    # Log parameters
    mlflow.log_params(params)

    # Initialize model and optimizer
    model = NeuralNetwork()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=params["learning_rate"])

    # Training loop
    for epoch in range(params["epochs"]):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        # Log metrics per epoch
        avg_loss = train_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        mlflow.log_metrics({"train_loss": avg_loss, "train_accuracy": accuracy}, step=epoch)

    # Log final model
    mlflow.pytorch.log_model(model, name="model")
```

### 2.5 PyTorch Lightning with MLflow (alternative training stack)

This section duplicates `mlflow-with-lightning.md` in full substance for Lightning users.


#### Configure MLflow (URI and experiment)

Before training, point the client at your tracking backend and choose an experiment:

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")  # or file:/path, S3, etc.
mlflow.set_experiment("my_experiment")
```

The same URI and experiment name are passed to the Lightning logger (below).

#### Create an `MLFlowLogger`

```python
from lightning.pytorch.loggers import MLFlowLogger

mlf_logger = MLFlowLogger(
    experiment_name="my_experiment",
    tracking_uri="http://localhost:5000",
    run_name="optional_run_name",
    # log_model=True uploads checkpoints; set False if you only want metrics
    log_model=False,
)
```

**Avoid nested runs for the same training job:** if you use `MLFlowLogger`, you typically **do not** wrap `trainer.fit()` in `mlflow.start_run(...)` for the same experiment, unless you intentionally want a parent/child run hierarchy.

#### Wire the logger into `Trainer`

```python
import lightning.pytorch as pl

trainer = pl.Trainer(
    logger=mlf_logger,
)
```


| Trainer argument           | Typical use with MLflow                              |
| -------------------------- | ---------------------------------------------------- |
| `logger`                   | `MLFlowLogger` (or a list of loggers).               |
| `log_every_n_steps`        | How often step-level metrics are flushed to the run. |
| `max_steps` / `max_epochs` | Stopping policy; does not change how logging works.  |


#### Log from a `LightningModule`

In `training_step`, `validation_step`, `test_step`, and hooks like `on_train_epoch_end`, use `**self.log**`. Lightning routes these to all attached loggers, including MLflow.

```python
def training_step(self, batch, batch_idx):
    loss = ...
    self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
    return loss
```


| `self.log` flags (common) | Effect                                                   |
| ------------------------- | -------------------------------------------------------- |
| `on_step=True`            | Log at each step (and optionally aggregate).             |
| `on_epoch=True`           | Emit epoch-level values (e.g. averaged validation loss). |
| `prog_bar=True`           | Show in the training progress bar.                       |


**Hyperparameters:** call `self.save_hyperparameters()` in `__init__` (optionally with `ignore=[...]` for large non-serializable objects). They are sent to loggers, including MLflow, as run parameters.

```python
def __init__(self, lr: float, ...):
    super().__init__()
    self.save_hyperparameters(ignore=["large_buffer"])
```

#### System metrics

`mlflow.start_run(log_system_metrics=True)` only applies when you open a run that way. Lightning’s `MLFlowLogger` creates runs through the client API, so that flag on `start_run` is not in play.

**Global enable (typical in MLflow 2/3+):**

```python
import mlflow

mlflow.enable_system_metrics_logging()
```

If system metrics still do not appear for runs created by the logger, check MLflow’s behavior for your version: you may need a small callback that starts MLflow’s `SystemMetricsMonitor` for the active run id (as returned by the logger) to mirror `start_run` behavior.


#### Checkpoints and logged artifacts (experiments focus)


| Mechanism                                                 | Role                                                                     |
| --------------------------------------------------------- | ------------------------------------------------------------------------ |
| `mlflow.pytorch.log_model` (in a callback or after `fit`) | Log a `torch.nn.Module` as an MLflow model artifact.                     |
| `MLFlowLogger(log_model=...)`                             | Controls whether Lightning uploads model/checkpoint artifacts to MLflow. |


If you use both a checkpoint callback and `mlflow.pytorch.log_model`, you can log **one** “best” model from the best checkpoint after training to avoid duplicating every intermediate file as a large artifact.

#### What appears in MLflow


| Source                                           | In MLflow                                                             |
| ------------------------------------------------ | --------------------------------------------------------------------- |
| `MLFlowLogger`                                   | A run in the given experiment, linked to the tracking URI.            |
| `self.log(...)`                                  | Metrics (names and step/epoch as configured).                         |
| `save_hyperparameters`                           | Parameters (unless ignored).                                          |
| `enable_system_metrics_logging` / custom monitor | System metrics (OS/resource), when supported for that run.            |
| `mlflow.pytorch.log_model` (if used)             | Artifacts under the chosen `artifact_path` (e.g. `MLmodel`, weights). |


#### Practical tips (Lightning)

1. **One run per `trainer.fit`:** the logger usually creates a single run; metrics from that run belong together.
2. **Resume training:** use Lightning’s checkpointing (`ckpt_path`) rather than depending on MLflow for optimizer state; MLflow remains for **tracking**, not the primary source of resumable training state unless you build that flow explicitly.
3. **Reproducibility:** log git commit, data version, or seeds as extra parameters (manual `mlflow.log_param` in setup code, or hyperparameters) if you need them in the run metadata.

#### Quick reference wiring (Lightning)

```python
import mlflow
import lightning.pytorch as pl
from lightning.pytorch.loggers import MLFlowLogger

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("my_experiment")
mlflow.enable_system_metrics_logging()  # optional; verify for your MLflow + logger combo

mlf_logger = MLFlowLogger(
    experiment_name="my_experiment",
    tracking_uri="http://localhost:5000",
    run_name="try-1",
    log_model=False,
)

trainer = pl.Trainer(max_epochs=10, logger=mlf_logger, accelerator="auto", devices=1)
trainer.fit(lit_module)  # lit_module: pl.LightningModule
```

Minimal contract: **config → logger → `Trainer` → `fit`**, with **metrics and params** flowing through `**self.log`** and `**save_hyperparameters**` on the `LightningModule`.

### 2.6 Configure MLflow and Hugging Face autologging

Call this after parsing arguments and before constructing the trainer (order may matter for autolog hooks):

```python
mlflow.set_tracking_uri(args.mlflow_uri)
mlflow.set_experiment(args.mlflow_experiment)
mlflow.transformers.autolog(log_models=False)
```

Note: prevent using `log_models=true` with HuggingFace Trainer as it may result in a deadlock (freezing the train script) while trying to save the model

### 2.7 Send training metrics to MLflow from `Trainer`

In your `TrainingArguments`, set:

```python
report_to="mlflow",
```

so the Hugging Face `Trainer` streams scalars and related logs to the active MLflow run.

### 2.8 Wrap training in an MLflow run and log the final model bundle

Place `trainer.train()` inside `mlflow.start_run`. Log the full Transformers bundle only from the **global main process** to avoid duplicate uploads in distributed training:

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

For **pure experiment tracking**, the critical parts are `run_name`, `log_system_metrics=True`, `report_to="mlflow"`, and `autolog`. Logging the full model is optional depending on policy and storage.

### 2.9 Launch training with the new arguments

Set the environment variables (for example from `current_uri.env` and your own names), then invoke your entrypoint (your training script):

```bash
python scripts/train.py \
  --mlflow_uri "$MLFLOW_TRACKING_URI" \
  --mlflow_experiment "$MLFLOW_EXPERIMENT_NAME" \
  --mlflow_run_group "$MLFLOW_RUN_GROUP"
```

### 2.10 Adding custom metrics (optional)

```python
import mlflow

with mlflow.start_run():
    # ... model training ...
    accuracy = 0.95
    f1_score = 0.92
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metrics({"f1_score": f1_score, "precision": 0.93})
```

### 2.11 Run grouping note (important for sweeps and batches)

MLflow does not expose a single first-class object called “run group” in the CLI. Teams usually implement **grouping** via **tags**, **run name**, or **parent/child runs**. In this repository’s training jobs, `--mlflow_run_group` is passed as `run_name` so all jobs in one logical batch share the same display label — useful when comparing runs in exports or JSON.

**Reproducibility checklist** when reviewing a group of runs:

1. Same **code**: tag or param with **git SHA** (or container image digest).
2. Same **data**: artifact or tag pointing to dataset version / snapshot ID.
3. Same **environment**: logged `requirements.txt` or conda env under artifacts.
4. **Params** that actually control the sweep (learning rate, seed, etc.).

---

## Part 3 — Using the CLI to monitor experiments and runs (and why the browser UI is often secondary on air-gapped HPC)

Official CLI reference: [MLflow Command-Line Interface](https://mlflow.org/docs/latest/cli.html).

### 3.1 Why you might not rely on MLflow UI on Shaheen 3–class systems — and whether the CLI “replaces” it

**Why MLflow UI can be hard to use as your main tool**

- **Network and policy:** Air-gapped or tightly controlled systems often **do not** allow arbitrary long-lived HTTP services, port forwarding from laptops, or browser access to internal node IPs. Even when a UI job is technically possible, **policy**, **firewall rules**, or **operational friction** mean many scientists work **only** from batch jobs and login nodes.
- **Headless workflows:** Training runs on compute nodes; **inspection** may happen from a login node or a post-processing machine with **no graphical browser** or no route to the tracking host.
- **No “tail log” subcommand:** As the CLI guide states, there is **no** separate “tail training log” command in core MLflow. **Structured** training history is what was logged as **metrics** and **params**; **unstructured** logs are usually **artifact files**. The CLI exposes that truth directly via `runs describe`, CSV export, and artifact download.

**Is the CLI a good substitute for MLflow UI?**

- **For experiments — largely yes**, for listing experiments and runs, pulling **full JSON** for a run (metrics, params, tags), exporting **CSV** for spreadsheet comparison, and downloading **artifacts**. These map to what data scientists need for **reproducibility audits** (sort by a validation metric, open matching `run_id`, inspect tags and artifacts).
- **Not feature-identical to a full browser UI:** On a laptop with a fast path to an MLflow server, **clicking** through compare views can feel faster for some tasks. On Shaheen 3, **that path is often unavailable** — so the comparison that matters is **CLI + CSV + notebooks + YouPlot** versus **no systematic tracking at all**. In that framing, the CLI is not a “second-class” UI; it is **how you ship reproducible experiment review** without a GUI route to the tracker.

So: **the CLI does not replicate every dashboard pixel**, but for **air-gapped, SSH-first** science it is the **primary, supported** way to query the backend **without** browser access — plus CSV/`jq` and terminal plots for curves.

### 3.2 Tracking URI and concepts (same as Python client)

Before any CLI command, MLflow needs to know **where metadata and (usually) artifact pointers live**.


| Concept          | Role                                                                                                               |
| ---------------- | ------------------------------------------------------------------------------------------------------------------ |
| **Tracking URI** | Address of the tracking server or local store (e.g. `http://mlflow.example.com:5000` or `file:///path/to/mlruns`). |
| **Experiment**   | A named bucket of runs (one problem or project line of work).                                                      |
| **Run**          | One execution: parameters, metrics, tags, artifacts.                                                               |
| **Artifacts**    | Files attached to a run (plots, checkpoints, `requirements.txt`, training logs saved as files, etc.).              |


The CLI reads the same tracking configuration as the Python client. If you set nothing, many setups default to a **local** file store under `./mlruns`.

### 3.3 Connect to an existing MLflow server

#### Set the tracking URI (required for a remote server)

Point every shell session (or job script) at the server your team runs:

```bash
export MLFLOW_TRACKING_URI="http://<host>:<port>"
```

Examples:

```bash
# HTTP tracking server on the LAN
export MLFLOW_TRACKING_URI="http://10.0.0.42:5000"
```

Verify that the CLI sees the same backend:

```bash
mlflow experiments search --max-results 5
```

sample output:
```
Experiment Id       Name              Artifact Location                                                  
------------------  ----------------  -------------------------------------------------------------------
0                   Default           mlflow-artifacts:/0                                                
399780980698506702  test-experiment1  file:///mlruns/399780980698506702
407043311271952138  finance spdm      mlflow-artifacts:/407043311271952138                               
543114047979669464  test-experiment2  file:///mlruns/543114047979669464
866406674350565752  test-experiment3  file:////mlruns/866406674350565752
```

If this errors or returns an empty list unexpectedly, fix **network reachability**, **URI spelling**, and **authentication** before debugging individual commands.

#### Persisting the URI

Typical patterns:

- Add `export MLFLOW_TRACKING_URI=...` to your shell profile or project `.env` that you `source`.


Until `mlflow experiments search` works from your machine, treat connectivity as **not** solved.

### 3.4 List experiments and resolve IDs

Experiments have a stable **numeric or string ID** (depending on backend) and a **name**. Many CLI commands require `--experiment-id`.

**Search experiments** (active, deleted, or all):

```bash
mlflow experiments search
mlflow experiments search --view all
mlflow experiments search --max-results 50
```

**Inspect one experiment** (by ID or name, table or JSON):

```bash
mlflow experiments get --experiment-id 12
mlflow experiments get --experiment-name "my-baseline-sweep" --output json
```

From the JSON or table output, note **artifact location** and **experiment ID**; you will reuse the ID in `runs list` and CSV export.

### 3.5 Runs: list, describe, compare

#### List runs in an experiment

```bash
mlflow runs list --experiment-id 12
mlflow runs list --experiment-id 12 --view all
```

Use this to copy **run IDs** for deeper inspection or downloads.

#### Full run payload (metrics, params, tags, status)

Get everything the server stores for one run as **JSON** (good for `jq`, scripts, or pasting into a notebook):

```bash
mlflow runs describe --run-id <run_id>
```

sample output:

```json
{
    "info": {
        "artifact_uri": "mlflow-artifacts:/407043311271952138/9d15995ec7b4425ca0637d1894822dd2/artifacts",
        "end_time": 1777241108495,
        "experiment_id": "407043311271952138",
        "lifecycle_stage": "active",
        "run_id": "9d15995ec7b4425ca0637d1894822dd2",
        "run_name": "real estate rg",
        "start_time": 1777241058188,
        "status": "FAILED",
        "user_id": "user1"
    },
    "data": {
        "metrics": {
            "train_loss_epoch": 0.025587234646081924,
            "train_loss_step": 0.023293079808354378,
            "epoch": 5.0,
            "val_loss": 0.0429033599793911,
            "val_loss_avg": 0.15990781784057617,
            "train_loss_avg": 0.05968168005347252,
            "system/gpu_0_utilization_percentage": 99.0,
            "system/gpu_0_memory_usage_megabytes": 3292.5,
            "system/system_memory_usage_megabytes": 13113.6,
            "system/gpu_0_power_usage_percentage": 99.7,
            "system/cpu_utilization_percentage": 10.1,
            "system/disk_available_megabytes": 317001.6,
            "system/disk_usage_megabytes": 159795.6,
            "system/network_receive_megabytes": 0.6235329999999522,
            "system/system_memory_usage_percentage": 79.7,
            "system/network_transmit_megabytes": 0.6278859999999895,
            "system/gpu_0_power_usage_watts": 54.9,
            "system/disk_usage_percentage": 33.5,
            "system/gpu_0_memory_usage_percentage": 51.1
        },
        "params": {
            "batch_size": "512",
            "block_size": "64",
            "learning_rate": "0.0003",
            "eval_iters": "200",
            "eval_every_n_epochs": "33"
        },
        "tags": {
            "mlflow.source.name": "train.py",
            "mlflow.source.git.commit": "f328948b4beed94a693875eda6f9bbd592ee0e91",
            "mlflow.user": "user1",
            "mlflow.source.type": "LOCAL",
            "mlflow.runName": "real estate rg"
        }
    },
    "inputs": {
        "model_inputs": [],
        "dataset_inputs": []
    },
    "outputs": {
        "model_outputs": []
    }
}

```

Example with filtering (requires `jq`):

```bash
mlflow runs describe --run-id <run_id> |  jq '.data.metrics | to_entries[] | select(.key == "val_loss_avg")'
```

sample output:
```
{
  "key": "val_loss_avg",
  "value": 0.15990781784057617
}
```

#### Spreadsheet-friendly comparison for one experiment

Export all runs in an experiment to CSV (metrics/params columns—useful for offline pivots):

```bash
mlflow experiments csv --experiment-id <experiment-id> -o <output-file-name>.csv
```

This supports **reproducibility audits**: sort by a validation metric, then open the matching `run_id` for artifacts and tags.

Use `mlflow runs describe` on two runs and diff the JSON (or use the UI compare view when available) to see what changed.

### 3.6 Artifacts: list and download

Artifacts live under each run’s artifact root (implementation depends on artifact store: local path, S3, GCS, etc.).

**List top-level artifacts** (JSON list):

```bash
mlflow artifacts list --run-id <run_id>
```

Sample output:

```sh
[{
  "path": "figures",
  "is_dir": true
}]
```

**List a subdirectory** (e.g. `figures` or `plots/` or `checkpoints/`):

```bash
mlflow artifacts list --run-id <run_id> --artifact-path plots
```

Sample output:

```
[{
  "path": "figures/val_pred_epoch_0000.png",
  "is_dir": false,
  "file_size": 192857
}]
```

<!-- **Copy** everything for a run (or one path) to your selected destination directory:

```bash
mlflow artifacts download --run-id <run_id> --dst-path ./downloaded_run
mlflow artifacts download --run-id <run_id> --artifact-path model --dst-path ./model_only
``` -->

If your training code wrote **stdout** to a file and logged it with `mlflow.log_artifact`, that file appears here like any other artifact.


### 3.7 Minimal end-to-end CLI workflow (copy-paste)

Replace placeholders with values from your environment.

```bash
export MLFLOW_TRACKING_URI="http://<host>:<port>"

# 1) Find your experiment
mlflow experiments search | head

# 2) Confirm ID and artifact root
mlflow experiments get --experiment-name "<your_experiment_name>" --output json

# 3) List recent runs
mlflow runs list --experiment-id <experiment_id>

# 4) Inspect one run (metrics, params, tags)
mlflow runs describe --run-id <run_id> | jq .

# 5) Copy artifacts to your desired location
mlflow artifacts list --run-id <run_id>
mlflow artifacts download --run-id <run_id> --dst-path ./artifacts_<run_id>

# 6) Optional: tabular export of all runs in the experiment
mlflow experiments csv --experiment-id <experiment_id> -o all_runs.csv
```

---

## Part 4 — CLI plotting with YouPlot (`uplot`):

### 4.1 Why we need a CLI plotting tool

In SSH-only or air-gapped workflows you still need **quick visual feedback** on **scalar metrics** (loss curves, GPU usage, etc.). A **terminal plotting** tool lets you:

- Plot **plain-text metric files** without starting a browser or Jupyter.
- Compose with **Unix tools**: `watch`, shell loops, `jq`, and file paths.
- Stay in the **same environment** where you already run `mlflow` CLI and job scripts.

This does **not** replace full notebook analysis or every UI feature; it **closes the loop** between “logging to MLflow” and “seeing the curve **now**.”

### 4.2 What YouPlot is

[YouPlot](https://github.com/red-data-tools/YouPlot) is a command-line program that draws charts **in the terminal**, backed by [UnicodePlot](https://github.com/red-data-tools/unicode_plot.rb). For our workflow the important properties are:


| Property              | Relevance                                                                                                     |
| --------------------- | ------------------------------------------------------------------------------------------------------------- |
| **Subcommands**       | `uplot line` fits **time-ordered** metric files without GNUplot scripts or Python glue.                       |
| `**stdin` and files** | Matches stream or path at end of pipeline in the upstream README.                                             |
| **Delimiter (`-d`)**  | **Space** is common for MLflow metrics; tab is YouPlot’s default when you omit `-d`, so `-d " "` is explicit. |
| `**--fmt`**           | Map columns to **x** / **y** (e.g. `xy` for step vs value).                                                   |
| **Unicode output**    | Readable in modern terminals; sizing via `-w`, `-h`.                                                          |
| **Pipelines**         | `-O` passes data through; default plot to stderr per project docs.                                            |
| **License**           | MIT; installable via `gem`, Homebrew, Nix, Guix, etc.                                                         |


YouPlot is **not** claimed to replace notebooks, or the full MLflow UI; it is the **low-friction default** for the **documented shell + MLflow** path.

### 4.3 Why YouPlot versus other CLI tools

Our case = **terminal + MLflow text metrics + shell pipelines**.


| Alternative                       | Why it is a weaker *default here*                                                            |
| --------------------------------- | -------------------------------------------------------------------------------------------- |
| **GNUplot**                       | Powerful but high setup cost for one-liners; less natural as a pipe stage.                   |
| **ttyplot**                       | Great for streaming one series; less general for arbitrary on-disk MLflow files and `--fmt`. |
| **termgraph** / **graph-cli**     | Often Python-backed and bar/summary oriented.                                                |
| **plotext** / **bashplotlib**     | Need Python per plot; awkward if the flow is bash + `mlflow` + `watch` with no venv.         |
| **chafa** / **img2txt** / **viu** | Need an image first; not data → chart from delimited text.                                   |
| **Rich** / **Textual**            | Libraries for apps; not the same as a pipe-friendly CLI binary.                              |


We still use those tools when the task fits. YouPlot wins here when you want **fast feedback on logged metrics in the same shell session**.

### 4.4 Limitations (upstream and practical)

- **Not a dedicated time-series engine**; we plot **step vs value**, which matches normal MLflow logging.
- **Ruby implementation**; `count` is not fast for huge data — `**line`** on metric-sized files is usually fine; pre-aggregate with `awk` if needed.
- For **interactive sub-millisecond dashboards** or rich desktop GUIs, add notebooks or a web UI — outside the scope of **lightweight terminal-first monitoring**.

### 4.5 On-disk layout of a local `mlruns` tree (file store)

When you use the **file store** (default `mlruns` under the project directory), MLflow arranges data like this (IDs are examples):

```sh
<project>/mlruns/
└── 407043311271952138/                    # Experiment ID (directory name)
    └── 1ec66186e2064779901587b73000a9ee/  # Run ID
        ├── artifacts/                    # Logged files (models, plots, etc.)
        ├── meta.yaml                     # Run metadata
        ├── metrics/                      # One file per metric name (whitespace-separated step/value lines)
        │   ├── system/                   # System metrics (subdirectory)
        │   │   ├── gpu_0_power_usage_percentage
        │   │   └── …
        │   ├── train_loss_avg
        │   ├── train_loss_epoch
        │   └── val_loss
        ├── params/                       # One file per param (single line value)
        │   ├── batch_size
        │   ├── learning_rate
        │   └── …
        └── tags/                         # One file per tag
            ├── mlflow.runName
            ├── mlflow.source.name
            └── …
```

- `**metrics/<name>**`: Plain text; each line is typically **step**, **value**, and **wall time** (space-separated), as MLflow logs them.
- `**params/`** and `**tags/**`: One file per key; values stored as text.

This layout is what YouPlot reads when you plot **directly from disk** (even if you also use the remote CLI for metadata).

### 4.6 Plot a single metric file with YouPlot

Assume [YouPlot](https://github.com/red-data-tools/YouPlot) is installed; see the upstream project for installation (this guide does **not** duplicate install steps from `install_uplot.md`).

Point `uplot line` at a concrete metric file under `metrics/`. MLflow’s metric files are usually **space-delimited**; set the delimiter accordingly.

```bash
uplot line -d " " \
  /path/to/your/project/mlruns/<experiment-id>/<run-id>/metrics/train_loss_avg
```

Sample output:

![plot](./Screenshot%20from%202026-04-27%2010-22-22.png)


| Item             | Description                                                                                                                         |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `**uplot line**` | Renders a line chart of the metric series.                                                                                          |
| `**-d " "**`     | **Space** as the field delimiter, matching the default MLflow metric file format.                                                   |
| **File path**    | The metric file to read. If the layout includes headers or a non-default column order, use `**-H`** and/or `**--fmt**` (see below). |


For the full set of subcommand options, run `uplot line --help`.

### 4.7 Monitor a metric with periodic refresh (`watch`)

`**watch**` re-runs a command at a fixed interval and redraws the terminal — useful while a run is still appending to a metric file.

```bash
watch -n 5 'uplot line -d " " /path/to/your/project/mlruns/<experiment-id>/<run-id>/metrics/system/gpu_0_memory_usage_percentage'
```
![plot](./Screenshot%20from%202026-04-28%2018-03-22.png)

| Item                    | Description                                                                                         |
| ----------------------- | --------------------------------------------------------------------------------------------------- |
| `**watch**`             | Periodically executes the **command** and updates the display.                                      |
| `**-n 5`**              | **Interval in seconds** between invocations (here, every 5 seconds).                                |
| **Outer single quotes** | The entire `uplot` line is one argument to `watch`, so the shell does not split the path on spaces. |


On systems without GNU `watch`, use an equivalent utility or a short `while` loop with `sleep`.

### 4.8 Relative paths from a run’s `metrics` directory and `--fmt`

If you `cd` to a run’s `metrics` folder, you can pass **relative** paths and use `**--fmt xy`** so the first column is **x** and the second **y** (step vs value).

Then:

```bash
watch -n 5 'uplot line -d " " system/gpu_0_power_usage_percentage --fmt xy'
```


| Item                  | Description                                                                                                    |
| --------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Working directory** | Must be the run’s `metrics` directory; otherwise `system/...` will not resolve.                                |
| `**-d " "`**          | Delimiter for fields (space, per MLflow).                                                                      |
| **Relative path**     | File **relative to** the current `metrics` directory.                                                          |
| `**--fmt xy`**        | First column = **x** (e.g. step), second = **y** (e.g. value). Use `uplot line --help` for other `fmt` values. |


Screenshots illustrating YouPlot and `watch` were included in the original `mlflow-cli-plotting.md` (same directory as this guide).

### 4.9 Monitor run status with the MLflow CLI and `watch`

To poll run metadata (status, params, metrics summary) **without** the UI, wrap `**mlflow runs describe`** in `watch`:

```bash
watch -n 5 'mlflow runs describe --run-id 5c8b298aeb2a41a0b7091853adb409b2'
```

Use **single quotes** around the `mlflow` invocation so the shell passes it unchanged. Some users prefer double quotes when the run ID is literal:

```sh
watch -n 5 "mlflow runs describe --run-id 5c8b298aeb2a41a0b7091853adb409b2"
```

#### `mlflow runs describe` reference


| Option / behavior          | Description                                                                                                                    |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `**mlflow runs describe**` | Writes a **JSON** document to **stdout** describing one run (metadata, params, metrics, tags; exact fields depend on version). |
| `**--run-id <RUN_ID>`**    | **Required.** The run’s UUID — from the UI, `mlflow runs list`, or the run directory name under `mlruns/<experiment_id>/`.     |
| **Output**                 | JSON for `**jq`** or other tools. Many versions do not offer a non-JSON output mode; see `mlflow runs describe --help`.        |


```bash
mlflow runs describe --help
```

#### `watch` in this context

Every **5 seconds** (or your chosen interval), the command runs again so you see updates after the backend or file store changes. Screenshot references: see original `mlflow-cli-plotting.md`.

### 4.10 How this integrates “with your system”


| Layer                 | Role                                                                                                                          |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **MLflow tracking**   | Python loggers / `start_run` write metrics, params, tags, and optionally artifacts to the **tracking URI** or local `mlruns`. |
| **CLI**               | `mlflow experiments`, `mlflow runs`, `mlflow artifacts` query the **same** metadata — primary path when UI is impractical.    |
| **File store**        | Under `mlruns/.../metrics/`, metric time series are **plain text** files.                                                     |
| **YouPlot + `watch`** | Read those files (or paths you copy to scratch) for **live line plots** in the terminal.                                      |


Together: **log → inspect via CLI or JSON → plot raw metric files with `uplot` → refresh with `watch`**.

### 4.11 Quick reference table


| Goal                                       | Command or pattern                                                 |
| ------------------------------------------ | ------------------------------------------------------------------ |
| Understand file-store paths                | `mlruns/<experiment_id>/<run_id>/metrics/...`                      |
| Plot a metric (absolute path)              | `uplot line -d " " <path-to-metric-file>`                          |
| Plot from `metrics/` with column semantics | `cd .../metrics` then `uplot line -d " " <relative-path> --fmt xy` |
| Auto-refresh terminal output               | `watch -n <seconds> '<command>'`                                   |
| Run details as JSON                        | `mlflow runs describe --run-id <id>`                               |


Replace example paths and run IDs with values from your environment.

---

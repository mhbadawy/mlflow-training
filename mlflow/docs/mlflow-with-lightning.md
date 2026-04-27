# Using PyTorch Lightning with MLflow

This guide explains how to connect **PyTorch Lightning** training to **MLflow** for experiment tracking: runs, metrics, parameters, and optional artifacts.

Use **Lightning’s `MLFlowLogger`** so `self.log(...)` from your `LightningModule` is written to MLflow without hand-written `mlflow.log_metric` calls in the training loop.

---

## Dependencies

| Package | Role |
|---------|------|
| `lightning` | `lightning.pytorch` — `LightningModule`, `Trainer`, loggers. |
| `mlflow` | Tracking server / file store; `MLFlowLogger` uses the MLflow client under the hood. |

Install (example):

```bash
pip install lightning mlflow
```

---

## 1. Configure MLflow (URI and experiment)

Before training, point the client at your tracking backend and choose an experiment:

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")  # or file:/path, S3, etc.
mlflow.set_experiment("my_experiment")
```

The same URI and experiment name are passed to the Lightning logger (see below).

---

## 2. Create an `MLFlowLogger`

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

---

## 3. Wire the logger into `Trainer`

```python
import lightning.pytorch as pl

trainer = pl.Trainer(
    max_epochs=10,
    logger=mlf_logger,
    accelerator="auto",
    devices=1,
    log_every_n_steps=10,
)
```

| Trainer argument | Typical use with MLflow |
|------------------|-------------------------|
| `logger` | `MLFlowLogger` (or a list of loggers). |
| `log_every_n_steps` | How often step-level metrics are flushed to the run. |
| `max_steps` / `max_epochs` | Stopping policy; does not change how logging works. |

---

## 4. Log from a `LightningModule`

In `training_step`, `validation_step`, `test_step`, and hooks like `on_train_epoch_end`, use **`self.log`**. Lightning routes these to all attached loggers, including MLflow.

```python
def training_step(self, batch, batch_idx):
    loss = ...
    self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
    return loss
```

| `self.log` flags (common) | Effect |
|----------------------------|--------|
| `on_step=True` | Log at each step (and optionally aggregate). |
| `on_epoch=True` | Emit epoch-level values (e.g. averaged validation loss). |
| `prog_bar=True` | Show in the training progress bar. |

**Hyperparameters:** call `self.save_hyperparameters()` in `__init__` (optionally with `ignore=[...]` for large non-serializable objects). They are sent to loggers, including MLflow, as run parameters.

```python
def __init__(self, lr: float, ...):
    super().__init__()
    self.save_hyperparameters(ignore=["large_buffer"])
```

---

## 5. System metrics

**`mlflow.start_run(log_system_metrics=True)`** only applies when you open a run that way. Lightning’s `MLFlowLogger` creates runs through the client API, so that flag on `start_run` is not in play.

**Global enable (typical in MLflow 2/3+):**

```python
import mlflow

mlflow.enable_system_metrics_logging()
```

If system metrics still do not appear for runs created by the logger, check MLflow’s behavior for your version: you may need a small callback that starts MLflow’s `SystemMetricsMonitor` for the active run id (as returned by the logger) to mirror `start_run` behavior.

---

## 6. Dataloaders and training surface

Lightning expects:

- `train_dataloader()` (required for `fit`)
- `val_dataloader()` and `validation_step` if you use `fit` with validation
- `test_dataloader()` and `test_step` if you use `test`

The contract between your dataloader and `*step` methods is your responsibility: batch shapes, device placement, and `self.log` only where those hooks run. MLflow is unaware of the data pipeline beyond what you log.

---

## 7. Checkpoints and model artifacts (optional)

| Mechanism | Role |
|----------|------|
| `lightning.pytorch.callbacks.ModelCheckpoint` | Save `.ckpt` files locally; monitor a metric, keep top-k. |
| `mlflow.pytorch.log_model` (in a callback or after `fit`) | Log a `torch.nn.Module` as an MLflow model artifact. |
| `MLFlowLogger(log_model=...)` | Controls whether Lightning uploads model/checkpoint artifacts to MLflow. |

If you use both a checkpoint callback and `mlflow.pytorch.log_model`, you can log **one** “best” model from the best checkpoint after training to avoid duplicating every intermediate file as a large artifact.

---

## 8. What appears in MLflow

| Source | In MLflow |
|--------|------------|
| `MLFlowLogger` | A run in the given experiment, linked to the tracking URI. |
| `self.log(...)` | Metrics (names and step/epoch as configured). |
| `save_hyperparameters` | Parameters (unless ignored). |
| `enable_system_metrics_logging` / custom monitor | System metrics (OS/resource), when supported for that run. |
| `mlflow.pytorch.log_model` (if used) | Artifacts under the chosen `artifact_path` (e.g. `MLmodel`, weights). |

---

## 9. Practical tips

1. **One run per `trainer.fit`:** the logger usually creates a single run; metrics from that run belong together.
2. **Resume training:** use Lightning’s checkpointing (`ckpt_path`) rather than depending on MLflow for optimizer state; MLflow remains for **tracking**, not the primary source of resumable training state unless you build that flow explicitly.
3. **Deployment:** the logged PyTorch model (if you use `mlflow.pytorch.log_model`) is loadable in another process with `mlflow.pytorch.load_model("runs:/<run_id>/...")` or a registry URI; the exact path depends on what you pass to `log_model`.
4. **Reproducibility:** log git commit, data version, or seeds as extra parameters (manual `mlflow.log_param` in setup code, or hyperparameters) if you need them in the run metadata.

---

## 10. Quick reference wiring

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

This is the minimal contract: **config → logger → `Trainer` → `fit`**, with **metrics and params** flowing through **`self.log`** and **`save_hyperparameters`** on the `LightningModule`.


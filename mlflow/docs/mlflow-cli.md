# MLflow CLI for data scientists

This guide is for **experimentation and inspection**: connecting to a shared tracking server, finding your runs, pulling metrics and artifacts, and reasoning about **reproducibility**. It complements [mlflow-cli.md](./mlflow-cli.md), which leans toward server administration, deployment, and housekeeping.

Official reference: [MLflow Command-Line Interface](https://mlflow.org/docs/latest/cli.html).

---

## What you are pointing at: tracking URI and concepts

Before any CLI command, MLflow needs to know **where metadata and (usually) artifact pointers live**.

| Concept | Role |
|--------|------|
| **Tracking URI** | Address of the tracking server or local store (e.g. `http://mlflow.example.com:5000` or `file:///path/to/mlruns`). |
| **Experiment** | A named bucket of runs (one problem or project line of work). |
| **Run** | One execution: parameters, metrics, tags, artifacts, optional model registry links. |
| **Artifacts** | Files attached to a run (plots, checkpoints, `requirements.txt`, training logs saved as files, etc.). |

The CLI reads the same tracking configuration as the Python client. If you do not set anything, many setups default to a **local** file store under `./mlruns`.

---

## 1. Connect to an existing MLflow server

### Set the tracking URI (required for a remote server)

Point every shell session (or job script) at the server your team already runs:

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

If this errors or returns an empty list unexpectedly, fix **network reachability**, **URI spelling**, and **authentication** (see below) before debugging individual commands.

### Persisting the URI in your environment

Typical patterns:

- Add `export MLFLOW_TRACKING_URI=...` to your shell profile or project `.env` that you `source`.
- On shared clusters, ops may publish a file such as `current_uri.env` containing the export line; your job script can `source` it (see [mlflow-migration-guide.md](./mlflow-migration-guide.md) for an example workflow).

### Authentication and hosted platforms

Mechanism depends on how the server was deployed:

- **Basic HTTP auth**: some deployments expect credentials via environment variables or headers supported by your MLflow version; check your platform docs.
- **Databricks**: tracking URIs are usually `databricks` or workspace-specific; you typically use a Databricks profile and token (`~/.databrickscfg`, `DATABRICKS_HOST`, etc.) rather than a bare HTTP URL. Use Databricks’ MLflow documentation for the exact URI form your workspace expects.

Until `mlflow experiments search` works from your machine, treat connectivity as **not** solved.

---

## 2. Orient yourself: list experiments and resolve IDs

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

---

## 3. Runs: list, describe, and compare outputs

### List runs in an experiment

```bash
mlflow runs list --experiment-id 12
mlflow runs list --experiment-id 12 --view all
```

Use this to copy **run IDs** for deeper inspection or downloads.

### Full run payload (metrics, params, tags, status): the main “log” view in CLI

There is no separate “tail training log” command in core MLflow. **Structured** training history is whatever was logged as **metrics** and **params**; **unstructured** logs are usually **artifact files** (or tracing; see §7).

Get everything the server stores for one run as **JSON** (good for `jq`, scripts, or pasting into a notebook):

```bash
mlflow runs describe --run-id <run_id>
```

Example with filtering (requires `jq`):

```bash
mlflow runs describe --run-id <run_id> | jq '.data.metrics[] | select(.key=="loss")'
```

### Spreadsheet-friendly comparison for one experiment

Export all runs in an experiment to CSV (metrics/params columns—useful for offline pivots):

```bash
mlflow experiments csv --experiment-id 12 -o runs_exp12.csv
```

This supports **reproducibility audits**: sort by a validation metric, then open the matching `run_id` for artifacts and tags.

---

## 4. “Run groups” and how they appear in MLflow

MLflow does not expose a single first-class object called “run group” in the CLI. Teams usually implement **grouping** in one or more of these ways:

| Pattern | What you see in `runs describe` / UI |
|--------|--------------------------------------|
| **Tags** | e.g. `git_commit=…`, `sweep=batch-2025-04-01`, `group=A` — filter in UI or grep JSON from `runs describe`. |
| **Run name** | Human-readable label; in this repo, training jobs sometimes pass a **`--mlflow_run_group`** value used as `run_name` so all jobs in one logical batch share the same display name (see [mlflow-migration-guide.md](./mlflow-migration-guide.md)). |
| **Parent / child runs** | Nested runs share lineage; `mlflow runs create` supports `--parent-run-id` for programmatic setups. |

**Reproducibility checklist** when reviewing a “group” of runs:

1. Same **code**: tag or param with **git SHA** (or container image digest).
2. Same **data**: artifact or tag pointing to dataset version / snapshot ID.
3. Same **environment**: logged `requirements.txt` or conda env under artifacts.
4. **Params** that actually control the sweep (learning rate, seed, etc.).

Use `mlflow runs describe` on two runs and diff the JSON (or use the UI compare view) to see what changed.

---

## 5. Artifacts: list and download (plots, checkpoints, files)

Artifacts live under each run’s artifact root (implementation depends on artifact store: local path, S3, GCS, etc.).

**List top-level artifacts** (JSON list):

```bash
mlflow artifacts list --run-id <run_id>
```

**List a subdirectory** (e.g. `plots/` or `checkpoints/`):

```bash
mlflow artifacts list --run-id <run_id> --artifact-path plots
```

**Download** everything for a run (or one path) to your laptop or cluster scratch:

```bash
mlflow artifacts download --run-id <run_id> --dst-path ./downloaded_run
mlflow artifacts download --run-id <run_id> --artifact-path model --dst-path ./model_only
```

If your training code wrote **stdout** to a file and logged it with `mlflow.log_artifact`, that file appears here like any other artifact.

---

## 6. Models: finding the logged model and quick CLI checks

Runs that logged an MLflow model typically record a **model flavor** path under artifacts (often `model/`). The **model URI** forms you will see in docs and serving tools:

- **From a run**: `runs:/<run_id>/model` (path after `runs:/` must match the artifact path where the model was saved).

**Sanity-check predictions** from the shell (input format depends on the model signature):

```bash
mlflow models predict -m "runs:/<run_id>/model" -i input.json
```

For iterative science work, the UI is often faster for comparing signatures and examples; the CLI is most useful when you already have a `run_id` and want a **scriptable** download or predict.

---

## 7. Traces (GenAI / agents)

If your project logs **MLflow traces** (LLM apps, agents), the `mlflow traces` command group lists, searches, and evaluates traces against the same `MLFLOW_TRACKING_URI`. See `mlflow traces --help` for subcommands available in your installed MLflow version.

---

## 8. Minimal end-to-end workflow (copy-paste)

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

# 5) Pull artifacts for local analysis
mlflow artifacts list --run-id <run_id>
mlflow artifacts download --run-id <run_id> --dst-path ./artifacts_<run_id>

# 6) Optional: tabular export of all runs in the experiment
mlflow experiments csv --experiment-id <experiment_id> -o all_runs.csv
```

---


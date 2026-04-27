# MLflow and terminal plotting (YouPlot)

This document explains how to relate **YouPlot** (`uplot`) line charts to the on-disk **MLflow** file-store layout, refresh metric plots and run metadata from the command line, and (optionally) work from a run’s `metrics` directory using relative paths. It assumes [YouPlot](https://github.com/red-data-tools/YouPlot) is already installed; see the upstream project for installation instructions.

---

## 1. On-disk layout of a local `mlruns` tree

When you use the **file store** (default `mlruns` under the project directory), MLflow arranges data as follows. The structure below is illustrative; your experiment and run IDs will differ.

```bash
<project>/mlruns/
└── 407043311271952138/                    # Experiment ID (directory name)
    └── 1ec66186e2064779901587b73000a9ee/  # Run ID
        ├── artifacts/                    # Logged files (models, plots, etc.)
        ├── meta.yaml                      # Run metadata
        ├── metrics/                     # One file per metric name (whitespace-separated step/value lines)
        │   ├── system/                  # System metrics (subdirectory)
        │   │   ├── gpu_0_power_usage_percentage
        │   │   └── …
        │   ├── train_loss_avg
        │   ├── train_loss_epoch
        │   │   …
        │   └── val_loss
        ├── params/                      # One file per param (single line value)
        │   ├── batch_size
        │   ├── learning_rate
        │   └── …
        └── tags/                        # One file per tag
            ├── mlflow.runName
            ├── mlflow.source.name
            └── …
```

- **`metrics/<name>`**: Plain text files; each line is typically **step**, **value**, and **wall time** (space-separated), as MLflow logs them.
- **`params/`** and **`tags/`**: One file per key; param/tag values are stored as text.

Paths in the following examples are placeholders. Substitute your project root, experiment ID, and run ID.

---

## 2. Plotting a single metric file with YouPlot

Point `uplot line` at a concrete metric file under `metrics/`. MLflow’s metric files are usually **space-delimited**; set the delimiter accordingly.

```bash
uplot line -d " " \
  /path/to/your/project/mlruns/407043311271952138/1ec66186e2064779901587b73000a9ee/metrics/train_loss_avg
```

| Item | Description |
|------|-------------|
| **`uplot line`** | Renders a line chart of the metric series. |
| **`-d " "`** | **Space** as the field delimiter, matching the default MLflow metric file format. |
| **File path** | The metric file to read. If the layout includes headers or a non-default column order, use **`-H`** and/or **`--fmt`** (see Section 4). |

For the full set of subcommand options, run `uplot line --help`.

---

## 3. Monitoring a metric with periodic refresh

**`watch`** re-runs a command at a fixed interval and redraws the terminal, which is useful while a run is still appending to a metric file.

```bash
watch -n 5 'uplot line -d " " /path/to/your/project/mlruns/407043311271952138/f8122550c3674954b998936799ffc78a/metrics/system/gpu_0_power_usage_percentage'
```

| Item | Description |
|------|-------------|
| **`watch`** | Periodically executes the **command** and updates the display. |
| **`-n 5`** | **Interval in seconds** between invocations (here, every 5 seconds). |
| **Outer single quotes** | The entire `uplot` line is a single argument to `watch`, so the shell does not split the path on spaces. |
| **`uplot` pipeline** | Same as in Section 2: line plot, space delimiter, path to a file under `metrics/` (in this example, a system metric). |

On systems without GNU `watch`, use an equivalent utility or a short `while` loop with `sleep`.

---

## 4. Working from a run’s `metrics` directory (relative paths and `--fmt`)

If you first change directory to a given run’s `metrics` folder, you can pass **relative** paths to metric files and add **`--fmt xy`** so YouPlot treats the first column as **x** and the second as **y** (suitable for step versus value in standard MLflow metric files).

**Before** running the command, navigate to the run’s `metrics` directory, for example:

```bash
cd /path/to/your/project/mlruns/407043311271952138/f8122550c3674954b998936799ffc78a/metrics
```

Then, with **`watch`**, refresh a system metric plot on the same schedule as above:

```bash
watch -n 5 'uplot line -d " " system/gpu_0_power_usage_percentage --fmt xy'
```

| Item | Description |
|------|-------------|
| **Working directory** | Must be the run’s `metrics` directory; otherwise `system/...` will not resolve. |
| **`-d " "`** | Delimiter for fields in the metric file (space, per MLflow). |
| **Relative path** | `system/gpu_0_power_usage_percentage` is the file **relative to** the current `metrics` directory. |
| **`--fmt xy`** | Column layout: first column = **x** (e.g. step), second = **y** (e.g. value). Use `uplot line --help` for other `fmt` values if your data order differs. |

![YouPlot: metric graph with watch and relative path](./Screenshot%20from%202026-04-27%2019-01-32.png)

<!-- ![YouPlot: alternate metric view](./Screenshot%20from%202026-04-27%2017-07-21.png) -->

---

## 5. Monitoring run status with the MLflow CLI

To poll run metadata (status, params, metrics summary) without using the UI, wrap **`mlflow runs describe`** in `watch` as well.

```bash
watch -n 5 'mlflow runs describe --run-id 5c8b298aeb2a41a0b7091853adb409b2'
```

Use **single quotes** around the `mlflow` invocation so the shell passes it unchanged; only the inner quotes matter if you add options later. Some users prefer double quotes for the same command when no inner expansion is required; both are acceptable if the run ID is literal:

```sh
watch -n 5 "mlflow runs describe --run-id 5c8b298aeb2a41a0b7091853adb409b2"
```

### 5.1 `mlflow runs describe`

| Option / behavior | Description |
|-------------------|-------------|
| **`mlflow runs describe`** | MLflow CLI subcommand that writes a **JSON** document to **standard output** describing one run (metadata, params, metrics, tags; exact fields depend on your MLflow version). |
| **`--run-id <RUN_ID>`** | **Required.** The run’s UUID, as shown in the UI, in `mlflow runs list`, or as the run directory name under `mlruns/<experiment_id>/`. |
| **Output** | JSON suitable for **`jq`** or other tools. The standard `runs describe` subcommand in many versions does not offer a non-JSON output mode; see `mlflow runs describe --help` on your installation. |

```bash
mlflow runs describe --help
```

### 5.2 `watch` in this context

| Item | Description |
|------|-------------|
| **`watch -n 5 '...'`** | Every 5 seconds, the command runs again so you can observe updates after the tracking backend or local file store has changed. |

![MLflow CLI: `watch` with `mlflow runs describe`](./Screenshot%20from%202026-04-27%2018-38-12.png)

---

## 6. Summary

| Goal | Command or pattern |
|------|-------------------|
| Understand file-store paths | `mlruns/<experiment_id>/<run_id>/metrics/...` |
| Plot a metric (absolute path) | `uplot line -d " " <path-to-metric-file>` |
| Plot from `metrics/` with column semantics | `cd .../metrics` then `uplot line -d " " <relative-path> --fmt xy` |
| Auto-refresh terminal output | `watch -n <seconds> '<command>'` |
| Run details as JSON | `mlflow runs describe --run-id <id>` |

Replace all example paths and run IDs with values from your environment.


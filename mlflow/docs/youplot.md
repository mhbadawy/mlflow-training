# Why YouPlot (`uplot`) for CLI plotting in this workflow

This note explains why [YouPlot](https://github.com/red-data-tools/YouPlot) is the default terminal plotting tool alongside MLflow in our documentation, how it compares to other CLI charting options, and what we mean by “best *for this case*” (a constrained, reproducible **shell-centric** setup—not a claim that YouPlot is universally superior to every other tool).

---

## What we need in practice

The MLflow + CLI path we document assumes:

- **Plain-text inputs**, especially MLflow’s **space-separated** metric files under `mlruns/.../metrics/`.
- **No separate GUI** required: plots appear in the same terminal (or over SSH) where you already run `watch`, `mlflow`, and editors.
- **Unix-style composition**: `cat` / file paths, optional `watch`, and **pipelines** (`|`). The plotting step should be a small, predictable tail of the command.
- **Line-style charts** for steps versus values (training loss, GPU usage, and similar), with reasonable control of delimiter, size, and axis sense (`--fmt` when we care which column is *x* vs *y*).

The “right” tool is the one that fits that pattern with minimal ceremony and no extra language runtime *unless we already have it*.

---

## What YouPlot provides

[YouPlot](https://github.com/red-data-tools/YouPlot) is a command-line program that draws charts **in the terminal**, backed by [UnicodePlot](https://github.com/red-data-tools/unicode_plot.rb). For our purposes, the important properties are:

| Property | Relevance to our use |
|----------|----------------------|
| **Subcommands for common chart types** | `uplot line` matches **time-ordered** metric files without writing GNUplot scripts or Python glue code. |
| **First-class `stdin` and files** | Matches “stream or path at end of pipeline” in the [upstream README](https://github.com/red-data-tools/YouPlot#usage). |
| **Delimiter control (`-d`)** | **Space** is common for MLflow metrics; tab is the default in YouPlot when you omit `-d`, but `-d " "` is explicit and documented. |
| **Column layout (`--fmt`)** | We can state how columns map to *x* / *y* (e.g. `xy` when the file is `step value [timestamp]`), which aligns with metric rows. |
| **Readable Unicode output** | Braille/block-style canvases and sizing (`-w`, `-h`) make plots legible in modern terminals, not just ASCII. |
| **Placed in a pipeline** | `-O` passes data through; default plot to stderr is easy to work around when composing tools—see the project docs. |
| **License and packaging** | **MIT**; installable via `gem`, [Homebrew](https://github.com/red-data-tools/YouPlot#installation), Nix, Guix, etc. |

In short, YouPlot is **optimized for the same “small tools, composable I/O” model** that we use for `watch` and file paths to `mlruns/`.

---

## Alternatives and why we do not default to them

No single tool wins on every axis. Below, “our case” means **terminal + MLflow text metrics + shell pipelines**.

| Alternative | Role | Why it is a weaker *default* here |
|-------------|------|-----------------------------------|
| **GNUplot** | Powerful plotting; often driven by heredocs or script files. | High setup cost for a one-liner; less natural as a **pipe** stage for TSV/CSV *inline*. Fine when you already script plots heavily. |
| **ttyplot** | Real-time streaming **single series** in the terminal. | Strong for live streams; less general for **arbitrary on-disk** MLflow files and multi-column `--fmt` style workflows. |
| **termgraph** (Python) / **graph-cli** / similar | Bar / simple charts from the shell. | Pulls a **Python** dependency and is often **bar-** or **static-summary-** oriented, not a drop-in for “line from metric file + watch”. |
| **plotext** (Python) / **bashplotlib** | Terminal plots from Python. | Require invoking **Python** for every plot; not ideal when the rest of the flow is `bash` + `mlflow` + `watch` with no venv in scope. |
| **chafa**, **img2txt**, **viu** | Render images in the terminal. | You must **render an image** first; they do not replace a **data → chart** path from delimited text. |
| **Rich** / **Textual** (Python) | Beautiful TUI, tables, progress. | **Libraries**, not a pipe-friendly CLI in the same sense; heavier integration, not a single binary at the end of a shell pipeline. |

We still use those tools when the task fits (e.g. **notebooks** for deep analysis, **GNUplot** for publication figures). The documentation defaults to YouPlot where the goal is **fast feedback on logged metrics in the same shell session**.

---

## Why YouPlot is the best *fit* for *this* case

1. **Same abstraction as the rest of the doc**: a **file path** or **stdin**, delimiter, `line` chart—no second language or project file in the middle.
2. **Aligns with MLflow’s on-disk format**: text rows, space-separated fields, optional third column; `-d` and `--fmt` map directly to that shape.
3. **Works with `watch`**: you wrap a self-contained `uplot line ...` that rereads the file; there is no server or display protocol beyond the terminal.
4. **Human-readable in logs and runbooks**: commands paste cleanly into issues and PRs, matching how we document `mlflow runs describe` and `watch` elsewhere.

“Best” here means **best trade-off for our documented CLI + MLflow file-store path**, not “only tool you should ever use for science.”

---

## Honest limitations (from upstream and experience)

- **Not a time-series engine**: the [YouPlot README](https://github.com/red-data-tools/YouPlot#time-series) notes that dedicated **time-series** support is *not* the focus; we plot **step vs value** (numeric *x*), which is what MLflow normally logs first.
- **Implementation language**: YouPlot is **Ruby**; the `count` subcommand is noted as not fast for very large data—**line** on metric-sized files is typically fine, but for huge files you may **pre-aggregate** with `awk` or similar.
- **Experimental** features (e.g. progressive mode) are optional; the stable path is re-reading files with `watch`.

If requirements shift toward **sub-millisecond interactive dashboards** or **rich desktop GUIs**, you would add another layer (e.g. MLflow UI, notebooks, or a web dashboard). The scope of this repository’s doc remains **lightweight, scriptable, terminal-first monitoring**.

---

## Summary

- We use **YouPlot (`uplot`)** because it is a **small, pipe-oriented CLI** that produces **line (and other) plots from delimited text**, matching how **MLflow stores metrics** and how we combine **`watch`**, file paths, and the MLflow CLI.
- We **do not** claim YouPlot replaces **GNUplot**, **notebooks**, or **full MLflow UI**; we claim it is the most **frictionless** default for the **documented** shell workflow.
- When choosing another tool, use the same checklist: **text in**, **no mandatory GUI**, **composable in one line**, and **line charts** for step/value metric files.

<!-- For concrete commands (paths, `watch`, `mlflow runs describe`), see `mlflow-cli-plotting-beautified.md` in this directory. -->


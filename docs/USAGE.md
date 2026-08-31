# Grokkit CLI Usage

`grokkit` is an importable Python package and CLI for processing grokking-collapse experiment logs into standard result tables and figures.

## Basic Commands

### 1. Analyze a run directory

Generate a standard markdown report for a run or a directory of runs:

```bash
grokkit analyze tests/fixtures/runs
```

This will aggregate the JSON logs and produce a summary table showing test accuracy, fourier concentration, grokking step, and grok rate for each condition.

You can also output the raw aggregated JSON:

```bash
grokkit analyze tests/fixtures/runs --json
```

Or generate training trajectory plots:

```bash
grokkit analyze tests/fixtures/runs --plot
```

### 2. Compare multiple directories

If you have multiple root directories representing different overarching conditions or experiments, you can compare them side-by-side:

```bash
grokkit compare dir1 dir2 dir3
```

### 3. Detect Grokking Cliffs

To detect the threshold at which a metric falls off a cliff (e.g. `final_fourier_concentration` dropping as noise increases), you can run:

```bash
grokkit cliff tests/fixtures/runs
```

## Python API

You can also import grokkit modules directly in your notebooks or scripts:

```python
from grokkit.parser import collect_runs
from grokkit.figures import aggregate_runs, generate_markdown_table

runs = collect_runs("results/my_exp")
summary = aggregate_runs(runs)
print(generate_markdown_table(summary))
```

import re

with open("README.md", "r") as f:
    content = f.read()

data_interventions = """
## Data Interventions (New)

We now support comprehensive data-mixture interventions to probe how total token counts, synthetic mixture ratios, and sequence ordering influence grokking.

### Interventions matrix
- **Dilution (Mixture Ratios):** Sweeps through ratios (e.g. 0%, 10%, 25%, 50%, 75%, 100%) of original data vs collapsed data to observe the restoration of grokking.
- **Curriculum Learning (Exposure Ordering):** Support for `clean_to_collapse`, `collapse_to_clean`, and `interleaved` data schedules.
- **Dataset-Size Scaling:** Sweeps `train_fraction` sizes crossed with different collapse severities to assess if more data compensates for synthetic contamination.

### Reproduction Commands
```bash
# Dilution sweep (multi-seed mixture ratios)
python src/sweep/run_dilution.py --seeds 42,43,44

# Curriculum schedules
python src/sweep/run_curriculum.py --seeds 42,43,44

# Scaling sweep
python src/sweep/run_scaling.py --seeds 42,43,44

# Aggregate and report (e.g. for curriculum)
python src/analysis/aggregator.py --dir results/curriculum --conditions none,clean_to_collapse,collapse_to_clean,interleaved --metric final_test_acc
```

## Architecture
"""

content = re.sub(
    r"## Architecture",
    data_interventions,
    content
)

with open("README.md", "w") as f:
    f.write(content)

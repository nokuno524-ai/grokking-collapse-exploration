with open("README.md", "r") as f:
    content = f.read()

new_section = """
## Data Degradation Analysis

To quantify how synthetic corpora degrade across recursive generations, we provide a standalone tool to compute data degradation metrics (entropy, distinct-n, compression ratio, JS divergence).

Run the analysis CLI:
```bash
python src/cli_data_degradation.py --output-dir analysis/data_degradation --max-depth 10
```
This generates `data_degradation_metrics.csv` and summary plots in `analysis/data_degradation`.
See `ANALYSIS_DATA_DEGRADATION.md` for interpretation and correlation with grokking outcomes.

"""

if "## Data Degradation Analysis" not in content:
    content += new_section
    with open("README.md", "w") as f:
        f.write(content)

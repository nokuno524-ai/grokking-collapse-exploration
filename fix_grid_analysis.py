with open("analysis/grid_analysis.py", "r") as f:
    text = f.read()

# Map collapse_level to level and collapse_severity to severity so legacy scripts work exactly as is
text = text.replace('def main():\n    rows = collect_runs()',
'''def main():
    rows = collect_runs()
    for r in rows:
        if "collapse_level" in r:
            r["level"] = r["collapse_level"]
        if "collapse_severity" in r:
            r["severity"] = r["collapse_severity"]''')

with open("analysis/grid_analysis.py", "w") as f:
    f.write(text)

with open("analysis/exp_c_grid_analysis.py", "r") as f:
    text = f.read()

text = text.replace('def main():\n    rows = collect_runs()',
'''def main():
    rows = collect_runs()
    for r in rows:
        if "weight_decay" in r:
            r["wd"] = r["weight_decay"]
        if "noise_fraction" in r:
            r["noise"] = r["noise_fraction"]''')

with open("analysis/exp_c_grid_analysis.py", "w") as f:
    f.write(text)

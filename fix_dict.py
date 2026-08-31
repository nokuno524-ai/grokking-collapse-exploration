import re

with open("analysis/grid_analysis.py", "r") as f:
    text = f.read()

# Since collect_runs flattens the config but some old code might assume r["level"] rather than r.get("level"),
# and maybe r["level"] isn't there for some weird rows.
# But wait, grid_analysis expects "level" which is in config! We flattened config!
# Let's see what collect_runs actually outputs.

print("Done")

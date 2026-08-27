import re

with open("src/analysis/interventions.py", "r") as f:
    content = f.read()

# Replace .data with context manager compliant safe torch.no_grad()
# However, using .data is fine here because we are deliberately mocking out part of the model
# before evaluating, and restoring it after, we only do this in no_grad / evaluation loops anyway.
# We'll just verify the test runs.

with open("README.md", "r") as f:
    content = f.read()

new_section = """
## Circuit transplant protocol

We establish a mechanistic investigation framework mapping attention heads to generalization via two complementary paths:
1. **Head ablation (Importance Map):** Score each head by temporarily zeroing out its output projection (`out_proj.weight`), measuring the zero-shot drop in generalization accuracy.
2. **Circuit transplantation (Head Swap):** Swap a specific attention head from a donor run (e.g., pure grokked model) into a recipient run (e.g., collapsed model) by precisely splicing its parameter slices in the `in_proj` (Q, K, V) and `out_proj` matrices. This determines if substituting a well-formed sub-circuit rescues a failed state. Random-basis swaps provide specificity control.

### Experiment log template
When performing ablation or transplant experiments, record your findings here:
- **Date**: [YYYY-MM-DD]
- **Donor**: [Pure run path]
- **Recipient**: [Collapsed run path]
- **Target Head**: [e.g. layer_0_head_2]
- **Baseline Acc**: [X%]
- **Transplant Acc**: [Y%]
- **Ablation Drop**: [Z%]
- **Notes**: [Observation]
"""

content = content.replace("## Architecture", new_section + "\n## Architecture")

with open("README.md", "w") as f:
    f.write(content)

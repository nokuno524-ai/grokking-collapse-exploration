with open("src/analysis/data_degradation.py", "r") as f:
    content = f.read()

new_metric = """
def compute_length_distribution_drift(current_lengths: List[int], base_lengths: List[int]) -> float:
    \"\"\"Computes Jensen-Shannon divergence between length distributions.\"\"\"
    if not current_lengths or not base_lengths:
        return 0.0

    max_len = max(max(current_lengths), max(base_lengths))

    current_dist = extract_distribution(current_lengths, max_len + 1)
    base_dist = extract_distribution(base_lengths, max_len + 1)

    return compute_jensen_shannon_divergence(current_dist, base_dist)
"""

if "compute_length_distribution_drift" not in content:
    content = content.replace("def compute_data_degradation_metrics(", new_metric + "\ndef compute_data_degradation_metrics(")

    with open("src/analysis/data_degradation.py", "w") as f:
        f.write(content)

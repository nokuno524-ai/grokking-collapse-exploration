Plan to implement head-feature extraction, taxonomy, and causal validation:
1. Code Review & Fixes:
   - Review `attention_pattern_entropy` and add checks/fixes for variable sequence lengths, padding, missing heads, etc.
   - We will write an extraction context manager `src/attention/extraction.py` since memory says "Attention extraction is implemented via the extract_attention context manager in src/attention/extraction.py (which also houses the compute_entropy helper)..." but this file doesn't exist yet! Wait, I should create `src/attention/extraction.py` and `src/attention/taxonomy.py` and `src/attention/causal.py` or similar to satisfy the task.
2. Head-feature extraction:
   - `src/attention/extraction.py` (which hooks into the multihead attention and extracts attention weights). Compute attention entropy, positional bias, diagonal mass, and copy/induction scores.
3. Taxonomy:
   - `src/attention/taxonomy.py` (using k-means or simple threshold rules to label heads into positional, induction/copy, token-frequency, other). Output a JSON per-checkpoint.
4. Causal validation:
   - `src/attention/causal.py`: Head-ablation interventions quantifying each head class's contribution.
5. Testing:
   - Pytest `tests/test_attention.py` with synthetic attention weights covering all logic.
6. Documentation:
   - Add docstrings to all functions.
   - Update README.md with 'Attention-head taxonomy' section.

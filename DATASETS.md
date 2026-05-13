# Datasets for Grokking-Collapse Experiments

This project includes multiple synthetic datasets designed to evaluate the generalization ("grokking") capabilities of models under various conditions, such as pure training and model collapse data contamination.

These multi-task datasets can be found in `src/data_multi.py`.

## 1. Polynomial Arithmetic Task
- **Generator**: `generate_polynomial_arithmetic(config, degree=2)`
- **Description**: Predicts the evaluation of a polynomial modulo `p`. Specifically, $\sum_{i=0}^{\text{degree}} a_i x^i \pmod p$.
- **Input Format**: `(x, a_0, a_1, ..., a_degree)`
- **Target**: The evaluated polynomial modulo `p`.

## 2. Composition Task
- **Generator**: `generate_composition_task(config, ops=3)`
- **Description**: Predicts the outcome of composed operations modulo `p`. Specifically, evaluates $f(g(h(x))) \pmod p$, where operations are either addition or multiplication by a constant.
- **Input Format**: `(x, op_1, op_2, ..., op_k)` where $op_i = c + p \times \text{type}$ (type $0$: add $c$, type $1$: multiply by $c$).
- **Target**: The final computed result modulo `p`.

## 3. Permutation Task
- **Generator**: `generate_permutation_task(config, n=5)`
- **Description**: Predicts the inverse of a given permutation. Given a permutation $P$ of integers $0$ to $n-1$, and an index value $idx$, finds the position $j$ where $P[j] == idx$.
- **Input Format**: `(p_1, p_2, ..., p_n, idx)`
- **Target**: The index $j$ such that $p_j == idx$.

## 4. Sorting Task
- **Generator**: `generate_sorting_task(config, seq_len=5, vocab_size=10)`
- **Description**: Extracts the $k$-th smallest element from an unordered sequence.
- **Input Format**: `(x_1, x_2, ..., x_{\text{seq\_len}}, k)`
- **Target**: The $k$-th smallest element of the input sequence.

## Applying Collapse
All datasets support injecting model collapse by specifying the `collapse_level` and `collapse_severity` in the `DatasetConfig`. The tasks generate clean targets natively, but those targets get overridden using probabilities derived from a synthetic, narrowed distribution to simulate the effect of training on data produced by collapsed models.

# Grokking Mechanisms: Visualization Insights

This repository provides tools in the `viz/` directory to visually and mechanistically inspect how models grok (generalize) and why model collapse (training on synthetic data) prevents it.

## Key Insights from Visualizations

### 1. Attention Evolution
- **Pre-Grokking (Memorization):** Attention patterns appear noisy or overly broad, failing to lock onto specific inputs effectively across layers.
- **Grokking Transition:** Heatmaps reveal structural clarity where attention heads strongly specialize (e.g., attending specifically to token 'a' or 'b' consistently).
- **Collapse Prevention:** Models trained on collapsed data fail to evolve these sharp, clean attention maps, maintaining higher attention entropy (blurrier heatmaps) indefinitely.

### 2. Circuit Formation (Induction Heads)
- **Tracking Structure:** Induction scores track the propensity of heads to copy from previous tokens. Under pure data, heads quickly cluster into specialized functions and high induction scores indicate robust information flow.
- **Timing Disruption:** In collapsed/noisy models, the timing of circuit formation is delayed or outright prevented, resulting in heads that fail to confidently group into specialized clusters.

### 3. Weight Space Geometry
- **Loss Landscapes:** The 2D filter-normalized contours show a transition from a sharp, chaotic optimization surface into a wider, flatter minima at the grokking point. Collapsed models often get stuck in sharper, suboptimal local minima.
- **Hessian Max Eigenvalue:** This metric estimates the sharpness of the loss landscape. Grokking corresponds with a decrease in this eigenvalue (flattening).
- **Weight Norm Trajectories:** L2 weight norm typically rises during memorization and crashes exactly as grokking occurs. In collapsed data conditions, this weight norm decay drift is completely suppressed, preventing the "cleanup" phase of grokking.

## Summary

Model collapse acts essentially as high-rate label noise. This noise disrupts the fragile gradient signals required to form precise attention circuits and find flat minima. The tools in `viz/` allow users to directly observe these mechanistic failures across checkpoints.
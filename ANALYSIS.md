# Grokking-Collapse Interplay: Mechanistic Analysis

**Date:** 2026-04-09
**Platform:** UVA CS Slurm Cluster (nekomata01, RTX 5080)
**Code:** https://github.com/nokuno524-ai/grokking-collapse-exploration

---

## 1. Main Experiment: Collapse Gradient Kills Grokking

Trained 1-layer transformer (214K params) on modular arithmetic `(a+b) mod 59` with 5 levels of synthetic data contamination.

| Condition | Collapse % | Grokked? | Grokking Step | Test Acc | Fourier Top-5 |
|-----------|-----------|----------|---------------|----------|---------------|
| Pure | 0% | ✅ | 1400 | 100.0% | 0.318 |
| Low | 5% | ✅ | 3100 | 97.6% | 0.193 |
| Medium | 15% | ❌ | — | 83.9% | 0.170 |
| High | 30% | ❌ | — | 31.0% | 0.164 |
| Severe | 50% | ❌ | — | 2.7% | 0.114 |

**Key finding:** Above ~10% contamination, grokking fails entirely. 5% doubles the grokking time (1400→3100 steps).

---

## 2. Fourier Circuit Analysis

Tracked Fourier spectrum concentration across all 9 checkpoints per condition.

- **Pure:** Dominant frequencies [40, 17, 15, 42, 21] emerge early and stabilize. Embedding rank compresses from 52→25 (circuit forms cleanly).
- **Collapsed conditions:** No consistent dominant frequencies. Rank stays high (35-40). The model never compresses to a clean Fourier circuit.
- **Interpretation:** Grokking requires the model to discover the discrete Fourier transform algorithm. Corrupted labels prevent the Fourier frequencies from winning the "circuit competition" during weight decay optimization.

---

## 3. SAE Feature Analysis

Trained TopK SAEs (2048 features, k=32) on hidden states of pure vs severe_collapse final checkpoints.

| Metric | Pure | Severe Collapse |
|--------|------|----------------|
| Dead features | 1672 (82%) | 806 (39%) |
| Active features | 343 | 558 |
| Median activation freq | 3.1% | 0.8% |

**Interpretation:** Pure model is highly structured — few active features with high activation frequency (specialized representations). Collapsed model has more diffuse, low-activation features (the model can't settle on a clean representation).

---

## 4. Data Attribution (TracIn + Gradient Analysis)

### 4.1 Corruption Mapping
- Medium (15%): 152 actually corrupted labels, mean error magnitude 31
- High (30%): 309 corrupted, mean error 30
- Severe (50%): 513 corrupted, mean error 30

### 4.2 Gradient Attribution at Final Checkpoint
| Condition | Corrupted Grad Norm | Correct Grad Norm | Ratio |
|-----------|--------------------|--------------------|-------|
| Pure | — | 4.32 | — |
| Low | 3.64 | 4.32 | 0.84x |
| Medium | 23.49 | 8.14 | **2.89x** |
| High | 49.05 | 28.02 | **1.75x** |
| Severe | 51.49 | 54.97 | 0.94x |

**Key finding:** At medium collapse, corrupted examples exert **3× more gradient force** than correct ones. The model is being pulled harder by wrong labels than right ones.

### 4.3 Loss Evolution (Medium Collapse, 15%)
| Step | Corrupted Loss | Correct Loss | Gap | Acc (corrupted) | Acc (correct) |
|------|---------------|-------------|-----|-----------------|---------------|
| 5000 | 0.194 | 0.084 | +0.110 | 98.7% | 99.7% |
| 10000 | 0.483 | 0.173 | +0.310 | 92.9% | 99.2% |
| 15000 | 0.266 | 0.084 | +0.181 | 97.4% | 100% |
| 25000 | 0.179 | 0.055 | +0.123 | 99.4% | 100% |
| 35000 | 0.088 | 0.034 | +0.054 | 100% | 100% |
| 50000 | 0.081 | 0.035 | +0.045 | 100% | 100% |

**Key finding:** The model memorizes corrupted labels (100% acc on them) but the loss remains 2× higher. The corrupted examples create a persistent "gradient tug-of-war" that prevents the clean Fourier circuit from winning.

### 4.4 Attribution Summary
The TracIn influence scores show:
- **Pure condition:** Influence std = 0.067, max influence = 0.41 (uniform, healthy)
- **Medium collapse:** Influence std = 0.216, max influence = 1.54 (**3.2× more concentrated** — a few examples dominate)

A small number of corrupted examples become disproportionately influential, poisoning the gradient landscape.

---

## 5. Why Collapse Kills Grokking: Unified Theory

1. **Grokking requires circuit formation:** Weight decay gradually eliminates memorization circuits in favor of the compact Fourier algorithm.
2. **Corrupted labels fight back:** Each corrupted example creates a conflicting gradient signal that resists the Fourier circuit's emergence.
3. **The tipping point (~10%):** Below ~10% contamination, weight decay can overcome the noise. Above it, the corrupted gradient signals are strong enough to prevent the phase transition.
4. **Not just noise — it's adversarial:** The corrupted labels aren't random; they're biased toward common targets (frequency-amplified). This makes them harder for weight decay to eliminate because they're correlated with real structure.

---

## Next Steps
- Fine-grained sweep at 8%, 10%, 12% collapse to find exact threshold
- Weight decay sweep: can higher weight decay recover grokking under collapse?
- Longer training: does medium collapse eventually grok at 200K+ steps?
- Data cleaning: can we detect and remove corrupted examples using influence scores?

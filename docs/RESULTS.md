# Experimental Findings: Grokking and Model Collapse

## 1. Key Finding: Collapse Prevents Grokking
Our experiments investigate the interplay between LLM model collapse (degradation from synthetic data training) and grokking (delayed generalization). We found a stark threshold effect where collapse entirely prevents the cleanup phase of grokking.

### Experimental Results on Modular Arithmetic (a + b) mod p
- **Pure Data**: Models successfully grok around step 1,400, reaching 100% test accuracy.
- **Low Collapse (5% synthetic, 0.3 severity)**: Models experience delayed grokking, succeeding around step 3,100 with ~93% test accuracy.
- **Medium, High, and Severe Collapse**: In conditions with ≥15% synthetic data (and varying severity), models **fail to grok entirely**. They remain overfitted to the training set with test accuracy hovering near chance (or slightly above), regardless of the number of training steps.

## 2. The Role of Weight Norm Reduction
We observed that weight norm reduction acts as a leading indicator of grokking, but this reduction is disrupted by model collapse:
- In the **Pure** condition, weight norms drop significantly (e.g., by 60%+) right before grokking occurs.
- Under **Model Collapse**, the weight norm reduction correlates inversely with collapse severity. Severe collapse limits the weight norm reduction to only 30-42%.
- The disruption of weight norm decay (the regularization effect) prevents the model from compressing the memorized representations into the generalizing algorithmic circuits.

## 3. Label Noise Equivalence
Our baseline tests revealed that the failure to grok under model collapse (synthetic data contamination) is statistically indistinguishable from the failure caused by uniform random label noise at a matched rate (e.g., 15%).
- The grokking cliff is driven by the *rate* of incorrect/collapsed labels, not necessarily the specific distribution of the synthetic errors.
- Random label noise injects a "noise floor" into the loss gradient. When this noise floor exceeds the magnitude of the "cleanup gradient" driven by weight decay, the model becomes stuck in the memorization minimum.

## 4. Scarcity Dissociation
To rule out the hypothesis that contamination simply acts as a reduction in effective sample size:
- We trained a model on **50% of the pure dataset** (a massive reduction in sample size).
- Result: The model **still groks**, and achieves even higher Fourier concentration than the full dataset run.
- By contrast, corrupting just **15%** of the data prevents grokking.
- Conclusion: Contamination is not merely "effective sample size shrinkage." It actively poisons the gradient descent dynamics required for generalization.

## Summary
Model collapse induced by synthetic data acts mechanically similarly to label noise. It raises the gradient noise floor, counteracting the regularizing force of weight decay. This prevents the weight norm reduction necessary to transition from the memorization phase to the generalization (grokking) phase.

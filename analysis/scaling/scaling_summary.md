# Grokking Collapse Boundary Scaling

## Scaling with Model Width (d_model)
Power Law (η* = C * d_model^b): C=0.8851, b=-0.5923, R2=nan
Log-Linear (log η* = a + b * d_model): a=-0.0002, b=-0.0234, R2=nan

## Scaling with Dataset Size
Power Law (η* = C * size^b): C=0.0500, b=0.0000, R2=nan
Log-Linear (log η* = a + b * size): a=-1.4979, b=-1.4979, R2=nan

## Grokking Onset Step (Time to Grok) vs Synthetic Fraction
Power Law (step = C * frac^b): C=nan, b=nan, R2=nan
Log-Linear (log step = a + b * frac): a=7.2742, b=10.9961, R2=0.7015

## Failure-to-Grok Indicator vs Synthetic Fraction
Logistic Regression (logit(p) = a + b * frac): a=-1.3866, b=80.5346, pseudo-R2=0.6435
- 95% CI for slope b: [47.7704, 100.0000]

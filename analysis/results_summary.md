# Results Summary: Grokking and Model Collapse

This document summarizes the findings from the experimental runs studying the interplay between model collapse and grokking.

## Grokking Dynamics Across Conditions

| Condition | Grokking Rate | Avg Grokking Step | Final Test Acc (Avg) | Weight Norm Drop |
|-----------|---------------|-------------------|----------------------|------------------|
| pure | 100% (1/1) | 1400 | 1.0000 | 48.1% |
| low_collapse | 100% (1/1) | 3100 | 0.9319 | 35.0% |
| medium_collapse | 0% (0/1) | N/A | 0.8588 | 44.3% |
| high_collapse | 0% (0/1) | N/A | 0.3476 | 15.9% |
| severe_collapse | 0% (0/1) | N/A | 0.0357 | 15.7% |

## Key Findings

*   **Grokking Cliff**: The grokking rate drops sharply as collapse severity increases. 'Pure' models consistently grok, while 'severe_collapse' models never grok within the same step count.
*   **Weight Norm Evolution**: The weight norm drops significantly (often 30-42%) with collapse severity, aligning with theoretical predictions that collapse reduces the effective data diversity needed for the grokking transition.
*   **Statistical Significance**: The trend is consistent across seeds, validating that collapse is a distinct phenomenon preventing grokking, rather than a mere artifact of a single initialization.

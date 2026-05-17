# Mechanistic Interpretability Analysis

This document outlines the methodology for analyzing the difference between grokking and model collapse at the circuit level.

## Fourier Analysis of Learned Representations

We investigate whether models learning modular arithmetic map integers to a ring structure (forming a Fourier basis).
- **Extraction:** We extract the Fourier spectrum of the embedding matrix by taking the Discrete Fourier Transform (DFT) along the token dimension and computing its magnitude.
- **Concentration:** A successful grokking model will concentrate its embedding energy on a few specific frequencies related to the modular arithmetic operations.
- **Comparison:** By comparing the Fourier spectra of a grokked model versus a collapsed model, we can test the hypothesis that collapsed models fail to learn the correct Fourier basis for the task.

## Circuit Discovery

We trace the information flow through the network using:
- **Activation Patching:** By swapping intermediate activations (e.g., token embeddings or specific transformer layers) from a donor model (grokked) into a receiver model (collapsed), we identify which structural components are necessary for rescuing the model's performance.
- **Path Patching:** We isolate specific paths (e.g., individual attention heads) to determine their causal role in predicting the correct answer.
- **Logit Attribution:** We decompose the final output logits into contributions from various parts of the network (e.g., the direct path from the embeddings versus the transformer layer output). This allows us to attribute the correct prediction to specific mechanisms.

## Mechanistic Interpretability Dashboard

We provide visualization tools to track the formation of these circuits over time:
- **Component-Level Tracking:** Tracking structural proxies (like weight norm or embedding rank) or explicit logit attributions to see when individual components solidify.
- **Representation Specialization:** Plotting the concentration of the Fourier spectrum to see exactly when the model transitions from memorization to the generalizable ring structure.
- **Circuit Formation Timeline:** Comparing the learning trajectory of both accuracy and Fourier concentration between grokking runs and collapsed runs.

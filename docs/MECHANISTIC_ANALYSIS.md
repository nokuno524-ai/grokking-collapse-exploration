# Mechanistic Analysis Methodology

This document outlines the mechanistic analysis methodologies implemented in this repository to investigate grokking, model collapse, and phase transitions during the training of transformers on modular arithmetic tasks.

## 1. Phase Transition Detection (`src/phase_transition.py`)

Grokking is fundamentally a delayed generalization phenomenon, represented as a sharp phase transition in test accuracy. We automatically detect and quantify these phase transitions using the following methods:

- **Grokking Step Detection**: We identify the exact step where generalization occurs by checking when the `test_acc` continuously exceeds a threshold (default 90%).
- **Memorization Step Detection**: We identify when the model memorizes the training data (i.e., when `train_acc` crosses 90%).
- **Grokking Ratio**: Calculated as `grokking_step / memorization_step`. A higher ratio indicates more delayed generalization.
- **Weight Norm Rupture**: We detect structural changes in the model using a change-point detection algorithm on the weight norms. We apply piecewise linear fitting (exhaustively searching for the optimal split index) to locate the inflection point where weight norms stabilize or begin to drop.

## 2. Grokking Prediction (`src/grokking_predictor.py`)

A core area of our investigation is whether grokking can be predicted early in training before accuracy improves. We implement a predictive model using early training dynamics:

- **Early Features Extraction**: Using the first N steps of training, we extract predictive features:
  - **Final Loss Gap**: Difference between test and train loss at step N.
  - **Mean Loss Gap**: Average gap between test and train loss over the first N steps.
  - **Weight Norm Slope**: The linear trend (slope) of the weight norm trajectory, indicating the rate of weight growth.
  - **Gradient Noise Scale**: The average magnitude of gradient noise (if available).
  - **Attention Entropy**: Mean entropy of the attention weights across heads, tracking head specialization early on.
- **Logistic Regression Classifier**: We train an `sklearn` Logistic Regression model to predict a binary label (Grokking vs. No Grokking) based on these early features, reporting feature importances to determine which signals are the most predictive.

## 3. Collapse Mechanism Analysis (`src/collapse_analysis.py`)

Model collapse directly affects a model's ability to grok. We analyze the severity and mechanisms of model collapse using several distributional and structural metrics:

- **Distributional Shift**:
  - We compute the **Kullback-Leibler (KL) divergence** and **Jensen-Shannon (JS) divergence** between the clean and collapsed training data distributions.
- **Mode Collapse Indicators**:
  - **Vocabulary Size**: The fraction of the total vocabulary utilized in model outputs.
  - **Entropy**: Shannon entropy of the model's output distribution. Low entropy suggests mode collapse.
  - **N-gram Diversity**: Computes the uniqueness ratio of unigrams and bigrams to quantify output repetition and structural decay.
- **Specialization Correlation**:
  - We measure Pearson correlation between the varying severities of collapse and the degree of attention head specialization, observing how structural decay impacts the mechanistic formation of specific circuits.

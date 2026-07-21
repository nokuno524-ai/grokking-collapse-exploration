import json
import os
import glob
import numpy as np
from typing import List, Dict, Tuple, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import sys

# Try to import metrics if available, otherwise just use dummy values for prediction
try:
    from metrics.data_quality import (
        ngram_diversity, token_distribution_shift,
        sequence_length_comparison, memorization_detection, diversity_metrics
    )
    METRICS_AVAILABLE = True
except ImportError:
    # Handle the case where we're running from a different directory
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from metrics.data_quality import (
            ngram_diversity, token_distribution_shift,
            sequence_length_comparison, memorization_detection, diversity_metrics
        )
        METRICS_AVAILABLE = True
    except ImportError:
        METRICS_AVAILABLE = False


def extract_metrics_from_data(original_data: List[List[int]],
                              synthetic_data: List[List[int]]) -> Dict[str, float]:
    """
    Compute all data quality metrics for a pair of original and synthetic datasets.
    """
    if not METRICS_AVAILABLE:
        raise ImportError("Could not import metrics module.")

    metrics = {}

    # N-gram diversity (we just use n=1, 2, 3 since sequences are short)
    ngrams_orig = ngram_diversity(original_data, max_n=3)
    ngrams_synth = ngram_diversity(synthetic_data, max_n=3)

    for n in range(1, 4):
        metrics[f"ngram_div_orig_{n}"] = ngrams_orig.get(n, 0.0)
        metrics[f"ngram_div_synth_{n}"] = ngrams_synth.get(n, 0.0)
        metrics[f"ngram_div_diff_{n}"] = ngrams_orig.get(n, 0.0) - ngrams_synth.get(n, 0.0)

    # Distribution shift
    metrics["kl_divergence"] = token_distribution_shift(original_data, synthetic_data)

    # Sequence length
    seq_len_metrics = sequence_length_comparison(original_data, synthetic_data)
    metrics["length_ks_stat"] = seq_len_metrics["ks_statistic"]
    metrics["length_wasserstein"] = seq_len_metrics["wasserstein_distance"]

    # Memorization
    metrics["memorization_fraction"] = memorization_detection(original_data, synthetic_data)

    # Diversity
    div_orig = diversity_metrics(original_data)
    div_synth = diversity_metrics(synthetic_data)

    for k, v in div_orig.items():
        metrics[f"div_orig_{k}"] = v
    for k, v in div_synth.items():
        metrics[f"div_synth_{k}"] = v

    return metrics


class CollapsePredictor:
    """
    Predicts model collapse from data quality metrics using a simple classifier.
    """

    def __init__(self, model_type: str = "rf"):
        self.model_type = model_type
        if model_type == "lr":
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == "rf":
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False

    def _prepare_features(self, metrics_list: List[Dict[str, float]], fit: bool = False) -> np.ndarray:
        if fit:
            self.feature_names = sorted(list(metrics_list[0].keys()))

        X = np.zeros((len(metrics_list), len(self.feature_names)))
        for i, metrics in enumerate(metrics_list):
            for j, name in enumerate(self.feature_names):
                X[i, j] = metrics.get(name, 0.0)

        # Handle inf/nan
        X = np.nan_to_num(X, posinf=1e6, neginf=-1e6)

        if fit:
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)

    def train(self, metrics_list: List[Dict[str, float]], labels: List[int]) -> float:
        """
        Train the predictor. Labels should be 1 for collapse, 0 for no collapse.
        Returns the cross-validation score (accuracy).
        """
        if not metrics_list:
            return 0.0

        X = self._prepare_features(metrics_list, fit=True)
        y = np.array(labels)

        # Calculate CV score if we have enough samples and classes
        cv_score = 0.0
        if len(y) >= 10 and len(np.unique(y)) > 1:
            cv_scores = cross_val_score(self.model, X, y, cv=5)
            cv_score = np.mean(cv_scores)

        # Fit on all data
        self.model.fit(X, y)
        self.is_fitted = True

        return cv_score

    def predict(self, metrics: Dict[str, float]) -> Tuple[int, float]:
        """
        Predict whether the data will cause model collapse.
        Returns (prediction, probability).
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call train() first.")

        X = self._prepare_features([metrics], fit=False)
        pred = self.model.predict(X)[0]
        prob = self.model.predict_proba(X)[0][1] # Probability of class 1 (collapse)

        return int(pred), float(prob)

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importances.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call train() first.")

        importances = {}
        if self.model_type == "rf":
            imp = self.model.feature_importances_
        elif self.model_type == "lr":
            imp = np.abs(self.model.coef_[0])

        for name, val in zip(self.feature_names, imp):
            importances[name] = float(val)

        # Sort by importance
        return {k: v for k, v in sorted(importances.items(), key=lambda item: item[1], reverse=True)}

    def find_collapse_threshold(self, feature_name: str, metrics_list: List[Dict[str, float]], labels: List[int]) -> float:
        """
        Find a simple threshold on a single feature that best separates the classes.
        """
        values = [m.get(feature_name, 0.0) for m in metrics_list]
        values = np.array(values)
        labels = np.array(labels)

        # Handle inf/nan
        values = np.nan_to_num(values, posinf=1e6, neginf=-1e6)

        sorted_idx = np.argsort(values)
        sorted_values = values[sorted_idx]
        sorted_labels = labels[sorted_idx]

        best_acc = 0
        best_threshold = 0

        for i in range(len(sorted_values) - 1):
            threshold = (sorted_values[i] + sorted_values[i+1]) / 2

            # Try < threshold = class 0
            pred1 = (values >= threshold).astype(int)
            acc1 = np.mean(pred1 == labels)

            # Try < threshold = class 1
            pred2 = (values < threshold).astype(int)
            acc2 = np.mean(pred2 == labels)

            if acc1 > best_acc:
                best_acc = acc1
                best_threshold = threshold
            if acc2 > best_acc:
                best_acc = acc2
                best_threshold = threshold

        return best_threshold

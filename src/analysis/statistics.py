import numpy as np
import pandas as pd
from typing import Tuple, Optional
import scipy.stats as stats
import statsmodels.api as sm

def bootstrap_confidence_interval(data: np.ndarray, num_samples: int = 1000, ci: float = 95.0) -> Tuple[float, float]:
    """
    Compute bootstrap confidence intervals for the mean of 1D array of data.
    """
    if len(data) == 0:
        return np.nan, np.nan

    data = np.asarray(data)
    # Filter out NaNs if any
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return np.nan, np.nan

    means = []
    for _ in range(num_samples):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))

    lower_bound = (100.0 - ci) / 2.0
    upper_bound = 100.0 - lower_bound

    return float(np.percentile(means, lower_bound)), float(np.percentile(means, upper_bound))

def compute_correlation_matrix(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Compute Pearson correlation matrix for specified columns.
    """
    # Drop rows with NaNs in the specified columns before calculating correlation
    valid_data = df[columns].dropna()
    return valid_data.corr(method='pearson')

def significance_testing(df: pd.DataFrame, target_col: str, feature_cols: list) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Run OLS Multiple Regression predicting target_col from feature_cols.
    Handles NaN values in target_col by either dropping them or replacing them (caller should pre-process if replacement is needed).
    """
    valid_data = df[[target_col] + feature_cols].dropna()

    X = valid_data[feature_cols]
    X = sm.add_constant(X)
    y = valid_data[target_col]

    model = sm.OLS(y, X).fit()
    return model

def analyze_grokking_factors(df: pd.DataFrame, max_steps: int = 50000) -> dict:
    """
    Comprehensive statistical analysis of grokking factors.
    Replaces NaN grokking_step with max_steps for continuous analysis.
    """
    # Create a copy for analysis
    analysis_df = df.copy()

    # Cap non-grokking seeds to max_steps
    if 'grokking_step' in analysis_df.columns:
        analysis_df['grokking_step'] = analysis_df['grokking_step'].fillna(max_steps)

    # Weight norm change proxy: final_weight_norm
    # Collapse proxy: collapse_severity

    features = ['collapse_severity', 'final_weight_norm', 'final_embedding_rank', 'final_fourier_concentration', 'attention_specialization']

    # Ensure columns exist
    features = [f for f in features if f in analysis_df.columns]

    corr_matrix = compute_correlation_matrix(analysis_df, features + ['grokking_step'])

    ols_model = None
    if 'grokking_step' in analysis_df.columns and len(features) > 0:
        ols_model = significance_testing(analysis_df, 'grokking_step', features)

    return {
        'correlation_matrix': corr_matrix,
        'ols_summary': ols_model.summary() if ols_model else None
    }

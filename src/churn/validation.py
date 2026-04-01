import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Floor applied to the outlier threshold to avoid false positives when
# q99 is 0 (e.g. a column that is mostly zeros).
_OUTLIER_THRESHOLD_FLOOR = 1.0

# Default multiplier applied to p99 to define the outlier threshold.
_DEFAULT_OUTLIER_MULTIPLIER = 10


def validate_features(
    features: pd.DataFrame,
    window_id: str,
    skip_outlier_cols: set[str] | None = None,
    outlier_multipliers: dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    Audit generated features for common data quality issues.

    This is a non-destructive function: it logs any issues found but always
    returns the input DataFrame unchanged. It does not drop rows or columns.

    Checks performed
    ----------------
    1. Missing values   : any NaN remaining after fills.
    2. Infinite values  : any np.inf / -np.inf in numeric columns.
    3. Extreme outliers : values exceeding N x the 99th percentile, checked
                          only on continuous numeric columns. The multiplier
                          defaults to 10 but can be overridden per column via
                          outlier_multipliers (e.g. 20 for naturally skewed
                          unbounded features like RevenueTrend).

    Parameters
    ----------
    features : DataFrame
        Feature set to audit.
    window_id : str
        Window identifier used in log messages for traceability.
    skip_outlier_cols : set of str, optional
        Column names to exclude from outlier detection entirely. Use this for
        categorical, binary, or ordinal columns where a statistical outlier
        threshold is meaningless (e.g. IsChurned, FavoriteMonthSin).
        Defaults to an empty set if not provided.
    outlier_multipliers : dict of str to int, optional
        Per-column multipliers applied to p99 to define the outlier threshold.
        Columns not listed here use _DEFAULT_OUTLIER_MULTIPLIER (10).
        Use higher values for features that are naturally right-skewed or
        unbounded by design (e.g. {'RevenueTrend': 20}).

    Returns
    -------
    DataFrame
        The input DataFrame, unchanged.
    """
    if skip_outlier_cols is None:
        skip_outlier_cols = set()
    if outlier_multipliers is None:
        outlier_multipliers = {}

    if len(features) == 0:
        logger.warning("No features to validate for window %s", window_id)
        return features

    issues = []

    # 1. Missing values
    missing = features.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        issues.append(f"Missing values: {missing.to_dict()}")

    # 2. Infinite values
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    inf_cols = [col for col in numeric_cols if np.isinf(features[col]).any()]
    if inf_cols:
        issues.append(f"Infinite values in columns: {inf_cols}")

    # 3. Extreme outliers — continuous columns only
    continuous_cols = [col for col in numeric_cols if col not in skip_outlier_cols]
    for col in continuous_cols:
        multiplier = outlier_multipliers.get(col, _DEFAULT_OUTLIER_MULTIPLIER)
        q99 = features[col].quantile(0.99)
        # Apply a floor so columns with q99 == 0 don't flag every non-zero value
        outlier_threshold = max(q99 * multiplier, _OUTLIER_THRESHOLD_FLOOR)
        n_outliers = (features[col] > outlier_threshold).sum()
        if n_outliers > 0:
            issues.append(
                f"Extreme outliers in '{col}': {n_outliers} value(s) "
                f"> {outlier_threshold:.2f} ({multiplier}x p99)"
            )

    if issues:
        logger.warning("Validation issues for window %s:", window_id)
        for issue in issues:
            logger.warning("  - %s", issue)
    else:
        logger.debug("Validation passed for window %s", window_id)

    return features


def filter_valid_customers(
    feature_df: pd.DataFrame,
    min_history_days: int = 30,
) -> pd.DataFrame:
    """
    Filter out customers with insufficient transaction history.

    A customer is considered valid if the span between their first and last
    transaction in the observation window is at least min_history_days.
    Customers with only a single transaction always fail this check.

    Parameters
    ----------
    feature_df : DataFrame
        Transaction data from the observation window containing at minimum:
        - CustomerID  : customer identifier
        - InvoiceDate : date of transaction
    min_history_days : int, default 30
        Minimum number of days between first and last transaction required
        for a customer to be included.

    Returns
    -------
    DataFrame
        Filtered transaction data containing only customers with sufficient
        transaction history. Has the same columns as the input.
    """
    if len(feature_df) == 0:
        return feature_df

    history = (
        feature_df
        .groupby('CustomerID')['InvoiceDate']
        .agg(first='min', last='max')
    )
    history['days_active'] = (history['last'] - history['first']).dt.days

    valid_customers = history[history['days_active'] >= min_history_days].index

    return feature_df[feature_df['CustomerID'].isin(valid_customers)]
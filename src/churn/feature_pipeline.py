import logging
from .feature_builders import *
from .windowing import get_window_data
from .validation import validate_features, filter_valid_customers

logger = logging.getLogger(__name__)

"""
Feature extraction pipeline for customer churn prediction.
Handles window-based feature extraction, merging, and label creation.
"""

# ---------------------------------------------------------------------------
# Fill values applied after merging all feature sets.
#
# Philosophy: NaN after a left-merge means a builder produced no row for
# that customer (e.g. a customer with a single purchase has no interval
# data). Each column gets the fill value that is semantically correct for
# that case, rather than a blanket fillna(0) which would silently conflate
# "missing" with "zero".
#
# Columns not listed here are expected to be fully populated by their
# builder and will surface as NaN in validate_features if they are not.
# ---------------------------------------------------------------------------
_FILL_VALUES = {
    # Timing features: a customer with one purchase has no measured interval
    # or delay — NaN is replaced with the window size as a conservative
    # upper-bound signal rather than 0 (which would imply no delay).
    # These are overridden per-call using window_size_days; see usage below.
    'AvgPurchaseInterval':          None,   # set dynamically
    'PurchaseIntervalCV':           0.0,    # undefined CV → treat as perfectly regular
    'DelayRatio':                   None,   # set dynamically

    # Engagement: customers absent from a builder genuinely have 0 activity
    'EngagementDensity':            0.0,
    'RecentShareLog':               0.0,
    # RevenueTrend = 1.0 means neutral (no acceleration or deceleration)
    'RevenueTrend':                 1.0,
    # ProductDiversityRate: no records → 0 diversity
    'ProductDiversityRate':         0.0,

    # Early lifecycle: customers with no early-window data have 0 activity
    'FirstMonthPurchases':          0.0,
    'MonetaryFirstMonth':           0.0,
    'AvgOrderValueFirstMonth':      0.0,

    # Returns: no return records → 0 return rate
    'ReturnRate':                   0.0,

    # Product diversity: no records → 0 unique products
    'UniqueProducts':               0.0,

    # Seasonality: no records → neutral defaults
    # sin(2π × 6 / 12) = 0.0, cos(2π × 6 / 12) = -1.0 (mid-year neutral)
    'Q4Ratio':                      0.0,
    'FavoriteMonthSin':             0.0,
    'FavoriteMonthCos':            -1.0,
}


def merge_features(feature_sets: list, on: str = 'CustomerID', how: str = 'left') -> pd.DataFrame:
    """
    Merge multiple feature DataFrames on a common key.

    Parameters
    ----------
    feature_sets : list of DataFrames
        List of feature DataFrames to merge.
    on : str, default 'CustomerID'
        Column name to merge on.
    how : str, default 'left'
        Type of merge to perform ('left', 'inner', 'outer').

    Returns
    -------
    DataFrame
        Merged DataFrame containing all features.

    Raises
    ------
    ValueError
        If any DataFrame in feature_sets is missing the key column.
    """
    if not feature_sets:
        return pd.DataFrame()

    missing = [i for i, fs in enumerate(feature_sets) if on not in fs.columns]
    if missing:
        raise ValueError(
            f"Feature sets at indices {missing} are missing the merge key '{on}'. "
            "This indicates a bug in an upstream builder."
        )

    merged = feature_sets[0]
    for fs in feature_sets[1:]:
        merged = merged.merge(fs, on=on, how=how)

    return merged


def build_churn_label(features: pd.DataFrame, label_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create churn label based on activity in the prediction window.

    A customer is considered churned if they have no purchases
    in the prediction window.

    Parameters
    ----------
    features : DataFrame
        Customer features from the observation window.
    label_df : DataFrame
        Transaction data from the prediction window.

    Returns
    -------
    DataFrame
        Input DataFrame with an added IsChurned column (0 or 1).
    """
    if len(features) == 0:
        return features

    if len(label_df) == 0:
        features['IsChurned'] = 1
        return features

    active_customers = set(label_df['CustomerID'].unique())
    features['IsChurned'] = (~features['CustomerID'].isin(active_customers)).astype(int)
    return features


def extract_features_for_window(
    df: pd.DataFrame,
    window_start,
    window_size_days: int,
    churn_threshold_days: int,
    min_customer_history: int = 30,
) -> pd.DataFrame | None:
    """
    Extract all features for a single observation window.

    Returns None if there is insufficient data for the window. Callers
    iterating over multiple windows should filter out None results before
    concatenating, e.g.:
        windows = [extract_features_for_window(...) for ...]
        all_features = pd.concat([w for w in windows if w is not None])

    Parameters
    ----------
    df : DataFrame
        Full transaction dataset.
    window_start : datetime
        Start date of the observation window.
    window_size_days : int
        Length of observation window in days.
    churn_threshold_days : int
        Number of days without a purchase to consider a customer churned.
    min_customer_history : int, default 30
        Minimum number of days between first and last transaction required
        for a customer to be included.

    Returns
    -------
    DataFrame or None
        Complete feature set with labels for the window, or None if there
        is insufficient data.
    """
    feature_df, label_df, feature_start, feature_end = get_window_data(
        df, feature_end=window_start, window_size_days=window_size_days,
        churn_threshold_days=churn_threshold_days,
    )

    if len(feature_df) == 0:
        logger.warning("No data for window starting %s", window_start.date())
        return None

    feature_df = filter_valid_customers(feature_df, min_customer_history)

    if len(feature_df) == 0:
        logger.warning("No valid customers for window starting %s", window_start.date())
        return None

    # Core behavioral features
    rfm = build_rfm(feature_df, feature_end, window_size_days)
    aov = build_aov(feature_df)
    return_rate = build_return_rate(feature_df)

    # Temporal & lifecycle features
    lifetime = build_customer_lifetime(feature_df, feature_end)
    early = build_early_engagement(feature_df)

    # Behavioral dynamics
    trend_df = build_trend_features(feature_df, feature_end, window_size_days)

    # Timing & consistency
    purchase_intervals = build_purchase_intervals(feature_df)
    delay_features = build_purchase_delay_features(rfm, purchase_intervals)

    # Seasonality
    seasonality = build_seasonality(feature_df)

    # Product behavior
    product_diversity = build_product_diversity(feature_df)

    # Engagement intensity
    engagement_intensity = build_engagement_intensity(feature_df, window_size_days)

    feature_sets = [
        # Core behavioral
        rfm, aov, return_rate,
        # Temporal & lifecycle
        lifetime, early,
        # Behavioral dynamics
        trend_df,
        # Timing & consistency
        purchase_intervals, delay_features,
        # Seasonality
        seasonality,
        # Product behavior
        product_diversity,
        # Engagement intensity
        engagement_intensity,
    ]

    features = merge_features(feature_sets, on='CustomerID', how='left')

    # Apply intentional per-column fill values (see _FILL_VALUES for rationale).
    # AvgPurchaseInterval and DelayRatio use window_size_days as their fallback
    # since a customer with no interval history is treated as maximally overdue.
    fill_values = {
        **_FILL_VALUES,
        'AvgPurchaseInterval': float(window_size_days),
        'DelayRatio':          float(window_size_days),
    }
    features = features.fillna(fill_values)

    features = build_derived_features(features)
    features = build_churn_label(features, label_df)

    window_id = f"{feature_start.strftime('%Y%m')}_{feature_end.strftime('%Y%m')}"
    features['WindowStart'] = feature_start
    features['WindowEnd'] = feature_end
    features['WindowID'] = window_id

    features = validate_features(
        features,
        window_id,
        skip_outlier_cols={
            'IsChurned',
            'FavoriteMonthSin',
            'FavoriteMonthCos',
            'WindowStart',
            'WindowEnd',
            'WindowID',
            # DelayRatio is structurally unbounded for churned customers:
            # it equals recency / AvgPurchaseInterval, and recency can be
            # arbitrarily large. Outlier detection fires on every window
            # and produces no actionable signal.
            'DelayRatio',
        },
        outlier_multipliers={
            # RevenueTrend is intentionally unbounded (cap removed to improve
            # decile capture). A higher multiplier reduces noise while still
            # flagging genuinely extreme values.
            'RevenueTrend': 20,
        },
    )

    return features
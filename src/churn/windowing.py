import logging
from datetime import timedelta

import pandas as pd

logger = logging.getLogger(__name__)

def get_window_data(
    df: pd.DataFrame,
    feature_end,
    window_size_days: int,
    churn_threshold_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """
    Split transaction data into a feature window and a label window.

    The feature window is the observation period used to build customer
    features. The label window is the subsequent period used to determine
    whether each customer churned.

        |<--- window_size_days --->|<--- churn_threshold_days --->|
        feature_start          feature_end                    label_end

    Parameters
    ----------
    df : DataFrame
        Full transaction dataset containing at minimum:
        - InvoiceDate : date of transaction
    feature_end : datetime
        End of the feature window (exclusive) and start of the label window
        (inclusive). All feature data is drawn from before this date.
    window_size_days : int
        Length of the feature (observation) window in days.
    churn_threshold_days : int
        Length of the label (prediction) window in days.

    Returns
    -------
    feature_df : DataFrame
        Transactions in [feature_start, feature_end).
    label_df : DataFrame
        Transactions in [feature_end, label_end).
    feature_start : Timestamp
        Start of the feature window.
    feature_end : Timestamp
        End of the feature window (same as the input feature_end).

    Raises
    ------
    ValueError
        If df is empty or does not contain an InvoiceDate column.
    """
    if len(df) == 0:
        raise ValueError("Input DataFrame is empty.")
    if 'InvoiceDate' not in df.columns:
        raise ValueError("Input DataFrame must contain an 'InvoiceDate' column.")

    feature_start = feature_end - timedelta(days=window_size_days)
    label_end = feature_end + timedelta(days=churn_threshold_days)

    feature_df = df[
        (df['InvoiceDate'] >= feature_start) &
        (df['InvoiceDate'] < feature_end)
    ].copy()

    label_df = df[
        (df['InvoiceDate'] >= feature_end) &
        (df['InvoiceDate'] < label_end)
    ].copy()

    logger.debug(
        "Window: feature=[%s, %s), label=[%s, %s) — "
        "%d feature rows, %d label rows",
        feature_start.date(), feature_end.date(),
        feature_end.date(), label_end.date(),
        len(feature_df), len(label_df),
    )

    return feature_df, label_df, feature_start, feature_end
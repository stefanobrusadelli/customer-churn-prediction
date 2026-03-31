from .feature_builders import *
from .windowing import get_window_data
from .validation import validate_features, filter_valid_customers

"""
Feature extraction pipeline for customer churn prediction.
Handles window-based feature extraction, merging, and label creation.
"""

def merge_features(feature_sets, on='CustomerID', how='left'):
    """
    Merge multiple feature DataFrames on a common key.
    
    Parameters:
    -----------
    feature_sets : list of DataFrames
        List of feature DataFrames to merge
    on : str, default='CustomerID'
        Column name to merge on
    how : str, default='left'
        Type of merge to perform ('left', 'inner', 'outer')
    
    Returns:
    --------
    merged : DataFrame
        Merged DataFrame containing all features
    """
    if not feature_sets:
        return pd.DataFrame()
    
    merged = feature_sets[0]
    
    for fs in feature_sets[1:]:
        if on in fs.columns:  # Only merge if the key column exists
            merged = merged.merge(fs, on=on, how=how)
    
    return merged

def build_churn_label(features, label_df):
    """
    Create churn label based on activity in prediction window.
    
    A customer is considered churned if they have NO purchases
    in the prediction window.
    
    Parameters:
    -----------
    features : DataFrame
        Customer features from observation window
    label_df : DataFrame
        Transaction data from prediction window
    
    Returns:
    --------
    DataFrame with added IsChurned column
    """
    if len(features) == 0:
        return features
        
    if len(label_df) == 0:
        # If no label data, all customers churned
        features['IsChurned'] = 1
        return features
    
    customers_in_label = label_df['CustomerID'].unique()
    features['IsChurned'] = (~features['CustomerID'].isin(customers_in_label)).astype(int)
    return features

def extract_features_for_window(df, window_start, window_size_days, churn_threshold_days, min_customer_history=30):
    """
    Extract all features for a single window.
    
    Parameters:
    -----------
    df : DataFrame
        Full transaction dataset
    window_start : datetime
        Start date of the observation window
    window_size_days : int
        Length of observation window in days
    churn_threshold_days : int
        Number of days without purchase to consider as churned
    min_customer_history : int, default=30
        Minimum number of days between first and last transaction required
        for a customer to be considered valid
    
    Returns:
    --------
    features : DataFrame
        Complete feature set with labels for the window
    """
    feature_df, label_df, feature_start, feature_end = get_window_data(
        df, window_start, window_size_days, churn_threshold_days
    )
    
    if len(feature_df) == 0:
        print(f"No data for window starting {window_start.date()}")
        return None
    
    feature_df = filter_valid_customers(feature_df, min_customer_history)
    
    if len(feature_df) == 0:
        print(f"No valid customers for window starting {window_start.date()}")
        return None
    
    # Core behavioral features
    rfm = build_rfm(feature_df, feature_end, window_size_days)
    aov = build_aov(feature_df)
    return_rate = build_return_rate(feature_df)
    
    # Temporal & lifecycle features
    first_purchase = get_first_purchase_dates(feature_df)
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
        rfm,
        aov,
        return_rate,
        seasonality,
        early,
        trend_df,
        lifetime,
        purchase_intervals,
        product_diversity,
        delay_features,
        engagement_intensity
    ]
    features = merge_features(feature_sets, on='CustomerID', how='left')
    
    features = features.fillna(0)
    
    features = build_churn_label(features, label_df)
    
    features['WindowStart'] = feature_start
    features['WindowEnd'] = feature_end
    features['WindowID'] = f"{feature_start.strftime('%Y%m')}_{feature_end.strftime('%Y%m')}"

    features = validate_features(features, features['WindowID'].iloc[0])
    
    return features
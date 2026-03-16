from datetime import timedelta

def get_window_data(df, window_start, window_size_days, churn_threshold_days):
    """
    Split data into feature and label windows.
    
    Parameters:
    -----------
    df : DataFrame
        Full transaction data
    window_start : datetime
        Start of prediction window (end of feature window)
    window_size_days : int
        Length of observation window
    churn_threshold_days : int
        Length of prediction window
    
    Returns:
    --------
    feature_df, label_df, feature_start, feature_end
    """
    feature_end = window_start
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
    
    return feature_df, label_df, feature_start, feature_end
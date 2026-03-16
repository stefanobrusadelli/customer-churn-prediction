import numpy as np

def validate_features(features, window_id):
    """
    Check for common data quality issues in generated features.
    
    Parameters:
    -----------
    features : DataFrame
        Features to validate
    window_id : str
        Window identifier for logging
    
    Returns:
    --------
    DataFrame with validation passed/failed
    """
    if len(features) == 0:
        print(f"No features to validate for {window_id}")
        return features
    
    issues = []
    
    # Check for missing values
    missing = features.isnull().sum()
    if missing.sum() > 0:
        issues.append(f"Missing values: {missing[missing>0].to_dict()}")
    
    # Check for infinite values
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(features[col]).any():
            issues.append(f"Infinite values in {col}")
    
    # Check for extreme outliers
    for col in numeric_cols:
        q99 = features[col].quantile(0.99)
        outlier_threshold = q99 * 10
        if (features[col] > outlier_threshold).any():
            issues.append(f"Extreme outliers in {col} (values > {outlier_threshold:.2f})")
    
    if issues:
        print(f"Validation issues for {window_id}:")
        for issue in issues:
            print(f"    - {issue}")
    
    return features

def filter_valid_customers(feature_df, min_history_days=30):
    """
    Filter customers with enough transaction history in the observation window.
    
    Features:
    - Filters out customers whose first and last transaction span less than min_history_days
    
    Notes:
    - Default minimum history is 30 days
    
    Parameters:
    -----------
    feature_df : DataFrame
        Transaction data from observation window containing at minimum:
        - CustomerID: Customer identifier
        - InvoiceDate: Date of transaction
    min_history_days : int, default=30
        Minimum number of days between first and last transaction required
        for a customer to be considered valid
    
    Returns:
    --------
    filtered_df : DataFrame
        Filtered transaction data containing only customers with 
        sufficient transaction history
    """
    if len(feature_df) == 0:
        return feature_df
    
    history = feature_df.groupby('CustomerID')['InvoiceDate'].agg(['min', 'max'])
    history['days_active'] = (history['max'] - history['min']).dt.days
    
    valid_customers = history[history['days_active'] >= min_history_days].index
    
    filtered_df = feature_df[feature_df['CustomerID'].isin(valid_customers)]
    
    return filtered_df
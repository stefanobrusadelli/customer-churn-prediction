"""
Feature Builders Module

Feature groups:
- RFM features → customer value & recency
- Engagement intensity → activity density over time
- Trend features → behavioral change over time
- Early lifecycle → first purchase behavior
- Seasonality → temporal purchasing patterns
- Timing → purchase regularity & delay
- Product behavior → diversity & returns
"""
import pandas as pd
import numpy as np
from datetime import timedelta

def build_rfm(feature_df, feature_end, window_size_days):
    """
    Calculate Recency, Frequency, and Monetary features for each customer.
    
    Features:
    - Recency: Days since customer's last purchase (as of feature_end)
    - Frequency: Number of unique invoices/purchases made
    - Monetary: Total spend across all transactions (observation window)
    - LogMonetary: Natural log of (Monetary + 1) to handle skewness
    - LogFrequency: Natural log of (Frequency + 1) to handle skewness
    
    Parameters:
    -----------
    feature_df : DataFrame
        Transaction data from observation window containing at minimum:
        - CustomerID: Customer identifier
        - InvoiceDate: Date of transaction
        - Invoice: Invoice identifier
        - TotalSum: Transaction amount
    feature_end : datetime
        End of observation window (used to calculate recency)
    
    Returns:
    --------
    rfm : DataFrame
        RFM features with columns:
        - CustomerID: Unique customer identifier
        - Recency: Days since last purchase
        - Frequency: Number of purchases
        - Monetary: Total spend
        - LogMonetary: Log-transformed monetary value
        - LogFrequency: Log-transformed frequency
    """
    if len(feature_df) == 0:
        return pd.DataFrame(columns=[
            'CustomerID', 'Recency', 'Frequency', 'Monetary',
            'LogFrequency', 'LogMonetary'
        ])

    # Frequency & Recency (purchase only)
    purchase_df = feature_df[
        feature_df['TransactionType'] == 'Standard_Purchase'
    ].copy()

    freq_rec = purchase_df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (feature_end - x.max()).days,
        'Invoice': 'nunique'
    }).rename(columns={
        'InvoiceDate': 'Recency',
        'Invoice': 'Frequency'
    })

    # Monetary (all transactions)
    monetary = feature_df.groupby('CustomerID')['TotalSum'].sum().rename('Monetary')

    # Merge
    rfm = freq_rec.join(monetary, how='outer').reset_index()

    rfm['Frequency'] = rfm['Frequency'].fillna(0)
    rfm['Recency'] = rfm['Recency'].fillna(window_size_days)
    rfm['Monetary'] = rfm['Monetary'].fillna(0)

    # Log transforms help reduce skewness in transactional variables
    rfm['LogMonetary'] = np.log1p(np.maximum(0, rfm['Monetary']))
    rfm['LogFrequency'] = np.log1p(np.maximum(0, rfm['Frequency']))

    return rfm

def build_aov(feature_df):
    """
    Calculate average order value and variability of transaction values.

    Features:
    - AvgOrderValue: Average transaction amount
    - OrderValueCV: Coefficient of variation (std/mean) of transaction amounts
        - Measures spending consistency/volatility
        - Higher values indicate more variable spending patterns
    - IsOrderValueCVDefined: Indicator whether CV is well-defined (mean > 0 and std available)

    Parameters:
    -----------
    feature_df : DataFrame
        Transaction data from observation window containing at minimum:
        - CustomerID: Customer identifier
        - TotalSum: Transaction amount

    Returns:
    --------
    aov : DataFrame
        Order value features with columns:
        - CustomerID: Unique customer identifier
        - AvgOrderValue: Average transaction value
        - OrderValueCV: Coefficient of variation of order value
        - IsOrderValueCVDefined: Indicator for valid CV computation
    """

    if len(feature_df) == 0:
        return pd.DataFrame(columns=[
            'CustomerID', 'AvgOrderValue','OrderValueCV',
            'IsOrderValueCVDefined'
        ])

    agg = (
        feature_df
        .groupby('CustomerID')['TotalSum']
        .agg(['mean', 'std'])
        .reset_index()
    )

    agg.columns = ['CustomerID', 'AvgOrderValue', 'OrderValueStd']

    valid = (agg['AvgOrderValue'] > 0) & (agg['OrderValueStd'].notna())

    agg['OrderValueCV'] = np.where(
        valid,
        agg['OrderValueStd'] / agg['AvgOrderValue'],
        0
    )
    agg['IsOrderValueCVDefined'] = valid.astype(int)

    agg = agg.fillna(0)

    return agg[[
        'CustomerID',
        'AvgOrderValue',
        'OrderValueCV',
        'IsOrderValueCVDefined'
    ]]

def build_return_rate(feature_df):
    """
    Calculate return rate with proper handling of edge cases.
    
    Features:
    - ReturnRate: Proportion of purchases that resulted in returns (capped at 1.0)
    
    Notes:
    - Returns are identified by TransactionType 'Linked_Return' or 'Unlinked_Return'
    - Purchases are identified by TransactionType 'Standard_Purchase'
    - Return rate is calculated as: NumReturns / TotalPurchases
    - Rate is capped at 1.0 to handle data anomalies or extreme cases
    - Customers with no purchases receive a return rate of 0
    
    Parameters:
    -----------
    feature_df : DataFrame
        Transaction data from observation window containing at minimum:
        - CustomerID: Customer identifier
        - TransactionType: Type of transaction (Standard_Purchase, Linked_Return, Unlinked_Return)
        - Invoice: Invoice identifier (for counting unique purchases)
    
    Returns:
    --------
    rr : DataFrame
        Return rate features with columns:
        - CustomerID: Unique customer identifier
        - ReturnRate: Proportion of purchases returned (0.0 to 1.0)
    """
    if len(feature_df) == 0:
        return pd.DataFrame(columns=['CustomerID', 'ReturnRate'])
    
    returns = feature_df[
        feature_df['TransactionType'].isin(['Linked_Return', 'Unlinked_Return'])
    ].copy()
    return_counts = returns.groupby('CustomerID').size().rename('NumReturns')
    
    purchases = feature_df[
        feature_df['TransactionType'] == 'Standard_Purchase'
    ].groupby('CustomerID')['Invoice'].nunique().rename('TotalPurchases')
    
    rr = pd.concat([return_counts, purchases], axis=1).fillna(0)
    
    rr['ReturnRate'] = np.where(
        rr['TotalPurchases'] > 0,
        rr['NumReturns'] / rr['TotalPurchases'],
        0
    )
    
    return rr[['ReturnRate']].reset_index()

def build_seasonality(feature_df):
    """
    Extract seasonal purchasing patterns.
    
    Features:
    - Q4Ratio: Proportion of purchases in Q4 (Oct-Dec)
    - FavoriteMonth: Most common purchase month (1-12, defaults to 6 if no mode)
    
    Parameters:
    -----------
    feature_df : DataFrame
        Transaction data from observation window containing at minimum:
        - CustomerID: Customer identifier
        - InvoiceDate: Date of transaction (used to extract month)
    
    Returns:
    --------
    season : DataFrame
        Seasonality features with columns:
        - CustomerID: Unique customer identifier
        - Q4Ratio: Proportion of purchases occurring in Q4 (0.0 to 1.0)
        - FavoriteMonth: Most frequent purchase month (1-12)
    """
    if len(feature_df) == 0:
        return pd.DataFrame(columns=['CustomerID', 'Q4Ratio', 'FavoriteMonth'])
    
    df = feature_df.copy()

    df['Month'] = df['InvoiceDate'].dt.month
    df['IsQ4'] = df['Month'].isin([10, 11, 12])

    # Q4Ratio & FavoriteMonth 
    season = df.groupby('CustomerID').agg({
        'IsQ4': 'mean',
        'Month': lambda x: x.mode()[0] if len(x.mode()) > 0 else 6
    }).reset_index()

    season.columns = ['CustomerID', 'Q4Ratio', 'FavoriteMonth']

    return season

def build_early_engagement(feature_df, early_period_days=30):
    """
    Capture customer behavior in their first N days.
    
    Features:
    - FirstMonthPurchases: Number of purchases in first N days
    - MonetaryFirstMonth: Total spend in first N days
    - AvgOrderValueFirstMonth: Average order value in first N days
        - Captures quality of early engagement
        - Higher values indicate higher-value initial purchases
    
    Parameters:
    -----------
    feature_df : DataFrame
        Transaction data from observation window containing at minimum:
        - CustomerID: Customer identifier
        - InvoiceDate: Date of transaction
        - Invoice: Invoice identifier
        - TotalSum: Transaction amount
    
    Returns:
    --------
    early : DataFrame
        Early engagement metrics with columns:
        - CustomerID: Unique customer identifier
        - FirstMonthPurchases: Number of purchases in first N days
        - MonetaryFirstMonth: Total spend in first N days
        - AvgOrderValueFirstMonth: Average order value in first N days
    """

    if len(feature_df) == 0:
        return pd.DataFrame(columns=[
            'CustomerID',
            'FirstMonthPurchases',
            'FirstMonthSpend',
            'FirstMonthAvgOrderValue'
        ])
    
    first_purchase = get_first_purchase_dates(feature_df).copy()
    
    df = feature_df.merge(first_purchase, on='CustomerID')
    df['DaysSinceFirst'] = (df['InvoiceDate'] - df['FirstPurchaseDate']).dt.days

    # Calculate early engagement metrics
    early = df[df['DaysSinceFirst'] <= early_period_days] \
        .groupby('CustomerID') \
        .agg(
            FirstMonthPurchases=('Invoice', 'nunique'),
            MonetaryFirstMonth=('TotalSum', 'sum')
        ) \
        .reset_index()

    # Ensure all customers are represented
    early = first_purchase[['CustomerID']].merge(
        early, on='CustomerID', how='left'
    ).fillna(0)

    early['AvgOrderValueFirstMonth'] = np.where(
        early['FirstMonthPurchases'] > 0,
        early['MonetaryFirstMonth'] / early['FirstMonthPurchases'],
        0
    )

    return early

def get_first_purchase_dates(feature_df):
    """
    Extract the first transaction date for each customer.
    
    Features:
    - FirstPurchaseDate: Date of customer's earliest transaction in the dataset
    
    Parameters:
    -----------
    feature_df : DataFrame
        Transaction data from observation window containing at minimum:
        - CustomerID: Customer identifier
        - InvoiceDate: Date of transaction
    
    Returns:
    --------
    first_purchase : DataFrame
        First purchase dates with columns:
        - CustomerID: Unique customer identifier
        - FirstPurchaseDate: Date of customer's first transaction
    """
    if len(feature_df) == 0:
        return pd.DataFrame(columns=['CustomerID', 'FirstPurchaseDate'])

    first_purchase = (feature_df
        .groupby('CustomerID')['InvoiceDate']
        .min()
        .reset_index()
        .rename(columns={'InvoiceDate': 'FirstPurchaseDate'})
    )
    
    return first_purchase

def build_trend_features(
    feature_df,
    feature_end,
    window_size_days,
    recent_days=30,
    epsilon=1e-6
):
    """
    Compare recent behavior vs historical behavior to identify trends.
    
    Features:
    - RevenueTrend: Recent revenue rate / historical revenue rate
        - >1 indicates increasing revenue velocity
        - <1 indicates decreasing revenue velocity
    - FrequencyTrend: Recent frequency rate / historical frequency rate
        - >1 indicates increasing purchase frequency
        - <1 indicates decreasing purchase frequency
    - MonetaryRecent: Total spend in the recent period
        - Captures absolute recent spending level
    - FreqRecent: Number of purchases in the recent period
        - Captures absolute recent engagement level
    - IsNewCustomerTrend: Indicator for customers with no historical activity but recent activity
    - NoHistoricalActivity: Indicator for customers with no activity in the historical period
    
    Notes:
    - Recent period is defined as the last `recent_days` of the observation window
    - Historical period is the remaining portion of the observation window
    - Smooth ratios are used (with epsilon) to avoid division by zero
    - Extreme ratios are capped at 10 to prevent outliers from dominating
    
    Parameters:
    -----------
    feature_df : DataFrame
        Transaction data from observation window containing at minimum:
        - CustomerID: Customer identifier
        - InvoiceDate: Date of transaction
        - Invoice: Invoice identifier
        - TotalSum: Transaction amount
        - TransactionType: Must include 'Standard_Purchase' records
    feature_end : datetime
        End of observation window
    window_size_days : int
        Total observation window length in days
    recent_days : int, default=30
        Number of days to consider as "recent" period
    epsilon : float, default=1e-6
        Small constant to avoid division by zero
    
    Returns:
    --------
    trend_df : DataFrame
        Trend features with columns:
        - CustomerID: Unique customer identifier
        - RevenueTrend: Ratio of recent to historical revenue rate (capped at 10)
        - FrequencyTrend: Ratio of recent to historical frequency rate (capped at 10)
        - MonetaryRecent: Total revenue in recent period
        - FreqRecent: Number of purchases in recent period
        - IsNewCustomerTrend: Binary flag for new/emerging activity
        - NoHistoricalActivity: Binary flag for no past activity
    """

    if len(feature_df) == 0:
        return pd.DataFrame(columns=[
            'CustomerID',
            'RevenueTrend', 'FrequencyTrend',
            'MonetaryRecent', 'FreqRecent',
            'IsNewCustomerTrend', 'NoHistoricalActivity'
        ])

    # Define time windows
    recent_start = feature_end - timedelta(days=recent_days)

    # Frequency (purchase only) 
    purchase_df = feature_df[
        feature_df['TransactionType'] == 'Standard_Purchase'
    ].copy()
    
    purchase_df['is_recent'] = purchase_df['InvoiceDate'] >= recent_start
    
    freq = purchase_df.groupby('CustomerID').agg(
        FreqRecent=('Invoice', lambda x: x[purchase_df.loc[x.index, 'is_recent']].nunique()),
        FreqHistorical=('Invoice', lambda x: x[~purchase_df.loc[x.index, 'is_recent']].nunique())
    )
    
    # Revenue (all transactions)
    feature_df['is_recent'] = feature_df['InvoiceDate'] >= recent_start
    
    revenue = feature_df.groupby('CustomerID').agg(
        recent_revenue=('TotalSum', lambda x: x[feature_df.loc[x.index, 'is_recent']].sum()),
        hist_revenue=('TotalSum', lambda x: x[~feature_df.loc[x.index, 'is_recent']].sum())
    )
    
    # Merge
    trend_df = revenue.join(freq, how='outer')
    trend_df = trend_df.fillna(0)

    # Time normalization
    recent_period_days = max(recent_days, 1)
    historical_period_days = max(window_size_days - recent_days, 1)

    # Rates
    trend_df['RecentRevenueRate'] = trend_df['recent_revenue'] / recent_period_days
    trend_df['HistoricalRevenueRate'] = trend_df['hist_revenue'] / historical_period_days

    trend_df['RecentFreqRate'] = trend_df['FreqRecent'] / recent_period_days
    trend_df['HistoricalFreqRate'] = trend_df['FreqHistorical'] / historical_period_days

    # Activity momentum (recent vs overall baseline rate)
    overall_freq_rate = (
        (trend_df['FreqRecent'] + trend_df['FreqHistorical']) /
        max(window_size_days, 1)
    )

    # Flags
    trend_df['IsNewCustomerTrend'] = (
        (trend_df['HistoricalRevenueRate'] < epsilon) &
        (trend_df['RecentRevenueRate'] > 0)
    ).astype(int)

    trend_df['NoHistoricalActivity'] = (
        trend_df['HistoricalRevenueRate'] < epsilon
    ).astype(int)

    # Final output
    trend_df = trend_df.reset_index()

    return trend_df[['CustomerID', 'IsNewCustomerTrend', 'NoHistoricalActivity']]

def build_customer_lifetime(feature_df, feature_end):
    """
    Calculate customer tenure.
    
    Features:
    - CustomerLifetime: Days since first purchase (as of feature_end)
    
    Parameters:
    -----------
    first_purchase_df : DataFrame
        Contains first purchase dates per customer with columns:
        - CustomerID: Customer identifier
        - FirstPurchaseDate: Date of customer's first transaction
    feature_end : datetime
        End of observation window (used to calculate lifetime)
    
    Returns:
    --------
    lifetime : DataFrame
        Customer lifetime features with columns:
        - CustomerID: Unique customer identifier
        - CustomerLifetime: Days since first purchase
    """
    if len(feature_df) == 0:
        return pd.DataFrame(columns=['CustomerID', 'CustomerLifetime'])

    df = get_first_purchase_dates(feature_df).copy()
    
    df['CustomerLifetime'] = (
        feature_end - df['FirstPurchaseDate']
    ).dt.days

    df = df[['CustomerID', 'CustomerLifetime']]

    return df

def build_purchase_intervals(feature_df):
    """
    Calculate statistics of time between purchases.
    
    Features:
    - AvgPurchaseInterval: Mean number of days between consecutive purchases
    - PurchaseIntervalCV: Coefficient of variation (std/mean) of purchase intervals
        - Measures regularity of purchase timing
        - Lower values indicate more consistent purchasing patterns
        - Set to 0 when undefined (e.g., insufficient data or zero mean)
    - IsPurchaseIntervalCVDefined: Indicator whether CV is statistically defined
        - 1 if AvgPurchaseInterval > 0 and StdPurchaseInterval is available
        - 0 otherwise (e.g., single purchase or insufficient data)
        - Helps distinguish true regularity (CV = 0) from missing/undefined cases
    
    Parameters:
    -----------
    feature_df : DataFrame
        Transaction data from observation window containing at minimum:
        - CustomerID: Customer identifier
        - InvoiceDate: Date of transaction (for calculating intervals)
    
    Returns:
    --------
    intervals : DataFrame
        Purchase interval features with columns:
        - CustomerID: Unique customer identifier
        - AvgPurchaseInterval: Mean days between purchases
        - StdPurchaseInterval: Standard deviation of purchase intervals
        - PurchaseIntervalCV: Coefficient of variation of intervals (filled with 0 if undefined)
        - IsPurchaseIntervalCVDefined: Binary flag indicating whether CV is defined
    """
    if len(feature_df) == 0:
        return pd.DataFrame(columns=[
            'CustomerID',
            'AvgPurchaseInterval',
            'PurchaseIntervalCV',
            'IsPurchaseIntervalCVDefined'
        ])

    purchase_df = feature_df[feature_df['TransactionType'] == 'Standard_Purchase'].copy()
    df = purchase_df.sort_values(['CustomerID', 'InvoiceDate']).copy()

    df['PrevPurchase'] = df.groupby('CustomerID')['InvoiceDate'].shift(1)
    df['Interval'] = (df['InvoiceDate'] - df['PrevPurchase']).dt.days

    intervals = df.groupby('CustomerID')['Interval'].agg(['mean', 'std']).reset_index()

    intervals.columns = [
        'CustomerID',
        'AvgPurchaseInterval',
        'StdPurchaseInterval'
    ]

    valid = (intervals['AvgPurchaseInterval'] > 0) & (intervals['StdPurchaseInterval'].notna())
    intervals['PurchaseIntervalCV'] = np.where(
        valid,
        intervals['StdPurchaseInterval'] / intervals['AvgPurchaseInterval'],
        0
    )
    intervals['IsPurchaseIntervalCVDefined'] = valid.astype(int)

    return intervals[['CustomerID','AvgPurchaseInterval','PurchaseIntervalCV',
            'IsPurchaseIntervalCVDefined']]

def build_product_diversity(feature_df):
    """
    Measure how many unique products a customer buys.
    
    Features:
    - UniqueProducts: Number of distinct products (StockCode) purchased
    
    Parameters:
    -----------
    feature_df : DataFrame
        Transaction data from observation window containing at minimum:
        - CustomerID: Customer identifier
        - StockCode: Product identifier
        - Invoice: Invoice identifier
    
    Returns:
    --------
    diversity : DataFrame
        Product diversity features with columns:
        - CustomerID: Unique customer identifier
        - UniqueProducts: Count of distinct products purchased
    """

    if len(feature_df) == 0:
        return pd.DataFrame(columns=[
            'CustomerID', 'UniqueProducts'
        ])

    diversity = feature_df.groupby('CustomerID').agg({
        'StockCode': 'nunique',
    }).reset_index()

    diversity.columns = [
        'CustomerID',
        'UniqueProducts'
    ]

    return diversity

def build_purchase_delay_features(rfm_df, interval_df, epsilon=1e-6):
    """
    Measure whether the customer is purchasing later than expected.
    
    Features:
    - DelayRatio: Recency divided by average purchase interval
        - Values >1 indicate customer is inactive beyond their typical pattern
        - Values <1 indicate customer is active within their typical pattern
    
    Parameters:
    -----------
    rfm_df : DataFrame
        RFM features containing at minimum:
        - CustomerID: Customer identifier
        - Recency: Days since last purchase
    interval_df : DataFrame
        Purchase interval features containing at minimum:
        - CustomerID: Customer identifier
        - AvgPurchaseInterval: Mean days between purchases
    epsilon : float, default=1e-6
        Small constant to avoid division by zero
    
    Returns:
    --------
    delay_df : DataFrame
        Purchase delay features with columns:
        - CustomerID: Unique customer identifier
        - DelayRatio: Recency divided by average interval
    """

    if len(rfm_df) == 0:
        return pd.DataFrame(columns=[
            'CustomerID',
            'DelayRatio'
        ])

    df = rfm_df.merge(interval_df, on='CustomerID', how='left')
    
    df['DelayRatio'] = df['Recency'] / (df['AvgPurchaseInterval'] + epsilon)

    return df[['CustomerID', 'DelayRatio']]
    
def build_engagement_intensity(feature_df, window_size_days):
    """
    Capture how frequently and consistently a customer is active over time.
    
    Features:
    - EngagementDensity: Proportion of active days within observation window
        - High values indicate frequent engagement
        - Low values indicate sporadic activity
    
    Parameters:
    -----------
    feature_df : DataFrame
        Transaction data from observation window containing at minimum:
        - CustomerID
        - InvoiceDate
    window_size_days : int
        Length of observation window in days
    
    Returns:
    --------
    engagement : DataFrame
        Engagement intensity features with columns:
        - CustomerID
        - EngagementDensity
    """

    if len(feature_df) == 0:
        return pd.DataFrame(columns=['CustomerID', 'ActiveDays', 'EngagementDensity'])

    df = feature_df.copy()

    df['PurchaseDate'] = df['InvoiceDate'].dt.date

    engagement = df.groupby('CustomerID') \
        .agg(ActiveDays=('PurchaseDate', 'nunique')) \
        .reset_index()

    engagement['EngagementDensity'] = engagement['ActiveDays'] / max(window_size_days, 1)

    return engagement[['CustomerID','EngagementDensity']]
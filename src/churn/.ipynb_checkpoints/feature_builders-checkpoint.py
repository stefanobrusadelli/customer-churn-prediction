import pandas as pd
import numpy as np
from datetime import timedelta

def build_rfm(feature_df, feature_end):
    """
    Calculate Recency, Frequency, and Monetary features for each customer.
    
    Features:
    - Recency: Days since customer's last purchase (as of feature_end)
    - Frequency: Number of unique invoices/purchases made
    - Monetary: Total sum spent across all transactions
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
        return pd.DataFrame(columns=['CustomerID', 'Recency', 'Frequency', 'Monetary', 'LogMonetary', 'LogFrequency'])
    
    rfm = feature_df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (feature_end - x.max()).days,
        'Invoice': 'nunique',
        'TotalSum': 'sum'
    }).reset_index()
    
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

    # Log transforms help reduce skewness in transactional variables
    rfm['LogMonetary'] = np.log1p(np.maximum(0, rfm['Monetary']))
    rfm['LogFrequency'] = np.log1p(np.maximum(0, rfm['Frequency']))

    return rfm

def build_aov(feature_df):
    """
    Calculate average order value per customer.
    
    Features:
    - AvgOrderValue: Mean transaction amount across all purchases
    
    Parameters:
    -----------
    feature_df : DataFrame
        Transaction data from observation window containing at minimum:
        - CustomerID: Customer identifier
        - TotalSum: Transaction amount
    
    Returns:
    --------
    aov : DataFrame
        Average order value features with columns:
        - CustomerID: Unique customer identifier
        - AvgOrderValue: Mean spend per transaction
    """
    if len(feature_df) == 0:
        return pd.DataFrame(columns=['CustomerID', 'AvgOrderValue'])
    
    aov = feature_df.groupby('CustomerID')['TotalSum'].mean().reset_index()
    aov.columns = ['CustomerID', 'AvgOrderValue']
    
    return aov

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
    
    # Cap at 1.0 (100%) to avoid outliers
    rr['ReturnRate'] = (rr['NumReturns'] / rr['TotalPurchases']).fillna(0).clip(upper=1.0)
    
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
    - FirstMonthSpend: Total spend in first N days
    
    Parameters:
    -----------
    feature_df : DataFrame
        Transaction data from observation window containing at minimum:
        - CustomerID: Customer identifier
        - InvoiceDate: Date of transaction
        - Invoice: Invoice identifier
        - TotalSum: Transaction amount
    early_period_days : int, default=30
        Number of days to consider as the "early engagement" period
        (measured from first purchase date)
    
    Returns:
    --------
    early_features : DataFrame
        Early engagement metrics with columns:
        - CustomerID: Unique customer identifier
        - FirstMonthPurchases: Number of purchases in first N days
        - FirstMonthSpend: Total spend in first N days
    first_purchase : DataFrame
        First purchase dates with columns:
        - CustomerID: Unique customer identifier
        - FirstPurchaseDate: Date of customer's first transaction
    """

    if len(feature_df) == 0:
        return pd.DataFrame(columns=['CustomerID', 'FirstMonthPurchases', 'FirstMonthSpend'])
    
    first_purchase = get_first_purchase_dates(feature_df)
    
    df = feature_df.merge(first_purchase, on='CustomerID')
    df['DaysSinceFirst'] = (df['InvoiceDate'] - df['FirstPurchaseDate']).dt.days

    # Calculate early engagement metrics
    early = (
        df[df['DaysSinceFirst'] <= early_period_days]
        .groupby('CustomerID')
        .agg(
            FirstMonthPurchases=('Invoice', 'nunique'),
            FirstMonthSpend=('TotalSum', 'sum')
        )
        .reset_index()
    )

    # Ensure all customers are represented (even those with no early activity)
    early = first_purchase[['CustomerID']].merge(
        early, on='CustomerID', how='left'
    ).fillna(0)

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

def build_purchase_velocity(feature_df):
    """
    Calculate purchase timing patterns and regularity.
    
    Features:
    - AvgDaysBetweenPurchases: Average gap between purchases
    - PurchaseRegularity: Coefficient of variation (std/mean) of purchase gaps
        - Lower values indicate more regular purchasing patterns
        - Higher values indicate irregular or sporadic purchasing
    
    Notes:
    - PurchaseRegularity = 0 for customers with only one purchase
    - PurchaseRegularity is set to 0 when AvgDaysBetweenPurchases = 0
    
    Parameters:
    -----------
    feature_df : DataFrame
        Transaction data from observation window containing at minimum:
        - CustomerID: Customer identifier
        - InvoiceDate: Date of transaction (for calculating gaps)
    
    Returns:
    --------
    velocity : DataFrame
        Purchase velocity features with columns:
        - CustomerID: Unique customer identifier
        - AvgDaysBetweenPurchases: Mean number of days between purchases
        - PurchaseRegularity: Coefficient of variation of purchase gaps
    """
    if len(feature_df) == 0:
        return pd.DataFrame(columns=['CustomerID', 'AvgDaysBetweenPurchases', 'PurchaseRegularity'])
    
    df = feature_df.sort_values(['CustomerID', 'InvoiceDate']).copy()

    df['date_diff'] = df.groupby('CustomerID')['InvoiceDate'] \
        .diff() \
        .dt.days

    velocity = df.groupby('CustomerID')['date_diff'].agg(
        AvgDaysBetweenPurchases='mean',
        StdDaysBetweenPurchases='std'
    ).reset_index()

    velocity['PurchaseRegularity'] = np.where(
        (velocity['AvgDaysBetweenPurchases'] > 0) & (velocity['StdDaysBetweenPurchases'].notna()),
        velocity['StdDaysBetweenPurchases'] / velocity['AvgDaysBetweenPurchases'],
        0
    )

    velocity = velocity.drop(columns=['StdDaysBetweenPurchases'])

    return velocity

def build_trend_features(feature_df, feature_end, window_size_days, recent_days=30):
    """
    Compare recent behavior vs historical behavior to identify trends.
    
    Features:
    - RevenueTrend: Recent revenue rate / historical revenue rate
        - >1 indicates increasing revenue velocity
        - <1 indicates decreasing revenue velocity
        - Special values: 2 for new customers with only recent activity, 1 for no activity
    - FrequencyTrend: Recent frequency rate / historical frequency rate
        - >1 indicates increasing purchase frequency
        - <1 indicates decreasing purchase frequency
        - Special values: 2 for new customers with only recent activity, 1 for no activity
    
    Notes:
    - Recent period is defined as the last `recent_days` of the observation window
    - Rates are calculated per day to account for different time periods
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
    
    Returns:
    --------
    trend_df : DataFrame
        Trend features with columns:
        - CustomerID: Unique customer identifier
        - RevenueTrend: Ratio of recent to historical revenue rate (capped at 10)
        - FrequencyTrend: Ratio of recent to historical frequency rate (capped at 10)
    """
    if len(feature_df) == 0:
        return pd.DataFrame(columns=['CustomerID', 'RevenueTrend', 'FrequencyTrend'])

    recent_start = feature_end - timedelta(days=recent_days)

    purchase_df = feature_df[
        feature_df['TransactionType'] == 'Standard_Purchase'
    ].copy()
    
    purchase_df['is_recent'] = purchase_df['InvoiceDate'] >= recent_start

    # Revenue masked columns
    purchase_df['recent_revenue'] = purchase_df['TotalSum'].where(purchase_df['is_recent'], 0)
    purchase_df['hist_revenue'] = purchase_df['TotalSum'].where(~purchase_df['is_recent'], 0)

    grouped = purchase_df.groupby('CustomerID')

    revenue = grouped[['recent_revenue', 'hist_revenue']].sum()

    freq = grouped.agg(
        RecentFreq=('Invoice', lambda invoice: invoice[purchase_df.loc[invoice.index, 'is_recent']].nunique()),
        HistoricalFreq=('Invoice', lambda invoice: invoice[~purchase_df.loc[invoice.index, 'is_recent']].nunique())
    )

    trend_df = revenue.join(freq)

    historical_period_days = max(window_size_days - recent_days, 1)
    recent_period_days = max(recent_days, 1)
    
    trend_df['RecentRevenueRate'] = trend_df['recent_revenue'] / recent_period_days
    trend_df['HistoricalRevenueRate'] = trend_df['hist_revenue'] / historical_period_days

    trend_df['RecentFreqRate'] = trend_df['RecentFreq'] / recent_period_days
    trend_df['HistoricalFreqRate'] = trend_df['HistoricalFreq'] / historical_period_days

    trend_df['RevenueTrend'] = np.where(
        trend_df['HistoricalRevenueRate'] > 0,
        trend_df['RecentRevenueRate'] / trend_df['HistoricalRevenueRate'],
         # Customer had no historical revenue but has recent revenue, so indicate new or emerging activity (2)
         # No revenue both historically and recently, treat as neutral (1)
        np.where(trend_df['RecentRevenueRate'] > 0, 2, 1) 
    )
    
    trend_df['FrequencyTrend'] = np.where(
        trend_df['HistoricalFreqRate'] > 0,
        trend_df['RecentFreqRate'] / trend_df['HistoricalFreqRate'],
         # Customer had no historical revenue but has recent revenue, so indicate new or emerging activity (2)
         # No revenue both historically and recently, treat as neutral (1)
        np.where(trend_df['RecentFreqRate'] > 0, 2, 1)
    )

    # Limit extreme frequency ratios caused by very small historical values
    trend_df['RevenueTrend'] = trend_df['RevenueTrend'].clip(upper=10)
    trend_df['FrequencyTrend'] = trend_df['FrequencyTrend'].clip(upper=10)

    return trend_df[['RevenueTrend', 'FrequencyTrend']].reset_index()

def build_customer_lifetime(first_purchase_df, feature_end):
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
    if len(first_purchase_df) == 0:
        return pd.DataFrame(columns=['CustomerID', 'CustomerLifetime'])
    
    lifetime = first_purchase_df.copy()

    lifetime['CustomerLifetime'] = (
        feature_end - lifetime['FirstPurchaseDate']
    ).dt.days

    lifetime = lifetime[['CustomerID', 'CustomerLifetime']]

    return lifetime

def add_inactivity_ratio(features):
    """
    Calculate ratio of current inactivity to typical purchase gap.
    
    Features:
    - InactivityRatio: Recency / AvgDaysBetweenPurchases
        - Values >1 indicate customer is inactive beyond their typical pattern
        - Values <1 indicate customer is active within their typical pattern
        - Set to 0 when AvgDaysBetweenPurchases is 0 (single-purchase customers)
    
    Notes:
    - This feature helps identify customers who may be at risk of churning
    - Higher ratios suggest increasing inactivity relative to historical patterns
    
    Parameters:
    -----------
    features : DataFrame
        Customer features containing at minimum:
        - Recency: Days since last purchase
        - AvgDaysBetweenPurchases: Average gap between purchases
    
    Returns:
    --------
    features : DataFrame
        Original DataFrame with additional InactivityRatio column
    """
    features['InactivityRatio'] = np.where(
        features['AvgDaysBetweenPurchases'] > 0,
        features['Recency'] / features['AvgDaysBetweenPurchases'],
        0
    )

    return features


def build_frequency_trend(feature_df, feature_end, recent_days=30):
    """
    Calculate purchase frequency trend comparing recent vs previous period.

    Features:
    - FreqRecent: Number of purchases in the recent period
    - FreqPrev: Number of purchases in the previous period
    - FreqChange: Difference between recent and previous frequency
    - FreqRatio: Ratio between recent and previous frequency

    Parameters
    ----------
    feature_df : DataFrame
        Transaction data containing:
        - CustomerID
        - InvoiceDate
        - Invoice
    feature_end : datetime
        End of observation window
    recent_days : int
        Length of recent window in days

    Returns
    -------
    DataFrame
    """

    if len(feature_df) == 0:
        return pd.DataFrame(columns=[
            'CustomerID', 'FreqRecent', 'FreqPrev', 'FreqChange', 'FreqRatio'
        ])

    recent_start = feature_end - timedelta(days=recent_days)
    prev_start = feature_end - timedelta(days=2 * recent_days)

    recent = feature_df[feature_df['InvoiceDate'] >= recent_start]
    prev = feature_df[(feature_df['InvoiceDate'] >= prev_start) &
                      (feature_df['InvoiceDate'] < recent_start)]

    freq_recent = recent.groupby('CustomerID')['Invoice'].nunique()
    freq_prev = prev.groupby('CustomerID')['Invoice'].nunique()

    freq = pd.DataFrame({
        'FreqRecent': freq_recent,
        'FreqPrev': freq_prev
    }).fillna(0)

    freq['FreqChange'] = freq['FreqRecent'] - freq['FreqPrev']
    freq['FreqRatio'] = freq['FreqRecent'] / (freq['FreqPrev'] + 1)

    freq = freq.reset_index()

    return freq


def build_revenue_trend(feature_df, feature_end, recent_days=30):
    """
    Calculate revenue trend comparing recent vs previous period.
    
    Features:
    - RevRecent: Total revenue in the recent period
    - RevPrev: Total revenue in the previous period
    - RevChange: Absolute change in revenue (RevRecent - RevPrev)
    - RevRatio: Ratio of recent to previous revenue (capped by adding 1 to denominator)
    
    Parameters:
    -----------
    feature_df : DataFrame
        Transaction data from observation window containing at minimum:
        - CustomerID: Customer identifier
        - InvoiceDate: Date of transaction
        - TotalSum: Transaction amount
    feature_end : datetime
        End of observation window
    recent_days : int, default=30
        Number of days to consider as "recent" period
    
    Returns:
    --------
    trend : DataFrame
        Revenue trend features with columns:
        - CustomerID: Unique customer identifier
        - RevRecent: Total spend in recent period
        - RevPrev: Total spend in previous period
        - RevChange: Change in spend between periods
        - RevRatio: Ratio of recent to previous spend
    """

    if len(feature_df) == 0:
        return pd.DataFrame(columns=[
            'CustomerID', 'RevRecent', 'RevPrev', 'RevChange', 'RevRatio'
        ])

    recent_start = feature_end - timedelta(days=recent_days)
    prev_start = feature_end - timedelta(days=2 * recent_days)

    recent = feature_df[feature_df['InvoiceDate'] >= recent_start]
    prev = feature_df[(feature_df['InvoiceDate'] >= prev_start) &
                      (feature_df['InvoiceDate'] < recent_start)]

    rev_recent = recent.groupby('CustomerID')['TotalSum'].sum()
    rev_prev = prev.groupby('CustomerID')['TotalSum'].sum()

    rev = pd.DataFrame({
        'RevRecent': rev_recent,
        'RevPrev': rev_prev
    }).fillna(0)

    rev['RevChange'] = rev['RevRecent'] - rev['RevPrev']
    rev['RevRatio'] = rev['RevRecent'] / (rev['RevPrev'] + 1)

    rev = rev.reset_index()

    return rev

def build_purchase_intervals(feature_df):
    """
    Calculate statistics of time between purchases.
    
    Features:
    - AvgPurchaseInterval: Mean number of days between consecutive purchases
    - StdPurchaseInterval: Standard deviation of purchase intervals
    - PurchaseIntervalCV: Coefficient of variation (std/mean) of purchase intervals
        - Measures regularity of purchase timing
        - Lower values indicate more consistent purchasing patterns
    
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
        - PurchaseIntervalCV: Coefficient of variation of intervals
    """

    if len(feature_df) == 0:
        return pd.DataFrame(columns=[
            'CustomerID',
            'AvgPurchaseInterval',
            'StdPurchaseInterval',
            'PurchaseIntervalCV'
        ])

    df = feature_df.sort_values(['CustomerID', 'InvoiceDate'])

    df['PrevPurchase'] = df.groupby('CustomerID')['InvoiceDate'].shift(1)
    df['Interval'] = (df['InvoiceDate'] - df['PrevPurchase']).dt.days

    intervals = df.groupby('CustomerID')['Interval'].agg(['mean', 'std']).reset_index()

    intervals.columns = [
        'CustomerID',
        'AvgPurchaseInterval',
        'StdPurchaseInterval'
    ]

    intervals['PurchaseIntervalCV'] = (
        intervals['StdPurchaseInterval'] /
        (intervals['AvgPurchaseInterval'] + 1)
    )

    intervals = intervals.fillna(0)

    return intervals

def build_customer_age(feature_df, feature_end):
    """
    Calculate how long the customer has existed.
    
    Features:
    - CustomerAgeDays: Number of days since customer's first purchase
    
    Parameters:
    -----------
    feature_df : DataFrame
        Transaction data from observation window containing at minimum:
        - CustomerID: Customer identifier
        - InvoiceDate: Date of transaction (to find first purchase)
    feature_end : datetime
        End of observation window (used to calculate age)
    
    Returns:
    --------
    age_df : DataFrame
        Customer age features with columns:
        - CustomerID: Unique customer identifier
        - CustomerAgeDays: Days since first purchase
    """

    if len(feature_df) == 0:
        return pd.DataFrame(columns=['CustomerID', 'CustomerAgeDays'])

    first_purchase = feature_df.groupby('CustomerID')['InvoiceDate'].min()

    age = (feature_end - first_purchase).dt.days

    age_df = age.reset_index()
    age_df.columns = ['CustomerID', 'CustomerAgeDays']

    return age_df

def build_product_diversity(feature_df):
    """
    Measure how many unique products a customer buys.
    
    Features:
    - UniqueProducts: Number of distinct products (StockCode) purchased
    - UniqueInvoices: Number of distinct purchase transactions
    
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
        - UniqueInvoices: Count of distinct purchase transactions
    """

    if len(feature_df) == 0:
        return pd.DataFrame(columns=[
            'CustomerID', 'UniqueProducts', 'UniqueInvoices'
        ])

    diversity = feature_df.groupby('CustomerID').agg({
        'StockCode': 'nunique',
        'Invoice': 'nunique'
    }).reset_index()

    diversity.columns = [
        'CustomerID',
        'UniqueProducts',
        'UniqueInvoices'
    ]

    return diversity

def build_spend_volatility(feature_df):
    """
    Calculate variability of transaction values.
    
    Features:
    - SpendMean: Average transaction amount
    - SpendStd: Standard deviation of transaction amounts
    - SpendCV: Coefficient of variation (std/mean) of transaction amounts
        - Measures spending consistency/volatility
        - Higher values indicate more variable spending patterns
    
    Parameters:
    -----------
    feature_df : DataFrame
        Transaction data from observation window containing at minimum:
        - CustomerID: Customer identifier
        - TotalSum: Transaction amount
    
    Returns:
    --------
    spend : DataFrame
        Spend volatility features with columns:
        - CustomerID: Unique customer identifier
        - SpendMean: Average transaction value
        - SpendStd: Standard deviation of transaction values
        - SpendCV: Coefficient of variation of spend
    """

    if len(feature_df) == 0:
        return pd.DataFrame(columns=[
            'CustomerID', 'SpendMean', 'SpendStd', 'SpendCV'
        ])

    spend = feature_df.groupby('CustomerID')['TotalSum'].agg(['mean', 'std']).reset_index()

    spend.columns = ['CustomerID', 'SpendMean', 'SpendStd']

    spend['SpendCV'] = spend['SpendStd'] / (spend['SpendMean'] + 1)

    spend = spend.fillna(0)

    return spend

def build_purchase_delay_features(rfm_df, interval_df):
    """
    Measure whether the customer is purchasing later than expected.
    
    Features:
    - PurchaseDelay: Difference between recency and average purchase interval
        - Positive values indicate customer is overdue compared to their typical pattern
        - Negative values indicate customer is purchasing sooner than usual
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
    
    Returns:
    --------
    delay_df : DataFrame
        Purchase delay features with columns:
        - CustomerID: Unique customer identifier
        - PurchaseDelay: Recency minus average interval
        - DelayRatio: Recency divided by average interval
    """

    if len(rfm_df) == 0:
        return pd.DataFrame(columns=[
            "CustomerID",
            "PurchaseDelay",
            "DelayRatio"
        ])

    df = rfm_df.merge(interval_df, on="CustomerID", how="left")

    df["PurchaseDelay"] = df["Recency"] - df["AvgPurchaseInterval"]

    df["DelayRatio"] = df["Recency"] / (df["AvgPurchaseInterval"] + 1)

    return df[["CustomerID", "PurchaseDelay", "DelayRatio"]]

def build_activity_decay(rfm_df, freq_trend_df):
    """
    Detect decrease in purchasing activity.
    
    Features:
    - ActivityDecay: Ratio of recent purchase frequency to total frequency
        - Lower values indicate declining activity (recent purchases make up smaller portion)
        - Higher values indicate sustained or increasing activity
        - Values close to 1 indicate most purchases are recent
    
    Parameters:
    -----------
    rfm_df : DataFrame
        RFM features containing at minimum:
        - CustomerID: Customer identifier
        - Frequency: Total number of purchases
    freq_trend_df : DataFrame
        Frequency trend features containing at minimum:
        - CustomerID: Customer identifier
        - FreqRecent: Number of purchases in recent period
    
    Returns:
    --------
    decay_df : DataFrame
        Activity decay features with columns:
        - CustomerID: Unique customer identifier
        - ActivityDecay: Recent frequency divided by total frequency
    """

    if len(rfm_df) == 0:
        return pd.DataFrame(columns=[
            "CustomerID",
            "ActivityDecay"
        ])

    df = rfm_df.merge(freq_trend_df, on="CustomerID", how="left")

    df["ActivityDecay"] = df["FreqRecent"] / (df["Frequency"] + 1)

    return df[["CustomerID", "ActivityDecay"]]

def build_revenue_decay(rfm_df, revenue_trend_df):
    """
    Detect decrease in spending behaviour.
    
    Features:
    - RevenueDecay: Ratio of recent revenue to total monetary value
        - Lower values indicate declining spending (recent revenue makes up smaller portion)
        - Higher values indicate sustained or increasing spending
        - Values close to 1 indicate most spending is recent
    
    Parameters:
    -----------
    rfm_df : DataFrame
        RFM features containing at minimum:
        - CustomerID: Customer identifier
        - Monetary: Total spend across all transactions
    revenue_trend_df : DataFrame
        Revenue trend features containing at minimum:
        - CustomerID: Customer identifier
        - RevRecent: Total revenue in recent period
    
    Returns:
    --------
    decay_df : DataFrame
        Revenue decay features with columns:
        - CustomerID: Unique customer identifier
        - RevenueDecay: Recent revenue divided by total monetary value
    """

    if len(rfm_df) == 0:
        return pd.DataFrame(columns=[
            "CustomerID",
            "RevenueDecay"
        ])

    df = rfm_df.merge(revenue_trend_df, on="CustomerID", how="left")

    df["RevenueDecay"] = df["RevRecent"] / (df["Monetary"] + 1)

    return df[["CustomerID", "RevenueDecay"]]
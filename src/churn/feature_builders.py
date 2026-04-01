"""
Feature Builders Module

This module constructs customer-level behavioral features from transactional data.
Features are designed to capture value, engagement, timing, and behavioral patterns.

Feature groups
--------------
- RFM features       → Recency (days since last purchase), Frequency (unique invoices),
                        Monetary (total spend), LogMonetary
- Order value        → AvgOrderValue, OrderValueCV (spending variability)
- Returns behavior   → ReturnRate (returns / purchases)
- Engagement intensity → EngagementDensity (active days / window length)
- Early lifecycle    → FirstMonthPurchases, MonetaryFirstMonth,
                        AvgOrderValueFirstMonth (behavior in first N days)
- Seasonality        → Q4Ratio (Oct–Dec concentration), FavoriteMonthSin,
                        FavoriteMonthCos (cyclic month encoding)
- Trend signals      → RecentShareLog (log-scaled share of recent activity),
                        RevenueTrend (recent vs historical revenue rate, unbounded)
- Customer lifetime  → CustomerLifetime (days since first purchase)
- Purchase timing    → AvgPurchaseInterval, PurchaseIntervalCV
- Purchase delay     → DelayRatio (recency / avg interval)
- Product behavior   → UniqueProducts (distinct StockCodes purchased),
                        ProductDiversityRate (UniqueProducts / Frequency)
- Derived features   → ValueEngagement (LogMonetary x EngagementDensity)
"""
import pandas as pd
import numpy as np
from datetime import timedelta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_df(columns: list) -> pd.DataFrame:
    """Return an empty DataFrame with the given columns."""
    return pd.DataFrame(columns=columns)


def _get_first_purchase_dates(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the first transaction date for each customer.

    Parameters
    ----------
    feature_df : DataFrame
        Transaction data containing at minimum:
        - CustomerID  : customer identifier
        - InvoiceDate : date of transaction

    Returns
    -------
    DataFrame
        Columns: CustomerID, FirstPurchaseDate
    """
    if len(feature_df) == 0:
        return _empty_df(['CustomerID', 'FirstPurchaseDate'])

    return (
        feature_df
        .groupby('CustomerID')['InvoiceDate']
        .min()
        .reset_index()
        .rename(columns={'InvoiceDate': 'FirstPurchaseDate'})
    )


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------

def build_rfm(
    feature_df: pd.DataFrame,
    feature_end,
    window_size_days: int,
) -> pd.DataFrame:
    """
    Calculate Recency, Frequency, and Monetary features for each customer.

    Features
    --------
    - Recency     : Days since customer's last purchase (as of feature_end).
                    Customers with no purchases receive window_size_days.
    - Frequency   : Number of unique Standard_Purchase invoices.
    - Monetary    : Total spend across all transaction types.
    - LogMonetary : log1p(Monetary) to reduce right skew.

    Parameters
    ----------
    feature_df : DataFrame
        Transaction data from the observation window containing at minimum:
        - CustomerID      : customer identifier
        - InvoiceDate     : date of transaction
        - Invoice         : invoice identifier
        - TotalSum        : transaction amount
        - TransactionType : must include 'Standard_Purchase' records
    feature_end : datetime
        End of observation window (used to calculate recency).
    window_size_days : int
        Length of observation window in days (used as recency fallback).

    Returns
    -------
    DataFrame
        Columns: CustomerID, Recency, Frequency, Monetary, LogMonetary
    """
    _COLS = ['CustomerID', 'Recency', 'Frequency', 'Monetary', 'LogMonetary']

    if len(feature_df) == 0:
        return _empty_df(_COLS)

    purchase_df = feature_df[
        feature_df['TransactionType'] == 'Standard_Purchase'
    ]

    freq_rec = (
        purchase_df
        .groupby('CustomerID')
        .agg(
            Recency=('InvoiceDate', lambda x: (feature_end - x.max()).days),
            Frequency=('Invoice', 'nunique'),
        )
    )

    monetary = (
        feature_df
        .groupby('CustomerID')['TotalSum']
        .sum()
        .rename('Monetary')
    )

    rfm = freq_rec.join(monetary, how='outer').reset_index()
    rfm['Frequency'] = rfm['Frequency'].fillna(0)
    rfm['Recency'] = rfm['Recency'].fillna(window_size_days)
    rfm['Monetary'] = rfm['Monetary'].fillna(0)

    rfm['LogMonetary'] = np.log1p(np.maximum(0, rfm['Monetary']))

    return rfm[_COLS]


def build_aov(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average order value and variability of transaction values.

    Features
    --------
    - AvgOrderValue : Mean transaction amount per customer.
    - OrderValueCV  : Coefficient of variation (std / mean) of transaction
                      amounts. Measures spending consistency; higher values
                      indicate more variable spending. Set to 0 when
                      undefined (mean <= 0 or insufficient data).

    Parameters
    ----------
    feature_df : DataFrame
        Transaction data from the observation window containing at minimum:
        - CustomerID : customer identifier
        - TotalSum   : transaction amount

    Returns
    -------
    DataFrame
        Columns: CustomerID, AvgOrderValue, OrderValueCV
    """
    _COLS = ['CustomerID', 'AvgOrderValue', 'OrderValueCV']

    if len(feature_df) == 0:
        return _empty_df(_COLS)

    agg = (
        feature_df
        .groupby('CustomerID')['TotalSum']
        .agg(AvgOrderValue='mean', OrderValueStd='std')
        .reset_index()
    )

    valid = (agg['AvgOrderValue'] > 0) & agg['OrderValueStd'].notna()
    agg['OrderValueCV'] = np.where(
        valid,
        agg['OrderValueStd'] / agg['AvgOrderValue'],
        0.0,
    )

    return agg.drop(columns='OrderValueStd').fillna(0)[_COLS]


def build_return_rate(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate return rate with proper handling of edge cases.

    Features
    --------
    - ReturnRate : Proportion of purchases that resulted in returns,
                  calculated as NumReturns / TotalPurchases.
                  Customers with no purchases receive 0.

    Notes
    -----
    Returns are identified by TransactionType 'Linked_Return' or
    'Unlinked_Return'. Purchases are identified by 'Standard_Purchase'.

    Parameters
    ----------
    feature_df : DataFrame
        Transaction data from the observation window containing at minimum:
        - CustomerID      : customer identifier
        - TransactionType : 'Standard_Purchase', 'Linked_Return', or
                            'Unlinked_Return'
        - Invoice         : invoice identifier (for counting unique purchases)

    Returns
    -------
    DataFrame
        Columns: CustomerID, ReturnRate
    """
    if len(feature_df) == 0:
        return _empty_df(['CustomerID', 'ReturnRate'])

    return_counts = (
        feature_df[feature_df['TransactionType'].isin(['Linked_Return', 'Unlinked_Return'])]
        .groupby('CustomerID')
        .size()
        .rename('NumReturns')
        .reset_index()
    )

    purchases = (
        feature_df[feature_df['TransactionType'] == 'Standard_Purchase']
        .groupby('CustomerID')['Invoice']
        .nunique()
        .rename('TotalPurchases')
        .reset_index()
    )

    rr = purchases.merge(return_counts, on='CustomerID', how='outer').fillna(0)
    rr['ReturnRate'] = np.where(
        rr['TotalPurchases'] > 0,
        rr['NumReturns'] / rr['TotalPurchases'],
        0.0,
    )

    return rr[['CustomerID', 'ReturnRate']]


def build_seasonality(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract seasonal purchasing patterns.

    Features
    --------
    - Q4Ratio          : Proportion of purchases occurring in Q4 (Oct-Dec),
                         ranging from 0.0 to 1.0.
    - FavoriteMonthSin : sin(2pi x FavoriteMonth / 12). Cyclic encoding of
                         the customer's most frequent purchase month so that
                         the model sees December and January as adjacent
                         rather than maximally distant.
    - FavoriteMonthCos : cos(2pi x FavoriteMonth / 12). Paired with
                         FavoriteMonthSin to fully represent cyclic position.

    Notes
    -----
    Raw month integers (1-12) are dropped in favour of the sin/cos pair.
    Using a raw integer would incorrectly imply that month 12 is "greater
    than" month 1, and that the distance between month 11 and month 1 is
    larger than the distance between month 11 and month 12.

    Parameters
    ----------
    feature_df : DataFrame
        Transaction data from the observation window containing at minimum:
        - CustomerID  : customer identifier
        - InvoiceDate : date of transaction (used to extract month)

    Returns
    -------
    DataFrame
        Columns: CustomerID, Q4Ratio, FavoriteMonthSin, FavoriteMonthCos
    """
    _COLS = ['CustomerID', 'Q4Ratio', 'FavoriteMonthSin', 'FavoriteMonthCos']

    if len(feature_df) == 0:
        return _empty_df(_COLS)

    df = feature_df.copy()
    df['Month'] = df['InvoiceDate'].dt.month
    df['IsQ4'] = df['Month'].isin([10, 11, 12])

    season = (
        df.groupby('CustomerID')
        .agg(
            Q4Ratio=('IsQ4', 'mean'),
            FavoriteMonth=('Month', lambda x: x.mode()[0] if len(x.mode()) > 0 else 6),
        )
        .reset_index()
    )

    season['FavoriteMonthSin'] = np.sin(2 * np.pi * season['FavoriteMonth'] / 12)
    season['FavoriteMonthCos'] = np.cos(2 * np.pi * season['FavoriteMonth'] / 12)

    return season[_COLS]


def build_early_engagement(
    feature_df: pd.DataFrame,
    early_period_days: int = 30,
) -> pd.DataFrame:
    """
    Capture customer behavior in their first N days after their first purchase.

    Features
    --------
    - FirstMonthPurchases     : Number of unique Standard_Purchase invoices
                                in the first N days.
    - MonetaryFirstMonth      : Total spend (all transaction types) in the
                                first N days.
    - AvgOrderValueFirstMonth : MonetaryFirstMonth / FirstMonthPurchases.
                                Set to 0 when there are no purchases.

    Parameters
    ----------
    feature_df : DataFrame
        Transaction data from the observation window containing at minimum:
        - CustomerID      : customer identifier
        - InvoiceDate     : date of transaction
        - Invoice         : invoice identifier
        - TotalSum        : transaction amount
        - TransactionType : must include 'Standard_Purchase' records
    early_period_days : int, default 30
        Number of days from first purchase to consider as the early period.

    Returns
    -------
    DataFrame
        Columns: CustomerID, FirstMonthPurchases, MonetaryFirstMonth,
                 AvgOrderValueFirstMonth
    """
    _COLS = [
        'CustomerID', 'FirstMonthPurchases',
        'MonetaryFirstMonth', 'AvgOrderValueFirstMonth',
    ]

    if len(feature_df) == 0:
        return _empty_df(_COLS)

    first_purchase = _get_first_purchase_dates(feature_df)
    df = feature_df.merge(first_purchase, on='CustomerID')
    df['DaysSinceFirst'] = (df['InvoiceDate'] - df['FirstPurchaseDate']).dt.days

    early_window = df[df['DaysSinceFirst'] <= early_period_days]

    # Count only Standard_Purchase invoices (returns previously inflated count)
    purchase_counts = (
        early_window[early_window['TransactionType'] == 'Standard_Purchase']
        .groupby('CustomerID')['Invoice']
        .nunique()
        .rename('FirstMonthPurchases')
        .reset_index()
    )

    # Monetary includes all transaction types (returns reduce spend correctly)
    monetary = (
        early_window
        .groupby('CustomerID')['TotalSum']
        .sum()
        .rename('MonetaryFirstMonth')
        .reset_index()
    )

    early = (
        first_purchase[['CustomerID']]
        .merge(purchase_counts, on='CustomerID', how='left')
        .merge(monetary, on='CustomerID', how='left')
        .fillna(0)
    )

    early['AvgOrderValueFirstMonth'] = np.where(
        early['FirstMonthPurchases'] > 0,
        early['MonetaryFirstMonth'] / early['FirstMonthPurchases'],
        0.0,
    )

    return early[_COLS]


def build_trend_features(
    feature_df: pd.DataFrame,
    feature_end,
    window_size_days: int,
    recent_days: int = 30,
    epsilon: float = 1e-6,
) -> pd.DataFrame:
    """
    Compare recent behaviour vs historical behaviour to identify trends.

    Features
    --------
    - RecentShareLog : log1p(FreqRecent / (FreqRecent + FreqHistorical + e)).
                       Values closer to log1p(1) indicate activity concentrated
                       in the recent period; values closer to 0 indicate
                       primarily historical activity.
    - RevenueTrend   : RecentRevenueRate / HistoricalRevenueRate (unbounded).
                       Values > 1 indicate accelerating spend;
                       values < 1 indicate decelerating spend.
                       Customers with no historical revenue but recent revenue
                       receive a high value (emerging customers).
                       Customers with no activity in either period receive 1
                       (neutral).

    Notes
    -----
    - Recent period     : last `recent_days` of the observation window.
    - Historical period : the remaining portion of the window.
    - Frequency counts  : Standard_Purchase only.
    - Revenue rates     : all transaction types (returns reduce revenue).
    - Rates are per-day normalised to make recent and historical periods
      comparable regardless of their different lengths.

    Parameters
    ----------
    feature_df : DataFrame
        Transaction data from the observation window containing at minimum:
        - CustomerID      : customer identifier
        - InvoiceDate     : date of transaction
        - Invoice         : invoice identifier
        - TotalSum        : transaction amount
        - TransactionType : must include 'Standard_Purchase' records
    feature_end : datetime
        End of observation window.
    window_size_days : int
        Total observation window length in days.
    recent_days : int, default 30
        Number of days to consider as the "recent" period.
    epsilon : float, default 1e-6
        Small constant to avoid division by zero.

    Returns
    -------
    DataFrame
        Columns: CustomerID, RecentShareLog, RevenueTrend
    """
    _COLS = ['CustomerID', 'RecentShareLog', 'RevenueTrend']

    if len(feature_df) == 0:
        return _empty_df(_COLS)

    recent_start = feature_end - timedelta(days=recent_days)
    recent_period_days = max(recent_days, 1)
    historical_period_days = max(window_size_days - recent_days, 1)

    # Frequency (Standard_Purchase only)
    purchase_df = feature_df[
        feature_df['TransactionType'] == 'Standard_Purchase'
    ].copy()

    recent_purchases     = purchase_df[purchase_df['InvoiceDate'] >= recent_start]
    historical_purchases = purchase_df[purchase_df['InvoiceDate'] <  recent_start]

    freq_recent = (
        recent_purchases.groupby('CustomerID')['Invoice']
        .nunique().rename('FreqRecent')
    )
    freq_historical = (
        historical_purchases.groupby('CustomerID')['Invoice']
        .nunique().rename('FreqHistorical')
    )

    # Revenue (all transaction types)
    recent_revenue = (
        feature_df[feature_df['InvoiceDate'] >= recent_start]
        .groupby('CustomerID')['TotalSum'].sum().rename('RecentRevenue')
    )
    historical_revenue = (
        feature_df[feature_df['InvoiceDate'] < recent_start]
        .groupby('CustomerID')['TotalSum'].sum().rename('HistoricalRevenue')
    )

    # Merge
    trend_df = (
        pd.DataFrame({
            'FreqRecent':        freq_recent,
            'FreqHistorical':    freq_historical,
            'RecentRevenue':     recent_revenue,
            'HistoricalRevenue': historical_revenue,
        })
        .fillna(0)
        .reset_index()
    )

    # RecentShareLog
    trend_df['RecentShareLog'] = np.log1p(
        trend_df['FreqRecent']
        / (trend_df['FreqRecent'] + trend_df['FreqHistorical'] + epsilon)
    )

    # RevenueTrend
    recent_rate     = trend_df['RecentRevenue']     / recent_period_days
    historical_rate = trend_df['HistoricalRevenue'] / historical_period_days

    trend_df['RevenueTrend'] = (
        (recent_rate + epsilon) / (historical_rate + epsilon)
    )

    return trend_df[_COLS]


def build_customer_lifetime(
    feature_df: pd.DataFrame,
    feature_end,
) -> pd.DataFrame:
    """
    Calculate customer tenure since their first purchase.

    Features
    --------
    - CustomerLifetime : Days between the customer's first purchase and
                         feature_end.

    Parameters
    ----------
    feature_df : DataFrame
        Transaction data from the observation window containing at minimum:
        - CustomerID  : customer identifier
        - InvoiceDate : date of transaction
    feature_end : datetime
        End of observation window (used to calculate lifetime).

    Returns
    -------
    DataFrame
        Columns: CustomerID, CustomerLifetime
    """
    if len(feature_df) == 0:
        return _empty_df(['CustomerID', 'CustomerLifetime'])

    df = _get_first_purchase_dates(feature_df)
    df['CustomerLifetime'] = (feature_end - df['FirstPurchaseDate']).dt.days

    return df[['CustomerID', 'CustomerLifetime']]


def build_purchase_intervals(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate statistics of time between consecutive purchases.

    Features
    --------
    - AvgPurchaseInterval : Mean number of days between consecutive
                            unique-day purchases.
    - PurchaseIntervalCV  : Coefficient of variation (std / mean) of
                            purchase intervals. Lower values indicate more
                            consistent timing. Set to 0 when undefined.

    Notes
    -----
    Intervals are computed on deduplicated daily purchase dates to avoid
    spurious zero-day intervals when multiple invoices share the same date.

    Parameters
    ----------
    feature_df : DataFrame
        Transaction data from the observation window containing at minimum:
        - CustomerID      : customer identifier
        - InvoiceDate     : date of transaction
        - TransactionType : must include 'Standard_Purchase' records

    Returns
    -------
    DataFrame
        Columns: CustomerID, AvgPurchaseInterval, PurchaseIntervalCV
    """
    _COLS = ['CustomerID', 'AvgPurchaseInterval', 'PurchaseIntervalCV']

    if len(feature_df) == 0:
        return _empty_df(_COLS)

    # Deduplicate to one row per (customer, day) so that multiple invoices
    # on the same date don't produce spurious zero-day intervals.
    purchase_df = (
        feature_df[feature_df['TransactionType'] == 'Standard_Purchase']
        .drop_duplicates(subset=['CustomerID', 'InvoiceDate'])
        .sort_values(['CustomerID', 'InvoiceDate'])
        .copy()
    )

    purchase_df['PrevPurchase'] = purchase_df.groupby('CustomerID')['InvoiceDate'].shift(1)
    purchase_df['Interval'] = (purchase_df['InvoiceDate'] - purchase_df['PrevPurchase']).dt.days

    intervals = (
        purchase_df
        .groupby('CustomerID')['Interval']
        .agg(AvgPurchaseInterval='mean', StdPurchaseInterval='std')
        .reset_index()
    )

    valid = (intervals['AvgPurchaseInterval'] > 0) & intervals['StdPurchaseInterval'].notna()
    intervals['PurchaseIntervalCV'] = np.where(
        valid,
        intervals['StdPurchaseInterval'] / intervals['AvgPurchaseInterval'],
        0.0,
    )

    return intervals[_COLS]


def build_product_diversity(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Measure how many unique products a customer buys and how diverse their
    purchasing is relative to their order frequency.

    Features
    --------
    - UniqueProducts      : Number of distinct StockCode values purchased.
    - ProductDiversityRate : UniqueProducts / Frequency. Measures how many
                             distinct products the customer buys per order.
                             High values indicate exploratory behaviour;
                             low values indicate habitual repeat purchasing.
                             Set to 0 for customers with no purchases.

    Parameters
    ----------
    feature_df : DataFrame
        Transaction data from the observation window containing at minimum:
        - CustomerID      : customer identifier
        - StockCode       : product identifier
        - Invoice         : invoice identifier (for frequency denominator)
        - TransactionType : must include 'Standard_Purchase' records

    Returns
    -------
    DataFrame
        Columns: CustomerID, UniqueProducts, ProductDiversityRate
    """
    _COLS = ['CustomerID', 'UniqueProducts', 'ProductDiversityRate']

    if len(feature_df) == 0:
        return _empty_df(_COLS)

    unique_products = (
        feature_df
        .groupby('CustomerID')['StockCode']
        .nunique()
        .rename('UniqueProducts')
        .reset_index()
    )

    frequency = (
        feature_df[feature_df['TransactionType'] == 'Standard_Purchase']
        .groupby('CustomerID')['Invoice']
        .nunique()
        .rename('Frequency')
        .reset_index()
    )

    diversity = unique_products.merge(frequency, on='CustomerID', how='left').fillna(0)
    diversity['ProductDiversityRate'] = np.where(
        diversity['Frequency'] > 0,
        diversity['UniqueProducts'] / diversity['Frequency'],
        0.0,
    )

    return diversity[_COLS]


def build_purchase_delay_features(
    rfm_df: pd.DataFrame,
    interval_df: pd.DataFrame,
    epsilon: float = 1e-6,
) -> pd.DataFrame:
    """
    Measure whether the customer is purchasing later than expected.

    Features
    --------
    - DelayRatio : Recency / (AvgPurchaseInterval + e).
                   Values > 1 indicate the customer has been inactive beyond
                   their typical inter-purchase gap.
                   Values < 1 indicate the customer is still within their
                   normal cadence.

    Parameters
    ----------
    rfm_df : DataFrame
        RFM features containing at minimum:
        - CustomerID : customer identifier
        - Recency    : days since last purchase
    interval_df : DataFrame
        Purchase interval features containing at minimum:
        - CustomerID          : customer identifier
        - AvgPurchaseInterval : mean days between purchases
    epsilon : float, default 1e-6
        Small constant added to the denominator to avoid division by zero.

    Returns
    -------
    DataFrame
        Columns: CustomerID, DelayRatio
    """
    if len(rfm_df) == 0:
        return _empty_df(['CustomerID', 'DelayRatio'])

    df = rfm_df.merge(interval_df, on='CustomerID', how='left')
    df['DelayRatio'] = (
        df['Recency'] / (df['AvgPurchaseInterval'] + epsilon)
    )

    return df[['CustomerID', 'DelayRatio']]


def build_engagement_intensity(
    feature_df: pd.DataFrame,
    window_size_days: int,
) -> pd.DataFrame:
    """
    Capture how frequently a customer is active over the observation window.

    Features
    --------
    - EngagementDensity : Proportion of calendar days in the observation
                          window on which the customer made at least one
                          transaction. High values indicate frequent
                          engagement; low values indicate sporadic activity.

    Parameters
    ----------
    feature_df : DataFrame
        Transaction data from the observation window containing at minimum:
        - CustomerID  : customer identifier
        - InvoiceDate : date of transaction
    window_size_days : int
        Length of observation window in days.

    Returns
    -------
    DataFrame
        Columns: CustomerID, EngagementDensity
    """
    if len(feature_df) == 0:
        return _empty_df(['CustomerID', 'EngagementDensity'])

    df = feature_df.copy()
    df['PurchaseDate'] = df['InvoiceDate'].dt.date

    engagement = (
        df.groupby('CustomerID')
        .agg(ActiveDays=('PurchaseDate', 'nunique'))
        .reset_index()
    )
    engagement['EngagementDensity'] = engagement['ActiveDays'] / max(window_size_days, 1)

    return engagement[['CustomerID', 'EngagementDensity']]


def build_derived_features(features: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived features from already-merged base features.

    This function must be called after all base feature sets have been
    merged, since it depends on columns produced by multiple builders.

    Features
    --------
    - ValueEngagement : LogMonetary * EngagementDensity.
                        Captures the interaction between spending volume and
                        activity density. High values indicate customers who
                        are both high-value and highly engaged; low values
                        indicate either low spend or sporadic activity.

    Parameters
    ----------
    features : DataFrame
        Merged feature set containing at minimum:
        - LogMonetary       : log-transformed total spend (from build_rfm)
        - EngagementDensity : proportion of active days in window
                              (from build_engagement_intensity)

    Returns
    -------
    DataFrame
        Input DataFrame with ValueEngagement column added in place.
    """
    features['ValueEngagement'] = features['LogMonetary'] * features['EngagementDensity']
    return features
"""
Feature engineering module.
Creates lag features, rolling statistics, and calendar features.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def create_lag_features(df, target_col='Weekly_Sales', lags=[1, 2, 4, 12], 
                       groupby_cols=['Store', 'Dept']):
    """
    Create lag features for time series forecasting.
    
    Args:
        df: DataFrame with time series data
        target_col: Column name to create lags for
        lags: List of lag periods to create
        groupby_cols: Columns to group by when creating lags
    
    Returns:
        DataFrame with lag features added
    """
    df = df.copy()
    
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df.groupby(groupby_cols)[target_col].shift(lag)
    
    return df


def create_rolling_features(df, target_col='Weekly_Sales', windows=[4, 8, 12],
                           groupby_cols=['Store', 'Dept']):
    """
    Create rolling statistics features.
    
    Args:
        df: DataFrame with time series data
        target_col: Column name to create rolling stats for
        windows: List of window sizes for rolling statistics
        groupby_cols: Columns to group by when creating rolling stats
    
    Returns:
        DataFrame with rolling features added
    """
    df = df.copy()
    
    for window in windows:
        # Rolling mean
        df[f'{target_col}_rolling_mean_{window}'] = (
            df.groupby(groupby_cols)[target_col]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        
        # Rolling std
        df[f'{target_col}_rolling_std_{window}'] = (
            df.groupby(groupby_cols)[target_col]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .std()
            .reset_index(0, drop=True)
        )
        
        # Rolling max
        df[f'{target_col}_rolling_max_{window}'] = (
            df.groupby(groupby_cols)[target_col]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .max()
            .reset_index(0, drop=True)
        )
        
        # Rolling min
        df[f'{target_col}_rolling_min_{window}'] = (
            df.groupby(groupby_cols)[target_col]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .min()
            .reset_index(0, drop=True)
        )
    
    return df


def create_calendar_features(df, date_col='Date'):
    """
    Create calendar-based features from date column.
    
    Args:
        df: DataFrame with date column
        date_col: Name of the date column
    
    Returns:
        DataFrame with calendar features added
    """
    df = df.copy()
    
    df['week'] = df[date_col].dt.isocalendar().week
    df['month'] = df[date_col].dt.month
    df['quarter'] = df[date_col].dt.quarter
    df['year'] = df[date_col].dt.year
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['day_of_month'] = df[date_col].dt.day
    df['week_of_year'] = df[date_col].dt.isocalendar().week
    
    # Cyclical encoding for periodic features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week'] / 52)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    return df


def create_external_features(df):
    """
    Process external features (Temperature, Fuel_Price, CPI, Unemployment).
    
    Args:
        df: DataFrame with external features
    
    Returns:
        DataFrame with processed external features
    """
    df = df.copy()
    
    # Fill missing values if any
    external_cols = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    for col in external_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    # Create lagged external features
    for col in external_cols:
        if col in df.columns:
            df[f'{col}_lag_1'] = df.groupby('Store')[col].shift(1)
            df[f'{col}_lag_4'] = df.groupby('Store')[col].shift(4)
    
    return df


def create_all_features(df, target_col='Weekly_Sales', 
                       lags=[1, 2, 4, 12],
                       rolling_windows=[4, 8, 12],
                       groupby_cols=['Store', 'Dept']):
    """
    Create all features for the forecasting model.
    
    Args:
        df: DataFrame with sales data
        target_col: Target column name
        lags: List of lag periods
        rolling_windows: List of rolling window sizes
        groupby_cols: Columns to group by
    
    Returns:
        DataFrame with all features added
    """
    print("Creating features...")
    
    # Calendar features
    df = create_calendar_features(df)
    print("  ✓ Calendar features created")
    
    # Lag features
    df = create_lag_features(df, target_col, lags, groupby_cols)
    print("  ✓ Lag features created")
    
    # Rolling features
    df = create_rolling_features(df, target_col, rolling_windows, groupby_cols)
    print("  ✓ Rolling features created")
    
    # External features
    df = create_external_features(df)
    print("  ✓ External features created")
    
    # Store and Dept as categorical (if not already)
    if 'Store' in df.columns:
        df['Store'] = df['Store'].astype('category')
    if 'Dept' in df.columns:
        df['Dept'] = df['Dept'].astype('category')
    
    # Type as categorical if exists
    if 'Type' in df.columns:
        df['Type'] = df['Type'].astype('category')
    
    print("Feature engineering complete!")
    
    return df


def get_feature_columns(df, exclude_cols=['Date', 'Weekly_Sales', 'Store', 'Dept']):
    """
    Get list of feature columns to use for modeling.
    
    Args:
        df: DataFrame
        exclude_cols: Columns to exclude from features
    
    Returns:
        List of feature column names
    """
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols


# ---------- FreshRetailNet-50K specific ----------

def create_external_features_freshretail(df, groupby_col='store_id'):
    """
    Process external features for FreshRetailNet: discount, holiday, activity,
    precipitation, temperature, humidity, wind, stock_hour6_22_cnt.
    """
    df = df.copy()
    external_cols = [
        'discount', 'holiday_flag', 'activity_flag', 'precpt',
        'avg_temperature', 'avg_humidity', 'avg_wind_level', 'stock_hour6_22_cnt'
    ]
    for col in external_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    for col in external_cols:
        if col in df.columns and groupby_col in df.columns:
            df[f'{col}_lag_1'] = df.groupby(groupby_col)[col].shift(1)
            df[f'{col}_lag_4'] = df.groupby(groupby_col)[col].shift(4)
    return df


def create_all_features_freshretail(
    df,
    target_col='sale_amount',
    date_col='dt',
    lags=(1, 2, 4, 7),
    rolling_windows=(4, 7, 14),
    groupby_cols=('store_id', 'product_id'),
):
    """
    Create all features for FreshRetailNet-50K forecasting.
    Uses sale_amount, dt, store_id, product_id and dataset's external covariates.
    """
    print("Creating features (FreshRetail)...")
    df = df.copy()
    groupby_cols = list(groupby_cols)

    df = create_calendar_features(df, date_col=date_col)
    print("  ✓ Calendar features created")

    df = create_lag_features(df, target_col=target_col, lags=list(lags), groupby_cols=groupby_cols)
    print("  ✓ Lag features created")

    df = create_rolling_features(
        df, target_col=target_col, windows=list(rolling_windows), groupby_cols=groupby_cols
    )
    print("  ✓ Rolling features created")

    df = create_external_features_freshretail(df, groupby_col='store_id')
    print("  ✓ External features created")

    for col in ['store_id', 'product_id', 'city_id', 'management_group_id',
                'first_category_id', 'second_category_id', 'third_category_id']:
        if col in df.columns:
            df[col] = df[col].astype('category')

    print("Feature engineering complete!")
    return df


def get_feature_columns_freshretail(
    df,
    exclude_cols=('dt', 'sale_amount', 'store_id', 'product_id', '_source'),
):
    """Feature columns for FreshRetail; excludes identifiers and target."""
    exclude_cols = set(exclude_cols)
    return [c for c in df.columns if c not in exclude_cols]

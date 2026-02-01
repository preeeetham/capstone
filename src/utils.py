"""
Utility functions for data splitting and preprocessing.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def time_based_split(df, date_col='Date', train_ratio=0.8):
    """
    Split data based on time (not random).
    
    Args:
        df: DataFrame to split
        date_col: Name of the date column
        train_ratio: Ratio of data to use for training
    
    Returns:
        Tuple of (train_df, test_df)
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"Train period: {train_df[date_col].min()} to {train_df[date_col].max()}")
    print(f"Test period: {test_df[date_col].min()} to {test_df[date_col].max()}")
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    return train_df, test_df


def prepare_ml_data(df, target_col='Weekly_Sales', feature_cols=None, 
                    exclude_cols=['Date', 'Weekly_Sales']):
    """
    Prepare data for machine learning models.
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        feature_cols: List of feature columns to use (if None, auto-detect)
        exclude_cols: Columns to exclude from features
    
    Returns:
        Tuple of (X, y) where X is features and y is target
    """
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Handle categorical columns
    df_processed = df.copy()
    for col in df_processed.columns:
        if df_processed[col].dtype == 'category':
            df_processed[col] = df_processed[col].cat.codes
    
    # Select numeric columns only
    numeric_cols = []
    for col in feature_cols:
        if col in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[col]):
            numeric_cols.append(col)
    
    X = df_processed[numeric_cols].fillna(0)
    y = df_processed[target_col].values if target_col in df_processed.columns else None
    
    return X, y

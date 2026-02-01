"""
Data loading and preprocessing module.
Handles downloading and initial processing of Walmart sales dataset.
"""

import kagglehub
import pandas as pd
import os
from pathlib import Path


def _find_file(root: str | Path, filename: str) -> str:
    """
    Find a file within the kagglehub extracted directory.

    kagglehub datasets sometimes extract into nested folders like:
      .../train.csv/train.csv
    so we search recursively and return the first match.
    """
    root_path = Path(root)
    matches = [p for p in root_path.rglob(filename) if p.is_file()]
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} under {root_path}")
    return str(matches[0])


def load_dataset():
    """
    Download and load the Walmart store sales forecasting dataset.
    
    Returns:
        tuple: (train_df, test_df, features_df, stores_df) - DataFrames containing the dataset
    """
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("micgonzalez/walmart-store-sales-forecasting")
    print(f"Dataset downloaded to: {path}")
    
    # Load CSV files (robust to nested extract layouts)
    train_path = _find_file(path, "train.csv")
    test_path = _find_file(path, "test.csv")
    features_path = _find_file(path, "features.csv")
    stores_path = _find_file(path, "stores.csv")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    features_df = pd.read_csv(features_path)
    stores_df = pd.read_csv(stores_path)
    
    # Parse dates
    train_df['Date'] = pd.to_datetime(train_df['Date'])
    test_df['Date'] = pd.to_datetime(test_df['Date'])
    features_df['Date'] = pd.to_datetime(features_df['Date'])
    
    # Merge features and stores with train/test data
    train_df = train_df.merge(features_df, on=['Store', 'Date'], how='left')
    train_df = train_df.merge(stores_df, on='Store', how='left')
    
    test_df = test_df.merge(features_df, on=['Store', 'Date'], how='left')
    test_df = test_df.merge(stores_df, on='Store', how='left')

    # Resolve duplicate holiday columns after merges
    for df in (train_df, test_df):
        if 'IsHoliday_x' in df.columns and 'IsHoliday_y' in df.columns:
            # Conservative: mark holiday if either source says it's a holiday
            df['IsHoliday'] = (df['IsHoliday_x'].astype(bool) | df['IsHoliday_y'].astype(bool))
            df.drop(columns=['IsHoliday_x', 'IsHoliday_y'], inplace=True)
        elif 'IsHoliday_x' in df.columns and 'IsHoliday' not in df.columns:
            df.rename(columns={'IsHoliday_x': 'IsHoliday'}, inplace=True)
        elif 'IsHoliday_y' in df.columns and 'IsHoliday' not in df.columns:
            df.rename(columns={'IsHoliday_y': 'IsHoliday'}, inplace=True)
    
    # Sort by Store, Dept, Date
    train_df = train_df.sort_values(['Store', 'Dept', 'Date']).reset_index(drop=True)
    test_df = test_df.sort_values(['Store', 'Dept', 'Date']).reset_index(drop=True)
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    return train_df, test_df, features_df, stores_df


def save_processed_data(df, filename):
    """Save processed dataframe to data directory."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    filepath = data_dir / filename
    df.to_csv(filepath, index=False)
    print(f"Saved processed data to {filepath}")

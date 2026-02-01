"""
Calculate accuracy metrics for the sales forecasting models.
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_dataset
from src.feature_engineering import create_all_features, get_feature_columns
from src.utils import time_based_split, prepare_ml_data
from src.models import NaiveForecast, MovingAverage, LightGBMModel, XGBoostModel
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def calculate_r2_score(y_true, y_pred):
    """Calculate R² score (coefficient of determination)."""
    return r2_score(y_true, y_pred)


def calculate_accuracy_percentage(y_true, y_pred, tolerance=0.1):
    """
    Calculate accuracy as percentage of predictions within tolerance.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        tolerance: Percentage tolerance (e.g., 0.1 = 10%)
    
    Returns:
        Accuracy percentage
    """
    errors = np.abs((y_true - y_pred) / (y_true + 1e-8))  # Add small epsilon to avoid division by zero
    within_tolerance = (errors <= tolerance).sum()
    return (within_tolerance / len(y_true)) * 100


def main():
    """Calculate accuracy metrics for all models."""
    
    print("="*80)
    print("ACCURACY METRICS CALCULATION")
    print("="*80)
    
    # Load and prepare data
    print("\nLoading dataset and preparing features...")
    train_df, test_df, features_df, stores_df = load_dataset()
    
    train_df_featured = create_all_features(
        train_df.copy(),
        target_col='Weekly_Sales',
        lags=[1, 2, 4, 12],
        rolling_windows=[4, 8, 12]
    )
    train_df_featured = train_df_featured.dropna(subset=['Weekly_Sales']).reset_index(drop=True)
    
    train_data, val_data = time_based_split(train_df_featured, date_col='Date', train_ratio=0.8)
    
    feature_cols = get_feature_columns(train_data)
    X_train, y_train = prepare_ml_data(train_data, feature_cols=feature_cols)
    X_val, y_val = prepare_ml_data(val_data, feature_cols=feature_cols)
    
    # Get predictions from all models
    print("\nGenerating predictions from all models...")
    
    # Baseline models
    naive_model = NaiveForecast()
    naive_model.fit(train_data, target_col='Weekly_Sales')
    naive_pred = naive_model.predict(val_data)
    
    ma_model = MovingAverage(window=4)
    ma_model.fit(train_data, target_col='Weekly_Sales')
    ma_pred = ma_model.predict(val_data)
    
    # ML models
    lgb_model = LightGBMModel()
    lgb_model.fit(X_train, y_train, X_val, y_val)
    lgb_pred = lgb_model.predict(X_val)
    
    xgb_model = XGBoostModel()
    xgb_model.fit(X_train, y_train, X_val, y_val)
    xgb_pred = xgb_model.predict(X_val)
    
    # Calculate metrics for each model
    models = {
        'Naive Forecast': naive_pred,
        'Moving Average': ma_pred,
        'LightGBM': lgb_pred,
        'XGBoost': xgb_pred
    }
    
    print("\n" + "="*80)
    print("ACCURACY METRICS SUMMARY")
    print("="*80)
    print(f"\n{'Model':<25} {'R² Score':<12} {'Accuracy (±10%)':<18} {'Accuracy (±20%)':<18} {'RMSE':<12} {'MAE':<12}")
    print("-"*80)
    
    results = []
    for model_name, y_pred in models.items():
        r2 = calculate_r2_score(y_val, y_pred)
        acc_10 = calculate_accuracy_percentage(y_val, y_pred, tolerance=0.10)
        acc_20 = calculate_accuracy_percentage(y_val, y_pred, tolerance=0.20)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        
        results.append({
            'Model': model_name,
            'R² Score': r2,
            'Accuracy (±10%)': acc_10,
            'Accuracy (±20%)': acc_20,
            'RMSE': rmse,
            'MAE': mae
        })
        
        print(f"{model_name:<25} {r2:<12.4f} {acc_10:<18.2f}% {acc_20:<18.2f}% {rmse:<12.2f} {mae:<12.2f}")
    
    # Find best model
    best_r2 = max(results, key=lambda x: x['R² Score'])
    best_acc_10 = max(results, key=lambda x: x['Accuracy (±10%)'])
    
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    print(f"\nBest R² Score: {best_r2['Model']} ({best_r2['R² Score']:.4f})")
    print(f"  - R² Score: {best_r2['R² Score']:.4f} ({best_r2['R² Score']*100:.2f}% variance explained)")
    print(f"  - Accuracy (±10%): {best_r2['Accuracy (±10%)']:.2f}%")
    print(f"  - Accuracy (±20%): {best_r2['Accuracy (±20%)']:.2f}%")
    print(f"  - RMSE: ${best_r2['RMSE']:,.2f}")
    print(f"  - MAE: ${best_r2['MAE']:,.2f}")
    
    print(f"\nBest Accuracy (±10%): {best_acc_10['Model']} ({best_acc_10['Accuracy (±10%)']:.2f}%)")
    
    # Calculate average sales for context
    avg_sales = np.mean(y_val)
    print(f"\nAverage Weekly Sales: ${avg_sales:,.2f}")
    print(f"Best Model RMSE as % of Average: {(best_r2['RMSE']/avg_sales)*100:.2f}%")
    print(f"Best Model MAE as % of Average: {(best_r2['MAE']/avg_sales)*100:.2f}%")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/accuracy_metrics.csv', index=False)
    print(f"\nDetailed results saved to: results/accuracy_metrics.csv")
    print("="*80)


if __name__ == "__main__":
    main()

"""
Fast version of the pipeline - skips slow statistical models (SARIMA, Prophet)
Focuses on baseline and ML models for quick results.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_dataset
from src.eda import generate_eda_report
from src.feature_engineering import get_feature_columns
from src.preprocessing import Preprocessor, get_walmart_config
from src.utils import time_based_split, prepare_ml_data
from src.models import (
    NaiveForecast, MovingAverage,
    LightGBMModel, XGBoostModel, LSTMModel
)
from src.evaluation import (
    evaluate_model, compare_models, plot_predictions, plot_all_predictions
)
import pandas as pd
import numpy as np


def main():
    """Run the fast forecasting pipeline."""
    
    print("="*80)
    print("WALMART SALES FORECASTING PIPELINE (FAST VERSION)")
    print("="*80)
    
    # Step 1: Load dataset
    print("\n[Step 1/8] Loading dataset...")
    train_df, test_df, features_df, stores_df = load_dataset()
    
    # Step 2: EDA (skip if already done)
    print("\n[Step 2/8] Performing Exploratory Data Analysis...")
    if not os.path.exists('results/sales_trends_overview.png'):
        generate_eda_report(train_df, save_dir='results')
    else:
        print("  EDA already completed, skipping...")
    
    # Step 3: Time-based split (before preprocessing so we fit only on train)
    print("\n[Step 3/8] Splitting data (temporal)...")
    train_part, val_part = time_based_split(train_df, date_col='Date', train_ratio=0.8)
    
    # Step 4: 11-step preprocessing (no data leakage; fit on train only where needed)
    print("\n[Step 4/8] Preprocessing (11 steps)...")
    config = get_walmart_config()
    preprocessor = Preprocessor(config, verbose=True)
    train_data, val_data = preprocessor.fit_transform_train_val(train_part, val_part)
    
    train_data = train_data.dropna(subset=['Weekly_Sales']).reset_index(drop=True)
    val_data = val_data.dropna(subset=['Weekly_Sales']).reset_index(drop=True)
    train_data = train_data.fillna(0)
    val_data = val_data.fillna(0)
    
    feature_cols = get_feature_columns(train_data)
    X_train, y_train = prepare_ml_data(train_data, feature_cols=feature_cols)
    X_val, y_val = prepare_ml_data(val_data, feature_cols=feature_cols)
    
    # Step 5: Baseline Models
    print("\n[Step 5/8] Training baseline models...")
    
    naive_model = NaiveForecast()
    naive_model.fit(train_data, target_col='Weekly_Sales')
    naive_pred = naive_model.predict(val_data)
    naive_results = evaluate_model(y_val, naive_pred, "Naive Forecast")
    
    ma_model = MovingAverage(window=4)
    ma_model.fit(train_data, target_col='Weekly_Sales')
    ma_pred = ma_model.predict(val_data)
    ma_results = evaluate_model(y_val, ma_pred, "Moving Average (4 weeks)")
    
    # Step 6: ML Models
    print("\n[Step 6/8] Training ML models...")
    
    # LightGBM
    print("  Training LightGBM...")
    lgb_model = LightGBMModel()
    lgb_model.fit(X_train, y_train, X_val, y_val)
    lgb_pred = lgb_model.predict(X_val)
    lgb_results = evaluate_model(y_val, lgb_pred, "LightGBM")
    
    # XGBoost
    print("  Training XGBoost...")
    xgb_model = XGBoostModel()
    xgb_model.fit(X_train, y_train, X_val, y_val)
    xgb_pred = xgb_model.predict(X_val)
    xgb_results = evaluate_model(y_val, xgb_pred, "XGBoost")
    
    # LSTM (simplified)
    print("  Training LSTM...")
    lstm_model = LSTMModel(sequence_length=12, units=50, epochs=10, batch_size=32)
    lstm_model.fit(train_data, target_col='Weekly_Sales', feature_cols=feature_cols)
    lstm_pred = lstm_model.predict(val_data, feature_cols=feature_cols)
    lstm_results = evaluate_model(y_val, lstm_pred, "LSTM")
    
    # Step 7: Compare Models
    print("\n[Step 7/8] Comparing models...")
    all_results = [
        naive_results,
        ma_results,
        lgb_results,
        xgb_results,
        lstm_results
    ]
    comparison_df = compare_models(all_results, save_path='results/model_comparison.csv')
    
    # Step 8: Visualizations
    print("\n[Step 8/8] Creating visualizations...")
    predictions_dict = {
        'Naive Forecast': {'y_true': y_val, 'y_pred': naive_pred},
        'Moving Average': {'y_true': y_val, 'y_pred': ma_pred},
        'LightGBM': {'y_true': y_val, 'y_pred': lgb_pred},
        'XGBoost': {'y_true': y_val, 'y_pred': xgb_pred},
        'LSTM': {'y_true': y_val, 'y_pred': lstm_pred}
    }
    
    plot_all_predictions(predictions_dict, save_path='results/all_predictions.png')
    plot_predictions(y_val, lgb_pred, 'LightGBM', save_path='results/lightgbm_predictions.png')
    plot_predictions(y_val, xgb_pred, 'XGBoost', save_path='results/xgboost_predictions.png')
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Best model (by RMSE): {comparison_df.iloc[0]['Model']}")
    print(f"  RMSE: {comparison_df.iloc[0]['RMSE']:.2f}")
    print(f"  MAE: {comparison_df.iloc[0]['MAE']:.2f}")
    print(f"  MAPE: {comparison_df.iloc[0]['MAPE']:.2f}%")
    print("\nAll results saved in the 'results/' directory.")
    print("="*80)


if __name__ == "__main__":
    main()

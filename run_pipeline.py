"""
Main pipeline script for sales forecasting.
Can be run directly: python run_pipeline.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_dataset
from src.eda import generate_eda_report
from src.feature_engineering import create_all_features, get_feature_columns
from src.utils import time_based_split, prepare_ml_data
from src.models import (
    NaiveForecast, MovingAverage, SARIMAModel, ProphetModel,
    LightGBMModel, XGBoostModel, LSTMModel
)
from src.evaluation import (
    evaluate_model, compare_models, plot_predictions, plot_all_predictions
)
import pandas as pd
import numpy as np


def main():
    """Run the complete forecasting pipeline."""
    
    print("="*80)
    print("WALMART SALES FORECASTING PIPELINE")
    print("="*80)
    
    # Step 1: Load dataset
    print("\n[Step 1/9] Loading dataset...")
    train_df, test_df, features_df, stores_df = load_dataset()
    
    # Step 2: EDA
    print("\n[Step 2/9] Performing Exploratory Data Analysis...")
    generate_eda_report(train_df, save_dir='results')
    
    # Step 3: Feature Engineering
    print("\n[Step 3/9] Creating features...")
    train_df_featured = create_all_features(
        train_df.copy(),
        target_col='Weekly_Sales',
        lags=[1, 2, 4, 12],
        rolling_windows=[4, 8, 12]
    )
    train_df_featured = train_df_featured.dropna(subset=['Weekly_Sales']).reset_index(drop=True)
    
    # Step 4: Time-based split
    print("\n[Step 4/9] Splitting data...")
    train_data, val_data = time_based_split(train_df_featured, date_col='Date', train_ratio=0.8)
    
    feature_cols = get_feature_columns(train_data)
    X_train, y_train = prepare_ml_data(train_data, feature_cols=feature_cols)
    X_val, y_val = prepare_ml_data(val_data, feature_cols=feature_cols)
    
    # Step 5: Baseline Models
    print("\n[Step 5/9] Training baseline models...")
    
    naive_model = NaiveForecast()
    naive_model.fit(train_data, target_col='Weekly_Sales')
    naive_pred = naive_model.predict(val_data)
    naive_results = evaluate_model(y_val, naive_pred, "Naive Forecast")
    
    ma_model = MovingAverage(window=4)
    ma_model.fit(train_data, target_col='Weekly_Sales')
    ma_pred = ma_model.predict(val_data)
    ma_results = evaluate_model(y_val, ma_pred, "Moving Average (4 weeks)")
    
    # Step 6: Research Models
    print("\n[Step 6/9] Training research-style models...")
    
    # Use sample for statistical models (they're slow)
    sample_stores = train_data['Store'].unique()[:5]
    sample_train = train_data[train_data['Store'].isin(sample_stores)].copy()
    sample_val = val_data[val_data['Store'].isin(sample_stores)].copy()
    
    # SARIMA
    print("  Training SARIMA...")
    sarima_model = SARIMAModel(order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
    sarima_model.fit(sample_train, target_col='Weekly_Sales')
    sarima_pred = sarima_model.predict(sample_val)
    if len(sample_val) > 0:
        sarima_y_val = sample_val['Weekly_Sales'].values
        sarima_results = evaluate_model(sarima_y_val, sarima_pred, "SARIMA")
    else:
        sarima_results = {'Model': 'SARIMA', 'RMSE': np.inf, 'MAE': np.inf, 'MAPE': np.inf}
    
    # Prophet
    print("  Training Prophet...")
    prophet_model = ProphetModel(yearly_seasonality=True, weekly_seasonality=True)
    prophet_model.fit(sample_train, target_col='Weekly_Sales', date_col='Date')
    prophet_pred = prophet_model.predict(sample_val, date_col='Date')
    if len(sample_val) > 0:
        prophet_y_val = sample_val['Weekly_Sales'].values
        prophet_results = evaluate_model(prophet_y_val, prophet_pred, "Prophet")
    else:
        prophet_results = {'Model': 'Prophet', 'RMSE': np.inf, 'MAE': np.inf, 'MAPE': np.inf}
    
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
    
    # LSTM
    print("  Training LSTM...")
    lstm_model = LSTMModel(sequence_length=12, units=50, epochs=20, batch_size=32)
    lstm_model.fit(train_data, target_col='Weekly_Sales', feature_cols=feature_cols)
    lstm_pred = lstm_model.predict(val_data, feature_cols=feature_cols)
    lstm_results = evaluate_model(y_val, lstm_pred, "LSTM")
    
    # Step 7: Compare Models
    print("\n[Step 7/9] Comparing models...")
    all_results = [
        naive_results,
        ma_results,
        sarima_results,
        prophet_results,
        lgb_results,
        xgb_results,
        lstm_results
    ]
    comparison_df = compare_models(all_results, save_path='results/model_comparison.csv')
    
    # Step 8: Visualizations
    print("\n[Step 8/9] Creating visualizations...")
    predictions_dict = {
        'Naive Forecast': {'y_true': y_val, 'y_pred': naive_pred},
        'Moving Average': {'y_true': y_val, 'y_pred': ma_pred},
        'LightGBM': {'y_true': y_val, 'y_pred': lgb_pred},
        'XGBoost': {'y_true': y_val, 'y_pred': xgb_pred}
    }
    
    if len(sample_val) > 0:
        predictions_dict['SARIMA'] = {'y_true': sarima_y_val, 'y_pred': sarima_pred}
        predictions_dict['Prophet'] = {'y_true': prophet_y_val, 'y_pred': prophet_pred}
    
    plot_all_predictions(predictions_dict, save_path='results/all_predictions.png')
    plot_predictions(y_val, lgb_pred, 'LightGBM', save_path='results/lightgbm_predictions.png')
    plot_predictions(y_val, xgb_pred, 'XGBoost', save_path='results/xgboost_predictions.png')
    
    # Step 9: Summary
    print("\n[Step 9/9] Pipeline complete!")
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

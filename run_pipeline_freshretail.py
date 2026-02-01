"""
Forecasting pipeline for FreshRetailNet-50K (Hugging Face).
Uses the same models as the Kaggle pipeline; results saved to results_freshretail/.
No synthetic data or hardcoding - real data and real model training.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader_freshretail import load_dataset_freshretail
from src.feature_engineering import get_feature_columns_freshretail
from src.preprocessing import Preprocessor, get_freshretail_config
from src.utils import prepare_ml_data
from src.models import (
    NaiveForecast,
    MovingAverage,
    LightGBMModel,
    XGBoostModel,
    LSTMModel,
)
from src.evaluation import (
    evaluate_model,
    compare_models,
    plot_predictions,
    plot_all_predictions,
)
import pandas as pd
import numpy as np

RESULTS_DIR = "results_freshretail"
TARGET_COL = "sale_amount"
DATE_COL = "dt"
GROUPBY_COLS = ["store_id", "product_id"]

# Full dataset: None = use all data (~4.5M train, ~350k eval). Suitable for 24GB RAM.
# To cap for lower memory, set e.g. MAX_TRAIN_ROWS = 500_000, MAX_EVAL_ROWS = 100_000
MAX_TRAIN_ROWS = None
MAX_EVAL_ROWS = None


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 80)
    print("FRESHRETAILNET-50K FORECASTING PIPELINE")
    print("Dataset: Dingdong-Inc/FreshRetailNet-50K (Hugging Face)")
    print("=" * 80)
    if MAX_TRAIN_ROWS is None and MAX_EVAL_ROWS is None:
        print("Mode: FULL DATASET (~4.5M train + ~350k eval) - no row caps")
    else:
        print(f"Mode: Capped (train max {MAX_TRAIN_ROWS}, eval max {MAX_EVAL_ROWS})")

    # Step 1: Load real data from Hugging Face
    print("\n[Step 1/8] Loading dataset...")
    train_df, eval_df = load_dataset_freshretail(
        max_train_rows=MAX_TRAIN_ROWS,
        max_eval_rows=MAX_EVAL_ROWS,
    )

    # Step 2: 11-step preprocessing (no data leakage; fit on train only where needed)
    print("\n[Step 2/8] Preprocessing (11 steps)...")
    config = get_freshretail_config()
    preprocessor = Preprocessor(config, verbose=True)
    train_data, val_data = preprocessor.fit_transform_train_val(train_df, eval_df, source_col="_source")

    train_data = train_data.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    val_data = val_data.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    train_data = train_data.fillna(0)
    val_data = val_data.fillna(0)

    # Step 3: Prepare ML matrices
    print("\n[Step 3/8] Preparing ML data...")
    feature_cols = get_feature_columns_freshretail(train_data)
    X_train, y_train = prepare_ml_data(
        train_data,
        target_col=TARGET_COL,
        feature_cols=feature_cols,
        exclude_cols=[DATE_COL, TARGET_COL, "store_id", "product_id", "_source"],
    )
    X_val, y_val = prepare_ml_data(
        val_data,
        target_col=TARGET_COL,
        feature_cols=feature_cols,
        exclude_cols=[DATE_COL, TARGET_COL, "store_id", "product_id", "_source"],
    )

    print(f"Training features shape: {X_train.shape}")
    print(f"Validation features shape: {X_val.shape}")

    # Step 4: Baselines
    print("\n[Step 4/8] Training baseline models...")
    naive_model = NaiveForecast()
    naive_model.fit(train_data, target_col=TARGET_COL, groupby_cols=GROUPBY_COLS)
    naive_pred = naive_model.predict(val_data, groupby_cols=GROUPBY_COLS)
    naive_results = evaluate_model(y_val, naive_pred, "Naive Forecast")

    ma_model = MovingAverage(window=4)
    ma_model.fit(train_data, target_col=TARGET_COL, groupby_cols=GROUPBY_COLS)
    ma_pred = ma_model.predict(val_data, groupby_cols=GROUPBY_COLS)
    ma_results = evaluate_model(y_val, ma_pred, "Moving Average (4 periods)")

    # Step 5: ML models
    print("\n[Step 5/8] Training ML models...")
    print("  Training LightGBM...")
    lgb_model = LightGBMModel()
    lgb_model.fit(X_train, y_train, X_val, y_val)
    lgb_pred = lgb_model.predict(X_val)
    lgb_results = evaluate_model(y_val, lgb_pred, "LightGBM")

    print("  Training XGBoost...")
    xgb_model = XGBoostModel()
    xgb_model.fit(X_train, y_train, X_val, y_val)
    xgb_pred = xgb_model.predict(X_val)
    xgb_results = evaluate_model(y_val, xgb_pred, "XGBoost")

    print("  Training LSTM...")
    # LSTM uses 'Date' column internally; FreshRetail uses 'dt'
    train_for_lstm = train_data.rename(columns={DATE_COL: "Date"})
    val_for_lstm = val_data.rename(columns={DATE_COL: "Date"})
    lstm_model = LSTMModel(sequence_length=12, units=50, epochs=10, batch_size=32)
    lstm_model.fit(
        train_for_lstm,
        target_col=TARGET_COL,
        feature_cols=feature_cols,
        groupby_cols=GROUPBY_COLS,
    )
    lstm_pred = lstm_model.predict(val_for_lstm, feature_cols=feature_cols)
    lstm_results = evaluate_model(y_val, lstm_pred, "LSTM")

    # Step 6: Compare
    print("\n[Step 6/8] Comparing models...")
    all_results = [
        naive_results,
        ma_results,
        lgb_results,
        xgb_results,
        lstm_results,
    ]
    comparison_df = compare_models(
        all_results,
        save_path=os.path.join(RESULTS_DIR, "model_comparison.csv"),
    )

    # Step 7: Plots
    print("\n[Step 7/8] Creating visualizations...")
    predictions_dict = {
        "Naive Forecast": {"y_true": y_val, "y_pred": naive_pred},
        "Moving Average": {"y_true": y_val, "y_pred": ma_pred},
        "LightGBM": {"y_true": y_val, "y_pred": lgb_pred},
        "XGBoost": {"y_true": y_val, "y_pred": xgb_pred},
        "LSTM": {"y_true": y_val, "y_pred": lstm_pred},
    }
    plot_all_predictions(
        predictions_dict,
        save_path=os.path.join(RESULTS_DIR, "all_predictions.png"),
    )
    plot_predictions(
        y_val,
        lgb_pred,
        "LightGBM",
        save_path=os.path.join(RESULTS_DIR, "lightgbm_predictions.png"),
    )
    plot_predictions(
        y_val,
        xgb_pred,
        "XGBoost",
        save_path=os.path.join(RESULTS_DIR, "xgboost_predictions.png"),
    )

    # Step 8: Summary
    print("\n[Step 8/8] Summary")
    print("=" * 80)
    print("SUMMARY (FreshRetailNet-50K)")
    print("=" * 80)
    print(f"Best model (by RMSE): {comparison_df.iloc[0]['Model']}")
    print(f"  RMSE: {comparison_df.iloc[0]['RMSE']:.4f}")
    print(f"  MAE: {comparison_df.iloc[0]['MAE']:.4f}")
    print(f"  MAPE: {comparison_df.iloc[0]['MAPE']:.2f}%")
    print(f"\nAll results saved in: {RESULTS_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()

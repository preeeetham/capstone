"""
Quick test script to verify all imports work correctly.
Run: python test_imports.py
"""

import sys
import os

print("Testing imports...")

try:
    from src.data_loader import load_dataset
    print("✓ data_loader imported successfully")
except Exception as e:
    print(f"✗ data_loader import failed: {e}")

try:
    from src.eda import generate_eda_report, check_missing_values, detect_outliers
    print("✓ eda imported successfully")
except Exception as e:
    print(f"✗ eda import failed: {e}")

try:
    from src.feature_engineering import create_all_features, get_feature_columns
    print("✓ feature_engineering imported successfully")
except Exception as e:
    print(f"✗ feature_engineering import failed: {e}")

try:
    from src.models import (
        NaiveForecast, MovingAverage, SARIMAModel, ProphetModel,
        LightGBMModel, XGBoostModel, LSTMModel
    )
    print("✓ models imported successfully")
except Exception as e:
    print(f"✗ models import failed: {e}")

try:
    from src.evaluation import (
        evaluate_model, compare_models, plot_predictions
    )
    print("✓ evaluation imported successfully")
except Exception as e:
    print(f"✗ evaluation import failed: {e}")

try:
    from src.utils import time_based_split, prepare_ml_data
    print("✓ utils imported successfully")
except Exception as e:
    print(f"✗ utils import failed: {e}")

print("\nAll imports tested!")

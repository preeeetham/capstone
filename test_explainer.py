from core.explainable_ai import generate_explanation

metrics_summary = {
    "linear": {"accuracy": 47.18, "mae": 269.8651, "rmse": 593.9554},
    "random_forest": {"accuracy": 58.18, "mae": 210.8596, "rmse": 528.5282},
    "xgboost": {"accuracy": 70.82, "mae": 202.6974, "rmse": 441.4607}
}

preprocessing_meta = {
    "rows_loaded": 3054348,
    "rows_after_outliers": 2887181,
    "outliers_removed": 167167,
    "negative_target_dropped": 0,
    "columns_dropped": [],
    "numeric_imputation": "median",
    "features_total": 41,
    "features_selected": 5,
    "models_skipped": ["knn", "dnn"]
}

config = {
    "data": {"target_col": "sales", "files": [{"path": "1"}, {"path": "2"}]}
}

if __name__ == "__main__":
    generate_explanation(metrics_summary, preprocessing_meta, config)
    print("Done! Check explainable_ai/explain.md")

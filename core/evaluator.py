"""
evaluator.py
============
Evaluates model predictions, generates plots, saves CSVs,
and triggers the Gemini-powered Explainable AI module.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from core.explainable_ai import generate_explanation


class Evaluator:
    def __init__(self, config):
        self.config = config
        os.makedirs('plots', exist_ok=True)
        os.makedirs('predictions', exist_ok=True)

    def evaluate(self, results_dict, preprocessing_meta=None):
        logging.info("Starting Evaluation Pipeline...")
        accuracies = []
        metrics_summary = {}

        for model_name, info in results_dict.items():
            y_test = info['y_test']
            y_pred = info['y_pred']

            mae  = metrics.mean_absolute_error(y_test, y_pred)
            mse  = metrics.mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2   = metrics.r2_score(y_test, y_pred)
            acc  = r2 * 100

            logging.info(f"--- {model_name.upper()} ---")
            logging.info(f"Accuracy (R² %): {acc:.2f}")
            logging.info(f"MAE:  {mae:.4f}")
            logging.info(f"RMSE: {rmse:.4f}")

            metrics_summary[model_name] = {
                "accuracy": acc,
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "r2": r2,
            }
            accuracies.append({'model': model_name, 'accuracy': acc})

            # Save predictions CSV
            df_out = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
            df_out.to_csv(f'predictions/{model_name}_real_pred.csv', index=False)

            # Save actual-vs-predicted line plot
            plt.figure(figsize=(15, 6))
            plt.plot(y_pred[:200], label="Prediction", color='steelblue', linewidth=1.5)
            plt.plot(y_test.values[:200], label="Actual", color='coral', linewidth=1.5)
            plt.legend()
            plt.title(f"{model_name.upper()} — Predictions vs Actual (first 200 samples)")
            plt.tight_layout()
            plt.savefig(f'plots/{model_name}_real_pred.png', dpi=150)
            plt.close()

        # Comparison bar chart
        if accuracies:
            acc_df = pd.DataFrame(accuracies)
            plt.figure(figsize=(10, 6))
            sns.barplot(x='model', y='accuracy', data=acc_df, palette='viridis')
            plt.title('Model Accuracy Comparison (R² %)')
            plt.ylabel('Accuracy (%)')
            plt.xlabel('Model')
            plt.tight_layout()
            plt.savefig('plots/compared_models.png', dpi=150)
            plt.close()
            logging.info("Evaluation plots saved to 'plots/'")

        # ── Explainable AI ──────────────────────────────────────────────────
        if preprocessing_meta is not None:
            try:
                output_path = generate_explanation(metrics_summary, preprocessing_meta, self.config)
                logging.info(f"✅  Retailer-friendly explanation saved → {output_path}")
            except Exception as e:
                logging.error(f"Explainable AI generation failed (pipeline still succeeded): {e}")
        # ────────────────────────────────────────────────────────────────────

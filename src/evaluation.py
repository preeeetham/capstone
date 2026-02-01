"""
Evaluation metrics and model comparison module.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List


def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_mae(y_true, y_pred):
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error."""
    # Avoid division by zero
    mask = y_true != 0
    if mask.sum() == 0:
        return np.inf
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Evaluate a model using multiple metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model
    
    Returns:
        Dictionary with evaluation metrics
    """
    rmse = calculate_rmse(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)
    
    results = {
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }
    
    return results


def compare_models(results_list: List[Dict], save_path="results/model_comparison.csv"):
    """
    Compare multiple models and create a comparison table.
    
    Args:
        results_list: List of evaluation result dictionaries
        save_path: Path to save comparison table
    """
    comparison_df = pd.DataFrame(results_list)
    
    # Sort by RMSE (lower is better)
    comparison_df = comparison_df.sort_values('RMSE')
    
    # Save to CSV
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(save_path, index=False)
    
    # Print formatted table
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    print(comparison_df.to_string(index=False))
    print("="*80)
    print(f"\nResults saved to {save_path}")
    
    return comparison_df


def plot_predictions(y_true, y_pred, model_name="Model", save_path="results/predictions.png"):
    """
    Plot predicted vs actual values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model
        save_path: Path to save the plot
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.5, s=20)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual Sales')
    axes[0].set_ylabel('Predicted Sales')
    axes[0].set_title(f'{model_name}: Predicted vs Actual')
    axes[0].grid(True, alpha=0.3)
    
    # Calculate metrics for display
    rmse = calculate_rmse(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)
    
    axes[0].text(0.05, 0.95, f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nMAPE: {mape:.2f}%',
                transform=axes[0].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Time series plot (sample for readability; use more points when data is large)
    sample_size = min(max(2000, len(y_true) // 50), len(y_true))
    indices = np.random.choice(len(y_true), sample_size, replace=False)
    indices = np.sort(indices)
    
    axes[1].plot(range(sample_size), y_true[indices], label='Actual', alpha=0.7, linewidth=1.5)
    axes[1].plot(range(sample_size), y_pred[indices], label='Predicted', alpha=0.7, linewidth=1.5)
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Sales')
    axes[1].set_title(f'{model_name}: Time Series Comparison (n={sample_size} of {len(y_true)})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved prediction plot to {save_path}")
    plt.close()


def plot_all_predictions(results_dict: Dict[str, Dict], save_path="results/all_predictions.png"):
    """
    Plot predictions for all models in a grid.
    
    Args:
        results_dict: Dictionary with model names as keys and {'y_true': ..., 'y_pred': ...} as values
        save_path: Path to save the plot
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    n_models = len(results_dict)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (model_name, results) in enumerate(results_dict.items()):
        y_true = results['y_true']
        y_pred = results['y_pred']
        
        ax = axes[idx]
        ax.scatter(y_true, y_pred, alpha=0.5, s=20)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Sales')
        ax.set_ylabel('Predicted Sales')
        ax.set_title(model_name)
        ax.grid(True, alpha=0.3)
        
        # Add metrics
        rmse = calculate_rmse(y_true, y_pred)
        mae = calculate_mae(y_true, y_pred)
        mape = calculate_mape(y_true, y_pred)
        
        ax.text(0.05, 0.95, f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nMAPE: {mape:.2f}%',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved all predictions plot to {save_path}")
    plt.close()

"""
Generate correlation heatmap for FreshRetailNet-50K dataset.
Shows relationships between features and target (sale_amount).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader_freshretail import load_dataset_freshretail
from src.preprocessing import Preprocessor, get_freshretail_config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Use a sample for speed (set to None for full data if you have time/memory)
MAX_TRAIN_ROWS = 100_000
MAX_EVAL_ROWS = 20_000

RESULTS_DIR = "results_freshretail"


def create_correlation_heatmap(df, target_col='sale_amount', save_path=None):
    """
    Create correlation heatmap showing relationships between features and target.
    
    Args:
        df: DataFrame with features
        target_col: Target column name
        save_path: Path to save plot
    """
    # Select only numeric columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude identifier columns if they're still present
    exclude = ['store_id', 'product_id', 'city_id', 'management_group_id',
               'first_category_id', 'second_category_id', 'third_category_id']
    numeric_cols = [c for c in numeric_cols if c not in exclude]
    
    # Compute correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Get correlations with target
    target_corr = corr_matrix[target_col].sort_values(ascending=False)
    
    # Select top correlations (positive and negative)
    n_top = min(25, len(target_corr) - 1)  # Show top 25 or all if less
    top_features = target_corr[1:n_top+1].index.tolist()  # Exclude target itself
    
    # Create heatmap with top features + target
    features_to_plot = [target_col] + top_features
    corr_subset = df[features_to_plot].corr()
    
    # Plot
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        corr_subset,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title('Feature Correlation Heatmap (FreshRetailNet-50K)\nTop Features by Correlation with sale_amount',
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved correlation heatmap to: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Print top correlations
    print("\n" + "="*70)
    print("TOP CORRELATIONS WITH SALE_AMOUNT")
    print("="*70)
    print(f"\n{'Feature':<40} {'Correlation':>10}")
    print("-" * 70)
    for feat in top_features[:15]:
        corr_val = corr_subset.loc[target_col, feat]
        print(f"{feat:<40} {corr_val:>10.4f}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("="*80)
    print("FRESHRETAILNET-50K CORRELATION HEATMAP GENERATOR")
    print("="*80)
    
    # Load data
    print("\n[1/4] Loading dataset...")
    train_df, eval_df = load_dataset_freshretail(
        max_train_rows=MAX_TRAIN_ROWS,
        max_eval_rows=MAX_EVAL_ROWS,
    )
    
    # Preprocess with 11 steps (to get all features)
    print("\n[2/4] Preprocessing (11 steps to create all features)...")
    config = get_freshretail_config()
    preprocessor = Preprocessor(config, verbose=False)
    train_processed, val_processed = preprocessor.fit_transform_train_val(train_df, eval_df)
    
    # Drop NaN and combine for correlation analysis
    print("\n[3/4] Preparing data for correlation analysis...")
    train_processed = train_processed.dropna(subset=['sale_amount']).fillna(0)
    
    # Sample if still very large (for heatmap readability)
    if len(train_processed) > 50_000:
        train_sample = train_processed.sample(n=50_000, random_state=42)
        print(f"  Sampled {len(train_sample)} rows for correlation computation")
    else:
        train_sample = train_processed
    
    # Generate heatmap
    print("\n[4/4] Generating correlation heatmap...")
    create_correlation_heatmap(
        train_sample,
        target_col='sale_amount',
        save_path=os.path.join(RESULTS_DIR, "correlation_heatmap.png")
    )
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"Heatmap saved to: {RESULTS_DIR}/correlation_heatmap.png")


if __name__ == "__main__":
    main()

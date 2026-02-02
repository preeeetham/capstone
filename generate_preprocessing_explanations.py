"""
Generate individual, focused explanation images for each of the 11 preprocessing methods.
Each method gets its own clear, interpretable visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.data_loader_freshretail import load_dataset_freshretail
from src.preprocessing import (
    PreprocessorConfig,
    step1_parse_dates,
    step2_select_columns,
    step3_sort_by_entity_and_date,
    step4_impute_missing_fit,
    step4_impute_missing_transform,
    step5_add_temporal_features,
    step6_add_lag_features,
    step7_add_rolling_features,
    step8_add_external_features,
    step9_encode_categoricals_fit,
    step9_encode_categoricals_transform,
    step10_cap_outliers_fit,
    step10_cap_outliers_transform,
    step11_scale_numerical_fit,
    step11_scale_numerical_transform,
    get_freshretail_config,
)

# Style settings
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Output directory
OUTPUT_DIR = Path("preprocessing_explanations")
OUTPUT_DIR.mkdir(exist_ok=True)


def save_figure(filename, title=None):
    """Save figure with consistent settings."""
    if title:
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {filename}")
    plt.close()


def method_1_parse_dates(df_before, df_after, config):
    """Step 1: Parse Dates - Show date type conversion."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    date_col = config.date_col
    
    # Left: Before - show date as string
    ax1 = axes[0]
    sample_dates_before = df_before[date_col].head(10).astype(str).values
    y_pos = np.arange(len(sample_dates_before))
    ax1.barh(y_pos, [1]*len(sample_dates_before), color='#e74c3c', alpha=0.7)
    for i, date_str in enumerate(sample_dates_before):
        ax1.text(0.5, i, date_str, ha='center', va='center', fontsize=10, fontweight='bold')
    ax1.set_yticks([])
    ax1.set_xlim(0, 1)
    ax1.set_xlabel('Date Format: STRING (object)', fontweight='bold')
    ax1.set_title('BEFORE: Date as String', fontweight='bold', color='#e74c3c')
    ax1.set_xticks([])
    
    # Right: After - show date as datetime
    ax2 = axes[1]
    sample_dates_after = df_after[date_col].head(10)
    formatted_dates = sample_dates_after.dt.strftime('%Y-%m-%d').values
    y_pos = np.arange(len(formatted_dates))
    ax2.barh(y_pos, [1]*len(formatted_dates), color='#2ecc71', alpha=0.7)
    for i, date_str in enumerate(formatted_dates):
        ax2.text(0.5, i, date_str, ha='center', va='center', fontsize=10, fontweight='bold')
    ax2.set_yticks([])
    ax2.set_xlim(0, 1)
    ax2.set_xlabel('Date Format: DATETIME (datetime64[ns])', fontweight='bold')
    ax2.set_title('AFTER: Date as Datetime Object', fontweight='bold', color='#2ecc71')
    ax2.set_xticks([])
    
    save_figure('method_01_parse_dates.png', 'Method 1: Parse Dates - Convert String to Datetime')


def method_2_select_columns(df_before, df_after):
    """Step 2: Select Columns - Show column filtering."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    before_cols = set(df_before.columns)
    after_cols = set(df_after.columns)
    kept_cols = sorted(before_cols & after_cols)
    dropped_cols = sorted(before_cols - after_cols)
    
    # Create visualization
    y_kept = np.arange(len(kept_cols))
    y_dropped = np.arange(len(kept_cols), len(kept_cols) + len(dropped_cols))
    
    if kept_cols:
        ax.barh(y_kept, [1]*len(kept_cols), color='#2ecc71', alpha=0.7, label=f'Kept ({len(kept_cols)})')
        for i, col in enumerate(kept_cols):
            ax.text(0.5, i, col, ha='center', va='center', fontsize=9, fontweight='bold')
    
    if dropped_cols:
        ax.barh(y_dropped, [1]*len(dropped_cols), color='#e74c3c', alpha=0.7, label=f'Dropped ({len(dropped_cols)})')
        for i, col in enumerate(dropped_cols):
            ax.text(0.5, len(kept_cols) + i, col, ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_xlabel('Column Status', fontweight='bold', fontsize=12)
    ax.legend(loc='upper right', fontsize=12)
    ax.set_title('Column Selection', fontweight='bold', fontsize=14, pad=20)
    
    # Add statistics
    ax.text(0.02, 0.98, f'Total Original Columns: {len(before_cols)}\nTotal Kept Columns: {len(after_cols)}',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    save_figure('method_02_select_columns.png', 'Method 2: Select Columns - Focus on Relevant Features')


def method_3_sort_data(df_before, df_after, config):
    """Step 3: Sort by Entity and Date - Show sorting impact."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Sample one store-product combination
    sample_entity = df_after.groupby(config.groupby_cols).size().index[0]
    mask_before = (df_before[config.groupby_cols[0]] == sample_entity[0]) & \
                  (df_before[config.groupby_cols[1]] == sample_entity[1])
    mask_after = (df_after[config.groupby_cols[0]] == sample_entity[0]) & \
                 (df_after[config.groupby_cols[1]] == sample_entity[1])
    
    sample_before = df_before[mask_before].head(15)
    sample_after = df_after[mask_after].head(15)
    
    # Left: Before (unsorted)
    ax1 = axes[0]
    if len(sample_before) > 0:
        dates_before = pd.to_datetime(sample_before[config.date_col])
        ax1.plot(range(len(dates_before)), range(len(dates_before)), 'o-', 
                color='#e74c3c', linewidth=2, markersize=8, alpha=0.7)
        ax1.set_xlabel('Row Index', fontweight='bold')
        ax1.set_ylabel('Chronological Order', fontweight='bold')
        ax1.set_title('BEFORE: Potentially Unsorted', fontweight='bold', color='#e74c3c')
        ax1.grid(True, alpha=0.3)
    
    # Right: After (sorted)
    ax2 = axes[1]
    if len(sample_after) > 0:
        dates_after = sample_after[config.date_col]
        ax2.plot(range(len(dates_after)), range(len(dates_after)), 'o-', 
                color='#2ecc71', linewidth=2, markersize=8, alpha=0.7)
        ax2.set_xlabel('Row Index', fontweight='bold')
        ax2.set_ylabel('Chronological Order', fontweight='bold')
        ax2.set_title('AFTER: Sorted Chronologically', fontweight='bold', color='#2ecc71')
        ax2.grid(True, alpha=0.3)
    
    save_figure('method_03_sort_data.png', 'Method 3: Sort by Entity and Date - Enable Temporal Features')


def method_4_impute_missing(df_before, df_after, stats):
    """Step 4: Impute Missing Values - Show before/after missing values."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top left: Missing values count before
    ax1 = axes[0, 0]
    missing_before = df_before.isnull().sum().sort_values(ascending=False).head(10)
    if len(missing_before) > 0:
        missing_before.plot(kind='barh', ax=ax1, color='#e74c3c', alpha=0.7)
        ax1.set_xlabel('Missing Value Count', fontweight='bold')
        ax1.set_title('BEFORE: Missing Values per Column', fontweight='bold', color='#e74c3c')
        ax1.grid(axis='x', alpha=0.3)
    
    # Top right: Missing values count after
    ax2 = axes[0, 1]
    missing_after = df_after.isnull().sum().sort_values(ascending=False).head(10)
    if len(missing_after) > 0:
        missing_after.plot(kind='barh', ax=ax2, color='#2ecc71', alpha=0.7)
        ax2.set_xlabel('Missing Value Count', fontweight='bold')
        ax2.set_title('AFTER: Missing Values per Column', fontweight='bold', color='#2ecc71')
        ax2.grid(axis='x', alpha=0.3)
    
    # Bottom: Imputation statistics
    ax3 = fig.add_subplot(2, 1, 2)
    ax3.axis('off')
    
    if stats:
        impute_text = "Imputation Statistics (Fit on Training Data Only):\n\n"
        for col, (kind, val) in list(stats.items())[:10]:
            if isinstance(val, (int, float)):
                impute_text += f"• {col}: {kind} = {val:.3f}\n"
            else:
                impute_text += f"• {col}: {kind} = {val}\n"
        
        ax3.text(0.1, 0.9, impute_text, transform=ax3.transAxes, 
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    save_figure('method_04_impute_missing.png', 'Method 4: Impute Missing Values - Fill NaN with Train Statistics')


def method_5_temporal_features(df_before, df_after, config):
    """Step 5: Temporal Features - Show new temporal features created."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Get temporal columns
    temporal_cols = [c for c in df_after.columns if c not in df_before.columns and 
                     any(x in c.lower() for x in ['year', 'month', 'week', 'day', 'quarter'])]
    
    # Plot 1: Month distribution
    if 'month' in df_after.columns:
        ax1 = axes[0, 0]
        df_after['month'].value_counts().sort_index().plot(kind='bar', ax=ax1, color='#3498db', alpha=0.7)
        ax1.set_xlabel('Month', fontweight='bold')
        ax1.set_ylabel('Count', fontweight='bold')
        ax1.set_title('Monthly Distribution', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Day of week distribution
    if 'day_of_week' in df_after.columns:
        ax2 = axes[0, 1]
        df_after['day_of_week'].value_counts().sort_index().plot(kind='bar', ax=ax2, color='#e74c3c', alpha=0.7)
        ax2.set_xlabel('Day of Week (0=Mon, 6=Sun)', fontweight='bold')
        ax2.set_ylabel('Count', fontweight='bold')
        ax2.set_title('Day of Week Distribution', fontweight='bold')
        ax2.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=45)
        ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Quarter distribution
    if 'quarter' in df_after.columns:
        ax3 = axes[1, 0]
        df_after['quarter'].value_counts().sort_index().plot(kind='bar', ax=ax3, color='#2ecc71', alpha=0.7)
        ax3.set_xlabel('Quarter', fontweight='bold')
        ax3.set_ylabel('Count', fontweight='bold')
        ax3.set_title('Quarterly Distribution', fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: List of new features
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    feature_text = f"New Temporal Features Created: {len(temporal_cols)}\n\n"
    for i, feat in enumerate(sorted(temporal_cols)[:12], 1):
        feature_text += f"{i}. {feat}\n"
    if len(temporal_cols) > 12:
        feature_text += f"... and {len(temporal_cols) - 12} more"
    
    ax4.text(0.1, 0.9, feature_text, transform=ax4.transAxes, 
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    save_figure('method_05_temporal_features.png', 'Method 5: Temporal Features - Extract Time Patterns')


def method_6_lag_features(df_before, df_after, config):
    """Step 6: Lag Features - Show lag feature correlations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    target = config.target_col
    lag_cols = [c for c in df_after.columns if 'lag' in c.lower() and target in c]
    
    # Plot scatter plots for each lag
    for idx, lag_col in enumerate(lag_cols[:4]):
        ax = axes[idx // 2, idx % 2]
        
        # Sample data to avoid overcrowding
        valid_data = df_after[[target, lag_col]].dropna()
        sample_size = min(1000, len(valid_data))
        if sample_size == 0:
            continue
        sample_data = valid_data.sample(sample_size, random_state=42) if len(valid_data) > 0 else valid_data
        
        ax.scatter(sample_data[lag_col], sample_data[target], alpha=0.3, s=20, color='#3498db')
        
        # Calculate and show correlation
        corr = sample_data[target].corr(sample_data[lag_col])
        
        # Add trend line
        z = np.polyfit(sample_data[lag_col], sample_data[target], 1)
        p = np.poly1d(z)
        ax.plot(sample_data[lag_col].sort_values(), p(sample_data[lag_col].sort_values()), 
               "r--", linewidth=2, alpha=0.8)
        
        ax.set_xlabel(f'{lag_col}', fontweight='bold')
        ax.set_ylabel(f'{target}', fontweight='bold')
        ax.set_title(f'{lag_col} vs Target\nCorrelation: {corr:.3f}', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    save_figure('method_06_lag_features.png', 'Method 6: Lag Features - Past Values Predict Future')


def method_7_rolling_features(df_before, df_after, config):
    """Step 7: Rolling Features - Show rolling statistics visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    target = config.target_col
    rolling_cols = [c for c in df_after.columns if 'rolling' in c.lower()]
    
    # Sample one entity's time series
    sample_entity = df_after.groupby(config.groupby_cols).size().index[0]
    mask = (df_after[config.groupby_cols[0]] == sample_entity[0]) & \
           (df_after[config.groupby_cols[1]] == sample_entity[1])
    sample_ts = df_after[mask].sort_values(config.date_col).head(50)
    
    if len(sample_ts) > 10:
        # Plot 1: Original target vs rolling mean
        ax1 = axes[0, 0]
        ax1.plot(range(len(sample_ts)), sample_ts[target].values, 'o-', 
                label='Original', color='#3498db', alpha=0.6)
        rolling_mean_col = [c for c in rolling_cols if 'mean' in c][0] if rolling_cols else None
        if rolling_mean_col and rolling_mean_col in sample_ts.columns:
            ax1.plot(range(len(sample_ts)), sample_ts[rolling_mean_col].values, 's-', 
                    label='Rolling Mean', color='#e74c3c', linewidth=2)
        ax1.set_xlabel('Time Index', fontweight='bold')
        ax1.set_ylabel('Value', fontweight='bold')
        ax1.set_title('Original vs Rolling Mean', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Rolling std (volatility)
        ax2 = axes[0, 1]
        rolling_std_col = [c for c in rolling_cols if 'std' in c][0] if rolling_cols else None
        if rolling_std_col and rolling_std_col in sample_ts.columns:
            ax2.plot(range(len(sample_ts)), sample_ts[rolling_std_col].values, 'o-', 
                    color='#2ecc71', linewidth=2)
            ax2.set_xlabel('Time Index', fontweight='bold')
            ax2.set_ylabel('Rolling Std', fontweight='bold')
            ax2.set_title('Rolling Standard Deviation (Volatility)', fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Rolling min and max
        ax3 = axes[1, 0]
        rolling_min_col = [c for c in rolling_cols if 'min' in c][0] if rolling_cols else None
        rolling_max_col = [c for c in rolling_cols if 'max' in c][0] if rolling_cols else None
        if rolling_min_col and rolling_max_col:
            ax3.fill_between(range(len(sample_ts)), 
                           sample_ts[rolling_min_col].values, 
                           sample_ts[rolling_max_col].values, 
                           alpha=0.3, color='#9b59b6', label='Rolling Range')
            ax3.plot(range(len(sample_ts)), sample_ts[target].values, 'o-', 
                    color='#3498db', label='Original', alpha=0.7)
            ax3.set_xlabel('Time Index', fontweight='bold')
            ax3.set_ylabel('Value', fontweight='bold')
            ax3.set_title('Rolling Min/Max Range', fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
    
    # Plot 4: List of rolling features
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    feature_text = f"New Rolling Features Created: {len(rolling_cols)}\n\n"
    for i, feat in enumerate(sorted(rolling_cols)[:10], 1):
        feature_text += f"{i}. {feat}\n"
    if len(rolling_cols) > 10:
        feature_text += f"... and {len(rolling_cols) - 10} more"
    
    ax4.text(0.1, 0.9, feature_text, transform=ax4.transAxes, 
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    save_figure('method_07_rolling_features.png', 'Method 7: Rolling Features - Smooth Trends and Capture Volatility')


def method_8_external_features(df_before, df_after, config):
    """Step 8: External Features - Show external feature impact."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    external_feature_cols = [c for c in df_after.columns if c not in df_before.columns and 
                            any(ext in c for ext in (config.external_cols or []))]
    
    # Plot correlations with target
    target = config.target_col
    
    # Get base external features (non-lagged)
    base_external = [c for c in config.external_cols if c in df_after.columns] if config.external_cols else []
    
    # Plot 1-3: Show relationship between external features and target
    for idx, ext_col in enumerate(base_external[:3]):
        if ext_col in df_after.columns:
            ax = axes[idx // 2, idx % 2]
            valid_data = df_after[[target, ext_col]].dropna()
            sample_size = min(1000, len(valid_data))
            if sample_size == 0:
                continue
            sample_data = valid_data.sample(sample_size, random_state=42) if len(valid_data) > 0 else valid_data
            
            ax.scatter(sample_data[ext_col], sample_data[target], alpha=0.3, s=20, color='#e74c3c')
            corr = sample_data[target].corr(sample_data[ext_col])
            
            ax.set_xlabel(ext_col, fontweight='bold')
            ax.set_ylabel(target, fontweight='bold')
            ax.set_title(f'{ext_col} vs Sales\nCorrelation: {corr:.3f}', fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    # Plot 4: List of external features
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    feature_text = f"Base External Features: {len(base_external)}\n"
    for feat in base_external:
        feature_text += f"• {feat}\n"
    feature_text += f"\nTotal with Lags: {len(external_feature_cols)}\n"
    feature_text += "(Each feature gets lag_1, lag_2, lag_4)"
    
    ax4.text(0.1, 0.9, feature_text, transform=ax4.transAxes, 
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    save_figure('method_08_external_features.png', 'Method 8: External Features - Weather, Promotions, Stock')


def method_9_encode_categoricals(df_before, df_after, encoders):
    """Step 9: Encode Categoricals - Show encoding transformation."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Show encoding for first few categorical columns
    cat_cols = list(encoders.keys())[:3] if encoders else []
    
    for idx, col in enumerate(cat_cols):
        ax = axes[idx // 2, idx % 2]
        
        # Before: show unique values
        unique_before = df_before[col].value_counts().head(10)
        
        # After: show encoded values
        unique_after = df_after[col].value_counts().head(10)
        
        # Plot side by side
        x = np.arange(len(unique_before))
        width = 0.35
        
        ax.barh(x - width/2, unique_before.values, width, label='Count', alpha=0.7, color='#3498db')
        ax.set_yticks(x)
        ax.set_yticklabels([f'{idx}: {val}' for idx, val in enumerate(unique_before.index[:10])], fontsize=9)
        ax.set_xlabel('Frequency', fontweight='bold')
        ax.set_title(f'{col}: Encoded to Numeric', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    
    # Plot 4: Encoding summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    encoding_text = f"Categorical Encoding Summary:\n\n"
    encoding_text += f"Total Columns Encoded: {len(encoders)}\n\n"
    for col, mapping in list(encoders.items())[:5]:
        encoding_text += f"• {col}: {len(mapping)} unique values\n"
    if len(encoders) > 5:
        encoding_text += f"... and {len(encoders) - 5} more\n"
    encoding_text += "\nMethod: Label Encoding\nFit on: Training data only\nUnknown values → -1"
    
    ax4.text(0.1, 0.9, encoding_text, transform=ax4.transAxes, 
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))
    
    save_figure('method_09_encode_categoricals.png', 'Method 9: Encode Categoricals - Convert to Numeric')


def method_10_cap_outliers(df_before, df_after, bounds):
    """Step 10: Cap Outliers - Show outlier capping effect."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Get numeric columns that were capped
    capped_cols = list(bounds.keys())[:4] if bounds else []
    
    for idx, col in enumerate(capped_cols):
        if col in df_before.columns and col in df_after.columns:
            ax = axes[idx // 2, idx % 2]
            
            # Create box plots
            data_to_plot = [df_before[col].dropna(), df_after[col].dropna()]
            bp = ax.boxplot(data_to_plot, labels=['Before', 'After'], patch_artist=True)
            
            # Color boxes
            bp['boxes'][0].set_facecolor('#e74c3c')
            bp['boxes'][0].set_alpha(0.7)
            bp['boxes'][1].set_facecolor('#2ecc71')
            bp['boxes'][1].set_alpha(0.7)
            
            ax.set_ylabel(col, fontweight='bold')
            ax.set_title(f'Outlier Capping: {col}', fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Add bounds text
            lower, upper = bounds[col]
            ax.text(0.02, 0.98, f'Bounds: [{lower:.2f}, {upper:.2f}]', 
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    save_figure('method_10_cap_outliers.png', 'Method 10: Cap Outliers - Winsorize at Train Percentiles')


def method_11_scale_numerical(df_before, df_after, scaler, scale_cols):
    """Step 11: Scale Numerical - Show scaling effect."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Get first few columns that were scaled
    cols_to_plot = scale_cols[:4] if scale_cols else []
    
    for idx, col in enumerate(cols_to_plot):
        if col in df_before.columns and col in df_after.columns:
            ax = axes[idx // 2, idx % 2]
            
            # Plot distributions
            before_data = df_before[col].dropna()
            after_data = df_after[col].dropna()
            
            ax.hist(before_data, bins=50, alpha=0.6, label='Before (Original)', 
                   color='#e74c3c', density=True)
            ax.hist(after_data, bins=50, alpha=0.6, label='After (Scaled)', 
                   color='#2ecc71', density=True)
            
            ax.set_xlabel(col, fontweight='bold')
            ax.set_ylabel('Density', fontweight='bold')
            ax.set_title(f'Scaling: {col}', fontweight='bold')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Add statistics
            before_mean = before_data.mean()
            before_std = before_data.std()
            after_mean = after_data.mean()
            after_std = after_data.std()
            
            stats_text = f'Before: μ={before_mean:.2f}, σ={before_std:.2f}\nAfter: μ={after_mean:.2f}, σ={after_std:.2f}'
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=8, 
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    save_figure('method_11_scale_numerical.png', 'Method 11: Scale Numerical - Standardize to Mean=0, Std=1')


def main():
    """Generate all preprocessing explanation images."""
    print("="*80)
    print("GENERATING INDIVIDUAL PREPROCESSING METHOD EXPLANATIONS")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR.absolute()}")
    print("\nThis will create 11 separate, focused images - one for each method.\n")
    
    # Load data
    print("[1/3] Loading dataset...")
    train_df, eval_df = load_dataset_freshretail(max_train_rows=5000, max_eval_rows=1000)
    config = get_freshretail_config()
    
    print("[2/3] Processing data through all steps...")
    current_df = train_df.copy()
    
    # Step 1: Parse dates
    print("\n  → Method 1: Parse Dates...")
    before_1 = current_df.copy()
    current_df = step1_parse_dates(current_df, config.date_col)
    method_1_parse_dates(before_1, current_df, config)
    
    # Step 2: Select columns
    print("  → Method 2: Select Columns...")
    before_2 = current_df.copy()
    current_df = step2_select_columns(current_df, config.keep_columns, config.drop_columns)
    method_2_select_columns(before_2, current_df)
    
    # Step 3: Sort
    print("  → Method 3: Sort by Entity and Date...")
    before_3 = current_df.copy()
    current_df = step3_sort_by_entity_and_date(current_df, config.groupby_cols, config.date_col)
    method_3_sort_data(before_3, current_df, config)
    
    # Step 4: Impute
    print("  → Method 4: Impute Missing Values...")
    before_4 = current_df.copy()
    impute_stats = step4_impute_missing_fit(current_df)
    current_df = step4_impute_missing_transform(current_df, impute_stats)
    method_4_impute_missing(before_4, current_df, impute_stats)
    
    # Step 5: Temporal features
    print("  → Method 5: Add Temporal Features...")
    before_5 = current_df.copy()
    current_df = step5_add_temporal_features(current_df, config.date_col)
    method_5_temporal_features(before_5, current_df, config)
    
    # Step 6: Lag features
    print("  → Method 6: Add Lag Features...")
    before_6 = current_df.copy()
    current_df = step6_add_lag_features(current_df, config.target_col, config.lags, config.groupby_cols)
    method_6_lag_features(before_6, current_df, config)
    
    # Step 7: Rolling features
    print("  → Method 7: Add Rolling Features...")
    before_7 = current_df.copy()
    current_df = step7_add_rolling_features(current_df, config.target_col, config.rolling_windows, config.groupby_cols)
    method_7_rolling_features(before_7, current_df, config)
    
    # Step 8: External features
    print("  → Method 8: Add External Features...")
    before_8 = current_df.copy()
    current_df = step8_add_external_features(current_df, config.external_cols, config.groupby_cols[0], use_freshretail=True)
    method_8_external_features(before_8, current_df, config)
    
    # Step 9: Encode categoricals
    print("  → Method 9: Encode Categoricals...")
    before_9 = current_df.copy()
    encoders = step9_encode_categoricals_fit(current_df, config.categorical_cols)
    current_df = step9_encode_categoricals_transform(current_df, encoders)
    method_9_encode_categoricals(before_9, current_df, encoders)
    
    # Step 10: Cap outliers
    print("  → Method 10: Cap Outliers...")
    before_10 = current_df.copy()
    numeric_cols = [c for c in current_df.columns 
                   if pd.api.types.is_numeric_dtype(current_df[c]) 
                   and c not in [config.target_col, config.date_col] + config.groupby_cols]
    cap_bounds = step10_cap_outliers_fit(current_df, numeric_cols, 
                                        config.cap_percentiles[0], config.cap_percentiles[1])
    current_df = step10_cap_outliers_transform(current_df, cap_bounds)
    method_10_cap_outliers(before_10, current_df, cap_bounds)
    
    # Step 11: Scale numerical
    print("  → Method 11: Scale Numerical...")
    before_11 = current_df.copy()
    exclude_cols = [config.target_col, config.date_col] + config.groupby_cols
    scaler, scale_cols = step11_scale_numerical_fit(current_df, exclude_cols=exclude_cols)
    current_df = step11_scale_numerical_transform(current_df, scaler, scale_cols)
    method_11_scale_numerical(before_11, current_df, scaler, scale_cols)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nGenerated 11 explanation images in: {OUTPUT_DIR.absolute()}")
    print("\nFiles created:")
    for i in range(1, 12):
        print(f"  • method_{i:02d}_*.png")


if __name__ == "__main__":
    main()

"""
Exploratory Data Analysis module.
Performs comprehensive EDA on the sales dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def check_missing_values(df, name="Dataset"):
    """
    Check and report missing values in the dataset.
    
    Args:
        df: DataFrame to check
        name: Name of the dataset for reporting
    """
    print(f"\n{'='*60}")
    print(f"Missing Values Analysis: {name}")
    print(f"{'='*60}")
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing Count': missing.values,
        'Missing Percentage': missing_pct.values
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    
    if len(missing_df) > 0:
        print(missing_df.to_string(index=False))
    else:
        print("No missing values found!")
    
    return missing_df


def detect_outliers(df, column='Weekly_Sales', method='IQR'):
    """
    Detect outliers in the specified column.
    
    Args:
        df: DataFrame
        column: Column name to check for outliers
        method: Method to use ('IQR' or 'Z-score')
    
    Returns:
        DataFrame with outlier information
    """
    print(f"\n{'='*60}")
    print(f"Outlier Detection: {column}")
    print(f"{'='*60}")
    
    if method == 'IQR':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        
        print(f"Q1: {Q1:.2f}")
        print(f"Q3: {Q3:.2f}")
        print(f"IQR: {IQR:.2f}")
        print(f"Lower bound: {lower_bound:.2f}")
        print(f"Upper bound: {upper_bound:.2f}")
        print(f"Number of outliers: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
        
    elif method == 'Z-score':
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        outliers = df[z_scores > 3]
        print(f"Number of outliers (|Z| > 3): {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
    
    return outliers


def plot_sales_trends(df, save_dir="results"):
    """
    Create comprehensive sales trend plots.
    
    Args:
        df: DataFrame with sales data
        save_dir: Directory to save plots
    """
    Path(save_dir).mkdir(exist_ok=True)
    
    # Set style
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        try:
            plt.style.use('seaborn-darkgrid')
        except:
            plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Overall weekly sales trend over time
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Aggregate sales by date
    daily_sales = df.groupby('Date')['Weekly_Sales'].sum().reset_index()
    
    # Plot 1: Overall trend
    axes[0, 0].plot(daily_sales['Date'], daily_sales['Weekly_Sales'], linewidth=2)
    axes[0, 0].set_title('Overall Weekly Sales Trend', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Total Weekly Sales')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Sales by store
    store_sales = df.groupby('Store')['Weekly_Sales'].sum().sort_values(ascending=False)
    axes[0, 1].bar(range(len(store_sales)), store_sales.values)
    axes[0, 1].set_title('Total Sales by Store', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Store')
    axes[0, 1].set_ylabel('Total Sales')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Sales by department
    dept_sales = df.groupby('Dept')['Weekly_Sales'].sum().sort_values(ascending=False).head(20)
    axes[1, 0].barh(range(len(dept_sales)), dept_sales.values)
    axes[1, 0].set_yticks(range(len(dept_sales)))
    axes[1, 0].set_yticklabels(dept_sales.index)
    axes[1, 0].set_title('Top 20 Departments by Sales', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Total Sales')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # Plot 4: Monthly sales pattern
    df['Month'] = df['Date'].dt.month
    monthly_sales = df.groupby('Month')['Weekly_Sales'].mean()
    axes[1, 1].plot(monthly_sales.index, monthly_sales.values, marker='o', linewidth=2, markersize=8)
    axes[1, 1].set_title('Average Monthly Sales Pattern', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('Average Weekly Sales')
    axes[1, 1].set_xticks(range(1, 13))
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/sales_trends_overview.png", dpi=300, bbox_inches='tight')
    print(f"\nSaved sales trends plot to {save_dir}/sales_trends_overview.png")
    plt.close()
    
    # 2. Holiday impact analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    holiday_sales = df.groupby('IsHoliday')['Weekly_Sales'].mean()
    axes[0].bar(['Non-Holiday', 'Holiday'], holiday_sales.values, color=['skyblue', 'coral'])
    axes[0].set_title('Average Sales: Holiday vs Non-Holiday', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Average Weekly Sales')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Sales distribution
    axes[1].hist(df[df['IsHoliday']==False]['Weekly_Sales'], bins=50, alpha=0.6, label='Non-Holiday', density=True)
    axes[1].hist(df[df['IsHoliday']==True]['Weekly_Sales'], bins=50, alpha=0.6, label='Holiday', density=True)
    axes[1].set_title('Sales Distribution: Holiday vs Non-Holiday', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Weekly Sales')
    axes[1].set_ylabel('Density')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/holiday_analysis.png", dpi=300, bbox_inches='tight')
    print(f"Saved holiday analysis plot to {save_dir}/holiday_analysis.png")
    plt.close()
    
    # 3. External variables correlation
    external_vars = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, var in enumerate(external_vars):
        if var in df.columns:
            axes[i].scatter(df[var], df['Weekly_Sales'], alpha=0.3, s=10)
            axes[i].set_title(f'Sales vs {var}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel(var)
            axes[i].set_ylabel('Weekly Sales')
            axes[i].grid(True, alpha=0.3)
            
            # Calculate correlation
            corr = df[[var, 'Weekly_Sales']].corr().iloc[0, 1]
            axes[i].text(0.05, 0.95, f'Corr: {corr:.3f}', transform=axes[i].transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/external_variables_analysis.png", dpi=300, bbox_inches='tight')
    print(f"Saved external variables analysis to {save_dir}/external_variables_analysis.png")
    plt.close()


def generate_eda_report(df, save_dir="results"):
    """
    Generate comprehensive EDA report.
    
    Args:
        df: DataFrame to analyze
        save_dir: Directory to save reports
    """
    Path(save_dir).mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS REPORT")
    print("="*60)
    
    # Basic statistics
    print(f"\nDataset Shape: {df.shape}")
    print(f"\nColumn Names: {list(df.columns)}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nBasic Statistics:\n{df.describe()}")
    
    # Missing values
    missing_df = check_missing_values(df)
    
    # Outliers
    outliers = detect_outliers(df, 'Weekly_Sales')
    
    # Sales trends
    plot_sales_trends(df, save_dir)
    
    print("\n" + "="*60)
    print("EDA Complete!")
    print("="*60)

"""
Comprehensive analysis of all 11 preprocessing steps with visualizations.
Shows before/after for each step, with statistics and explanations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Import our modules
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

# Set up visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

# Create output directory
OUTPUT_DIR = Path("preprocessing_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)


class PreprocessingAnalyzer:
    """Analyzes each preprocessing step with visualizations."""
    
    def __init__(self, train_df: pd.DataFrame, config: PreprocessorConfig):
        self.original_train = train_df.copy()
        self.config = config
        self.current_df = train_df.copy()
        self.step_results = {}
        
    def analyze_step(self, step_num: int, step_name: str, 
                     before_df: pd.DataFrame, after_df: pd.DataFrame,
                     metadata: Dict[str, Any] = None):
        """Analyze and visualize a preprocessing step."""
        print(f"\n{'='*80}")
        print(f"STEP {step_num}: {step_name}")
        print(f"{'='*80}")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # Store results
        result = {
            'step_num': step_num,
            'step_name': step_name,
            'before_shape': before_df.shape,
            'after_shape': after_df.shape,
            'columns_added': list(set(after_df.columns) - set(before_df.columns)),
            'columns_removed': list(set(before_df.columns) - set(after_df.columns)),
            'metadata': metadata or {}
        }
        
        # Plot 1: Data shape comparison
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_shape_comparison(ax1, before_df, after_df)
        
        # Plot 2: Column count comparison
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_column_comparison(ax2, before_df, after_df, result)
        
        # Plot 3: Missing values comparison
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_missing_values(ax3, before_df, after_df)
        
        # Plot 4: Data types comparison
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_dtype_comparison(ax4, before_df, after_df)
        
        # Plot 5: Numeric features statistics
        ax5 = fig.add_subplot(gs[1, 1:])
        self._plot_numeric_stats(ax5, before_df, after_df)
        
        # Plot 6: Target variable distribution (if available)
        ax6 = fig.add_subplot(gs[2, :])
        if self.config.target_col in after_df.columns:
            self._plot_target_distribution(ax6, before_df, after_df)
        else:
            ax6.text(0.5, 0.5, 'Target column not available in this step', 
                    ha='center', va='center', fontsize=12)
            ax6.axis('off')
        
        # Plot 7: Step-specific visualization
        ax7 = fig.add_subplot(gs[3, :])
        self._plot_step_specific(ax7, step_num, before_df, after_df, metadata)
        
        # Add title
        fig.suptitle(f'Step {step_num}: {step_name}\n' + 
                     f'Shape: {before_df.shape} → {after_df.shape}',
                     fontsize=16, fontweight='bold', y=0.995)
        
        # Save figure
        filename = OUTPUT_DIR / f"step_{step_num:02d}_{step_name.lower().replace(' ', '_')}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to {filename}")
        plt.close()
        
        # Print summary
        self._print_summary(result)
        
        self.step_results[step_num] = result
        return result
    
    def _plot_shape_comparison(self, ax, before_df, after_df):
        """Plot shape comparison bar chart."""
        metrics = ['Rows', 'Columns', 'Total Cells']
        before_vals = [before_df.shape[0], before_df.shape[1], 
                      before_df.shape[0] * before_df.shape[1]]
        after_vals = [after_df.shape[0], after_df.shape[1], 
                     after_df.shape[0] * after_df.shape[1]]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax.bar(x - width/2, before_vals, width, label='Before', alpha=0.8, color='#3498db')
        ax.bar(x + width/2, after_vals, width, label='After', alpha=0.8, color='#e74c3c')
        
        ax.set_xlabel('Metric', fontweight='bold')
        ax.set_ylabel('Count', fontweight='bold')
        ax.set_title('Data Shape Comparison', fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, (b, a) in enumerate(zip(before_vals, after_vals)):
            ax.text(i - width/2, b, f'{b:,}', ha='center', va='bottom', fontsize=8)
            ax.text(i + width/2, a, f'{a:,}', ha='center', va='bottom', fontsize=8)
    
    def _plot_column_comparison(self, ax, before_df, after_df, result):
        """Plot column changes."""
        added = len(result['columns_added'])
        removed = len(result['columns_removed'])
        unchanged = len(set(before_df.columns) & set(after_df.columns))
        
        sizes = [unchanged, added, removed]
        labels = [f'Unchanged\n({unchanged})', f'Added\n({added})', f'Removed\n({removed})']
        colors = ['#95a5a6', '#2ecc71', '#e74c3c']
        explode = (0, 0.1 if added > 0 else 0, 0.1 if removed > 0 else 0)
        
        if sum(sizes) > 0:
            ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                  autopct='%1.1f%%', startangle=90, textprops={'fontsize': 9})
            ax.set_title('Column Changes', fontweight='bold', pad=10)
        else:
            ax.text(0.5, 0.5, 'No columns', ha='center', va='center')
            ax.axis('off')
    
    def _plot_missing_values(self, ax, before_df, after_df):
        """Plot missing values comparison."""
        before_missing = before_df.isnull().sum().sum()
        after_missing = after_df.isnull().sum().sum()
        before_pct = (before_missing / (before_df.shape[0] * before_df.shape[1])) * 100
        after_pct = (after_missing / (after_df.shape[0] * after_df.shape[1])) * 100
        
        categories = ['Before', 'After']
        missing_counts = [before_missing, after_missing]
        missing_pcts = [before_pct, after_pct]
        
        x = np.arange(len(categories))
        bars = ax.bar(x, missing_counts, color=['#3498db', '#e74c3c'], alpha=0.8)
        
        ax.set_ylabel('Missing Values Count', fontweight='bold')
        ax.set_title('Missing Values Comparison', fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.grid(axis='y', alpha=0.3)
        
        # Add labels with counts and percentages
        for i, (bar, count, pct) in enumerate(zip(bars, missing_counts, missing_pcts)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count:,}\n({pct:.2f}%)',
                   ha='center', va='bottom', fontsize=9)
    
    def _plot_dtype_comparison(self, ax, before_df, after_df):
        """Plot data types comparison."""
        def get_dtype_counts(df):
            dtype_map = {
                'int': 0, 'float': 0, 'object': 0, 'datetime': 0, 'bool': 0, 'other': 0
            }
            for dtype in df.dtypes:
                if pd.api.types.is_integer_dtype(dtype):
                    dtype_map['int'] += 1
                elif pd.api.types.is_float_dtype(dtype):
                    dtype_map['float'] += 1
                elif pd.api.types.is_object_dtype(dtype):
                    dtype_map['object'] += 1
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    dtype_map['datetime'] += 1
                elif pd.api.types.is_bool_dtype(dtype):
                    dtype_map['bool'] += 1
                else:
                    dtype_map['other'] += 1
            return dtype_map
        
        before_dtypes = get_dtype_counts(before_df)
        after_dtypes = get_dtype_counts(after_df)
        
        dtype_names = list(before_dtypes.keys())
        x = np.arange(len(dtype_names))
        width = 0.35
        
        before_vals = [before_dtypes[k] for k in dtype_names]
        after_vals = [after_dtypes[k] for k in dtype_names]
        
        ax.bar(x - width/2, before_vals, width, label='Before', alpha=0.8, color='#3498db')
        ax.bar(x + width/2, after_vals, width, label='After', alpha=0.8, color='#e74c3c')
        
        ax.set_xlabel('Data Type', fontweight='bold')
        ax.set_ylabel('Count', fontweight='bold')
        ax.set_title('Data Type Distribution', fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(dtype_names)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_numeric_stats(self, ax, before_df, after_df):
        """Plot numeric features statistics comparison."""
        # Get numeric columns that exist in both
        before_numeric = before_df.select_dtypes(include=[np.number]).columns
        after_numeric = after_df.select_dtypes(include=[np.number]).columns
        common_numeric = list(set(before_numeric) & set(after_numeric))
        
        if len(common_numeric) == 0:
            ax.text(0.5, 0.5, 'No common numeric columns', ha='center', va='center')
            ax.axis('off')
            return
        
        # Take first 5 numeric columns for comparison
        cols_to_plot = common_numeric[:5]
        
        stats_data = []
        for col in cols_to_plot:
            before_mean = before_df[col].mean()
            after_mean = after_df[col].mean()
            before_std = before_df[col].std()
            after_std = after_df[col].std()
            
            stats_data.append({
                'Column': col[:20],  # Truncate long names
                'Before Mean': before_mean,
                'After Mean': after_mean,
                'Before Std': before_std,
                'After Std': after_std,
            })
        
        stats_df = pd.DataFrame(stats_data)
        
        # Plot grouped bar chart
        x = np.arange(len(stats_df))
        width = 0.2
        
        ax.bar(x - width*1.5, stats_df['Before Mean'], width, 
              label='Before Mean', alpha=0.8, color='#3498db')
        ax.bar(x - width*0.5, stats_df['After Mean'], width, 
              label='After Mean', alpha=0.8, color='#e74c3c')
        ax.bar(x + width*0.5, stats_df['Before Std'], width, 
              label='Before Std', alpha=0.8, color='#9b59b6')
        ax.bar(x + width*1.5, stats_df['After Std'], width, 
              label='After Std', alpha=0.8, color='#f39c12')
        
        ax.set_xlabel('Column', fontweight='bold')
        ax.set_ylabel('Value', fontweight='bold')
        ax.set_title('Numeric Statistics Comparison (Mean & Std)', fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(stats_df['Column'], rotation=45, ha='right')
        ax.legend(ncol=2, loc='upper left')
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_target_distribution(self, ax, before_df, after_df):
        """Plot target variable distribution."""
        target = self.config.target_col
        
        if target not in before_df.columns and target not in after_df.columns:
            ax.text(0.5, 0.5, f'Target column "{target}" not found', 
                   ha='center', va='center')
            ax.axis('off')
            return
        
        # Create subplots for before and after distributions
        if target in before_df.columns:
            before_data = before_df[target].dropna()
            ax.hist(before_data, bins=50, alpha=0.6, label='Before', 
                   color='#3498db', density=True)
        
        if target in after_df.columns:
            after_data = after_df[target].dropna()
            ax.hist(after_data, bins=50, alpha=0.6, label='After', 
                   color='#e74c3c', density=True)
        
        ax.set_xlabel('Target Value', fontweight='bold')
        ax.set_ylabel('Density', fontweight='bold')
        ax.set_title(f'Target Variable Distribution: {target}', fontweight='bold', pad=10)
        ax.legend()
        ax.grid(alpha=0.3)
    
    def _plot_step_specific(self, ax, step_num, before_df, after_df, metadata):
        """Create step-specific visualizations."""
        
        if step_num == 1:  # Parse dates
            # Show date parsing success
            date_col = self.config.date_col
            if date_col in after_df.columns:
                before_type = str(before_df[date_col].dtype) if date_col in before_df.columns else 'N/A'
                after_type = str(after_df[date_col].dtype)
                ax.text(0.5, 0.6, f'Date Column: {date_col}', 
                       ha='center', va='center', fontsize=14, fontweight='bold')
                ax.text(0.5, 0.4, f'Before: {before_type}\nAfter: {after_type}', 
                       ha='center', va='center', fontsize=12)
                if pd.api.types.is_datetime64_any_dtype(after_df[date_col]):
                    date_range = f"{after_df[date_col].min()} to {after_df[date_col].max()}"
                    ax.text(0.5, 0.2, f'Date Range: {date_range}', 
                           ha='center', va='center', fontsize=10)
            ax.axis('off')
        
        elif step_num == 2:  # Select columns
            # Show which columns were kept/dropped
            added = metadata.get('columns_added', []) if metadata else []
            removed = metadata.get('columns_removed', []) if metadata else []
            
            text = "Columns Selection:\n\n"
            if removed:
                text += f"Dropped: {', '.join(removed[:5])}"
                if len(removed) > 5:
                    text += f"... (+{len(removed)-5} more)"
            else:
                text += "No columns dropped"
            
            ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=11)
            ax.axis('off')
        
        elif step_num == 3:  # Sort
            # Show sorting impact
            date_col = self.config.date_col
            if date_col in after_df.columns:
                is_sorted = after_df[date_col].is_monotonic_increasing
                ax.text(0.5, 0.5, 
                       f'Data sorted by: {", ".join(self.config.groupby_cols + [date_col])}\n\n'
                       f'Date column monotonic: {is_sorted}',
                       ha='center', va='center', fontsize=12)
            ax.axis('off')
        
        elif step_num == 4:  # Impute missing
            # Show imputation statistics
            if metadata:
                impute_stats = metadata.get('impute_stats', {})
                text = f"Imputed {len(impute_stats)} columns\n\n"
                for col, (kind, val) in list(impute_stats.items())[:5]:
                    if isinstance(val, (int, float)):
                        text += f"{col}: {kind} = {val:.2f}\n"
                    else:
                        text += f"{col}: {kind} = {val}\n"
                ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=10, 
                       family='monospace')
            ax.axis('off')
        
        elif step_num == 5:  # Temporal features
            # Show temporal features added
            temporal_cols = [c for c in after_df.columns 
                           if any(x in c.lower() for x in ['year', 'month', 'week', 'day', 'quarter'])]
            new_temporal = [c for c in temporal_cols if c not in before_df.columns]
            
            if new_temporal:
                # Plot distribution of one temporal feature
                if 'month' in after_df.columns:
                    month_counts = after_df['month'].value_counts().sort_index()
                    ax.bar(month_counts.index, month_counts.values, color='#3498db', alpha=0.7)
                    ax.set_xlabel('Month', fontweight='bold')
                    ax.set_ylabel('Count', fontweight='bold')
                    ax.set_title('Month Distribution in Data', fontweight='bold')
                    ax.grid(axis='y', alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 
                           f'Added temporal features:\n{", ".join(new_temporal)}',
                           ha='center', va='center', fontsize=11)
                    ax.axis('off')
            else:
                ax.text(0.5, 0.5, 'No temporal features added', 
                       ha='center', va='center', fontsize=11)
                ax.axis('off')
        
        elif step_num == 6:  # Lag features
            # Show lag features distribution
            lag_cols = [c for c in after_df.columns if 'lag' in c.lower()]
            new_lags = [c for c in lag_cols if c not in before_df.columns]
            
            if new_lags and len(new_lags) > 0:
                # Plot correlation between target and first lag
                target = self.config.target_col
                if target in after_df.columns and new_lags[0] in after_df.columns:
                    valid_data = after_df[[target, new_lags[0]]].dropna()
                    if len(valid_data) > 10:
                        ax.scatter(valid_data[new_lags[0]], valid_data[target], 
                                 alpha=0.3, s=10, color='#3498db')
                        ax.set_xlabel(new_lags[0], fontweight='bold')
                        ax.set_ylabel(target, fontweight='bold')
                        corr = valid_data[target].corr(valid_data[new_lags[0]])
                        ax.set_title(f'Target vs {new_lags[0]} (corr={corr:.3f})', 
                                   fontweight='bold')
                        ax.grid(alpha=0.3)
                    else:
                        ax.text(0.5, 0.5, f'Added {len(new_lags)} lag features:\n' + 
                               ', '.join(new_lags[:5]), ha='center', va='center', fontsize=10)
                        ax.axis('off')
                else:
                    ax.text(0.5, 0.5, f'Added {len(new_lags)} lag features', 
                           ha='center', va='center', fontsize=11)
                    ax.axis('off')
            else:
                ax.text(0.5, 0.5, 'No lag features added', 
                       ha='center', va='center', fontsize=11)
                ax.axis('off')
        
        elif step_num == 7:  # Rolling features
            # Show rolling features
            rolling_cols = [c for c in after_df.columns 
                          if any(x in c.lower() for x in ['rolling', 'mean', 'std'])]
            new_rolling = [c for c in rolling_cols if c not in before_df.columns]
            
            if new_rolling:
                ax.text(0.5, 0.5, 
                       f'Added {len(new_rolling)} rolling features:\n' + 
                       ', '.join([c[:30] for c in new_rolling[:8]]),
                       ha='center', va='center', fontsize=10)
            else:
                ax.text(0.5, 0.5, 'No rolling features added', 
                       ha='center', va='center', fontsize=11)
            ax.axis('off')
        
        elif step_num == 8:  # External features
            # Show external feature lags
            if self.config.external_cols:
                external_feature_cols = [c for c in after_df.columns 
                                       if any(ext in c for ext in self.config.external_cols)]
                new_external = [c for c in external_feature_cols if c not in before_df.columns]
                
                text = f'External features tracked: {len(self.config.external_cols)}\n'
                text += f'New feature columns created: {len(new_external)}\n\n'
                text += f'Base features: {", ".join(self.config.external_cols[:5])}'
                ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=10)
            else:
                ax.text(0.5, 0.5, 'No external features configured', 
                       ha='center', va='center', fontsize=11)
            ax.axis('off')
        
        elif step_num == 9:  # Encode categoricals
            # Show encoding information
            if metadata and 'encoders' in metadata:
                encoders = metadata['encoders']
                text = f'Encoded {len(encoders)} categorical columns:\n\n'
                for col, mapping in list(encoders.items())[:5]:
                    text += f'{col}: {len(mapping)} unique values\n'
                ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=10, 
                       family='monospace')
            else:
                ax.text(0.5, 0.5, 'No categorical encoding performed', 
                       ha='center', va='center', fontsize=11)
            ax.axis('off')
        
        elif step_num == 10:  # Cap outliers
            # Show outlier capping effect
            if metadata and 'cap_bounds' in metadata:
                bounds = metadata['cap_bounds']
                if bounds and len(bounds) > 0:
                    # Show capping for first numeric column
                    col = list(bounds.keys())[0]
                    if col in before_df.columns and col in after_df.columns:
                        before_vals = before_df[col].dropna()
                        after_vals = after_df[col].dropna()
                        
                        ax.boxplot([before_vals, after_vals], labels=['Before', 'After'])
                        ax.set_ylabel(col, fontweight='bold')
                        ax.set_title(f'Outlier Capping Effect: {col}', fontweight='bold')
                        ax.grid(axis='y', alpha=0.3)
                    else:
                        ax.text(0.5, 0.5, f'Capped {len(bounds)} columns', 
                               ha='center', va='center', fontsize=11)
                        ax.axis('off')
                else:
                    ax.text(0.5, 0.5, 'No outlier capping performed', 
                           ha='center', va='center', fontsize=11)
                    ax.axis('off')
            else:
                ax.text(0.5, 0.5, 'No outlier capping metadata', 
                       ha='center', va='center', fontsize=11)
                ax.axis('off')
        
        elif step_num == 11:  # Scale numerical
            # Show scaling effect
            if metadata and 'scale_cols' in metadata:
                scale_cols = metadata['scale_cols']
                if scale_cols and len(scale_cols) > 0:
                    col = scale_cols[0]
                    if col in before_df.columns and col in after_df.columns:
                        before_vals = before_df[col].dropna()
                        after_vals = after_df[col].dropna()
                        
                        ax.hist(before_vals, bins=50, alpha=0.5, label='Before', 
                               color='#3498db', density=True)
                        ax.hist(after_vals, bins=50, alpha=0.5, label='After', 
                               color='#e74c3c', density=True)
                        ax.set_xlabel(col, fontweight='bold')
                        ax.set_ylabel('Density', fontweight='bold')
                        ax.set_title(f'Scaling Effect: {col}', fontweight='bold')
                        ax.legend()
                        ax.grid(alpha=0.3)
                    else:
                        ax.text(0.5, 0.5, f'Scaled {len(scale_cols)} columns', 
                               ha='center', va='center', fontsize=11)
                        ax.axis('off')
                else:
                    ax.text(0.5, 0.5, 'No numerical scaling performed', 
                           ha='center', va='center', fontsize=11)
                    ax.axis('off')
            else:
                ax.text(0.5, 0.5, 'No scaling metadata', 
                       ha='center', va='center', fontsize=11)
                ax.axis('off')
    
    def _print_summary(self, result):
        """Print text summary of the step."""
        print(f"\nShape: {result['before_shape']} → {result['after_shape']}")
        
        if result['columns_added']:
            print(f"Columns Added ({len(result['columns_added'])}): " + 
                  f"{', '.join(result['columns_added'][:5])}")
            if len(result['columns_added']) > 5:
                print(f"  ... and {len(result['columns_added'])-5} more")
        
        if result['columns_removed']:
            print(f"Columns Removed ({len(result['columns_removed'])}): " + 
                  f"{', '.join(result['columns_removed'][:5])}")
        
        if result['metadata']:
            print(f"Metadata: {list(result['metadata'].keys())}")


def main():
    """Run comprehensive analysis of all 11 preprocessing steps."""
    
    print("="*80)
    print("COMPREHENSIVE PREPROCESSING ANALYSIS")
    print("="*80)
    print("\nThis script will:")
    print("  1. Load FreshRetailNet-50K dataset")
    print("  2. Apply each preprocessing step sequentially")
    print("  3. Generate visualizations showing before/after for each step")
    print("  4. Explain what each step does and why it's important")
    print(f"\nOutput directory: {OUTPUT_DIR.absolute()}")
    print("="*80)
    
    # Load dataset (sample for faster processing)
    print("\n[1/3] Loading dataset...")
    train_df, eval_df = load_dataset_freshretail(max_train_rows=10000, max_eval_rows=2000)
    
    # Get configuration
    config = get_freshretail_config()
    
    # Initialize analyzer
    print("\n[2/3] Initializing analyzer...")
    analyzer = PreprocessingAnalyzer(train_df, config)
    
    # Create a cumulative dataframe that we'll transform step by step
    current_df = train_df.copy()
    
    print("\n[3/3] Analyzing each preprocessing step...")
    print("="*80)
    
    # STEP 1: Parse dates
    before_1 = current_df.copy()
    current_df = step1_parse_dates(current_df, config.date_col)
    analyzer.analyze_step(1, "Parse Dates", before_1, current_df)
    
    # STEP 2: Select columns
    before_2 = current_df.copy()
    current_df = step2_select_columns(current_df, config.keep_columns, config.drop_columns)
    analyzer.analyze_step(2, "Select Columns", before_2, current_df)
    
    # STEP 3: Sort by entity and date
    before_3 = current_df.copy()
    current_df = step3_sort_by_entity_and_date(current_df, config.groupby_cols, config.date_col)
    analyzer.analyze_step(3, "Sort by Entity and Date", before_3, current_df)
    
    # STEP 4: Impute missing
    before_4 = current_df.copy()
    impute_stats = step4_impute_missing_fit(current_df)
    current_df = step4_impute_missing_transform(current_df, impute_stats)
    analyzer.analyze_step(4, "Impute Missing Values", before_4, current_df, 
                         metadata={'impute_stats': impute_stats})
    
    # STEP 5: Temporal features
    before_5 = current_df.copy()
    current_df = step5_add_temporal_features(current_df, config.date_col)
    analyzer.analyze_step(5, "Add Temporal Features", before_5, current_df)
    
    # STEP 6: Lag features
    before_6 = current_df.copy()
    current_df = step6_add_lag_features(current_df, config.target_col, 
                                        config.lags, config.groupby_cols)
    analyzer.analyze_step(6, "Add Lag Features", before_6, current_df)
    
    # STEP 7: Rolling features
    before_7 = current_df.copy()
    current_df = step7_add_rolling_features(current_df, config.target_col, 
                                            config.rolling_windows, config.groupby_cols)
    analyzer.analyze_step(7, "Add Rolling Features", before_7, current_df)
    
    # STEP 8: External features
    before_8 = current_df.copy()
    current_df = step8_add_external_features(current_df, config.external_cols, 
                                             config.groupby_cols[0], use_freshretail=True)
    analyzer.analyze_step(8, "Add External Features", before_8, current_df)
    
    # STEP 9: Encode categoricals
    before_9 = current_df.copy()
    encoders = step9_encode_categoricals_fit(current_df, config.categorical_cols)
    current_df = step9_encode_categoricals_transform(current_df, encoders)
    analyzer.analyze_step(9, "Encode Categoricals", before_9, current_df, 
                         metadata={'encoders': encoders})
    
    # STEP 10: Cap outliers
    before_10 = current_df.copy()
    numeric_cols = [c for c in current_df.columns 
                   if pd.api.types.is_numeric_dtype(current_df[c]) 
                   and c not in [config.target_col, config.date_col] + config.groupby_cols]
    cap_bounds = step10_cap_outliers_fit(current_df, numeric_cols, 
                                        config.cap_percentiles[0], 
                                        config.cap_percentiles[1])
    current_df = step10_cap_outliers_transform(current_df, cap_bounds)
    analyzer.analyze_step(10, "Cap Outliers", before_10, current_df, 
                         metadata={'cap_bounds': cap_bounds})
    
    # STEP 11: Scale numerical
    before_11 = current_df.copy()
    exclude_cols = [config.target_col, config.date_col] + config.groupby_cols
    scaler, scale_cols = step11_scale_numerical_fit(current_df, exclude_cols=exclude_cols)
    current_df = step11_scale_numerical_transform(current_df, scaler, scale_cols)
    analyzer.analyze_step(11, "Scale Numerical", before_11, current_df, 
                         metadata={'scaler': scaler, 'scale_cols': scale_cols})
    
    # Create summary report
    print("\n" + "="*80)
    print("CREATING SUMMARY REPORT")
    print("="*80)
    create_summary_report(analyzer, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll visualizations saved to: {OUTPUT_DIR.absolute()}")
    print(f"Total files created: {len(list(OUTPUT_DIR.glob('*.png')))}")


def create_summary_report(analyzer: PreprocessingAnalyzer, output_dir: Path):
    """Create a comprehensive summary report."""
    
    # Create summary figure
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
    
    # Plot 1: Overall data transformation
    ax1 = fig.add_subplot(gs[0, :])
    steps = [f"Step {r['step_num']}" for r in analyzer.step_results.values()]
    rows = [r['after_shape'][0] for r in analyzer.step_results.values()]
    cols = [r['after_shape'][1] for r in analyzer.step_results.values()]
    
    x = np.arange(len(steps))
    ax1.plot(x, rows, marker='o', linewidth=2, markersize=8, 
            label='Rows', color='#3498db')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(x, cols, marker='s', linewidth=2, markersize=8, 
                 label='Columns', color='#e74c3c')
    
    ax1.set_xlabel('Preprocessing Step', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Number of Rows', fontweight='bold', fontsize=12, color='#3498db')
    ax1_twin.set_ylabel('Number of Columns', fontweight='bold', fontsize=12, color='#e74c3c')
    ax1.set_title('Data Transformation Across All Steps', fontweight='bold', fontsize=14, pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(steps, rotation=45, ha='right')
    ax1.tick_params(axis='y', labelcolor='#3498db')
    ax1_twin.tick_params(axis='y', labelcolor='#e74c3c')
    ax1.grid(alpha=0.3)
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    
    # Plot 2: Columns added per step
    ax2 = fig.add_subplot(gs[1, 0])
    steps_short = [f"{r['step_num']}" for r in analyzer.step_results.values()]
    cols_added = [len(r['columns_added']) for r in analyzer.step_results.values()]
    colors = ['#2ecc71' if c > 0 else '#95a5a6' for c in cols_added]
    ax2.bar(steps_short, cols_added, color=colors, alpha=0.8)
    ax2.set_xlabel('Step', fontweight='bold')
    ax2.set_ylabel('Columns Added', fontweight='bold')
    ax2.set_title('Feature Engineering Impact', fontweight='bold', pad=10)
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Processing complexity
    ax3 = fig.add_subplot(gs[1, 1])
    complexity_scores = [
        1, 1, 1,  # Steps 1-3: simple
        3,        # Step 4: imputation (fit-transform)
        2,        # Step 5: temporal
        4,        # Step 6: lags
        4,        # Step 7: rolling
        3,        # Step 8: external
        3,        # Step 9: encoding (fit-transform)
        3,        # Step 10: capping (fit-transform)
        3,        # Step 11: scaling (fit-transform)
    ]
    colors_complexity = ['#2ecc71' if c <= 2 else '#f39c12' if c <= 3 else '#e74c3c' 
                        for c in complexity_scores]
    ax3.bar(steps_short, complexity_scores, color=colors_complexity, alpha=0.8)
    ax3.set_xlabel('Step', fontweight='bold')
    ax3.set_ylabel('Complexity Score', fontweight='bold')
    ax3.set_title('Processing Complexity (1=Simple, 5=Complex)', fontweight='bold', pad=10)
    ax3.set_ylim(0, 5)
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Data leakage risk
    ax4 = fig.add_subplot(gs[1, 2])
    leakage_risk = [
        0, 0, 0,  # Steps 1-3: no risk
        1,        # Step 4: fit on train only
        0,        # Step 5: no risk
        0,        # Step 6: no risk (past only)
        0,        # Step 7: no risk (past only)
        0,        # Step 8: no risk
        1,        # Step 9: fit on train only
        1,        # Step 10: fit on train only
        1,        # Step 11: fit on train only
    ]
    colors_risk = ['#2ecc71' if r == 0 else '#f39c12' for r in leakage_risk]
    ax4.bar(steps_short, leakage_risk, color=colors_risk, alpha=0.8)
    ax4.set_xlabel('Step', fontweight='bold')
    ax4.set_ylabel('Leakage Prevention Required', fontweight='bold')
    ax4.set_title('Data Leakage Risk Management', fontweight='bold', pad=10)
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['No Fit', 'Fit on Train'])
    ax4.grid(axis='y', alpha=0.3)
    
    # Plot 5: Purpose of each step (text summary)
    ax5 = fig.add_subplot(gs[2:, :])
    ax5.axis('off')
    
    purposes = [
        "1. Parse Dates: Convert date strings to datetime objects for temporal operations",
        "2. Select Columns: Remove unnecessary columns to reduce memory and computation",
        "3. Sort Data: Order by entity and date for correct lag/rolling calculations",
        "4. Impute Missing: Fill missing values using train statistics (median/mode)",
        "5. Temporal Features: Extract time-based features (year, month, day, etc.)",
        "6. Lag Features: Add past values of target variable for temporal patterns",
        "7. Rolling Features: Add rolling statistics (mean, std) over windows",
        "8. External Features: Incorporate external covariates and their lags",
        "9. Encode Categoricals: Convert categorical variables to numeric (label encoding)",
        "10. Cap Outliers: Winsorize extreme values using train percentiles",
        "11. Scale Numerical: Standardize numerical features (zero mean, unit variance)",
    ]
    
    summary_text = "PREPROCESSING STEPS SUMMARY\n" + "="*90 + "\n\n"
    summary_text += "\n".join(purposes)
    summary_text += "\n\n" + "="*90 + "\n"
    summary_text += "KEY PRINCIPLES:\n"
    summary_text += "  • No Data Leakage: Fit operations use training data only\n"
    summary_text += "  • Temporal Integrity: Lag/rolling features use past values only\n"
    summary_text += "  • Feature Engineering: Create informative features from raw data\n"
    summary_text += "  • Standardization: Ensure consistent scales for model training\n"
    summary_text += "  • Efficiency: Remove unnecessary data, handle missing values\n"
    
    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, 
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    fig.suptitle('Preprocessing Pipeline: Complete Summary', 
                fontsize=18, fontweight='bold', y=0.995)
    
    # Save summary
    summary_file = output_dir / "00_complete_summary.png"
    plt.savefig(summary_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved summary report to {summary_file}")
    plt.close()
    
    # Create detailed text report
    report_file = output_dir / "preprocessing_report.txt"
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE PREPROCESSING ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        for step_num, result in analyzer.step_results.items():
            f.write(f"\nSTEP {step_num}: {result['step_name']}\n")
            f.write("-"*80 + "\n")
            f.write(f"Shape: {result['before_shape']} → {result['after_shape']}\n")
            f.write(f"Columns Added: {len(result['columns_added'])}\n")
            f.write(f"Columns Removed: {len(result['columns_removed'])}\n")
            
            if result['columns_added']:
                f.write(f"\nNew Columns:\n")
                for col in result['columns_added'][:10]:
                    f.write(f"  • {col}\n")
                if len(result['columns_added']) > 10:
                    f.write(f"  ... and {len(result['columns_added'])-10} more\n")
            
            f.write("\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"✓ Saved text report to {report_file}")


if __name__ == "__main__":
    main()

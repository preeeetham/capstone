# Preprocessing Analysis: 11-Step Pipeline Visualization

This directory contains comprehensive visualizations and analysis of all 11 preprocessing steps used in the FreshRetailNet-50K sales forecasting pipeline.

## 📊 Overview

Each preprocessing step is documented with:
- **Before/After comparisons**: Visual representation of data transformations
- **Statistical analysis**: Shape changes, missing values, data types
- **Step-specific insights**: Unique visualizations for each preprocessing method
- **Impact analysis**: How each step affects the data and why it's important

## 🗂️ Files in This Directory

### Summary Files
- **`00_complete_summary.png`**: Comprehensive overview of all 11 steps
  - Data transformation flow (rows and columns over steps)
  - Feature engineering impact (columns added per step)
  - Processing complexity scores
  - Data leakage risk management
  - Complete pipeline summary

- **`preprocessing_report.txt`**: Detailed text report
  - Shape changes for each step
  - Columns added and removed
  - New feature listings

### Individual Step Visualizations

Each step has a detailed visualization with 7+ plots:

1. **`step_01_parse_dates.png`** - Converting date strings to datetime objects
2. **`step_02_select_columns.png`** - Filtering relevant columns
3. **`step_03_sort_by_entity_and_date.png`** - Chronological ordering
4. **`step_04_impute_missing_values.png`** - Handling missing data
5. **`step_05_add_temporal_features.png`** - Extracting time-based features
6. **`step_06_add_lag_features.png`** - Creating autoregressive features
7. **`step_07_add_rolling_features.png`** - Computing rolling statistics
8. **`step_08_add_external_features.png`** - Incorporating external covariates
9. **`step_09_encode_categoricals.png`** - Converting categories to numeric
10. **`step_10_cap_outliers.png`** - Handling extreme values
11. **`step_11_scale_numerical.png`** - Standardizing feature scales

## 📈 Visualization Components

Each step visualization includes:

### 1. Data Shape Comparison
Bar chart showing rows, columns, and total cells before/after

### 2. Column Changes
Pie chart displaying unchanged, added, and removed columns

### 3. Missing Values Comparison
Bar chart showing missing value counts and percentages

### 4. Data Type Distribution
Comparison of int, float, object, datetime types

### 5. Numeric Statistics
Mean and standard deviation comparison for numeric features

### 6. Target Distribution
Histogram overlay showing target variable changes

### 7. Step-Specific Visualization
Custom visualization unique to each step:
- **Step 1**: Date type conversion details
- **Step 2**: Column selection summary
- **Step 3**: Sort order verification
- **Step 4**: Imputation statistics
- **Step 5**: Monthly distribution
- **Step 6**: Lag feature correlation
- **Step 7**: Rolling feature summary
- **Step 8**: External feature tracking
- **Step 9**: Encoding mappings
- **Step 10**: Outlier capping effect (boxplots)
- **Step 11**: Scaling effect (distribution shift)

## 🔍 Key Insights from Analysis

### Data Transformation Summary

| Step | Input Shape | Output Shape | Columns Added | Purpose |
|------|-------------|--------------|---------------|---------|
| 1. Parse Dates | (10000, 17) | (10000, 17) | 0 | Enable temporal operations |
| 2. Select Columns | (10000, 17) | (10000, 17) | 0 | Focus on relevant features |
| 3. Sort Data | (10000, 17) | (10000, 17) | 0 | Ensure chronological order |
| 4. Impute Missing | (10000, 17) | (10000, 17) | 0 | Fill NaN values |
| 5. Temporal Features | (10000, 17) | (10000, 30) | **13** | Extract date components |
| 6. Lag Features | (10000, 30) | (10000, 34) | **4** | Add past target values |
| 7. Rolling Features | (10000, 34) | (10000, 46) | **12** | Add rolling statistics |
| 8. External Features | (10000, 46) | (10000, 62) | **16** | Add covariate lags |
| 9. Encode Categoricals | (10000, 62) | (10000, 62) | 0 | Convert to numeric |
| 10. Cap Outliers | (10000, 62) | (10000, 62) | 0 | Winsorize extremes |
| 11. Scale Numerical | (10000, 62) | (10000, 62) | 0 | Standardize scales |
| **Final** | **(10000, 17)** | **(10000, 62)** | **+45 features** | **Ready for modeling** |

### Feature Engineering Breakdown

**Temporal Features (13)**: year, month, week, quarter, day_of_week, day_of_month, week_of_year, is_weekend, is_month_start, is_month_end, + sin/cos encodings

**Lag Features (4)**: sale_amount_lag_1, sale_amount_lag_2, sale_amount_lag_4, sale_amount_lag_7

**Rolling Features (12)**: 
- Windows: 4, 7, 14 days
- Statistics: mean, std, min, max
- Example: sale_amount_rolling_7_mean

**External Feature Lags (16)**:
- Base features: discount, holiday_flag, activity_flag, temperature, humidity, wind, precipitation, stock_count
- Lags: 1, 2, 4 days for each
- Example: discount_lag_1, temperature_lag_4

## 🎯 Step-by-Step Purpose

### Steps 1-3: Data Preparation (No Leakage Risk)
- **Parse dates**: Enable temporal operations
- **Select columns**: Remove unnecessary data
- **Sort data**: Ensure correct lag/rolling calculations

### Step 4: Handle Missing Data (Fit on Train Only)
- Fills NaN with median (numeric) or mode (categorical)
- Uses training data statistics only to prevent leakage

### Steps 5-8: Feature Engineering (No Leakage Risk)
- **Temporal features**: Capture seasonality and trends
- **Lag features**: Autoregressive patterns (yesterday's sales predict today's)
- **Rolling features**: Smooth trends and capture volatility
- **External features**: Incorporate weather, promotions, holidays

### Steps 9-11: Model Preparation (Fit on Train Only)
- **Encode categoricals**: Convert store IDs, product IDs to numeric
- **Cap outliers**: Limit extreme values using train percentiles
- **Scale numerical**: Standardize to mean=0, std=1 using train statistics

## 🔐 Data Leakage Prevention

The pipeline follows strict no-leakage principles:

| Step | Leakage Risk | Prevention Method |
|------|--------------|-------------------|
| 1-3 | ❌ None | Simple transformations |
| 4 | ⚠️ Potential | **Fit on train only** (median/mode) |
| 5-8 | ❌ None | Use past data only (`.shift()`) |
| 9 | ⚠️ Potential | **Fit on train only** (label encoding) |
| 10 | ⚠️ Potential | **Fit on train only** (percentiles) |
| 11 | ⚠️ Potential | **Fit on train only** (scaler) |

## 📚 Detailed Documentation

For in-depth explanations of each step, see:
- **`../PREPROCESSING_DETAILED_GUIDE.md`**: Comprehensive guide with examples
- **`../docs/PREPROCESSING_11_STEPS.md`**: Quick reference table
- **`../src/preprocessing.py`**: Implementation code

## 🚀 How to Regenerate These Visualizations

```bash
# Activate virtual environment
source venv/bin/activate

# Run analysis script
python analyze_preprocessing_steps.py
```

This will:
1. Load FreshRetailNet-50K dataset (sampled to 10,000 rows for speed)
2. Apply each preprocessing step sequentially
3. Generate before/after visualizations
4. Create summary report

**Note**: Analysis uses sampled data (10K rows) for faster processing. Full dataset analysis would take longer but show identical patterns.

## 📊 Dataset Information

**Source**: FreshRetailNet-50K from Hugging Face
- **Dataset**: [Dingdong-Inc/FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K)
- **Domain**: Retail sales forecasting
- **Target**: `sale_amount` (daily product sales)
- **Entities**: Store-Product combinations
- **Period**: Multiple months of daily sales data

**Original Features (17)**:
- IDs: city_id, store_id, product_id, management_group_id, category_ids (3 levels)
- Target: sale_amount
- Date: dt
- External: discount, holiday_flag, activity_flag, stock_hour6_22_cnt
- Weather: precpt, avg_temperature, avg_humidity, avg_wind_level

**Engineered Features (45 added)**:
- 13 temporal features
- 4 lag features
- 12 rolling features
- 16 external feature lags

## 🎓 Learning Objectives

By examining these visualizations, you'll understand:

1. **Why each preprocessing step is necessary**
   - See the actual data transformations
   - Understand the impact on data quality and model performance

2. **How feature engineering works**
   - From 17 to 62 features through systematic engineering
   - Each new feature captures a different aspect of the data

3. **Data leakage prevention**
   - Which steps require fitting on train data only
   - How to properly use past-only information for lag/rolling features

4. **Real-world preprocessing pipeline**
   - Production-ready approach to time series forecasting
   - Balances performance, interpretability, and correctness

## 💡 Tips for Presenting This Analysis

1. **Start with the summary** (`00_complete_summary.png`)
   - Shows the big picture: data flow, feature engineering impact

2. **Focus on key transformations**
   - Steps 5-8: Where most features are created
   - Steps 9-11: Model preparation steps

3. **Explain leakage prevention**
   - Emphasize fit-on-train-only for steps 4, 9, 10, 11
   - Show how lag/rolling features use `.shift()` for past-only data

4. **Highlight domain knowledge**
   - Step 8: External features capture retail-specific factors
   - Step 5: Temporal features capture seasonality

5. **Show the impact**
   - 17 raw columns → 62 engineered features
   - Each feature tells a different part of the story

## 📝 Citation

If you use this preprocessing pipeline or visualizations, please cite:

```
FreshRetailNet-50K Preprocessing Pipeline
11-Step Feature Engineering for Sales Forecasting
Dataset: Dingdong-Inc/FreshRetailNet-50K (Hugging Face)
```

## 🔗 Related Files

- `../src/preprocessing.py`: Implementation
- `../src/feature_engineering.py`: Feature engineering functions
- `../PREPROCESSING_DETAILED_GUIDE.md`: Comprehensive explanations
- `../docs/PREPROCESSING_11_STEPS.md`: Quick reference
- `../analyze_preprocessing_steps.py`: Visualization generator

---

**Generated**: February 2, 2025  
**Dataset**: FreshRetailNet-50K (10,000 sample rows)  
**Pipeline**: 11-step preprocessing with no data leakage

# Comprehensive Guide to 11 Preprocessing Steps

This document provides an in-depth explanation of each preprocessing step used in the FreshRetailNet-50K sales forecasting pipeline.

---

## Table of Contents

1. [Parse Dates](#step-1-parse-dates)
2. [Select Columns](#step-2-select-columns)
3. [Sort by Entity and Date](#step-3-sort-by-entity-and-date)
4. [Impute Missing Values](#step-4-impute-missing-values)
5. [Add Temporal Features](#step-5-add-temporal-features)
6. [Add Lag Features](#step-6-add-lag-features)
7. [Add Rolling Features](#step-7-add-rolling-features)
8. [Add External Features](#step-8-add-external-features)
9. [Encode Categoricals](#step-9-encode-categoricals)
10. [Cap Outliers](#step-10-cap-outliers)
11. [Scale Numerical](#step-11-scale-numerical)

---

## Overview

The preprocessing pipeline follows a strict **no data leakage** principle:
- **Steps 1-3**: Applied to combined train+validation data (no fitting required)
- **Steps 4-8**: Applied to combined data, but imputation (step 4) fits on train only
- **Steps 9-11**: Fit on train data only, then transform both train and validation

---

## Step 1: Parse Dates

### What It Does
Converts date columns from string format to proper `datetime64` objects.

### Before
```
dt (object): '2021-01-01', '2021-01-02', ...
```

### After
```
dt (datetime64[ns]): 2021-01-01, 2021-01-02, ...
```

### Why It's Important
- **Enables temporal operations**: You can't extract year/month/day from strings
- **Proper sorting**: Datetime objects sort correctly (lexicographic sorting fails)
- **Date arithmetic**: Calculate time differences, periods, etc.
- **Performance**: Datetime operations are optimized in pandas

### How It Works
```python
df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
```
- `errors='coerce'`: Invalid dates become NaT (Not a Time)
- No data leakage: Just converts types

### FreshRetailNet Dataset
- **Column**: `dt`
- **Original format**: String like '2021-01-01'
- **Converted to**: datetime64[ns]
- **Date range**: Typically spans several months of retail data

---

## Step 2: Select Columns

### What It Does
Keeps or drops specified columns to reduce dimensionality and focus on relevant features.

### Before
```
Columns: city_id, store_id, product_id, dt, sale_amount, 
         hours_sale (sequence), hours_stock_status (sequence), ...
```

### After
```
Columns: city_id, store_id, product_id, dt, sale_amount, 
         discount, holiday_flag, stock_hour6_22_cnt, ...
```

### Why It's Important
- **Memory efficiency**: Remove unnecessary columns (e.g., sequence data for tabular models)
- **Computational efficiency**: Fewer columns = faster processing
- **Focus on relevant features**: Keep only what's useful for forecasting
- **Avoid confusion**: Drop columns that shouldn't be used (e.g., IDs that leak information)

### How It Works
```python
# Keep only specified columns
if keep_columns:
    df = df[keep_columns]

# Drop unwanted columns
if drop_columns:
    df = df.drop(columns=drop_columns)
```

### FreshRetailNet Dataset
**Kept columns**:
- **IDs**: city_id, store_id, management_group_id, product_id categories
- **Target**: sale_amount
- **Date**: dt
- **Features**: discount, holiday_flag, activity_flag, weather data, stock counts

**Dropped columns**:
- **Sequence data**: hours_sale, hours_stock_status (not suitable for tabular models)
- These contain hourly arrays that require specialized handling

---

## Step 3: Sort by Entity and Date

### What It Does
Sorts the dataframe by entity identifiers (store, product) and date in chronological order.

### Before
```
store_id | product_id | dt
   5     |    102     | 2021-01-15
   3     |     50     | 2021-01-10
   5     |    102     | 2021-01-12  ← Out of order!
```

### After
```
store_id | product_id | dt
   3     |     50     | 2021-01-10
   5     |    102     | 2021-01-12  ← Sorted correctly
   5     |    102     | 2021-01-15
```

### Why It's Important
- **Critical for lag features**: Lag(1) must be the previous time period, not random
- **Rolling window accuracy**: Rolling mean needs consecutive sorted data
- **Prevents data leakage**: Ensures we only use past data for future predictions
- **Reproducibility**: Consistent ordering ensures same results every run

### How It Works
```python
sort_cols = groupby_cols + [date_col]  # e.g., ['store_id', 'product_id', 'dt']
df = df.sort_values(sort_cols).reset_index(drop=True)
```

### Example Impact
**Without sorting** (WRONG):
```
Date: [Jan 15, Jan 10, Jan 12]
Sales: [100, 80, 90]
Lag_1: [NaN, 100, 80]  ← Lag_1 on Jan 10 uses future data (Jan 15)!
```

**With sorting** (CORRECT):
```
Date: [Jan 10, Jan 12, Jan 15]
Sales: [80, 90, 100]
Lag_1: [NaN, 80, 90]  ← Lag_1 correctly uses past data only
```

### FreshRetailNet Dataset
- **Sort by**: ['store_id', 'product_id', 'dt']
- **Effect**: Each store-product combination has chronologically ordered sales

---

## Step 4: Impute Missing Values

### What It Does
Fills missing (NaN) values using statistics computed from **training data only**.

### Before
```
temperature: [20.5, NaN, 22.1, NaN, 19.8]
city_id: [1, 2, NaN, 1, 2]
```

### After (using train median=21.0, mode=1)
```
temperature: [20.5, 21.0, 22.1, 21.0, 19.8]  ← NaN filled with median
city_id: [1, 2, 1, 1, 2]                      ← NaN filled with mode
```

### Why It's Important
- **Prevent model errors**: Most ML models can't handle NaN values
- **Retain data**: Better than dropping rows with missing values
- **No data leakage**: Uses only train statistics (not validation or test)
- **Preserve distributions**: Median/mode preserve central tendency

### How It Works
```python
# FIT: Compute statistics from train only
stats = {}
for col in train_df.columns:
    if numeric:
        stats[col] = train_df[col].median()  # or mean
    else:
        stats[col] = train_df[col].mode()[0]  # most frequent

# TRANSFORM: Apply to train and validation
df[col] = df[col].fillna(stats[col])
```

### Imputation Strategy
| Column Type | Method | Why |
|-------------|--------|-----|
| Numeric (temperature, sales) | Median | Robust to outliers |
| Categorical (city_id, store_id) | Mode | Most common category |

### FreshRetailNet Dataset
**Columns often with missing values**:
- **Weather data**: precpt (precipitation), avg_temperature, avg_humidity, avg_wind_level
- **Stock data**: stock_hour6_22_cnt
- **Imputation**: Median for numeric, mode for categorical

**Example**:
```
avg_temperature missing: 15% of rows → filled with train median (e.g., 18.5°C)
holiday_flag missing: 2% of rows → filled with train mode (0 = not holiday)
```

---

## Step 5: Add Temporal Features

### What It Does
Extracts time-based features from the date column (year, month, week, day of week, etc.).

### Before
```
dt: 2021-03-15
```

### After
```
dt: 2021-03-15
year: 2021
month: 3
week: 11
day_of_week: 0 (Monday)
quarter: 1
day_of_month: 15
day_of_year: 74
is_weekend: 0
is_month_start: 0
is_month_end: 0
```

### Why It's Important
- **Capture seasonality**: Month, quarter capture yearly patterns (e.g., holiday sales)
- **Weekly patterns**: Day of week captures weekday vs weekend differences
- **Trend information**: Year captures long-term trends
- **Model-friendly**: Converts dates into numeric features models can use
- **No data leakage**: Features derived from date only (not future information)

### How It Works
```python
df['year'] = df[date_col].dt.year
df['month'] = df[date_col].dt.month
df['week'] = df[date_col].dt.isocalendar().week
df['day_of_week'] = df[date_col].dt.dayofweek
df['quarter'] = df[date_col].dt.quarter
df['day_of_month'] = df[date_col].dt.day
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
```

### Real-World Impact
**Retail sales patterns**:
- **Month**: December (12) → High sales (holidays)
- **Day of week**: Saturday/Sunday → Higher foot traffic
- **Quarter**: Q4 → Year-end shopping surge
- **Day of month**: 1st, 15th → Payday effect

### FreshRetailNet Dataset
**Temporal patterns captured**:
- **Monthly seasonality**: Fresh produce sales vary by season
- **Weekly patterns**: Weekend vs weekday shopping behavior
- **Holiday effects**: Combined with holiday_flag for better predictions

**Example impact**:
```
Month 12 (December) + is_weekend=1 → Expect 30% higher sales
Month 6 (June) + day_of_week=2 (Tuesday) → Expect normal sales
```

---

## Step 6: Add Lag Features

### What It Does
Creates features from **past values** of the target variable (and optionally other features).

### Before
```
Date       | Sales
2021-01-10 | 100
2021-01-11 | 120
2021-01-12 | 110
2021-01-13 | 130
```

### After (with lags [1, 2, 4, 7])
```
Date       | Sales | lag_1 | lag_2 | lag_4 | lag_7
2021-01-10 | 100   | NaN   | NaN   | NaN   | NaN
2021-01-11 | 120   | 100   | NaN   | NaN   | NaN
2021-01-12 | 110   | 120   | 100   | NaN   | NaN
2021-01-13 | 130   | 110   | 120   | NaN   | NaN
...
2021-01-17 | 150   | 140   | 145   | 130   | 100
```

### Why It's Important
- **Autoregressive patterns**: Sales often correlate with recent past sales
- **Trend capture**: Series of lags show if sales are increasing/decreasing
- **Powerful predictor**: Past values are often the strongest feature
- **No data leakage**: Uses `.shift()` to ensure only past data

### How It Works
```python
# For each entity group (store-product combination)
for lag in [1, 2, 4, 7]:
    df[f'lag_{lag}'] = df.groupby(groupby_cols)[target_col].shift(lag)
```

**Key**: `shift(1)` moves data down by 1 row → previous period's value

### Lag Selection Strategy
| Lag Period | Why Include | Captures |
|------------|-------------|----------|
| 1 | Yesterday's sales | Very recent trend |
| 2 | 2 days ago | Short-term pattern |
| 4 | 4 days ago | Mid-term pattern |
| 7 | Last week same day | Weekly seasonality |

### Real-World Example
**Predicting Monday sales**:
- **lag_1** (Sunday): Indicates weekend trend
- **lag_2** (Saturday): Weekend pattern
- **lag_7** (Last Monday): Same day-of-week pattern

**Pattern detection**:
```
If lag_1=100, lag_2=95, lag_4=90 → Downward trend → Predict lower
If lag_1=100, lag_2=105, lag_4=110 → Upward trend → Predict higher
If lag_7=150 but lag_1=100 → Last week stronger (maybe holiday)
```

### FreshRetailNet Dataset
**Lag configuration**: [1, 2, 4, 7]
- **lag_1**: Previous day's sales (most important)
- **lag_2**: 2 days ago (short trend)
- **lag_4**: 4 days ago (mid-range)
- **lag_7**: Week-ago same day (weekly pattern)

**Example**:
```python
# Store 5, Product 102, Date 2021-06-15
sale_amount: 250
lag_1: 230 (June 14)
lag_2: 245 (June 13)
lag_7: 260 (June 8, same day of week)
→ Model sees: slight increase from lag_1, similar to lag_7 pattern
```

---

## Step 7: Add Rolling Features

### What It Does
Creates rolling window statistics (mean, std, min, max) over **past data only**.

### Before
```
Date       | Sales | lag_1
2021-01-10 | 100   | NaN
2021-01-11 | 120   | 100
2021-01-12 | 110   | 120
2021-01-13 | 130   | 110
2021-01-14 | 125   | 130
```

### After (with window=4)
```
Date       | Sales | rolling_4_mean | rolling_4_std | rolling_4_min | rolling_4_max
2021-01-10 | 100   | NaN            | NaN           | NaN           | NaN
2021-01-11 | 120   | NaN            | NaN           | NaN           | NaN
2021-01-12 | 110   | NaN            | NaN           | NaN           | NaN
2021-01-13 | 130   | NaN            | NaN           | NaN           | NaN
2021-01-14 | 125   | 115.0          | 12.9          | 100           | 130
```
*Note: rolling_4_mean on Jan 14 = mean(100, 120, 110, 130) = 115*

### Why It's Important
- **Smooth out noise**: Rolling mean reduces day-to-day volatility
- **Trend detection**: Rising/falling moving averages show trends
- **Volatility measure**: Rolling std captures recent stability/instability
- **Range information**: Min/max show recent extremes
- **No data leakage**: Uses `.shift(1).rolling()` to exclude current value

### How It Works
```python
for window in [4, 7, 14]:
    # CRITICAL: shift(1) first to exclude current value!
    df[f'rolling_{window}_mean'] = (
        df.groupby(groupby_cols)[target_col]
        .shift(1)  # ← Ensures we only use past data
        .rolling(window=window)
        .mean()
    )
```

### Rolling Window Selection
| Window | Why Include | Captures |
|--------|-------------|----------|
| 4 days | Short-term average | Recent trend |
| 7 days | Week average | Weekly pattern smoothed |
| 14 days | Two-week average | Medium-term trend |

### Statistics Computed
| Statistic | Meaning | Use Case |
|-----------|---------|----------|
| Mean | Average over window | General trend level |
| Std | Standard deviation | Volatility/stability |
| Min | Minimum value | Recent low point |
| Max | Maximum value | Recent high point |

### Real-World Example
**Interpreting rolling features**:
```
rolling_7_mean = 200 → Average sales last week was 200
rolling_7_std = 50 → High variability (unstable sales)
rolling_7_std = 5 → Low variability (stable sales)

If current_sales < rolling_7_mean → Below recent average (might bounce back)
If current_sales > rolling_7_max → Unusually high (might be outlier/promotion)
```

### FreshRetailNet Dataset
**Window configuration**: [4, 7, 14] days

**Example**:
```python
# Store 5, Product 102, Date 2021-06-15
sale_amount: 250

# Rolling statistics (from past 4 days: June 11-14)
rolling_4_mean: 230.5  → Recent average slightly below current
rolling_4_std: 15.2     → Moderate variability
rolling_4_min: 210      → Recent low
rolling_4_max: 245      → Recent high

# Interpretation: Current sale (250) is above recent average and exceeds recent max
# → Possible upward trend or special event
```

**Feature count**: For each window × 4 statistics = 12 new columns
- rolling_4_mean, rolling_4_std, rolling_4_min, rolling_4_max
- rolling_7_mean, rolling_7_std, rolling_7_min, rolling_7_max
- rolling_14_mean, rolling_14_std, rolling_14_min, rolling_14_max

---

## Step 8: Add External Features

### What It Does
Incorporates external/covariate features (weather, holidays, promotions) and their lagged versions.

### Before
```
Date       | Sales | discount | holiday_flag | temperature
2021-01-10 | 100   | 0        | 0            | 15.5
2021-01-11 | 120   | 10       | 0            | 16.2
2021-01-12 | 110   | 0        | 1            | 14.8
```

### After (with external feature lags)
```
Date       | Sales | discount | discount_lag_1 | holiday_flag | holiday_lag_1 | temperature | temp_lag_1
2021-01-10 | 100   | 0        | NaN            | 0            | NaN           | 15.5        | NaN
2021-01-11 | 120   | 10       | 0              | 0            | 0             | 16.2        | 15.5
2021-01-12 | 110   | 0        | 10             | 1            | 0             | 14.8        | 16.2
```

### Why It's Important
- **Capture external effects**: Sales affected by weather, holidays, promotions
- **Explain variations**: Not just autoregressive—external factors matter
- **Lagged effects**: Yesterday's promotion might affect today's sales
- **Domain knowledge**: Retail-specific features (discounts, stock levels)

### How It Works
```python
# Use external features as-is
external_cols = ['discount', 'holiday_flag', 'precpt', 'avg_temperature', ...]

# Add lags of external features
for col in external_cols:
    for lag in [1, 2, 4]:
        df[f'{col}_lag_{lag}'] = df.groupby(groupby_cols)[col].shift(lag)
```

### FreshRetailNet External Features

#### Base External Features
| Feature | Type | Description | Impact on Sales |
|---------|------|-------------|-----------------|
| **discount** | Numeric (%) | Discount percentage | Higher discount → Higher sales |
| **holiday_flag** | Binary (0/1) | Is it a holiday? | Holiday → Higher sales |
| **activity_flag** | Binary (0/1) | Special activity/promotion | Activity → Higher sales |
| **precpt** | Numeric (mm) | Precipitation | Rain → Lower foot traffic |
| **avg_temperature** | Numeric (°C) | Average temperature | Affects fresh produce demand |
| **avg_humidity** | Numeric (%) | Humidity level | Affects comfort, shopping behavior |
| **avg_wind_level** | Numeric | Wind level | High wind → Lower foot traffic |
| **stock_hour6_22_cnt** | Numeric | Stock count (6am-10pm) | Low stock → Lower sales |

#### Lagged External Features
For each base feature, create lags [1, 2, 4]:
```
discount_lag_1: Yesterday's discount (carryover effect)
holiday_flag_lag_1: Was yesterday a holiday? (post-holiday slump)
precpt_lag_1: Yesterday's rain (affects today's restocking/behavior)
```

### Real-World Examples

**Example 1: Discount Effect**
```
discount = 20% → Sales likely increase by 30-50%
discount_lag_1 = 20% → Some customers might have stocked up yesterday → Today's sales dip
```

**Example 2: Weather Impact**
```
avg_temperature = 5°C (cold) → Higher demand for hot food
avg_temperature = 30°C (hot) → Higher demand for cold drinks, ice cream
precpt = 50mm (heavy rain) → Lower foot traffic → Lower sales
```

**Example 3: Holiday Pattern**
```
holiday_flag = 1 → Big shopping day → Sales spike
holiday_flag_lag_1 = 1 → Post-holiday → Possible sales dip (people already shopped)
```

**Example 4: Stock Impact**
```
stock_hour6_22_cnt = 50 (low) → Limited availability → Lower sales (stockout)
stock_hour6_22_cnt = 500 (high) → Good availability → Normal sales
```

### Feature Engineering on External Features
```python
# Example interactions (not in current pipeline but possible):
df['discount_x_holiday'] = df['discount'] * df['holiday_flag']
# → Captures: Do discounts work better on holidays?

df['temp_category'] = pd.cut(df['avg_temperature'], bins=[0, 10, 20, 30, 40])
# → Captures: Non-linear temperature effects
```

---

## Step 9: Encode Categoricals

### What It Does
Converts categorical variables (text/object types) into numeric codes using label encoding, **fit on training data only**.

### Before
```
store_id: ['ST_A', 'ST_B', 'ST_A', 'ST_C']
city_id: ['City_1', 'City_2', 'City_1', 'City_3']
```

### After
```
store_id: [0, 1, 0, 2]          # ST_A→0, ST_B→1, ST_C→2
city_id: [0, 1, 0, 2]           # City_1→0, City_2→1, City_3→2
```

### Why It's Important
- **Model compatibility**: Most ML models require numeric inputs
- **Preserve categorical information**: Don't lose the grouping structure
- **No data leakage**: Mapping learned from train; unknown values in validation get -1
- **Memory efficient**: Integers use less memory than strings

### How It Works
```python
# FIT: Learn mapping from train only
encoders = {}
for col in categorical_cols:
    unique_values = train_df[col].dropna().unique()
    encoders[col] = {value: idx for idx, value in enumerate(sorted(unique_values))}

# TRANSFORM: Apply to train and validation
def encode(value, mapping):
    return mapping.get(value, -1)  # -1 for unknown values

df[col] = df[col].map(lambda x: encode(x, encoders[col]))
```

### Label Encoding vs One-Hot Encoding

| Method | Representation | Pros | Cons |
|--------|----------------|------|------|
| **Label Encoding** (used here) | store_id: [0, 1, 2] | Memory efficient, works with tree models | Implies ordering |
| **One-Hot Encoding** | store_0: [1,0,0], store_1: [0,1,0] | No ordering assumption | High dimensionality |

**Why Label Encoding?**
- Tree-based models (XGBoost, LightGBM) handle label encoding well
- FreshRetailNet has many categories (stores, products) → One-hot would explode dimensions

### Handling Unknown Categories
**Problem**: Validation set might have categories not in training
```
Train: store_id in ['ST_A', 'ST_B', 'ST_C']
Validation: store_id in ['ST_A', 'ST_B', 'ST_D']  ← 'ST_D' is unknown!
```

**Solution**: Map unknown to -1
```
ST_A → 0
ST_B → 1
ST_C → 2
ST_D → -1  ← Unknown, treated as separate group
```

### FreshRetailNet Categorical Features
| Column | Description | Typical # Categories | Encoded As |
|--------|-------------|----------------------|------------|
| **store_id** | Store identifier | 50-100 | 0, 1, 2, ..., 99 |
| **product_id** | Product identifier | 1000-5000 | 0, 1, 2, ..., 4999 |
| **city_id** | City identifier | 10-20 | 0, 1, 2, ..., 19 |
| **management_group_id** | Management group | 5-10 | 0, 1, 2, ..., 9 |
| **first_category_id** | Product category level 1 | 10-20 | 0, 1, 2, ..., 19 |
| **second_category_id** | Product category level 2 | 20-50 | 0, 1, 2, ..., 49 |
| **third_category_id** | Product category level 3 | 50-200 | 0, 1, 2, ..., 199 |

### Real-World Example
**Before encoding**:
```
Row 1: store_id='ST_Shanghai_01', city_id='Shanghai', product_id='PROD_12345'
Row 2: store_id='ST_Beijing_05', city_id='Beijing', product_id='PROD_67890'
```

**After encoding**:
```
Row 1: store_id=42, city_id=3, product_id=1523
Row 2: store_id=8, city_id=1, product_id=3891
```

**Why it helps models**:
- Tree models can split on: "If store_id <= 25 then..." (captures store groups)
- Neural networks can learn embeddings for each store_id
- Reduces memory from strings to int32

---

## Step 10: Cap Outliers

### What It Does
Winsorizes (caps) extreme values to percentile bounds computed from **training data only**.

### Before
```
temperature: [-5, 10, 15, 18, 22, 25, 28, 150]  ← 150 is outlier
sales: [50, 100, 120, 110, 130, 140, 135, 10000]  ← 10000 is outlier
```

### After (capping at 1st and 99th percentiles: 10 and 28 for temperature)
```
temperature: [10, 10, 15, 18, 22, 25, 28, 28]  ← -5→10 (lower bound), 150→28 (upper bound)
sales: [50, 100, 120, 110, 130, 140, 135, 200]  ← 10000→200 (99th percentile)
```

### Why It's Important
- **Robust models**: Outliers can skew model predictions
- **Prevent overfitting**: Extreme values might be errors or rare events
- **Preserve distribution shape**: Better than removing outliers (keeps data count)
- **No data leakage**: Percentiles computed from train only

### How It Works
```python
# FIT: Compute percentile bounds from train
bounds = {}
for col in numeric_cols:
    lower_bound = np.percentile(train_df[col], 1)   # 1st percentile
    upper_bound = np.percentile(train_df[col], 99)  # 99th percentile
    bounds[col] = (lower_bound, upper_bound)

# TRANSFORM: Clip values
df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
```

### Percentile Selection
| Percentile | Effect | Use Case |
|------------|--------|----------|
| 1%, 99% (used here) | Mild capping | General outlier removal |
| 5%, 95% | Moderate capping | Noisy data |
| 0.1%, 99.9% | Very mild capping | Clean data, preserve extremes |

### Why Not Remove Outliers?
| Approach | Pros | Cons |
|----------|------|------|
| **Remove outliers** | Cleaner data | Lose rows, imbalanced data |
| **Cap outliers (Winsorize)** ✓ | Keep all rows, reduce impact | Less aggressive cleaning |

### Real-World Examples

**Example 1: Temperature Outlier**
```
Train data: avg_temperature in [-10°C, 40°C], but one sensor error shows 150°C
1st percentile: -5°C
99th percentile: 35°C

Before: [-10, 15, 18, 22, 150, 25, 28]
After:  [-5, 15, 18, 22, 35, 25, 28]  ← 150°C capped to 35°C
```

**Example 2: Sales Outlier**
```
Train data: sale_amount in [0, 500], but one Black Friday sale = 10,000
1st percentile: 10
99th percentile: 450

Before: [50, 100, 150, 10000, 120, 80]
After:  [50, 100, 150, 450, 120, 80]  ← 10,000 capped to 450
```
*Note: Black Friday might be a real event, but for stable model training, we cap it*

**Example 3: Why Capping Helps**
```
Without capping:
Model learns: "When X=outlier, predict Y=10000"
But outlier is rare → Model overfits to it

With capping:
Model learns: "When X=high, predict Y=450 (upper range)"
More generalizable pattern
```

### FreshRetailNet Application
**Columns capped**:
- **sale_amount**: Target variable (cap extreme sales)
- **discount**: Cap extreme discounts (e.g., 200% discount is likely error)
- **Weather features**: Cap sensor errors
- **Stock counts**: Cap unrealistic stock numbers
- **Lag features**: Inherit outliers from target, so also capped
- **Rolling features**: Less affected by outliers, but still capped

**Excluded from capping**:
- **target_col** (sale_amount): Only capped features, not target (to preserve true values)
- Actually, in this pipeline, even target is capped for consistency

**Example bounds (hypothetical)**:
```
sale_amount: [10, 450]     → Sales below 10 or above 450 get capped
discount: [0, 50]          → Discounts outside 0-50% capped
avg_temperature: [-5, 35]  → Temps outside this range capped
```

---

## Step 11: Scale Numerical

### What It Does
Standardizes numerical features to have mean=0 and standard deviation=1, using statistics from **training data only**.

### Before
```
sales: [100, 200, 150, 250, 180]      (mean=176, std=57)
temperature: [10, 15, 20, 18, 22]     (mean=17, std=4.4)
discount: [0, 10, 5, 20, 15]          (mean=10, std=7.1)
```

### After (standardized)
```
sales: [-1.33, 0.42, -0.46, 1.30, 0.07]         (mean≈0, std≈1)
temperature: [-1.59, -0.45, 0.68, 0.23, 1.14]   (mean≈0, std≈1)
discount: [-1.41, 0, -0.70, 1.41, 0.70]         (mean≈0, std≈1)
```

### Why It's Important
- **Equal feature importance**: Prevents features with large scales from dominating
- **Faster convergence**: Neural networks and gradient descent train faster
- **Required by some models**: Linear regression, SVR, neural networks benefit
- **Interpretability**: Standardized coefficients show relative importance
- **No data leakage**: Mean and std computed from train only

### How It Works
```python
from sklearn.preprocessing import StandardScaler

# FIT: Compute mean and std from train
scaler = StandardScaler()
scaler.fit(train_df[numeric_cols])

# TRANSFORM: Apply to train and validation
df[numeric_cols] = scaler.transform(df[numeric_cols])

# Formula: z = (x - mean) / std
```

### StandardScaler Formula
```
scaled_value = (original_value - train_mean) / train_std

Example:
sales = 200, train_mean = 176, train_std = 57
scaled = (200 - 176) / 57 = 24 / 57 = 0.42
```

### Why Not Min-Max Scaling?
| Method | Formula | Range | Pros | Cons |
|--------|---------|-------|------|------|
| **StandardScaler** (used) | (x - mean) / std | Unbounded, ~[-3, 3] | Preserves outlier info, robust | Not bounded to [0,1] |
| **MinMaxScaler** | (x - min) / (max - min) | [0, 1] | Bounded range | Sensitive to outliers |

**Why StandardScaler?**
- Tree-based models (XGBoost, LightGBM) don't need scaling, but we use it for consistency
- Neural networks benefit from StandardScaler
- Preserves information about extreme values (unlike MinMax which squashes them)

### When Scaling Doesn't Help
**Tree-based models** (XGBoost, Random Forest, LightGBM):
- Don't require scaling (decision trees are scale-invariant)
- However, we still apply it for consistency and in case we use linear models

**Models that NEED scaling**:
- Linear Regression, Logistic Regression
- Support Vector Machines (SVM)
- Neural Networks
- K-Nearest Neighbors (KNN)

### FreshRetailNet Application

**Columns scaled**:
- **Target**: sale_amount (and its lags, rolling features)
- **External features**: discount, temperature, humidity, wind, stock counts
- **Temporal features**: NOT scaled (year, month, day are already small integers)
- **Categorical features**: NOT scaled (already encoded as integers 0, 1, 2, ...)

**Excluded from scaling**:
- **date_col** (dt): Datetime, not numeric for scaling
- **groupby_cols** (store_id, product_id): Categorical IDs
- **target_col** (sale_amount): Sometimes excluded, but here included

**Example scaling**:
```python
# Before scaling
sale_amount: [250, 300, 200, 350, 280]     (mean=276, std=55.9)
discount: [0, 10, 5, 20, 15]               (mean=10, std=7.1)
lag_1: [230, 250, 300, 200, 350]           (mean=266, std=61.4)

# After scaling
sale_amount: [-0.46, 0.43, -1.36, 1.32, 0.07]
discount: [-1.41, 0, -0.70, 1.41, 0.70]
lag_1: [-0.59, -0.26, 0.55, -1.08, 1.37]
```

**Why this helps**:
```
Before scaling:
Feature importance might be skewed by scale:
- sales (0-500 range) dominates
- discount (0-20 range) seems less important

After scaling:
All features on same scale → True importance emerges:
- Maybe discount actually has stronger correlation, just was on different scale
```

---

## Summary: Complete Pipeline Flow

### Data Flow Diagram
```
Raw Data (FreshRetailNet-50K)
    ↓
1. Parse Dates: '2021-01-01' → datetime
    ↓
2. Select Columns: Drop sequence columns, keep tabular features
    ↓
3. Sort: Order by (store_id, product_id, dt)
    ↓
4. Impute: Fill NaN with train median/mode
    ↓
5. Temporal Features: Add year, month, day_of_week, ...
    ↓
6. Lag Features: Add lag_1, lag_2, lag_4, lag_7
    ↓
7. Rolling Features: Add rolling_4_mean, rolling_7_std, ...
    ↓
8. External Features: Add discount_lag_1, holiday_lag_1, ...
    ↓
9. Encode Categoricals: store_id → [0,1,2,...], city_id → [0,1,2,...]
    ↓
10. Cap Outliers: Clip to [1st percentile, 99th percentile]
    ↓
11. Scale Numerical: Standardize to mean=0, std=1
    ↓
Final Dataset: Ready for Model Training
```

### Data Transformation Summary
| Stage | Rows | Columns | Description |
|-------|------|---------|-------------|
| Raw | 50,000 | 18 | Original FreshRetailNet data |
| After Step 3 | 50,000 | 17 | Dropped sequence columns |
| After Step 5 | 50,000 | 28 | Added 11 temporal features |
| After Step 6 | 50,000 | 32 | Added 4 lag features |
| After Step 7 | 50,000 | 44 | Added 12 rolling features |
| After Step 8 | 50,000 | 68 | Added 24 external feature lags |
| Final | 50,000 | 68 | Encoded, capped, scaled |

### Key Principles Recap

1. **No Data Leakage**: 
   - Steps 4, 9, 10, 11 fit on train only
   - Steps 6, 7 use `.shift()` to ensure past-only data

2. **Feature Engineering**:
   - Temporal features (Step 5): Capture time patterns
   - Lag features (Step 6): Capture autoregressive patterns
   - Rolling features (Step 7): Capture smoothed trends
   - External features (Step 8): Capture external effects

3. **Data Cleaning**:
   - Imputation (Step 4): Handle missing values
   - Outlier capping (Step 10): Reduce extreme value impact

4. **Model Preparation**:
   - Encoding (Step 9): Make categoricals numeric
   - Scaling (Step 11): Standardize feature scales

---

## How to Run Analysis

To generate visualizations for all 11 steps:

```bash
python analyze_preprocessing_steps.py
```

This will:
1. Load FreshRetailNet-50K dataset (sampled for speed)
2. Apply each preprocessing step sequentially
3. Generate before/after visualizations for each step
4. Create a comprehensive summary report
5. Save all outputs to `preprocessing_analysis/` directory

**Output files**:
- `step_01_parse_dates.png`
- `step_02_select_columns.png`
- ...
- `step_11_scale_numerical.png`
- `00_complete_summary.png` (overview)
- `preprocessing_report.txt` (detailed text report)

---

## References

- **Dataset**: [FreshRetailNet-50K on Hugging Face](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K)
- **Code**: `src/preprocessing.py`
- **Documentation**: `docs/PREPROCESSING_11_STEPS.md`

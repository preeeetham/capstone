# Complete Guide to Imported Functions & Models
## Understanding What's Inside Each Module

This guide explains **every function and model** that the notebook imports from the `src/` folder.

---

# 📦 Table of Contents

1. [data_loader.py](#data_loaderpy) - Downloading and loading Kaggle data
2. [eda.py](#edapy) - Exploratory Data Analysis functions
3. [feature_engineering.py](#feature_engineeringpy) - Creating 49 features
4. [utils.py](#utilspy) - Helper functions
5. [models.py](#modelspy) - All 5 models (Naive, MA, LightGBM, XGBoost, LSTM)
6. [evaluation.py](#evaluationpy) - Metrics and visualization

---

# 1. data_loader.py

## Purpose
Downloads Walmart dataset from Kaggle and prepares it for analysis.

---

### Function: `_find_file(root, filename)`

**What it does:**  
Searches for a file in nested directories.

**Why needed:**  
Kaggle sometimes extracts files into weird nested folders like:
```
.../train.csv/train.csv  (file nested inside folder with same name)
```

**How it works:**
```python
def _find_file(root, filename):
    root_path = Path(root)  # Convert to Path object
    matches = [p for p in root_path.rglob(filename) if p.is_file()]
    # rglob = recursive glob (searches all subdirectories)
    if not matches:
        raise FileNotFoundError(f"Could not find {filename}")
    return str(matches[0])  # Return first match
```

**Example:**
```python
path = _find_file("/path/to/download", "train.csv")
# Returns: "/path/to/download/train.csv/train.csv"
```

---

### Function: `load_dataset()` ⭐ MAIN FUNCTION

**What it does:**  
Downloads and merges all Walmart data files.

**Returns:**
- `train_df` - Historical sales (421,570 rows)
- `test_df` - Future period (115,064 rows)
- `features_df` - External variables (Temperature, CPI, etc.)
- `stores_df` - Store characteristics (Type, Size)

**Step-by-step breakdown:**

#### Step 1: Download from Kaggle
```python
path = kagglehub.dataset_download("micgonzalez/walmart-store-sales-forecasting")
```
- Uses `kagglehub` library
- Downloads to cache folder: `~/.cache/kagglehub/...`
- If already downloaded, uses cached version (fast!)

#### Step 2: Find and Load CSV Files
```python
train_path = _find_file(path, "train.csv")
test_path = _find_file(path, "test.csv")
features_path = _find_file(path, "features.csv")
stores_path = _find_file(path, "stores.csv")

train_df = pd.read_csv(train_path)  # Load into pandas DataFrame
test_df = pd.read_csv(test_path)
features_df = pd.read_csv(features_path)
stores_df = pd.read_csv(stores_path)
```

**What's in each file:**

| File | Columns | What It Contains |
|------|---------|------------------|
| `train.csv` | Store, Dept, Date, Weekly_Sales, IsHoliday | Historical sales data |
| `test.csv` | Store, Dept, Date, IsHoliday | Future period (no sales - that's what we predict) |
| `features.csv` | Store, Date, Temperature, Fuel_Price, MarkDown1-5, CPI, Unemployment, IsHoliday | External factors by store and date |
| `stores.csv` | Store, Type, Size | Store characteristics |

#### Step 3: Parse Dates
```python
train_df['Date'] = pd.to_datetime(train_df['Date'])
```
- Converts string "2010-02-05" to datetime object
- Why? So we can do date operations (extract month, sort by date, etc.)

#### Step 4: Merge DataFrames
```python
train_df = train_df.merge(features_df, on=['Store', 'Date'], how='left')
train_df = train_df.merge(stores_df, on='Store', how='left')
```

**What merge does:**

Before merge:
```
train_df:
Store | Dept | Date       | Weekly_Sales
1     | 5    | 2010-02-05 | 24924.50

features_df:
Store | Date       | Temperature | Fuel_Price
1     | 2010-02-05 | 42.31       | 2.572
```

After merge:
```
Store | Dept | Date       | Weekly_Sales | Temperature | Fuel_Price
1     | 5    | 2010-02-05 | 24924.50     | 42.31       | 2.572
```

**Merge parameters:**
- `on=['Store', 'Date']` - Match rows where both Store AND Date are equal
- `how='left'` - Keep all rows from train_df, add features_df columns

#### Step 5: Handle Duplicate IsHoliday Columns
```python
if 'IsHoliday_x' in df.columns and 'IsHoliday_y' in df.columns:
    df['IsHoliday'] = (df['IsHoliday_x'].astype(bool) | df['IsHoliday_y'].astype(bool))
    df.drop(columns=['IsHoliday_x', 'IsHoliday_y'], inplace=True)
```

**Why this happens:**  
Both `train.csv` and `features.csv` have `IsHoliday` column.  
After merge, pandas renames them to `IsHoliday_x` and `IsHoliday_y`.

**Solution:**  
Combine with OR logic - mark as holiday if EITHER says it's a holiday.
```python
IsHoliday_x = True,  IsHoliday_y = False  →  IsHoliday = True
IsHoliday_x = False, IsHoliday_y = True   →  IsHoliday = True
IsHoliday_x = False, IsHoliday_y = False  →  IsHoliday = False
```

#### Step 6: Sort Data
```python
train_df = train_df.sort_values(['Store', 'Dept', 'Date']).reset_index(drop=True)
```
- Sorts by Store (1,2,3...), then Dept (1,2,3...), then Date (oldest first)
- `.reset_index(drop=True)` - Renumber rows 0, 1, 2, 3...
- **Why?** Time series models need data in chronological order

**Final result:**
```python
return train_df, test_df, features_df, stores_df
```

Four DataFrames ready for analysis!

---

### Function: `save_processed_data(df, filename)`

**What it does:**  
Saves DataFrame to CSV file.

```python
def save_processed_data(df, filename):
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)  # Create data/ folder if doesn't exist
    filepath = data_dir / filename
    df.to_csv(filepath, index=False)  # Save without row numbers
```

**Example:**
```python
save_processed_data(train_df, "processed_train.csv")
# Saves to: data/processed_train.csv
```

---

# 2. eda.py

## Purpose
Performs Exploratory Data Analysis - understanding data before modeling.

---

### Function: `check_missing_values(df, name="Dataset")`

**What it does:**  
Finds and reports missing data.

**Line-by-line:**

```python
missing = df.isnull().sum()
```
- `.isnull()` - Returns True/False for each cell (True = missing)
- `.sum()` - Counts True values per column

**Example:**
```python
MarkDown1    270889  # 270,889 missing values
MarkDown2    310322
Temperature       0  # No missing values
```

```python
missing_pct = (missing / len(df)) * 100
```
- Converts count to percentage
- `len(df)` = total rows (421,570)

**Example:**
```python
MarkDown1: 270889 / 421570 * 100 = 64.26%
```

```python
missing_df = pd.DataFrame({
    'Column': missing.index,
    'Missing Count': missing.values,
    'Missing Percentage': missing_pct.values
})
```
Creates a nice table:
```
Column          Missing Count    Missing Percentage
MarkDown2       310322           73.61%
MarkDown4       286603           67.98%
MarkDown3       284479           67.48%
```

```python
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
```
- Filter: Only show columns with missing values
- Sort: Most missing at top

**Returns:**  
DataFrame with missing value report

---

### Function: `detect_outliers(df, column='Weekly_Sales', method='IQR')`

**What it does:**  
Finds unusual values (outliers) in sales data.

**Method: IQR (Interquartile Range)**

```python
Q1 = df[column].quantile(0.25)  # 25th percentile
Q3 = df[column].quantile(0.75)  # 75th percentile
IQR = Q3 - Q1                    # Interquartile range
```

**What these mean:**
- Q1 = 25% of data is below this value (e.g., $2,079.65)
- Q3 = 75% of data is below this value (e.g., $20,205.85)
- IQR = Range of middle 50% of data

**Visual representation:**
```
Min        Q1        Median      Q3        Max
|----------|-----------|----------|---------|
           |<---  IQR  --->|
```

```python
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
```

**Outlier rule:**  
Value is outlier if < lower_bound OR > upper_bound

**Example:**
```
Q1 = 2,079.65
Q3 = 20,205.85
IQR = 18,126.20
lower_bound = 2,079.65 - 1.5 * 18,126.20 = -25,109.65
upper_bound = 20,205.85 + 1.5 * 18,126.20 = 47,395.16

Weekly_Sales = -1,000  → outlier (< -25,109.65? No, but unusual negative)
Weekly_Sales = 100,000 → outlier (> 47,395.16? YES!)
```

```python
outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
```
- Filters rows that are outliers
- `|` means OR

**Our results:**  
35,521 outliers (8.43% of data)

**Returns:**  
DataFrame containing only outlier rows

---

### Function: `plot_sales_trends(df, save_dir="results")`

**What it does:**  
Creates 3 visualization files showing sales patterns.

#### Plot 1: Sales Trends Overview (4 subplots)

```python
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
```
- Creates 2×2 grid of plots
- `figsize=(16, 12)` - Width 16 inches, height 12 inches

**Subplot 1: Overall Trend**
```python
daily_sales = df.groupby('Date')['Weekly_Sales'].sum().reset_index()
axes[0, 0].plot(daily_sales['Date'], daily_sales['Weekly_Sales'], linewidth=2)
```
- Groups by Date, sums all stores/depts for that date
- Line plot showing total sales over time
- Shows seasonality, trends, spikes

**Subplot 2: Sales by Store**
```python
store_sales = df.groupby('Store')['Weekly_Sales'].sum().sort_values(ascending=False)
axes[0, 1].bar(range(len(store_sales)), store_sales.values)
```
- Bar chart: Each bar = one store
- Sorted from highest to lowest sales
- Shows which stores are top performers

**Subplot 3: Top 20 Departments**
```python
dept_sales = df.groupby('Dept')['Weekly_Sales'].sum().sort_values(ascending=False).head(20)
axes[1, 0].barh(range(len(dept_sales)), dept_sales.values)
```
- Horizontal bar chart (`.barh`)
- Shows top 20 departments only
- Helps identify key product categories

**Subplot 4: Monthly Pattern**
```python
df['Month'] = df['Date'].dt.month  # Extract month (1-12)
monthly_sales = df.groupby('Month')['Weekly_Sales'].mean()
axes[1, 1].plot(monthly_sales.index, monthly_sales.values, marker='o')
```
- Average sales per month
- Shows seasonality (Dec higher? Summer lower?)
- `marker='o'` adds dots on line

```python
plt.tight_layout()
plt.savefig(f"{save_dir}/sales_trends_overview.png", dpi=300, bbox_inches='tight')
```
- `.tight_layout()` - Adjusts spacing so labels don't overlap
- `.savefig()` - Saves to file
- `dpi=300` - High resolution (300 dots per inch)
- `bbox_inches='tight'` - Crop whitespace

#### Plot 2: Holiday Analysis

```python
holiday_sales = df.groupby('IsHoliday')['Weekly_Sales'].mean()
axes[0].bar(['Non-Holiday', 'Holiday'], holiday_sales.values, color=['skyblue', 'coral'])
```
- Compares average sales: Holiday vs Non-Holiday weeks
- Two bars showing difference

```python
axes[1].hist(df[df['IsHoliday']==False]['Weekly_Sales'], bins=50, alpha=0.6, label='Non-Holiday')
axes[1].hist(df[df['IsHoliday']==True]['Weekly_Sales'], bins=50, alpha=0.6, label='Holiday')
```
- Histogram = distribution of values
- `bins=50` - Divide range into 50 buckets
- `alpha=0.6` - Transparency (can see overlap)
- Shows if holiday sales have different distribution

#### Plot 3: External Variables

```python
external_vars = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
for i, var in enumerate(external_vars):
    axes[i].scatter(df[var], df['Weekly_Sales'], alpha=0.3, s=10)
```
- 4 scatter plots (one per variable)
- X-axis = external variable (Temperature)
- Y-axis = Weekly_Sales
- Each point = one week
- Shows relationship (correlation)

```python
corr = df[[var, 'Weekly_Sales']].corr().iloc[0, 1]
axes[i].text(0.05, 0.95, f'Corr: {corr:.3f}', ...)
```
- Calculates correlation coefficient
- `.corr()` - Correlation matrix
- `.iloc[0, 1]` - Gets correlation value (row 0, col 1)
- Displays as text on plot

---

### Function: `generate_eda_report(df, save_dir="results")` ⭐ MAIN FUNCTION

**What it does:**  
Runs all EDA functions and prints comprehensive report.

```python
def generate_eda_report(df, save_dir="results"):
    Path(save_dir).mkdir(exist_ok=True)  # Create results/ folder
    
    print(f"\nDataset Shape: {df.shape}")           # (421570, 16)
    print(f"\nColumn Names: {list(df.columns)}")    # ['Store', 'Dept', ...]
    print(f"\nData Types:\n{df.dtypes}")            # int64, float64, datetime64
    print(f"\nBasic Statistics:\n{df.describe()}")  # mean, std, min, max, etc.
    
    missing_df = check_missing_values(df)           # Call missing values function
    outliers = detect_outliers(df, 'Weekly_Sales')  # Call outlier detection
    plot_sales_trends(df, save_dir)                 # Call plotting function
```

**Output:**
- Prints to screen: shape, columns, statistics
- Prints missing value table
- Prints outlier statistics
- Creates 3 PNG files

---

# 3. feature_engineering.py

## Purpose
Creates 49 new features from original 16 columns.

---

### Function: `create_lag_features(df, target_col='Weekly_Sales', lags=[1, 2, 4, 12])`

**What it does:**  
Creates features that look back in time.

**How it works:**

```python
for lag in lags:
    df[f'{target_col}_lag_{lag}'] = df.groupby(['Store', 'Dept'])[target_col].shift(lag)
```

**Breaking it down:**

1. `df.groupby(['Store', 'Dept'])` - Separate data by Store-Dept combination
2. `[target_col]` - Select Weekly_Sales column
3. `.shift(lag)` - Move values down by `lag` rows

**Example with shift(1):**

Before:
```
Store | Dept | Date       | Weekly_Sales
1     | 5    | 2010-02-05 | 10000
1     | 5    | 2010-02-12 | 12000
1     | 5    | 2010-02-19 | 11000
```

After `.shift(1)`:
```
Store | Dept | Date       | Weekly_Sales | Weekly_Sales_lag_1
1     | 5    | 2010-02-05 | 10000        | NaN (no previous week)
1     | 5    | 2010-02-12 | 12000        | 10000 (last week's sales)
1     | 5    | 2010-02-19 | 11000        | 12000 (last week's sales)
```

**Why groupby is important:**

Without groupby (WRONG):
```
Store 1, Dept 5, Week 10  → lag_1 = Store 1, Dept 5, Week 9  ✓
Store 1, Dept 6, Week 1   → lag_1 = Store 1, Dept 5, Week 10  ✗ (wrong dept!)
```

With groupby (CORRECT):
```
Store 1, Dept 5, Week 10  → lag_1 = Store 1, Dept 5, Week 9   ✓
Store 1, Dept 6, Week 1   → lag_1 = NaN (first week for this group) ✓
```

**All lags created:**
- `lag_1` = Last week
- `lag_2` = 2 weeks ago
- `lag_4` = 4 weeks ago (1 month)
- `lag_12` = 12 weeks ago (3 months)

---

### Function: `create_rolling_features(df, target_col='Weekly_Sales', windows=[4, 8, 12])`

**What it does:**  
Creates moving averages and statistics over past N weeks.

**Example: 4-week rolling mean**

```python
df['Weekly_Sales_rolling_mean_4'] = (
    df.groupby(['Store', 'Dept'])['Weekly_Sales']
    .shift(1)                          # Don't include current week
    .rolling(window=4, min_periods=1)  # Look at past 4 weeks
    .mean()                            # Calculate average
    .reset_index(0, drop=True)
)
```

**Step-by-step example:**

Data:
```
Week 1: 5000
Week 2: 6000
Week 3: 7000
Week 4: 8000
Week 5: 9000  ← We want to predict this
```

For Week 5, rolling_mean_4:
```
.shift(1)  → Use weeks 1,2,3,4 (not week 5)
.rolling(window=4).mean() → (5000 + 6000 + 7000 + 8000) / 4 = 6500
```

**Why shift(1)?**  
In real forecasting, we don't know current week's sales yet!
```
WRONG: Use weeks 2,3,4,5 to predict week 5  (cheating - includes future!)
RIGHT: Use weeks 1,2,3,4 to predict week 5  (only past data)
```

**All rolling features created:**

For each window (4, 8, 12):
- `rolling_mean_{window}` - Average
- `rolling_std_{window}` - Volatility (standard deviation)
- `rolling_max_{window}` - Highest value
- `rolling_min_{window}` - Lowest value

Total: 3 windows × 4 stats = 12 rolling features

**What they capture:**
- Mean: General trend level
- Std: How volatile/stable sales are
- Max: Peak sales in recent period
- Min: Floor sales in recent period

---

### Function: `create_calendar_features(df, date_col='Date')`

**What it does:**  
Extracts time-based patterns from dates.

**Simple features:**

```python
df['week'] = df['Date'].dt.isocalendar().week       # 1-52
df['month'] = df['Date'].dt.month                   # 1-12
df['quarter'] = df['Date'].dt.quarter               # 1-4 (Q1, Q2, Q3, Q4)
df['year'] = df['Date'].dt.year                     # 2010, 2011, 2012
df['day_of_week'] = df['Date'].dt.dayofweek         # 0=Monday, 6=Sunday
df['day_of_month'] = df['Date'].dt.day              # 1-31
```

**Cyclical encoding (sin/cos):**

**Problem with raw numbers:**
```
December = 12
January = 1
Numerically: 12 - 1 = 11 (far apart)
Reality: December and January are next to each other!
```

**Solution: Sine and Cosine encoding**

```python
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
```

**How it works:**

Imagine months on a clock (circle):
```
        12 (Dec)
    11        1 (Jan)
 10              2
9                 3
 8               4
    7         5
        6
```

Convert to angle:
```
Month 1 (Jan):  1/12 * 360° = 30°
Month 6 (Jun):  6/12 * 360° = 180°
Month 12 (Dec): 12/12 * 360° = 360° = 0° (full circle)
```

Calculate sin and cos:
```
Month 1:  sin(30°) = 0.5,   cos(30°) = 0.87
Month 6:  sin(180°) = 0,    cos(180°) = -1
Month 12: sin(360°) = 0,    cos(360°) = 1
```

**Why this works:**

Distance between December (month 12) and January (month 1):
```
Raw numbers: |12 - 1| = 11  ✗ (seems far)

Using sin/cos:
Dec: (sin=0, cos=1)
Jan: (sin=0.5, cos=0.87)
Euclidean distance = √[(0-0.5)² + (1-0.87)²] = 0.27  ✓ (close!)
```

Model learns December ≈ January because their sin/cos values are similar!

---

### Function: `create_external_features(df)`

**What it does:**  
Processes and creates lagged versions of external variables.

```python
external_cols = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']

for col in external_cols:
    df[col] = df[col].fillna(df[col].median())  # Fill missing with median
```

**Fill missing values:**
- Some stores might have missing Temperature, etc.
- `.median()` - Middle value (robust to outliers)
- Fill missing with typical value

```python
for col in external_cols:
    df[f'{col}_lag_1'] = df.groupby('Store')[col].shift(1)
    df[f'{col}_lag_4'] = df.groupby('Store')[col].shift(4)
```

**Create lags:**
- `Temperature_lag_1` = Last week's temperature
- `Fuel_Price_lag_4` = Fuel price 4 weeks ago

**Why lags for external features?**  
Economic indicators take time to affect behavior.

Example:
```
Week 1: Fuel price jumps from $2.50 to $4.00
Week 1: People still shop normally (haven't adjusted yet)
Week 2-3: People start driving less, shopping less
Week 4: Reduced shopping is visible in sales
```

So `Fuel_Price_lag_4` might predict current sales better than current fuel price!

---

### Function: `create_all_features(df, ...)` ⭐ MAIN FUNCTION

**What it does:**  
Calls all feature engineering functions in sequence.

```python
def create_all_features(df, target_col='Weekly_Sales', 
                       lags=[1, 2, 4, 12],
                       rolling_windows=[4, 8, 12]):
    
    df = create_calendar_features(df)           # Add week, month, sin/cos
    df = create_lag_features(df, target_col, lags)  # Add lag_1, lag_2, lag_4, lag_12
    df = create_rolling_features(df, target_col, rolling_windows)  # Add rolling stats
    df = create_external_features(df)           # Process external vars
    
    # Convert to categorical (more efficient for tree models)
    df['Store'] = df['Store'].astype('category')
    df['Dept'] = df['Dept'].astype('category')
    df['Type'] = df['Type'].astype('category')
    
    return df
```

**Feature count:**
- Calendar: 13 features (week, month, quarter, year, day_of_week, day_of_month, week_of_year, + 6 cyclical)
- Lag: 4 lags × 5 variables (Sales, Temp, Fuel, CPI, Unemployment) = 20... wait, let me recount
  - Actually: 4 lags for Sales + 2 lags × 4 external vars = 4 + 8 = 12 lags
- Rolling: 3 windows × 4 stats = 12 features
- External: 9 original + 8 lagged = 17
- Store features: 3 (Store, Dept, Type)

Total ≈ 49 features

---

### Function: `get_feature_columns(df, exclude_cols=[...])`

**What it does:**  
Returns list of column names to use as features (excludes target and identifiers).

```python
def get_feature_columns(df, exclude_cols=['Date', 'Weekly_Sales', 'Store', 'Dept']):
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols
```

**Example:**
```python
All columns: ['Store', 'Dept', 'Date', 'Weekly_Sales', 'Temperature', 'lag_1', 'rolling_mean_4', ...]
Exclude: ['Date', 'Weekly_Sales', 'Store', 'Dept']
Feature cols: ['Temperature', 'lag_1', 'rolling_mean_4', ...]  ← 49 features
```

---

# 4. utils.py

## Purpose
Helper functions for data preparation.

---

### Function: `time_based_split(df, date_col='Date', train_ratio=0.8)`

**What it does:**  
Splits data by time (80% train, 20% validation).

```python
def time_based_split(df, date_col='Date', train_ratio=0.8):
    df = df.sort_values(date_col).reset_index(drop=True)
```
- Sort by date (oldest first)
- Reset row numbers

```python
split_idx = int(len(df) * train_ratio)
```
- `len(df)` = 421,570 rows
- `0.8 * 421,570` = 337,256
- Split at row 337,256

```python
train_df = df.iloc[:split_idx].copy()   # Rows 0 to 337,255
test_df = df.iloc[split_idx:].copy()    # Rows 337,256 to end
```
- `.iloc[:337256]` - First 337,256 rows
- `.iloc[337256:]` - Remaining rows
- `.copy()` - Make copy (not just view)

**Result:**
```
Training: 2010-02-05 to 2012-04-13 (80%)
Validation: 2012-04-13 to 2012-10-26 (20%)
```

**Why time-based?**  
Forecasting = predicting future from past.  
Random split would mix future into training (cheating!).

```
WRONG (random split):
Train: [2010-Jan, 2011-Mar, 2012-Jul, ...]  ← Has 2012 data
Val:   [2010-Feb, 2011-Apr, 2012-Aug, ...]  ← Model saw future!

RIGHT (time-based):
Train: [2010-Jan, 2010-Feb, ..., 2012-Apr]  ← All past
Val:   [2012-Apr, 2012-May, ..., 2012-Oct]  ← All future
```

---

### Function: `prepare_ml_data(df, target_col='Weekly_Sales', feature_cols=None)`

**What it does:**  
Converts DataFrame to X (features) and y (target) for ML models.

```python
def prepare_ml_data(df, target_col='Weekly_Sales', feature_cols=None):
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col not in ['Date', 'Weekly_Sales']]
```
- If no feature list provided, use all columns except Date and target

```python
# Handle categorical columns
df_processed = df.copy()
for col in df_processed.columns:
    if df_processed[col].dtype == 'category':
        df_processed[col] = df_processed[col].cat.codes
```

**Convert categorical to numbers:**

Before:
```
Type: ['A', 'B', 'C', 'A', 'B']
```

After:
```
Type: [0, 1, 2, 0, 1]
```

Why? ML models need numbers, not text.

```python
# Select numeric columns only
numeric_cols = []
for col in feature_cols:
    if col in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[col]):
        numeric_cols.append(col)
```
- Filter to only numeric columns
- Excludes text columns that can't be converted

```python
X = df_processed[numeric_cols].fillna(0)
y = df_processed[target_col].values
```
- `X` = Features DataFrame (337,256 × 49)
- `.fillna(0)` - Replace NaN with 0
- `y` = Target array (337,256 values)
- `.values` - Convert to numpy array

**Returns:**
```python
return X, y
```

**Example:**
```python
X = [[42.31, 2.572, 100, 12000, ...],   # Row 1: 49 features
     [45.07, 2.548, 95,  13000, ...],   # Row 2: 49 features
     ...]
     
y = [24924.50, 46039.49, ...]           # Target values
```

---

# 5. models.py (Covered separately - see next section)

This file contains all 5 model classes. I'll create a detailed explanation in the next section.

---

# 6. evaluation.py

## Purpose
Calculate metrics and create visualizations.

---

### Function: `calculate_rmse(y_true, y_pred)`

```python
def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
```

**Step-by-step:**

1. `(y_true - y_pred)` - Calculate errors
   ```
   actual = [10000, 12000, 11000]
   pred   = [10500, 11800, 10900]
   errors = [-500,   200,   100]
   ```

2. `** 2` - Square each error
   ```
   squared = [250000, 40000, 10000]
   ```

3. `np.mean(...)` - Average
   ```
   mean = (250000 + 40000 + 10000) / 3 = 100000
   ```

4. `np.sqrt(...)` - Square root
   ```
   rmse = √100000 = 316.23
   ```

**Why square then sqrt?**
- Penalizes large errors more heavily
- Returns error in original units (dollars)

---

### Function: `calculate_mae(y_true, y_pred)`

```python
def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
```

**Simpler than RMSE:**

1. `(y_true - y_pred)` - Errors: [-500, 200, 100]
2. `np.abs(...)` - Absolute values: [500, 200, 100]
3. `np.mean(...)` - Average: (500 + 200 + 100) / 3 = 266.67

**Interpretation:**  
"On average, predictions are off by $266.67"

---

### Function: `calculate_mape(y_true, y_pred)`

```python
def calculate_mape(y_true, y_pred):
    mask = y_true != 0  # Avoid division by zero
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
```

**Percentage error:**

```
actual = 10000,  pred = 10500
error% = |10000 - 10500| / 10000 = 500 / 10000 = 0.05 = 5%
```

**Why mask?**  
If actual = 0, we'd divide by zero → infinity!

---

### Function: `evaluate_model(y_true, y_pred, model_name)`

```python
def evaluate_model(y_true, y_pred, model_name="Model"):
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
```

**Returns dictionary:**
```python
{
    'Model': 'LightGBM',
    'RMSE': 2880.83,
    'MAE': 1393.87,
    'MAPE': 2220.36
}
```

---

### Function: `compare_models(results_list, save_path)`

```python
def compare_models(results_list, save_path="results/model_comparison.csv"):
    comparison_df = pd.DataFrame(results_list)
    comparison_df = comparison_df.sort_values('RMSE')  # Sort by RMSE
    comparison_df.to_csv(save_path, index=False)       # Save to CSV
    print(comparison_df.to_string(index=False))         # Print table
    return comparison_df
```

**Input:**
```python
[
    {'Model': 'Naive', 'RMSE': 32047.87, ...},
    {'Model': 'LightGBM', 'RMSE': 2880.83, ...},
    ...
]
```

**Output (sorted table):**
```
Model                RMSE       MAE        MAPE
LightGBM          2880.83   1393.87   2220.36
XGBoost           3022.41   1535.04   5412.53
...
```

---

### Function: `plot_predictions(y_true, y_pred, model_name, save_path)`

**What it creates:**  
Two-panel plot for one model.

**Left panel: Scatter plot**
```python
axes[0].scatter(y_true, y_pred, alpha=0.5, s=20)
axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
```
- Each point = one prediction
- X-axis = actual value
- Y-axis = predicted value
- Red diagonal line = perfect prediction
- Points near line = good predictions
- Points far from line = bad predictions

**Right panel: Time series**
```python
sample_size = min(200, len(y_true))
indices = np.random.choice(len(y_true), sample_size, replace=False)
axes[1].plot(range(sample_size), y_true[indices], label='Actual')
axes[1].plot(range(sample_size), y_pred[indices], label='Predicted')
```
- Shows 200 random samples
- Blue line = actual sales
- Orange line = predicted sales
- How closely lines track = how good model is

---

### Function: `plot_all_predictions(predictions_dict, save_path)`

**What it creates:**  
Grid of scatter plots (one per model).

```python
n_models = len(predictions_dict)
n_cols = 3
n_rows = (n_models + n_cols - 1) // n_cols  # Ceiling division
```

**Layout calculation:**
```
6 models ÷ 3 columns = 2 rows
7 models ÷ 3 columns = 3 rows (ceiling)
```

Creates 3-column grid with as many rows as needed.

---

**That's all the imported functions explained! Next, I'll explain the models in detail...**

Would you like me to continue with a detailed explanation of all 5 models (Naive, Moving Average, LightGBM, XGBoost, LSTM)?

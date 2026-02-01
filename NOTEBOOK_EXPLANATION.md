# Notebook Line-by-Line Explanation
## sales_forecasting_pipeline.ipynb

This document explains every cell in the Jupyter notebook step by step.

---

## 📝 Cell 0: Title and Overview (Markdown)

```markdown
# Walmart Sales Forecasting System
...
```

**What it does:** 
- This is just documentation (not code)
- Explains what the notebook is about
- Lists the dataset, goal, and evaluation metrics

**Key points:**
- Dataset: Walmart Store Sales from Kaggle
- Goal: Compare simple models vs advanced models
- Metrics: RMSE, MAE, MAPE

---

## 📝 Cell 1: Section Header (Markdown)

```markdown
## Step 1: Load Dataset
```

**What it does:** 
- Just a section divider
- Helps organize the notebook into logical steps

---

## 💻 Cell 2: Import Libraries and Load Data (Code)

```python
import sys
import os
sys.path.append('../')
```

**Line-by-line:**

1. `import sys` - Import Python's system module (for path manipulation)
2. `import os` - Import operating system module (for file operations)
3. `sys.path.append('../')` - Add parent directory to Python's search path
   - **Why?** The notebook is in `notebooks/` folder but needs to import from `src/` folder
   - `../` means "go up one directory" (from notebooks/ to capstone/)
   - Now Python can find `src.data_loader`, `src.models`, etc.

```python
from src.data_loader import load_dataset
import pandas as pd
import numpy as np
```

4. `from src.data_loader import load_dataset` - Import the function that downloads Kaggle data
5. `import pandas as pd` - Import pandas (data manipulation library), nickname it "pd"
6. `import numpy as np` - Import numpy (numerical computing), nickname it "np"

```python
# Load dataset
train_df, test_df, features_df, stores_df = load_dataset()
```

7. `train_df, test_df, features_df, stores_df = load_dataset()` - Call the function
   - **What it does:** Downloads Walmart data from Kaggle (if not cached)
   - **Returns 4 dataframes:**
     - `train_df` = Historical sales data (421,570 rows)
     - `test_df` = Future period for prediction (115,064 rows)
     - `features_df` = External factors (temperature, CPI, etc.)
     - `stores_df` = Store characteristics (size, type)
   - The function automatically merges features and stores into train/test

```python
print(f"\nTrain dataset shape: {train_df.shape}")
print(f"Test dataset shape: {test_df.shape}")
print(f"\nTrain columns: {list(train_df.columns)}")
```

8-10. **Print statements** to verify data loaded correctly:
   - `.shape` = (rows, columns) - e.g., (421570, 16)
   - `.columns` = list of column names
   - `f"..."` = f-string (formatted string) - inserts variables into text

**Output example:**
```
Train dataset shape: (421570, 16)
Test dataset shape: (115064, 15)
Train columns: ['Store', 'Dept', 'Date', 'Weekly_Sales', 'Temperature', ...]
```

---

## 📝 Cell 3: Section Header (Markdown)

```markdown
## Step 2: Exploratory Data Analysis
```

**What it does:** Section divider for EDA step

---

## 💻 Cell 4: Run EDA (Code)

```python
from src.eda import generate_eda_report, check_missing_values, detect_outliers, plot_sales_trends
```

**What it imports:**
- `generate_eda_report` - Main function that runs all EDA and creates plots
- `check_missing_values` - Finds missing data
- `detect_outliers` - Identifies unusual values
- `plot_sales_trends` - Creates time series visualizations

```python
# Generate comprehensive EDA report
generate_eda_report(train_df, save_dir='../results')
```

**What this does:**
1. Takes `train_df` (the training data)
2. Analyzes it:
   - Checks for missing values (finds MarkDown1-5 are 64-74% missing)
   - Detects outliers (8.43% of sales are outliers)
   - Calculates statistics (mean, median, std, min, max)
   - Creates visualizations
3. Saves 3 PNG files to `results/` folder:
   - `sales_trends_overview.png` - Sales over time, by store type, distribution
   - `holiday_analysis.png` - Holiday vs non-holiday comparison
   - `external_variables_analysis.png` - Temperature, fuel, CPI, unemployment trends

**Why important?** Understand data before modeling - find problems, patterns, relationships

---

## 📝 Cell 5: Section Header (Markdown)

```markdown
## Step 3: Feature Engineering
```

---

## 💻 Cell 6: Create Features (Code)

```python
from src.feature_engineering import create_all_features, get_feature_columns
```

**Imports:**
- `create_all_features` - Function that creates 49 new features
- `get_feature_columns` - Function that lists all feature column names

```python
train_df_featured = create_all_features(
    train_df.copy(),
    target_col='Weekly_Sales',
    lags=[1, 2, 4, 12],
    rolling_windows=[4, 8, 12]
)
```

**Line-by-line:**

1. `train_df.copy()` - Make a copy of original data (don't modify original)
2. `target_col='Weekly_Sales'` - What we're trying to predict
3. `lags=[1, 2, 4, 12]` - Create lag features:
   - lag_1 = last week's sales
   - lag_2 = 2 weeks ago
   - lag_4 = 4 weeks ago (1 month)
   - lag_12 = 12 weeks ago (3 months)
4. `rolling_windows=[4, 8, 12]` - Create rolling statistics:
   - 4-week window: mean, std, max, min
   - 8-week window: mean, std, max, min
   - 12-week window: mean, std, max, min

**What gets created:**
- 4 lag features × 3 variables (Sales, Temp, Fuel, CPI, Unemployment) = 12 lags
- 3 windows × 4 stats = 12 rolling features
- 8 calendar features (week, month, quarter, year, cyclical encoding)
- Plus external variables
- **Total: 49 features**

```python
train_df_featured = train_df_featured.dropna(subset=['Weekly_Sales']).reset_index(drop=True)
```

**What this does:**
1. `.dropna(subset=['Weekly_Sales'])` - Remove rows where Weekly_Sales is NaN
   - **Why?** Lag features create NaN in first few rows (can't look back 12 weeks for week 1)
   - We need Weekly_Sales for training, so drop rows without it
2. `.reset_index(drop=True)` - Renumber rows from 0, 1, 2, ...
   - After dropping rows, indices have gaps
   - Reset makes them sequential again

```python
print(f"\nDataset shape after feature engineering: {train_df_featured.shape}")
print(f"\nFeature columns: {len(get_feature_columns(train_df_featured))}")
```

**Prints:**
- New shape (fewer rows due to dropna, more columns due to new features)
- Number of feature columns (49)

**Example output:**
```
Dataset shape after feature engineering: (421570, 65)
Feature columns: 49
```

---

## 📝 Cell 7: Section Header (Markdown)

```markdown
## Step 4: Time-Based Train-Test Split
```

---

## 💻 Cell 8: Split Data for Training and Validation (Code)

```python
from src.utils import time_based_split
```

**Import:** Function to split data by time (not randomly)

```python
train_data, val_data = time_based_split(train_df_featured, date_col='Date', train_ratio=0.8)
```

**What this does:**

1. Takes `train_df_featured` (data with 49 features)
2. `date_col='Date'` - Column to sort by
3. `train_ratio=0.8` - Use 80% for training, 20% for validation

**How it works:**
- Sorts data by Date (oldest to newest)
- Splits at 80% point:
  - **Training**: 2010-02-05 to 2012-04-13 (first 80%)
  - **Validation**: 2012-04-13 to 2012-10-26 (last 20%)
- **Why time-based?** In real world, we predict future using past
  - Random split would "cheat" by training on future data

**Result:**
- `train_data` = 337,256 rows
- `val_data` = 84,314 rows

```python
from src.utils import prepare_ml_data
from src.feature_engineering import get_feature_columns
```

**Import more functions:**
- `prepare_ml_data` - Separates features (X) from target (y)
- `get_feature_columns` - Gets list of feature column names

```python
feature_cols = get_feature_columns(train_data)
```

**What this does:**
- Extracts list of 49 feature column names
- Excludes: 'Date', 'Weekly_Sales', 'Store', 'Dept' (not features for ML models)

```python
X_train, y_train = prepare_ml_data(train_data, feature_cols=feature_cols)
X_val, y_val = prepare_ml_data(val_data, feature_cols=feature_cols)
```

**What this does:**

For training data:
- `X_train` = Features only (337,256 rows × 49 columns) - the inputs
- `y_train` = Target only (337,256 values) - what we predict

For validation data:
- `X_val` = Features (84,314 rows × 49 columns)
- `y_val` = Target (84,314 values)

**Inside prepare_ml_data:**
- Selects only numeric columns
- Fills missing values with 0
- Converts categorical to numbers
- Returns X (features) and y (target)

```python
print(f"\nTraining features shape: {X_train.shape}")
print(f"Validation features shape: {X_val.shape}")
```

**Prints shapes to verify:**
```
Training features shape: (337256, 49)
Validation features shape: (84314, 49)
```

---

## 📝 Cell 9: Section Header (Markdown)

```markdown
## Step 5: Baseline Models
```

---

## 💻 Cell 10: Train Baseline Models (Code)

```python
from src.models import NaiveForecast, MovingAverage
from src.evaluation import evaluate_model
```

**Imports:**
- `NaiveForecast` - Simplest model (predict = last week's value)
- `MovingAverage` - Average of past N weeks
- `evaluate_model` - Calculates RMSE, MAE, MAPE

### Naive Forecast

```python
print("\n" + "="*60)
print("Training Baseline Models")
print("="*60)
```

**Just prints a header:**
```
============================================================
Training Baseline Models
============================================================
```

```python
naive_model = NaiveForecast()
```

**Creates naive model object** (doesn't train yet, just initializes)

```python
naive_model.fit(train_data, target_col='Weekly_Sales')
```

**Training the naive model:**

1. `train_data` - The training dataframe
2. `target_col='Weekly_Sales'` - What column to predict

**What fit() does internally:**
- Groups by Store and Dept (e.g., Store 1 Dept 5, Store 1 Dept 6, etc.)
- For each group, stores the **last value** of Weekly_Sales
- Example: Store 1, Dept 5 → last training week = $10,523.40
- Stores in a dictionary: `{(1, 5): 10523.40, (1, 6): 8234.12, ...}`

```python
naive_pred = naive_model.predict(val_data)
```

**Making predictions:**

1. Takes `val_data` (validation dataframe)
2. For each row:
   - Looks up Store and Dept
   - Returns stored last value for that Store-Dept
3. Returns array of predictions (84,314 values)

**Example:**
- Val row 1: Store 1, Dept 5 → predict $10,523.40
- Val row 2: Store 1, Dept 5 → predict $10,523.40 (same)
- Val row 50: Store 2, Dept 10 → predict $7,891.23 (different group)

```python
naive_results = evaluate_model(y_val, naive_pred, "Naive Forecast")
```

**Evaluation:**

1. `y_val` - Actual sales (ground truth)
2. `naive_pred` - Predicted sales
3. `"Naive Forecast"` - Model name for display

**What evaluate_model() does:**
- Calculates RMSE = √(mean((actual - predicted)²))
- Calculates MAE = mean(|actual - predicted|)
- Calculates MAPE = mean(|actual - predicted| / actual) × 100
- Returns dictionary: `{'Model': 'Naive Forecast', 'RMSE': 32047.87, 'MAE': 20458.72, 'MAPE': 142481.83}`

```python
print(f"\nNaive Forecast - RMSE: {naive_results['RMSE']:.2f}, MAE: {naive_results['MAE']:.2f}, MAPE: {naive_results['MAPE']:.2f}%")
```

**Prints results:**
```
Naive Forecast - RMSE: 32047.87, MAE: 20458.72, MAPE: 142481.83%
```

### Moving Average

```python
ma_model = MovingAverage(window=4)
```

**Creates moving average model:**
- `window=4` means use last 4 weeks

```python
ma_model.fit(train_data, target_col='Weekly_Sales')
```

**Training:**
- For each Store-Dept group
- Stores last 4 values of Weekly_Sales
- Example: Store 1, Dept 5 → [9800, 10200, 10500, 10700]

```python
ma_pred = ma_model.predict(val_data)
```

**Prediction:**
- For each validation row
- Looks up Store-Dept
- Predicts = average of stored 4 values
- Example: (9800 + 10200 + 10500 + 10700) / 4 = 10,300

```python
ma_results = evaluate_model(y_val, ma_pred, "Moving Average (4 weeks)")
print(f"Moving Average - RMSE: {ma_results['RMSE']:.2f}, MAE: {ma_results['MAE']:.2f}, MAPE: {ma_results['MAPE']:.2f}%")
```

**Evaluate and print:**
```
Moving Average - RMSE: 31094.09, MAE: 19891.78, MAPE: 136619.94%
```

**Slightly better than naive!**

---

## 📝 Cells 11-12: Section Headers (Markdown)

```markdown
## Step 6: Research-Style Models
### 6.1 Statistical Model: SARIMA
```

---

## 💻 Cell 13: Train SARIMA Model (Code)

```python
from src.models import SARIMAModel
```

**Import:** SARIMA (Seasonal ARIMA) statistical model

```python
print("\n" + "="*60)
print("Training SARIMA Model")
print("="*60)
print("Note: This may take a while. Using sample data for speed...")
```

**Just informational prints** - SARIMA is slow!

```python
sample_stores = train_data['Store'].unique()[:5]  # Use first 5 stores
```

**Create sample:**
- `.unique()` gets list of all store numbers
- `[:5]` takes first 5 stores only
- **Why?** SARIMA trains separate model for each Store-Dept combination
- Full data = 45 stores × 99 depts = 4,455 models (takes hours!)
- Sample = 5 stores reduces to ~500 models (faster demo)

```python
sample_train = train_data[train_data['Store'].isin(sample_stores)].copy()
sample_val = val_data[val_data['Store'].isin(sample_stores)].copy()
```

**Filter data:**
- `train_data['Store'].isin(sample_stores)` - Boolean mask (True if Store is in [1,2,3,4,5])
- `train_data[...]` - Keep only those rows
- `.copy()` - Make a copy

Result: Training and validation data for only 5 stores

```python
sarima_model = SARIMAModel(order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
```

**Create SARIMA model with parameters:**

- `order=(1, 1, 1)` - Non-seasonal parameters:
  - 1 = AR (autoregressive) - use 1 past value
  - 1 = I (integrated) - difference once to make stationary
  - 1 = MA (moving average) - use 1 past error
  
- `seasonal_order=(1, 1, 1, 52)` - Seasonal parameters:
  - 1 = Seasonal AR
  - 1 = Seasonal I
  - 1 = Seasonal MA
  - 52 = Seasonality period (52 weeks = 1 year)

**What SARIMA does:**
- Captures trends (increasing/decreasing)
- Captures seasonality (yearly patterns)
- Statistically models time series

```python
sarima_model.fit(sample_train, target_col='Weekly_Sales')
```

**Training:**
- For each Store-Dept in sample
- Fits individual SARIMA model
- Stores fitted models in dictionary
- If model fails (not enough data), stores mean as fallback

```python
sarima_pred = sarima_model.predict(sample_val)
```

**Prediction:**
- For each Store-Dept in validation sample
- Uses fitted SARIMA model to forecast
- Returns array of predictions

```python
if len(sample_val) > 0:
    sarima_y_val = sample_val['Weekly_Sales'].values
    sarima_results = evaluate_model(sarima_y_val, sarima_pred, "SARIMA")
    print(f"\nSARIMA - RMSE: {sarima_results['RMSE']:.2f}, MAE: {sarima_results['MAE']:.2f}, MAPE: {sarima_results['MAPE']:.2f}%")
else:
    sarima_results = {'Model': 'SARIMA', 'RMSE': np.inf, 'MAE': np.inf, 'MAPE': np.inf}
```

**Conditional evaluation:**
- `if len(sample_val) > 0` - Check if we have validation data
- If yes: evaluate and print
- If no: set metrics to infinity (infinity = invalid)

---

## 📝 Cell 14: Section Header (Markdown)

```markdown
### 6.2 Statistical Model: Prophet
```

---

## 💻 Cell 15: Train Prophet Model (Code)

```python
from src.models import ProphetModel
```

**Import:** Prophet (Facebook's forecasting library)

```python
print("\n" + "="*60)
print("Training Prophet Model")
print("="*60)
print("Note: This may take a while. Using sample data for speed...")
```

**Informational header** - Prophet is also slow

```python
# Use same sample for consistency
prophet_model = ProphetModel(yearly_seasonality=True, weekly_seasonality=True)
```

**Create Prophet model:**
- `yearly_seasonality=True` - Capture yearly patterns (holidays, seasons)
- `weekly_seasonality=True` - Capture weekly patterns (weekends)
- `daily_seasonality=False` - Not used (we have weekly data)

**What Prophet does:**
- Decomposes time series into trend + seasonality + holidays
- Flexible to missing data and outliers
- Designed for business forecasting

```python
prophet_model.fit(sample_train, target_col='Weekly_Sales', date_col='Date')
```

**Training:**
- Uses same 5-store sample
- `target_col='Weekly_Sales'` - What to predict
- `date_col='Date'` - Time column
- Fits separate Prophet model per Store-Dept

**Internally:**
- Converts to Prophet format: `{'ds': dates, 'y': sales}`
- Fits additive model: y = trend + yearly + weekly + error

```python
prophet_pred = prophet_model.predict(sample_val, date_col='Date')
```

**Prediction:**
- For each Store-Dept in validation
- Uses fitted Prophet model
- Forecasts for validation dates
- Returns predictions

```python
if len(sample_val) > 0:
    prophet_y_val = sample_val['Weekly_Sales'].values
    prophet_results = evaluate_model(prophet_y_val, prophet_pred, "Prophet")
    print(f"\nProphet - RMSE: {prophet_results['RMSE']:.2f}, MAE: {prophet_results['MAE']:.2f}, MAPE: {prophet_results['MAPE']:.2f}%")
else:
    prophet_results = {'Model': 'Prophet', 'RMSE': np.inf, 'MAE': np.inf, 'MAPE': np.inf}
```

**Evaluate and print** (same as SARIMA)

---

## 📝 Cell 16: Section Header (Markdown)

```markdown
### 6.3 Machine Learning Model: LightGBM
```

---

## 💻 Cell 17: Train LightGBM (Code)

```python
from src.models import LightGBMModel
```

**Import:** LightGBM (gradient boosting trees)

```python
print("\n" + "="*60)
print("Training LightGBM Model")
print("="*60)
```

**Header**

```python
lgb_model = LightGBMModel()
```

**Create LightGBM model** with default parameters:
- `objective='regression'` - Predicting continuous values
- `metric='rmse'` - Optimize for RMSE
- `num_leaves=31` - Tree complexity
- `learning_rate=0.05` - Slow, careful learning
- `feature_fraction=0.9` - Use 90% of features per tree
- `bagging_fraction=0.8` - Use 80% of data per tree

```python
lgb_model.fit(X_train, y_train, X_val, y_val)
```

**Training:**

1. `X_train` - Training features (337,256 × 49)
2. `y_train` - Training target (337,256 values)
3. `X_val` - Validation features (for early stopping)
4. `y_val` - Validation target (to monitor performance)

**What happens:**
- Builds decision trees sequentially (boosting)
- Tree 1 predicts, calculates errors
- Tree 2 corrects errors of Tree 1
- Tree 3 corrects errors of Trees 1+2
- Continues until validation error stops improving
- **Stops at iteration 336** (out of max 1000) - early stopping!

**Why it's fast:**
- Optimized for large datasets
- Uses histogram-based algorithm
- Trains in ~30 seconds

```python
lgb_pred = lgb_model.predict(X_val)
```

**Prediction:**
- Takes validation features
- Passes through all 336 trees
- Each tree votes (weighted sum)
- Returns final predictions (84,314 values)

```python
lgb_results = evaluate_model(y_val, lgb_pred, "LightGBM")
print(f"\nLightGBM - RMSE: {lgb_results['RMSE']:.2f}, MAE: {lgb_results['MAE']:.2f}, MAPE: {lgb_results['MAPE']:.2f}%")
```

**Evaluate and print:**
```
LightGBM - RMSE: 2880.83, MAE: 1393.87, MAPE: 2220.36%
```

**🏆 Best model!**

---

## 📝 Cell 18: Section Header (Markdown)

```markdown
### 6.4 Machine Learning Model: XGBoost
```

---

## 💻 Cell 19: Train XGBoost (Code)

```python
from src.models import XGBoostModel
```

**Import:** XGBoost (extreme gradient boosting)

```python
print("\n" + "="*60)
print("Training XGBoost Model")
print("="*60)
```

**Header**

```python
xgb_model = XGBoostModel()
```

**Create XGBoost model** with defaults:
- `objective='reg:squarederror'` - Regression with squared error
- `eval_metric='rmse'` - Evaluation metric
- `max_depth=6` - Maximum tree depth
- `learning_rate=0.05` - Learning speed
- `subsample=0.8` - Row sampling
- `colsample_bytree=0.8` - Column sampling

**Difference from LightGBM:**
- Different tree-building algorithm
- More regularization
- Slightly slower but very robust

```python
xgb_model.fit(X_train, y_train, X_val, y_val)
```

**Training:**
- Same inputs as LightGBM
- Builds 500 trees (no early stopping in our config)
- Each tree corrects previous errors
- Takes ~1-2 minutes

```python
xgb_pred = xgb_model.predict(X_val)
```

**Prediction:**
- Pass validation data through all trees
- Aggregate predictions
- Return results

```python
xgb_results = evaluate_model(y_val, xgb_pred, "XGBoost")
print(f"\nXGBoost - RMSE: {xgb_results['RMSE']:.2f}, MAE: {xgb_results['MAE']:.2f}, MAPE: {xgb_results['MAPE']:.2f}%")
```

**Evaluate and print:**
```
XGBoost - RMSE: 3022.41, MAE: 1535.04, MAPE: 5412.53%
```

**Second best!**

---

## 📝 Cell 20: Section Header (Markdown)

```markdown
### 6.5 Deep Learning Model: LSTM
```

---

## 💻 Cell 21: Train LSTM (Code)

```python
from src.models import LSTMModel
```

**Import:** LSTM (Long Short-Term Memory neural network)

```python
print("\n" + "="*60)
print("Training LSTM Model")
print("="*60)
```

**Header**

```python
# LSTM requires special handling
lstm_model = LSTMModel(sequence_length=12, units=50, epochs=20, batch_size=32)
```

**Create LSTM model with parameters:**

- `sequence_length=12` - Look at past 12 weeks to predict week 13
- `units=50` - Number of neurons per LSTM layer
- `epochs=20` - Training iterations (passes through data)
- `batch_size=32` - Process 32 samples at a time

**LSTM architecture:**
```
Input: (12 weeks × 5 features)
    ↓
LSTM Layer 1 (50 units, return sequences)
    ↓
Dropout (20% - prevents overfitting)
    ↓
LSTM Layer 2 (50 units)
    ↓
Dropout (20%)
    ↓
Dense Layer (25 units)
    ↓
Output (1 value - next week's sales)
```

```python
lstm_model.fit(train_data, target_col='Weekly_Sales', feature_cols=feature_cols)
```

**Training (special for LSTM):**

1. **Aggregates by Date** (not Store-Dept):
   - Sums Weekly_Sales per date
   - Averages features per date
   - Creates daily time series
   
2. **Scales data** to 0-1 range:
   - Features: scaled with MinMaxScaler
   - Target: scaled separately
   - **Why?** Neural networks train better with normalized inputs

3. **Creates sequences**:
   ```
   Date 1-12  → predict Date 13
   Date 2-13  → predict Date 14
   Date 3-14  → predict Date 15
   ...
   ```

4. **Trains neural network**:
   - Forward pass: input → layers → output
   - Calculate loss (error)
   - Backward pass: adjust weights
   - Repeat for 20 epochs

**Note:** If TensorFlow not installed, returns fallback (zeros or mean)

```python
lstm_pred = lstm_model.predict(val_data, feature_cols=feature_cols)
```

**Prediction (our fixed version):**

1. **Aggregates validation by Date**
2. **Combines with training data** (need history for sequences)
3. **For each validation date:**
   - Take past 12 weeks as sequence
   - Scale features
   - Pass through LSTM model
   - Get prediction
   - Inverse scale back to dollars
4. **Maps date-level prediction to all rows** (broadcast)

**If TensorFlow not installed:** Returns fallback values

```python
lstm_results = evaluate_model(y_val, lstm_pred, "LSTM")
print(f"\nLSTM - RMSE: {lstm_results['RMSE']:.2f}, MAE: {lstm_results['MAE']:.2f}, MAPE: {lstm_results['MAPE']:.2f}%")
```

**Evaluate and print**

---

## 📝 Cell 22: Section Header (Markdown)

```markdown
## Step 7: Model Comparison
```

---

## 💻 Cell 23: Compare All Models (Code)

```python
from src.evaluation import compare_models, plot_predictions, plot_all_predictions
```

**Import visualization functions:**
- `compare_models` - Creates comparison table
- `plot_predictions` - Single model plot
- `plot_all_predictions` - Grid of all models

```python
# Collect all results
all_results = [
    naive_results,
    ma_results,
    sarima_results,
    prophet_results,
    lgb_results,
    xgb_results,
    lstm_results
]
```

**Create list of all result dictionaries:**

Each dictionary contains:
```python
{
    'Model': 'LightGBM',
    'RMSE': 2880.83,
    'MAE': 1393.87,
    'MAPE': 2220.36
}
```

```python
# Compare models
comparison_df = compare_models(all_results, save_path='../results/model_comparison.csv')
```

**What compare_models() does:**

1. Converts list of dictionaries to DataFrame
2. Sorts by RMSE (lower = better)
3. Prints formatted table to screen
4. Saves to CSV file

**Output table:**
```
Model                      RMSE       MAE        MAPE
LightGBM                2880.83   1393.87   2220.36
XGBoost                 3022.41   1535.04   5412.53
LSTM                   27030.13  15778.64    100.00
Moving Average         31094.09  19891.78 136619.94
Naive Forecast         32047.87  20458.72 142481.83
```

**File saved:** `results/model_comparison.csv`

---

## 📝 Cell 24: Section Header (Markdown)

```markdown
## Step 8: Visualization
```

---

## 💻 Cell 25: Create Plots (Code)

```python
# Plot predictions for all models
predictions_dict = {
    'Naive Forecast': {'y_true': y_val, 'y_pred': naive_pred},
    'Moving Average': {'y_true': y_val, 'y_pred': ma_pred},
    'LightGBM': {'y_true': y_val, 'y_pred': lgb_pred},
    'XGBoost': {'y_true': y_val, 'y_pred': xgb_pred}
}
```

**Create dictionary for plotting:**

Each entry has:
- Key = model name
- Value = dictionary with:
  - `y_true` = actual values
  - `y_pred` = predicted values

```python
# Add SARIMA and Prophet if available
if len(sample_val) > 0:
    predictions_dict['SARIMA'] = {'y_true': sarima_y_val, 'y_pred': sarima_pred}
    predictions_dict['Prophet'] = {'y_true': prophet_y_val, 'y_pred': prophet_pred}
```

**Conditionally add SARIMA and Prophet:**
- Only if sample_val has data
- Uses sarima_y_val (not y_val) because SARIMA used sample

```python
plot_all_predictions(predictions_dict, save_path='../results/all_predictions.png')
```

**Create grid plot:**

1. Creates subplot grid (3 columns, enough rows)
2. For each model:
   - Scatter plot: actual vs predicted
   - Red diagonal line (perfect prediction)
   - Annotate with RMSE, MAE, MAPE
3. Saves to file

**Result:** `all_predictions.png` with 6 subplots

```python
# Individual plots for top models
plot_predictions(y_val, lgb_pred, 'LightGBM', save_path='../results/lightgbm_predictions.png')
plot_predictions(y_val, xgb_pred, 'XGBoost', save_path='../results/xgboost_predictions.png')
```

**Create detailed plots for best 2 models:**

Each plot has 2 subplots:
1. **Left: Scatter plot** - Predicted vs Actual (like above)
2. **Right: Time series** - Sample of 200 points showing actual vs predicted over time

**Files saved:**
- `lightgbm_predictions.png`
- `xgboost_predictions.png`

---

## 📝 Cell 26: Final Summary (Markdown)

```markdown
## Summary

The forecasting pipeline has been completed. All results are saved in the `results/` directory.
```

**Conclusion:** Just states pipeline is done

---

## 📊 Summary of Notebook Flow

```
1. Load Data from Kaggle (421,570 rows)
        ↓
2. Explore Data (EDA, missing values, outliers)
        ↓
3. Create 49 Features (lags, rolling, calendar)
        ↓
4. Split 80/20 (Time-based, not random)
        ↓
5. Train Baselines (Naive, Moving Average)
        ↓
6. Train Advanced Models:
   - SARIMA (statistical, sample)
   - Prophet (statistical, sample)  
   - LightGBM (ML, full data) 🏆
   - XGBoost (ML, full data)
   - LSTM (Deep Learning, full data)
        ↓
7. Compare All Models (RMSE, MAE, MAPE)
        ↓
8. Visualize Results (plots, charts)
        ↓
9. Save Everything to results/
```

---

## 🎯 Key Takeaways

1. **Modular design:** Each step imports from `src/` modules
2. **Proper workflow:** EDA → Feature Engineering → Split → Train → Evaluate
3. **Time-based split:** Respects temporal order (critical for forecasting)
4. **Multiple models:** Baseline to advanced (comprehensive comparison)
5. **Proper evaluation:** Calculate metrics, create visualizations
6. **Reproducible:** Run notebook top-to-bottom, gets same results

---

## ❓ Quick Q&A

**Q: Why sys.path.append('../')?**
**A:** Notebook is in `notebooks/` folder, needs to import from `src/` folder one level up.

**Q: Why dropna after creating features?**
**A:** Lag features create NaN in first rows (can't look back 12 weeks for week 1).

**Q: Why time-based split instead of random?**
**A:** Forecasting predicts future from past. Random split would let model cheat by seeing future data.

**Q: Why use sample for SARIMA/Prophet?**
**A:** They train separate model per Store-Dept (4,455 models). Takes hours. Sample = faster demo.

**Q: Why does LightGBM win?**
**A:** Lag features (correlation 0.95+) → trees capture this perfectly. Fast and accurate.

**Q: What's X_train, y_train, X_val, y_val?**
**A:**
- X_train = training features (inputs)
- y_train = training target (what to predict)
- X_val = validation features (test inputs)
- y_val = validation target (test actual values)

**Q: What does fit() do?**
**A:** Trains the model (learns patterns from training data).

**Q: What does predict() do?**
**A:** Uses trained model to make predictions on new data.

---

**You now understand every line! 🎉**

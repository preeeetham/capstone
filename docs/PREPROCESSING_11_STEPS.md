# 11 Preprocessing Steps (No Data Leakage)

All steps are designed to **improve training efficiency** without **losing important information** or introducing **data leakage**. Fit-based steps use **train data only** for fitting.

| Step | Name | Description | Leakage risk |
|------|------|-------------|--------------|
| **1** | Parse dates | Convert date column(s) to `datetime` | None |
| **2** | Select columns | Keep/drop columns (preserve needed features) | None |
| **3** | Sort by entity and date | Sort by (group keys, date) for correct lag/rolling order | None |
| **4** | Impute missing | Fill NaN with **train** median (numeric) or mode (categorical); transform train & val | None (fit on train only) |
| **5** | Temporal features | Extract year, month, week, day_of_week from date only | None |
| **6** | Lag features | Add past values of target (shift only; no future) | None |
| **7** | Rolling features | Add rolling mean/std/min/max over **past** window (shift(1) then rolling) | None |
| **8** | External features | Add covariates and their past lags | None |
| **9** | Encode categoricals | Label-encode categoricals; **fit mapping on train**; unknown in val → -1 | None (fit on train only) |
| **10** | Cap outliers | Winsorize numeric columns to **train** 1st–99th percentiles | None (fit on train only) |
| **11** | Scale numerical | StandardScaler **fit on train**; transform train & val | None (fit on train only) |

## Execution order

- **Steps 1–3**: Applied to combined (train + val) data; no fit.
- **Steps 4–8**: Applied to combined data so validation rows get correct lags from history; **step 4 imputation** is fit on train rows only, then applied to all.
- **Steps 9–11**: Fit on **train only**, then transform both train and validation.

## Where it lives

- **Module**: `src/preprocessing.py`
- **Class**: `Preprocessor` with configs `get_walmart_config()` and `get_freshretail_config()`
- **Entry point**: `Preprocessor(config).fit_transform_train_val(train_df, val_df)`

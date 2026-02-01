"""
11 preprocessing steps for efficient training with no data leakage.
All fit-based steps (imputation, encoding, capping, scaling) are fit on train only.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Any
from dataclasses import dataclass, field

# Optional sklearn for scaling (graceful fallback)
try:
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ---------- Step names (for logging) ----------
PREPROCESSING_STEPS = [
    "1_parse_dates",
    "2_select_columns",
    "3_sort_by_entity_and_date",
    "4_impute_missing",
    "5_add_temporal_features",
    "6_add_lag_features",
    "7_add_rolling_features",
    "8_add_external_features",
    "9_encode_categoricals",
    "10_cap_outliers",
    "11_scale_numerical",
]


@dataclass
class PreprocessorConfig:
    """Config for the 11-step preprocessor (Walmart or FreshRetail)."""
    date_col: str
    target_col: str
    groupby_cols: List[str]
    # Columns to keep (None = keep all except explicitly dropped)
    keep_columns: Optional[List[str]] = None
    # Columns to drop from raw data
    drop_columns: Optional[List[str]] = None
    # Categorical columns (will be label-encoded; fit on train)
    categorical_cols: Optional[List[str]] = None
    # External/covariate columns (for step 8)
    external_cols: Optional[List[str]] = None
    # Lag periods for target
    lags: List[int] = field(default_factory=lambda: [1, 2, 4, 12])
    rolling_windows: List[int] = field(default_factory=lambda: [4, 8, 12])
    # Outlier capping: use (lower_pct, upper_pct) e.g. (1, 99)
    cap_percentiles: Tuple[float, float] = (1.0, 99.0)
    # Scale numericals (step 11); set False to skip if models don't need it
    scale_numerical: bool = True


def _get_feature_eng_module():
    """Lazy import to avoid circular deps."""
    from src import feature_engineering
    return feature_engineering


# ---------- Step 1: Parse dates ----------
def step1_parse_dates(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Convert date column to datetime. No leakage."""
    df = df.copy()
    if df[date_col].dtype != "datetime64[ns]":
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    return df


# ---------- Step 2: Select columns ----------
def step2_select_columns(
    df: pd.DataFrame,
    keep_columns: Optional[List[str]] = None,
    drop_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Keep or drop columns. No leakage; preserves needed info."""
    df = df.copy()
    if keep_columns is not None:
        existing = [c for c in keep_columns if c in df.columns]
        df = df[existing]
    if drop_columns is not None:
        to_drop = [c for c in drop_columns if c in df.columns]
        df = df.drop(columns=to_drop)
    return df


# ---------- Step 3: Sort by entity and date ----------
def step3_sort_by_entity_and_date(
    df: pd.DataFrame, groupby_cols: List[str], date_col: str
) -> pd.DataFrame:
    """Sort by (entity, date) for correct lag/rolling order. No leakage."""
    df = df.copy()
    sort_cols = [c for c in groupby_cols + [date_col] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


# ---------- Step 4: Impute missing (fit on train) ----------
def step4_impute_missing_fit(df_train: pd.DataFrame, numeric_fill: str = "median") -> dict:
    """Compute imputation values from train only. No leakage."""
    stats = {}
    for col in df_train.columns:
        if df_train[col].isna().any():
            if pd.api.types.is_numeric_dtype(df_train[col]):
                stats[col] = ("numeric", getattr(df_train[col], numeric_fill)())
            else:
                mode_val = df_train[col].mode()
                stats[col] = ("category", mode_val.iloc[0] if len(mode_val) else None)
    return stats


def step4_impute_missing_transform(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Apply imputation using train-derived values. No leakage."""
    df = df.copy()
    for col, (kind, val) in stats.items():
        if col in df.columns and val is not None:
            df[col] = df[col].fillna(val)
    return df


# ---------- Step 5: Temporal features ----------
def step5_add_temporal_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Add year, month, week, day from date only. No leakage."""
    fe = _get_feature_eng_module()
    return fe.create_calendar_features(df, date_col=date_col)


# ---------- Step 6: Lag features ----------
def step6_add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lags: List[int],
    groupby_cols: List[str],
) -> pd.DataFrame:
    """Add lag features (past values only). No leakage."""
    fe = _get_feature_eng_module()
    return fe.create_lag_features(
        df, target_col=target_col, lags=lags, groupby_cols=groupby_cols
    )


# ---------- Step 7: Rolling features ----------
def step7_add_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    windows: List[int],
    groupby_cols: List[str],
) -> pd.DataFrame:
    """Add rolling stats over past window only. No leakage."""
    fe = _get_feature_eng_module()
    return fe.create_rolling_features(
        df, target_col=target_col, windows=windows, groupby_cols=groupby_cols
    )


# ---------- Step 8: External features ----------
def step8_add_external_features(
    df: pd.DataFrame,
    external_cols: Optional[List[str]] = None,
    groupby_col: Optional[str] = None,
    use_freshretail: bool = False,
) -> pd.DataFrame:
    """Add external/covariate features (and their lags). Past-only; no leakage."""
    df = df.copy()
    fe = _get_feature_eng_module()
    if use_freshretail:
        return fe.create_external_features_freshretail(df, groupby_col=groupby_col or "store_id")
    return fe.create_external_features(df)


# ---------- Step 9: Encode categoricals (fit on train) ----------
def step9_encode_categoricals_fit(
    df_train: pd.DataFrame, categorical_cols: Optional[List[str]] = None
) -> dict:
    """Build label mapping from train only. Unknown in val mapped to -1 or max+1. No leakage."""
    if not categorical_cols:
        return {}
    encoders = {}
    for col in categorical_cols:
        if col not in df_train.columns:
            continue
        uniques = df_train[col].dropna().unique()
        encoders[col] = {v: i for i, v in enumerate(sorted(str(x) for x in uniques))}
    return encoders


def step9_encode_categoricals_transform(
    df: pd.DataFrame, encoders: dict
) -> pd.DataFrame:
    """Apply label encoding. Unknown categories get -1. No leakage."""
    df = df.copy()
    for col, mapping in encoders.items():
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str).map(lambda x: mapping.get(x, -1))
    return df


# ---------- Step 10: Cap outliers (fit on train) ----------
def step10_cap_outliers_fit(
    df_train: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
) -> dict:
    """Compute percentile bounds from train only. No leakage."""
    if numeric_cols is None:
        numeric_cols = [
            c for c in df_train.columns
            if pd.api.types.is_numeric_dtype(df_train[c])
        ]
    bounds = {}
    for col in numeric_cols:
        if col not in df_train.columns:
            continue
        try:
            low = np.nanpercentile(df_train[col], lower_pct)
            high = np.nanpercentile(df_train[col], upper_pct)
            bounds[col] = (low, high)
        except Exception:
            pass
    return bounds


def step10_cap_outliers_transform(df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    """Winsorize using train-derived bounds. No leakage."""
    df = df.copy()
    for col, (low, high) in bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=low, upper=high)
    return df


# ---------- Step 11: Scale numerical (fit on train) ----------
def step11_scale_numerical_fit(
    df_train: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    exclude_cols: Optional[List[str]] = None,
) -> Tuple[Optional[Any], List[str]]:
    """Fit StandardScaler on train only. No leakage."""
    exclude_cols = set(exclude_cols or [])
    if numeric_cols is None:
        numeric_cols = [
            c for c in df_train.columns
            if pd.api.types.is_numeric_dtype(df_train[c]) and c not in exclude_cols
        ]
    else:
        numeric_cols = [c for c in numeric_cols if c in df_train.columns and c not in exclude_cols]
    if not HAS_SKLEARN or not numeric_cols:
        return None, numeric_cols
    scaler = StandardScaler()
    scaler.fit(df_train[numeric_cols].fillna(0))
    return scaler, numeric_cols


def step11_scale_numerical_transform(
    df: pd.DataFrame, scaler: Optional[Any], numeric_cols: List[str]
) -> pd.DataFrame:
    """Apply scaling. No leakage if scaler was fit on train."""
    if scaler is None or not numeric_cols:
        return df
    df = df.copy()
    df[numeric_cols] = scaler.transform(df[numeric_cols].fillna(0))
    return df


# ---------- Full pipeline: run all 11 steps ----------
class Preprocessor:
    """
    Runs the 11 preprocessing steps. Fit-based steps (4, 9, 10, 11) are fit on
    train portion only to avoid data leakage.
    """

    def __init__(self, config: PreprocessorConfig, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        self._impute_stats = None
        self._encoders = None
        self._cap_bounds = None
        self._scaler = None
        self._scale_cols = None

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def run_steps_1_to_3(self, df: pd.DataFrame) -> pd.DataFrame:
        """Steps 1–3: no fit required. Safe to run on combined train+val."""
        cfg = self.config
        df = step1_parse_dates(df, cfg.date_col)
        self._log("  ✓ Step 1: Parse dates")
        df = step2_select_columns(df, cfg.keep_columns, cfg.drop_columns)
        self._log("  ✓ Step 2: Select columns")
        df = step3_sort_by_entity_and_date(df, cfg.groupby_cols, cfg.date_col)
        self._log("  ✓ Step 3: Sort by entity and date")
        return df

    def run_steps_4_to_8(self, df: pd.DataFrame, fit: bool, train_mask: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Steps 4–8. If fit=True, use rows where train_mask True for fitting.
        train_mask must be provided when fit=True.
        """
        cfg = self.config
        # Step 4: Impute
        if fit and train_mask is not None:
            self._impute_stats = step4_impute_missing_fit(df.loc[train_mask])
            self._log("  ✓ Step 4: Impute missing (fit on train)")
        if self._impute_stats:
            df = step4_impute_missing_transform(df, self._impute_stats)
            if not fit:
                self._log("  ✓ Step 4: Impute missing (transform)")

        # Step 5–7: Temporal, lag, rolling (no fit; use existing feature_engineering)
        df = step5_add_temporal_features(df, cfg.date_col)
        self._log("  ✓ Step 5: Temporal features")
        df = step6_add_lag_features(df, cfg.target_col, cfg.lags, cfg.groupby_cols)
        self._log("  ✓ Step 6: Lag features")
        df = step7_add_rolling_features(df, cfg.target_col, cfg.rolling_windows, cfg.groupby_cols)
        self._log("  ✓ Step 7: Rolling features")

        # Step 8: External (dataset-specific)
        use_freshretail = cfg.external_cols is not None and "sale_amount" == cfg.target_col
        df = step8_add_external_features(
            df,
            external_cols=cfg.external_cols,
            groupby_col=cfg.groupby_cols[0] if cfg.groupby_cols else None,
            use_freshretail=use_freshretail,
        )
        self._log("  ✓ Step 8: External features")
        return df

    def run_steps_9_to_11_fit(self, df_train: pd.DataFrame) -> None:
        """Fit steps 9, 10, 11 on train only."""
        cfg = self.config
        self._encoders = step9_encode_categoricals_fit(df_train, cfg.categorical_cols)
        self._log("  ✓ Step 9: Encode categoricals (fit on train)")

        exclude_for_cap = [cfg.target_col, cfg.date_col] + cfg.groupby_cols
        numeric_for_cap = [
            c for c in df_train.columns
            if pd.api.types.is_numeric_dtype(df_train[c]) and c not in exclude_for_cap
        ]
        self._cap_bounds = step10_cap_outliers_fit(
            df_train,
            numeric_cols=numeric_for_cap,
            lower_pct=cfg.cap_percentiles[0],
            upper_pct=cfg.cap_percentiles[1],
        )
        self._log("  ✓ Step 10: Cap outliers (fit on train)")

        if cfg.scale_numerical and HAS_SKLEARN:
            exclude_scale = [cfg.target_col, cfg.date_col] + cfg.groupby_cols
            self._scaler, self._scale_cols = step11_scale_numerical_fit(
                df_train, exclude_cols=exclude_scale
            )
            self._log("  ✓ Step 11: Scale numerical (fit on train)")
        else:
            self._scaler, self._scale_cols = None, []

    def run_steps_9_to_11_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform steps 9, 10, 11 (using fitted params)."""
        if self._encoders:
            df = step9_encode_categoricals_transform(df, self._encoders)
        if self._cap_bounds:
            df = step10_cap_outliers_transform(df, self._cap_bounds)
        if self._scaler is not None and self._scale_cols:
            df = step11_scale_numerical_transform(df, self._scaler, self._scale_cols)
        return df

    def fit_transform_train_val(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        source_col: str = "_source",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Full 11-step preprocessing: combine train+val for steps 1–8 (so val gets
        correct lags), then split and fit steps 9–11 on train, transform both.
        Returns (train_processed, val_processed) with no data leakage.
        """
        cfg = self.config
        self._log("Preprocessing: Steps 1–3 (on combined)...")
        train_df = train_df.copy()
        val_df = val_df.copy()
        train_df[source_col] = "train"
        val_df[source_col] = "val"
        full = pd.concat([train_df, val_df], ignore_index=True)
        full = self.run_steps_1_to_3(full)

        self._log("Preprocessing: Steps 4–8 (on combined; impute fit on train)...")
        train_mask = full[source_col] == "train"
        full = self.run_steps_4_to_8(full, fit=True, train_mask=train_mask)

        train_processed = full[full[source_col] == "train"].drop(columns=[source_col])
        val_processed = full[full[source_col] == "val"].drop(columns=[source_col])

        self._log("Preprocessing: Steps 9–11 (fit on train, transform both)...")
        self.run_steps_9_to_11_fit(train_processed)
        train_processed = self.run_steps_9_to_11_transform(train_processed)
        val_processed = self.run_steps_9_to_11_transform(val_processed)

        return train_processed, val_processed


# ---------- Config presets ----------
def get_walmart_config() -> PreprocessorConfig:
    return PreprocessorConfig(
        date_col="Date",
        target_col="Weekly_Sales",
        groupby_cols=["Store", "Dept"],
        keep_columns=None,
        drop_columns=None,
        categorical_cols=["Store", "Dept", "Type"] if True else None,
        external_cols=["Temperature", "Fuel_Price", "CPI", "Unemployment"],
        lags=[1, 2, 4, 12],
        rolling_windows=[4, 8, 12],
        cap_percentiles=(1.0, 99.0),
        scale_numerical=True,
    )


def get_freshretail_config() -> PreprocessorConfig:
    return PreprocessorConfig(
        date_col="dt",
        target_col="sale_amount",
        groupby_cols=["store_id", "product_id"],
        keep_columns=None,
        drop_columns=None,
        categorical_cols=[
            "store_id", "product_id", "city_id", "management_group_id",
            "first_category_id", "second_category_id", "third_category_id",
        ],
        external_cols=[
            "discount", "holiday_flag", "activity_flag", "precpt",
            "avg_temperature", "avg_humidity", "avg_wind_level", "stock_hour6_22_cnt",
        ],
        lags=[1, 2, 4, 7],
        rolling_windows=[4, 7, 14],
        cap_percentiles=(1.0, 99.0),
        scale_numerical=True,
    )

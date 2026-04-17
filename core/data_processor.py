import pandas as pd
import numpy as np
from scipy import stats
import logging
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.data_cfg = config.get('data', {})
        self.prep_cfg = config.get('preprocessing', {})
        self.minmax_scale = MinMaxScaler(feature_range=(0, 1))
        # Metadata tracked for the Explainable AI module
        self.meta = {
            "rows_loaded": 0,
            "rows_after_outliers": 0,
            "outliers_removed": 0,
            "negative_target_dropped": 0,
            "columns_dropped": [],
            "numeric_imputation": self.prep_cfg.get('numeric_imputation', 'median'),
            "features_total": 0,
            "features_selected": "N/A",  # updated by model_trainer
        }

    def run_pipeline(self):
        df = self.load_data()
        df = self.handle_missing_target(df)
        df = self.engineer_date_features(df)
        df = self.handle_missing_values(df)
        df = self.handle_outliers(df)
        df = self.encode_categorical(df)
        df = self.normalize_numeric(df)
        self.meta["features_total"] = df.shape[1] - 1  # minus target
        return df, self.meta

    def load_data(self):
        files = self.data_cfg.get('files', [])
        merge_on = self.data_cfg.get('merge_on', None)

        dfs = []
        for file in files:
            logging.info(f"Loading {file['path']}...")
            dfs.append(pd.read_csv(file['path']))

        if len(dfs) == 1:
            df = dfs[0]
            self.meta["rows_loaded"] = len(df)
            return df

        if not merge_on:
            logging.warning("Multiple files provided but 'merge_on' is null. Using only first file.")
            self.meta["rows_loaded"] = len(dfs[0])
            return dfs[0]

        df_master = dfs[0]
        for i in range(1, len(dfs)):
            logging.info(f"Merging file {files[i]['path']}...")
            df_master = pd.merge(df_master, dfs[i], on=merge_on, how='left')

        self.meta["rows_loaded"] = len(df_master)
        return df_master

    def handle_missing_target(self, df):
        target_col = self.data_cfg['target_col']
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in data.")

        start_len = len(df)
        df = df.dropna(subset=[target_col])
        if len(df) < start_len:
            logging.info(f"Dropped {start_len - len(df)} rows with missing target values.")

        if self.prep_cfg.get('drop_negative_target', False):
            start_len = len(df)
            df = df[df[target_col] >= 0]
            dropped = start_len - len(df)
            self.meta["negative_target_dropped"] = dropped
            logging.info(f"Dropped {dropped} rows with negative target values.")

        return df

    def engineer_date_features(self, df):
        date_col = self.data_cfg['date_col']
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df['Year'] = df[date_col].dt.year
            df['Month'] = df[date_col].dt.month
            df['Week'] = df[date_col].dt.isocalendar().week
            df['DayOfWeek'] = df[date_col].dt.dayofweek
            df['DayOfYear'] = df[date_col].dt.dayofyear
            df.sort_values(by=[date_col], inplace=True)
            df.drop(columns=[date_col], inplace=True)
            logging.info("Extracted date features.")
        return df

    def handle_missing_values(self, df):
        num_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(exclude=[np.number]).columns

        num_strategy = self.prep_cfg.get('numeric_imputation', 'median')
        cat_strategy = self.prep_cfg.get('categorical_imputation', 'mode')

        drop_threshold = 0.3
        dropped_cols = []
        for col in list(df.columns):
            missing_ratio = df[col].isnull().mean()
            if missing_ratio > drop_threshold and col != self.data_cfg['target_col']:
                logging.warning(f"Dropping column '{col}' due to {missing_ratio*100:.2f}% missing values.")
                df.drop(columns=[col], inplace=True)
                dropped_cols.append(col)
        self.meta["columns_dropped"] = dropped_cols

        num_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(exclude=[np.number]).columns

        if num_strategy == 'median':
            for col in num_cols:
                df[col] = df[col].fillna(df[col].median())
        elif num_strategy == 'zero':
            for col in num_cols:
                df[col] = df[col].fillna(0)

        if cat_strategy == 'mode':
            for col in cat_cols:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        elif cat_strategy == 'unknown':
            for col in cat_cols:
                df[col] = df[col].fillna('Unknown')

        logging.info("Missing values handled.")
        return df

    def handle_outliers(self, df):
        method = self.prep_cfg.get('outlier_method', 'zscore')
        thresh = self.prep_cfg.get('outlier_threshold', 2.5)

        target_col = self.data_cfg['target_col']
        if method == 'zscore':
            num_cols = df.select_dtypes(include=[np.number]).columns.drop(target_col, errors='ignore')
            if len(num_cols) > 0:
                start_len = len(df)
                z_scores = np.abs((df[num_cols] - df[num_cols].mean()) / df[num_cols].std())
                df = df[(z_scores < thresh).all(axis=1)]
                removed = start_len - len(df)
                self.meta["outliers_removed"] = removed
                self.meta["rows_after_outliers"] = len(df)
                logging.info(f"Outliers dropped using Z-score method: {removed} rows.")
        return df

    def encode_categorical(self, df):
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if cat_cols:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
            logging.info(f"One-hot encoded {len(cat_cols)} categorical columns.")
        return df

    def normalize_numeric(self, df):
        target_col = self.data_cfg['target_col']
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in num_cols:
            num_cols.remove(target_col)

        for col in num_cols:
            arr = np.array(df[col]).reshape(-1, 1)
            df[col] = self.minmax_scale.fit_transform(arr)

        logging.info("Numeric features normalized.")
        return df

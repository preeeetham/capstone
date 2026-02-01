"""
Model implementations for sales forecasting.
Includes baseline models and research-style models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# ML models
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except Exception as e:
    LIGHTGBM_AVAILABLE = False
    print(f"Warning: LightGBM not available ({e})")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception as e:
    XGBOOST_AVAILABLE = False
    print(f"Warning: XGBoost not available ({e})")

# Statistical models
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception as e:
    PROPHET_AVAILABLE = False
    print(f"Warning: Prophet not available ({e})")

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except Exception as e:
    STATSMODELS_AVAILABLE = False
    print(f"Warning: statsmodels not available ({e})")

# Deep learning
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except Exception as e:
    TENSORFLOW_AVAILABLE = False
    print(f"Warning: TensorFlow not available ({e})")


class NaiveForecast:
    """Naive forecast: uses last observed value."""
    
    def __init__(self):
        self.last_values = {}
    
    def fit(self, df, target_col='Weekly_Sales', groupby_cols=['Store', 'Dept']):
        """Store last value for each group."""
        for name, group in df.groupby(groupby_cols):
            self.last_values[name] = group[target_col].iloc[-1]
        return self
    
    def predict(self, df, groupby_cols=['Store', 'Dept']):
        """Predict using last observed value."""
        predictions = []
        for name, group in df.groupby(groupby_cols):
            if name in self.last_values:
                pred = self.last_values[name]
            else:
                # If group not seen, use mean of all groups
                pred = np.mean(list(self.last_values.values())) if self.last_values else 0
            predictions.extend([pred] * len(group))
        return np.array(predictions)


class MovingAverage:
    """Moving average forecast."""
    
    def __init__(self, window=4):
        self.window = window
        self.last_values = {}
    
    def fit(self, df, target_col='Weekly_Sales', groupby_cols=['Store', 'Dept']):
        """Store last N values for each group."""
        for name, group in df.groupby(groupby_cols):
            self.last_values[name] = group[target_col].tail(self.window).tolist()
        return self
    
    def predict(self, df, groupby_cols=['Store', 'Dept']):
        """Predict using moving average."""
        predictions = []
        for name, group in df.groupby(groupby_cols):
            if name in self.last_values and len(self.last_values[name]) > 0:
                pred = np.mean(self.last_values[name])
            else:
                pred = np.mean([np.mean(vals) for vals in self.last_values.values()]) if self.last_values else 0
            predictions.extend([pred] * len(group))
        return np.array(predictions)


class SARIMAModel:
    """SARIMA model for time series forecasting."""
    
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.models = {}
        self.STATSMODELS_AVAILABLE = STATSMODELS_AVAILABLE
    
    def fit(self, df, target_col='Weekly_Sales', groupby_cols=['Store', 'Dept']):
        """Fit SARIMA model for each group."""
        if not self.STATSMODELS_AVAILABLE:
            print("statsmodels not available, skipping SARIMA")
            return self
        
        print("Fitting SARIMA models (this may take a while)...")
        for idx, (name, group) in enumerate(df.groupby(groupby_cols)):
            if idx % 50 == 0:
                print(f"  Fitting model {idx+1}/{len(df.groupby(groupby_cols))}")
            
            try:
                # Use simpler ARIMA if SARIMA fails
                try:
                    model = SARIMAX(group[target_col], 
                                   order=self.order,
                                   seasonal_order=self.seasonal_order,
                                   enforce_stationarity=False,
                                   enforce_invertibility=False)
                    fitted_model = model.fit(disp=False, maxiter=50)
                except:
                    # Fallback to simple ARIMA
                    model = ARIMA(group[target_col], order=self.order)
                    fitted_model = model.fit()
                
                self.models[name] = fitted_model
            except Exception as e:
                # Store mean as fallback
                self.models[name] = group[target_col].mean()
        
        return self
    
    def predict(self, df, groupby_cols=['Store', 'Dept'], steps=1):
        """Predict using fitted SARIMA models."""
        if not self.STATSMODELS_AVAILABLE:
            return np.zeros(len(df))
        
        predictions = []
        for name, group in df.groupby(groupby_cols):
            if name in self.models:
                if isinstance(self.models[name], (int, float, np.number)):
                    preds = np.full(len(group), self.models[name])
                else:
                    try:
                        forecast = self.models[name].forecast(steps=len(group))
                        preds = forecast.values if hasattr(forecast, 'values') else np.array(forecast)
                        if len(preds) != len(group):
                            preds = np.full(len(group), preds[-1] if len(preds) > 0 else 0)
                    except Exception:
                        last_val = self.models[name].fittedvalues.iloc[-1] if hasattr(self.models[name], 'fittedvalues') else 0
                        preds = np.full(len(group), last_val)
            else:
                fallback = np.mean([m.fittedvalues.iloc[-1] if hasattr(m, 'fittedvalues') else m for m in self.models.values() if not isinstance(m, (int, float))])
                preds = np.full(len(group), fallback)
            predictions.extend(preds)
        
        return np.array(predictions)


class ProphetModel:
    """Prophet model for time series forecasting."""
    
    def __init__(self, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False):
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.models = {}
        self.PROPHET_AVAILABLE = PROPHET_AVAILABLE
    
    def fit(self, df, target_col='Weekly_Sales', date_col='Date', 
            groupby_cols=['Store', 'Dept'], external_regressors=None):
        """Fit Prophet model for each group."""
        if not self.PROPHET_AVAILABLE:
            print("Prophet not available, skipping")
            return self
        
        print("Fitting Prophet models (this may take a while)...")
        for idx, (name, group) in enumerate(df.groupby(groupby_cols)):
            if idx % 50 == 0:
                print(f"  Fitting model {idx+1}/{len(df.groupby(groupby_cols))}")
            
            try:
                prophet_df = pd.DataFrame({
                    'ds': group[date_col],
                    'y': group[target_col]
                })
                
                model = Prophet(
                    yearly_seasonality=self.yearly_seasonality,
                    weekly_seasonality=self.weekly_seasonality,
                    daily_seasonality=self.daily_seasonality
                )
                
                # Add external regressors if provided
                if external_regressors:
                    for regressor in external_regressors:
                        if regressor in group.columns:
                            model.add_regressor(regressor)
                            prophet_df[regressor] = group[regressor].values
                
                model.fit(prophet_df)
                self.models[name] = model
            except Exception as e:
                # Store mean as fallback
                self.models[name] = group[target_col].mean()
        
        return self
    
    def predict(self, df, date_col='Date', groupby_cols=['Store', 'Dept']):
        """Predict using fitted Prophet models."""
        if not self.PROPHET_AVAILABLE:
            return np.zeros(len(df))
        
        predictions = []
        for name, group in df.groupby(groupby_cols):
            if name in self.models:
                if isinstance(self.models[name], (int, float, np.number)):
                    preds = np.full(len(group), self.models[name])
                else:
                    try:
                        # Use actual validation dates for proper out-of-sample evaluation
                        future = group[[date_col]].copy()
                        future.columns = ['ds']
                        forecast = self.models[name].predict(future)
                        preds = forecast['yhat'].values
                        if len(preds) != len(group):
                            preds = np.full(len(group), forecast['yhat'].mean())
                    except Exception:
                        fallback = self.models[name].history['y'].mean() if hasattr(self.models[name], 'history') else 0
                        preds = np.full(len(group), fallback)
            else:
                preds = np.zeros(len(group))
            predictions.extend(preds)
        
        return np.array(predictions)


class LightGBMModel:
    """LightGBM model for sales forecasting."""
    
    def __init__(self, **params):
        self.params = params or {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        self.model = None
        self.feature_cols = None
        self.LIGHTGBM_AVAILABLE = LIGHTGBM_AVAILABLE
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, categorical_features=None):
        """Fit LightGBM model."""
        if not self.LIGHTGBM_AVAILABLE:
            print("LightGBM not available")
            return self
        
        self.feature_cols = X_train.columns.tolist() if hasattr(X_train, 'columns') else None
        
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
        
        if X_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical_features)
            self.model = lgb.train(
                self.params,
                train_data,
                valid_sets=[train_data, val_data],
                num_boost_round=1000,
                callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
            )
        else:
            self.model = lgb.train(
                self.params,
                train_data,
                num_boost_round=500
            )
        
        return self
    
    def predict(self, X):
        """Predict using fitted LightGBM model."""
        if not self.LIGHTGBM_AVAILABLE or self.model is None:
            return np.zeros(len(X))
        
        return self.model.predict(X, num_iteration=self.model.best_iteration if hasattr(self.model, 'best_iteration') else None)


class XGBoostModel:
    """XGBoost model for sales forecasting."""
    
    def __init__(self, **params):
        self.params = params or {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'verbosity': 0
        }
        self.model = None
        self.XGBOOST_AVAILABLE = XGBOOST_AVAILABLE
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Fit XGBoost model."""
        if not self.XGBOOST_AVAILABLE:
            print("XGBoost not available")
            return self
        
        # Use simpler fit without early stopping for compatibility
        self.model = xgb.XGBRegressor(**self.params, n_estimators=500)
        self.model.fit(X_train, y_train, verbose=False)
        
        return self
    
    def predict(self, X):
        """Predict using fitted XGBoost model."""
        if not self.XGBOOST_AVAILABLE or self.model is None:
            return np.zeros(len(X))
        
        return self.model.predict(X)


class LSTMModel:
    """LSTM model for time series forecasting."""
    
    def __init__(self, sequence_length=12, units=50, epochs=50, batch_size=32):
        self.sequence_length = sequence_length
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.TENSORFLOW_AVAILABLE = TENSORFLOW_AVAILABLE
    
    def _create_sequences(self, data, target):
        """Create sequences for LSTM input."""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i+self.sequence_length])
            y.append(target[i+self.sequence_length])
        return np.array(X), np.array(y)
    
    def fit(self, df, target_col='Weekly_Sales', feature_cols=None, 
            groupby_cols=['Store', 'Dept']):
        """Fit LSTM model."""
        if not self.TENSORFLOW_AVAILABLE:
            print("TensorFlow not available")
            return self
        
        from sklearn.preprocessing import MinMaxScaler
        
        # Aggregate data for LSTM (simplified approach)
        # In practice, you might want to train separate models per group
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col not in ['Date', target_col, 'Store', 'Dept']]
        
        # Use numeric columns only
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            print("No numeric features available for LSTM")
            return self
        
        # Aggregate by date for simplicity
        daily_data = df.groupby('Date').agg({
            target_col: 'sum',
            **{col: 'mean' for col in numeric_cols[:5]}  # Use top 5 features
        }).reset_index()
        
        daily_data = daily_data.sort_values('Date')
        
        # Scale data
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
        X_scaled = self.scaler_X.fit_transform(daily_data[numeric_cols[:5]].values)
        y_scaled = self.scaler_y.fit_transform(daily_data[[target_col]].values)
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y_scaled.flatten())
        
        if len(X_seq) == 0:
            print("Not enough data for LSTM sequences")
            return self
        
        # Store for prediction: aggregated training data and feature column names
        self.daily_data_train_ = daily_data.copy()
        self.numeric_cols_used_ = numeric_cols[:5]
        self.target_col_ = target_col
        
        # Build model
        self.model = Sequential([
            LSTM(self.units, return_sequences=True, input_shape=(self.sequence_length, X_seq.shape[2])),
            Dropout(0.2),
            LSTM(self.units, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Train
        self.model.fit(
            X_seq, y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            validation_split=0.2
        )
        
        return self
    
    def predict(self, df, feature_cols=None):
        """Predict using fitted LSTM model with real sequence-based forecasting."""
        if not self.TENSORFLOW_AVAILABLE or self.model is None:
            return np.zeros(len(df))
        if not hasattr(self, 'daily_data_train_') or self.daily_data_train_ is None:
            fallback = df['Weekly_Sales'].mean() if 'Weekly_Sales' in df.columns else 0
            return np.full(len(df), fallback)
        
        cols = self.numeric_cols_used_
        target_col = self.target_col_
        train_daily = self.daily_data_train_
        
        # Aggregate validation data by Date (same schema as training)
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c not in ['Date', target_col, 'Store', 'Dept']]
        numeric_available = [c for c in cols if c in df.columns]
        if len(numeric_available) == 0:
            fallback = df[target_col].mean() if target_col in df.columns else 0
            return np.full(len(df), fallback)
        
        agg_dict = {target_col: 'sum', **{c: 'mean' for c in numeric_available}}
        val_daily = df.groupby('Date').agg(agg_dict).reset_index()
        val_daily = val_daily.sort_values('Date')
        
        # Combined timeline: train then validation (same columns for features)
        for c in cols:
            if c not in val_daily.columns:
                val_daily[c] = 0
        full_daily = pd.concat([
            train_daily[['Date'] + cols + [target_col]],
            val_daily[['Date'] + cols + [target_col]]
        ], ignore_index=True)
        full_daily = full_daily.sort_values('Date').reset_index(drop=True)
        
        # Align column order
        X_full = full_daily[cols].fillna(0).values
        X_scaled = self.scaler_X.transform(X_full)
        
        n = len(full_daily)
        n_train = len(train_daily)
        seq_len = self.sequence_length
        preds_by_date = {}
        
        for i in range(n_train, n):
            if i < seq_len:
                continue
            seq = X_scaled[i - seq_len:i]
            seq_batch = np.expand_dims(seq, axis=0)
            pred_scaled = self.model.predict(seq_batch, verbose=0)
            pred_val = self.scaler_y.inverse_transform(pred_scaled)[0, 0]
            d = full_daily['Date'].iloc[i]
            preds_by_date[d] = pred_val
        
        # Map each row in df to its date's prediction (broadcast date-level forecast to rows)
        pred_arr = np.zeros(len(df))
        for i, (_, row) in enumerate(df.iterrows()):
            d = row['Date']
            pred_arr[i] = preds_by_date.get(d, np.nan)
        
        # If any date had no forecast (edge case), fill with train target mean
        train_mean = train_daily[target_col].mean()
        pred_arr = np.where(np.isnan(pred_arr), train_mean, pred_arr)
        return pred_arr.astype(np.float64)

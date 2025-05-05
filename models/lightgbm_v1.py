# Financial Time Series Forecasting with LightGBM (API Fixed Version)

import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FinancialTimeSeriesLGBM:
    def __init__(self, forecast_horizon=1, lookback_period=30):
        """Initialize with parameters for financial forecasting"""
        self.forecast_horizon = forecast_horizon
        self.lookback_period = lookback_period
        self.model = None
        self.feature_names = None
        self.feature_importance = None
        
        # Default LightGBM parameters optimized for financial data
        self.default_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.01,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1
        }
        
    def create_features(self, df, price_col='price', volume_col='volume'):
        """Create comprehensive financial features"""
        df = df.copy()
        
        # Basic price features
        df['returns'] = df[price_col].pct_change()
        df['log_returns'] = np.log(df[price_col] / df[price_col].shift(1))
        
        # Lagged features
        for lag in [1, 3, 5, 7, 14, 21, 30]:
            df[f'price_lag_{lag}'] = df[price_col].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            
        # Rolling window features
        windows = [5, 10, 20, 50, 100]
        for window in windows:
            df[f'ma_{window}'] = df[price_col].rolling(window=window).mean()
            df[f'std_{window}'] = df[price_col].rolling(window=window).std()
            df[f'volatility_{window}'] = df['returns'].rolling(window=window).std() * np.sqrt(252)
            df[f'rsi_{window}'] = self._calculate_rsi(df[price_col], window)
            
            if volume_col in df.columns:
                df[f'volume_ma_{window}'] = df[volume_col].rolling(window=window).mean()
                df[f'volume_ratio_{window}'] = df[volume_col] / df[f'volume_ma_{window}']
        
        # Momentum indicators
        df['macd'] = self._calculate_macd(df[price_col])
        df['momentum_10'] = df[price_col] / df[price_col].shift(10) - 1
        df['momentum_30'] = df[price_col] / df[price_col].shift(30) - 1
        
        # Technical indicators
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._bollinger_bands(df[price_col], 20)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Calendar features
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['is_month_end'] = df.index.is_month_end.astype(int)
        df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        
        # Economic cycle features
        index_dt = pd.to_datetime(df.index)
        year_start = index_dt.year.map(lambda x: pd.Timestamp(year=x, month=1, day=1))
        df['days_since_year_start'] = (index_dt - year_start).days
        df['sin_day_of_year'] = np.sin(2 * np.pi * df['days_since_year_start'] / 365.25)
        df['cos_day_of_year'] = np.cos(2 * np.pi * df['days_since_year_start'] / 365.25)
        
        # Time-based features
        df['trend'] = np.arange(len(df))
        df['trend_squared'] = df['trend'] ** 2
        
        return df
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).ewm(span=period).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd - signal_line
    
    def _bollinger_bands(self, prices, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        return upper, middle, lower
    
    def prepare_train_data(self, df, target_col='price'):
        """Prepare data for training"""
        # Create features
        df_features = self.create_features(df)
        
        # Create target variable (future returns)
        df_features['target'] = df_features[target_col].shift(-self.forecast_horizon)
        
        # Remove rows with NaN values
        df_clean = df_features.dropna()
        
        # Separate features and target
        feature_cols = [col for col in df_clean.columns if col not in ['target', 'price', 'volume']]
        X = df_clean[feature_cols]
        y = df_clean['target']
        
        self.feature_names = feature_cols
        
        return X, y
    
    def train(self, X, y, params=None):
        """Train LightGBM model"""
        if params is None:
            params = self.default_params
        
        # Create LightGBM dataset
        train_data = lgb.Dataset(X, label=y)
        
        # Create callbacks for early stopping and verbose evaluation
        callbacks = [
            lgb.early_stopping(stopping_rounds=10),
            lgb.log_evaluation(period=10)
        ]
        
        # Train model
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[train_data],
            valid_names=['train'],
            callbacks=callbacks
        )
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importance(),
            'importance_gain': self.model.feature_importance(importance_type='gain')
        })
        self.feature_importance = self.feature_importance.sort_values('importance', ascending=False)
        
        return self.model
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        return self.model.predict(X)
    
    def forecast_recursive(self, df, steps=30, price_col='price'):
        """Recursive forecasting for multiple steps ahead"""
        forecasts = []
        current_df = df.copy()
        
        for i in range(steps):
            # Prepare features
            X, _ = self.prepare_train_data(current_df)
            
            # Get the last row of features
            X_latest = X.iloc[-1:].copy()
            
            # Make prediction
            pred = self.predict(X_latest)[0]
            forecasts.append(pred)
            
            # Update dataframe with prediction
            last_date = current_df.index[-1]
            next_date = last_date + pd.Timedelta(days=1)
            
            # Create new row with predicted values
            new_row = pd.DataFrame({price_col: [pred]}, index=[next_date])
            current_df = pd.concat([current_df, new_row])
        
        return pd.Series(forecasts, index=pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=steps))
    
    def evaluate_model(self, y_true, y_pred):
        """Evaluate model performance"""
        metrics = {
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
        return metrics
    
    def plot_feature_importance(self, top_n=20):
        """Plot feature importance"""
        plt.figure(figsize=(10, 8))
        top_features = self.feature_importance.head(top_n)
        
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'Top {top_n} Important Features')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, y_true, y_pred, title='Predictions vs Actual'):
        """Plot predictions against actual values"""
        plt.figure(figsize=(12, 6))
        plt.plot(y_true.index, y_true.values, label='Actual', alpha=0.7)
        plt.plot(y_true.index, y_pred, label='Predicted', alpha=0.7)
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

# Example usage:
if __name__ == "__main__":
    # Generate synthetic financial data for demonstration
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    
    # Create synthetic price data with trend and seasonality
    trend = np.linspace(100, 300, len(dates))
    seasonality = 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365.25)
    noise = np.random.normal(0, 5, len(dates))
    price = trend + seasonality + noise
    
    # Create synthetic volume data
    volume = np.random.randint(1000000, 2000000, len(dates))
    
    # Create DataFrame
    df = pd.DataFrame({
        'price': price,
        'volume': volume
    }, index=dates)
    
    # Split data
    train_size = int(len(df) * 0.8)
    train_df = df[:train_size]
    test_df = df[train_size:]
    
    # Initialize and train model
    lgbm_model = FinancialTimeSeriesLGBM(forecast_horizon=1)
    
    # Prepare training data
    X_train, y_train = lgbm_model.prepare_train_data(train_df)
    
    # Train model
    model = lgbm_model.train(X_train, y_train)
    
    # Prepare test data
    X_test, y_test = lgbm_model.prepare_train_data(test_df)
    
    # Make predictions
    y_pred = lgbm_model.predict(X_test)
    
    # Evaluate
    metrics = lgbm_model.evaluate_model(y_test, y_pred)
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot feature importance
    lgbm_model.plot_feature_importance(top_n=15)
    
    # Plot predictions
    lgbm_model.plot_predictions(y_test, y_pred)
    
    # Multi-step forecasting
    forecast_steps = 30
    forecasts = lgbm_model.forecast_recursive(train_df, steps=forecast_steps)
    
    # Plot forecast
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[-365:], df['price'][-365:], label='Historical', alpha=0.7)
    plt.plot(forecasts.index, forecasts, label=f'{forecast_steps}-Day Forecast', color='red', alpha=0.7)
    plt.title('Stock Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
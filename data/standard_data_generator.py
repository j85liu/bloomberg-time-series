import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import datetime
from scipy import signal
import warnings
warnings.filterwarnings('ignore')


class ComprehensiveSyntheticDataset:
    """
    Generates a comprehensive synthetic dataset to showcase features of various time series models.
    Features include:
    - Multiple time series components (trend, seasonality, cyclic patterns)
    - Exogenous variables with different patterns
    - Static covariates
    - Missing values and outliers
    - Hierarchical data
    - Different data types (continuous, count, binary)
    """
    
    def __init__(
        self,
        n_series: int = 500,
        n_timesteps: int = 365,
        freq: str = 'D',
        start_date: str = '2020-01-01',
        n_exog_vars: int = 5,
        n_static_vars: int = 3,
        noise_level: float = 0.1,
        missing_prob: float = 0.02,
        outlier_prob: float = 0.01,
        random_seed: int = 42,
        include_hierarchical: bool = True,
        n_levels: int = 2
    ):
        self.n_series = n_series
        self.n_timesteps = n_timesteps
        self.freq = freq
        self.start_date = start_date
        self.n_exog_vars = n_exog_vars
        self.n_static_vars = n_static_vars
        self.noise_level = noise_level
        self.missing_prob = missing_prob
        self.outlier_prob = outlier_prob
        self.random_seed = random_seed
        self.include_hierarchical = include_hierarchical
        self.n_levels = n_levels
        
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Generate time index
        self.time_index = pd.date_range(start=start_date, periods=n_timesteps, freq=freq)
        self.time_features = self._generate_time_features()
        
        # Generate series IDs and hierarchical structure
        self.series_ids = self._generate_series_ids()
        self.hierarchy = self._generate_hierarchy() if include_hierarchical else None
        
        # Generate static variables
        self.static_vars = self._generate_static_vars()
        
        # Generate exogenous variables
        self.exog_vars = self._generate_exog_vars()
        
        # Generate target time series with multiple patterns
        self.target_series = self._generate_target_series()
        
        # Generate count and binary series variations
        self.count_series = self._convert_to_count(self.target_series)
        self.binary_series = self._convert_to_binary(self.target_series)
        
        # Create final dataset
        self.data = self._create_final_dataset()
    
    def _generate_time_features(self) -> pd.DataFrame:
        """Generate comprehensive time-based features"""
        t = np.arange(self.n_timesteps)
        
        # Cyclical time features
        hour_of_day = np.sin(2 * np.pi * t / 24) if self.freq.count('H') else np.zeros(self.n_timesteps)
        day_of_week = np.sin(2 * np.pi * t / 7)
        day_of_month = np.sin(2 * np.pi * t / 30)
        month_of_year = np.sin(2 * np.pi * t / 365)
        quarter = np.sin(2 * np.pi * t / (365/4))
        
        # Linear time features
        time_linear = t / self.n_timesteps
        
        # Holiday indicator (simplified)
        is_holiday = np.zeros(self.n_timesteps)
        holiday_days = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]  # Major holidays
        for day in holiday_days:
            if day < self.n_timesteps:
                is_holiday[day:min(day+3, self.n_timesteps)] = 1
        
        # Weekend indicator
        dates = pd.date_range(self.start_date, periods=self.n_timesteps, freq=self.freq)
        is_weekend = dates.dayofweek.isin([5, 6]).astype(float)
        
        # Create DataFrame
        time_features = pd.DataFrame({
            'hour_sin': hour_of_day,
            'hour_cos': np.cos(2 * np.pi * t / 24) if self.freq.count('H') else np.zeros(self.n_timesteps),
            'day_of_week_sin': day_of_week,
            'day_of_week_cos': np.cos(2 * np.pi * t / 7),
            'day_of_month_sin': day_of_month,
            'day_of_month_cos': np.cos(2 * np.pi * t / 30),
            'month_of_year_sin': month_of_year,
            'month_of_year_cos': np.cos(2 * np.pi * t / 365),
            'quarter_sin': quarter,
            'quarter_cos': np.cos(2 * np.pi * t / (365/4)),
            'time_linear': time_linear,
            'time_quadratic': time_linear ** 2,
            'is_holiday': is_holiday,
            'is_weekend': is_weekend,
        }, index=self.time_index)
        
        return time_features
    
    def _generate_series_ids(self) -> pd.DataFrame:
        """Generate series identifiers with metadata"""
        series_info = []
        
        for i in range(self.n_series):
            info = {
                'series_id': f'ts_{i:04d}',
                'category': np.random.choice(['A', 'B', 'C', 'D'], p=[0.4, 0.3, 0.2, 0.1]),
                'region': np.random.choice(['North', 'South', 'East', 'West']),
                'size': np.random.choice(['Small', 'Medium', 'Large'], p=[0.3, 0.5, 0.2]),
                'group': f'group_{np.random.randint(0, 10):02d}'
            }
            series_info.append(info)
        
        return pd.DataFrame(series_info)
    
    def _generate_hierarchy(self) -> Dict[str, List[str]]:
        """Generate hierarchical groupings for series"""
        hierarchy = {}
        
        # Level 1: Total
        hierarchy['total'] = list(self.series_ids['series_id'])
        
        # Level 2: By category
        for category in self.series_ids['category'].unique():
            hierarchy[f'category_{category}'] = list(
                self.series_ids[self.series_ids['category'] == category]['series_id']
            )
        
        # Level 3: By region within category
        for category in self.series_ids['category'].unique():
            for region in self.series_ids['region'].unique():
                mask = (self.series_ids['category'] == category) & (self.series_ids['region'] == region)
                series_list = list(self.series_ids[mask]['series_id'])
                if series_list:
                    hierarchy[f'category_{category}_region_{region}'] = series_list
        
        return hierarchy
    
    def _generate_static_vars(self) -> pd.DataFrame:
        """Generate static variables for each series"""
        static_data = []
        
        for i in range(self.n_series):
            # Generate correlated static variables
            x1 = np.random.randn()
            x2 = 0.7 * x1 + 0.3 * np.random.randn()  # Correlated with x1
            x3 = np.random.choice([0, 1], p=[0.3, 0.7])  # Binary static variable
            
            # Category-dependent static variables
            category = self.series_ids.iloc[i]['category']
            category_effect = {'A': 1.5, 'B': 1.0, 'C': 0.5, 'D': 0.2}[category]
            x4 = category_effect + 0.5 * np.random.randn()
            
            # Size-dependent static variables
            size = self.series_ids.iloc[i]['size']
            size_effect = {'Small': 0.5, 'Medium': 1.0, 'Large': 2.0}[size]
            x5 = size_effect + 0.3 * np.random.randn()
            
            static_data.append({
                'series_id': self.series_ids.iloc[i]['series_id'],
                'static_1': x1,
                'static_2': x2,
                'static_3': x3,
                'static_4': x4,
                'static_5': x5,
                'static_6': np.random.gamma(2, 2),  # Skewed distribution
                'static_7': np.random.beta(2, 5),   # Different distribution
                'static_categorical': category
            })
        
        return pd.DataFrame(static_data)
    
    def _generate_exog_vars(self) -> Dict[str, pd.DataFrame]:
        """Generate exogenous variables with different characteristics"""
        t = np.arange(self.n_timesteps)
        
        # Generate exogenous variables with different patterns
        exog_data = {}
        
        # Exog 1: Seasonal pattern with trend
        seasonal_period = 365
        exog_1 = 10 + 0.01 * t + 5 * np.sin(2 * np.pi * t / seasonal_period)
        exog_1 += np.random.randn(self.n_timesteps) * 0.5
        
        # Exog 2: Weekly pattern
        exog_2 = 3 + 2 * np.sin(2 * np.pi * t / 7) + np.cos(2 * np.pi * t / 7)
        exog_2 += np.random.randn(self.n_timesteps) * 0.3
        
        # Exog 3: Stepwise changes
        exog_3 = np.zeros(self.n_timesteps)
        step_points = [0, 100, 200, 300]
        step_values = [1, 3, 2, 4]
        for i, point in enumerate(step_points):
            if point < self.n_timesteps:
                exog_3[point:] = step_values[i]
        exog_3 += np.random.randn(self.n_timesteps) * 0.2
        
        # Exog 4: Random walk
        exog_4 = np.cumsum(np.random.randn(self.n_timesteps) * 0.1)
        exog_4 += 5
        
        # Exog 5: Spike pattern (irregular events)
        exog_5 = np.zeros(self.n_timesteps)
        spike_prob = 0.05
        spike_locations = np.random.choice(self.n_timesteps, size=int(self.n_timesteps * spike_prob), replace=False)
        exog_5[spike_locations] = np.random.randn(len(spike_locations)) * 3 + 5
        exog_5 += np.random.randn(self.n_timesteps) * 0.1
        
        # Create DataFrames for each series
        for i in range(self.n_series):
            series_id = self.series_ids.iloc[i]['series_id']
            
            # Add some series-specific variation
            scale = 0.8 + 0.4 * np.random.rand()
            shift = np.random.randn() * 0.5
            
            exog_data[series_id] = pd.DataFrame({
                'exog_1': exog_1 * scale + shift,
                'exog_2': exog_2 * scale + shift,
                'exog_3': exog_3 * scale + shift,
                'exog_4': exog_4 * scale + shift,
                'exog_5': exog_5 * scale + shift,
            }, index=self.time_index)
        
        return exog_data
    
    def _generate_target_series(self) -> Dict[str, pd.DataFrame]:
        """Generate target time series with multiple patterns"""
        series_data = {}
        t = np.arange(self.n_timesteps)
        
        for i in range(self.n_series):
            series_id = self.series_ids.iloc[i]['series_id']
            
            # Base components
            base_level = 100 + 50 * np.random.randn()
            
            # Trend component
            trend_strength = np.random.uniform(0.5, 2.0)
            trend_direction = np.random.choice([-1, 1])
            trend = trend_direction * trend_strength * (t / self.n_timesteps) * base_level * 0.2
            
            # Seasonal components
            yearly_amp = np.random.uniform(0.1, 0.3) * base_level
            yearly_phase = np.random.uniform(0, 2*np.pi)
            yearly_seasonal = yearly_amp * np.sin(2 * np.pi * t / 365 + yearly_phase)
            
            weekly_amp = np.random.uniform(0.05, 0.15) * base_level
            weekly_phase = np.random.uniform(0, 2*np.pi)
            weekly_seasonal = weekly_amp * np.sin(2 * np.pi * t / 7 + weekly_phase)
            
            # Cyclic component (longer than yearly, e.g., business cycle)
            cycle_period = np.random.uniform(500, 800)
            cycle_amp = np.random.uniform(0.15, 0.25) * base_level
            cyclic = cycle_amp * np.sin(2 * np.pi * t / cycle_period)
            
            # AR component
            ar_coef = np.random.uniform(0.3, 0.7)
            ar_process = np.zeros(self.n_timesteps)
            for j in range(1, self.n_timesteps):
                ar_process[j] = ar_coef * ar_process[j-1] + np.random.randn() * 0.5
            
            # Exogenous effects
            exog_effect = 0
            for j in range(1, 6):
                exog_coef = np.random.uniform(-0.1, 0.1)
                exog_effect += exog_coef * self.exog_vars[series_id][f'exog_{j}']
            
            # Static variable effects
            static_effect = 0
            for j in range(1, 6):
                static_coef = np.random.uniform(-0.2, 0.2)
                static_val = self.static_vars.loc[i, f'static_{j}']
                static_effect += static_coef * static_val * base_level
            
            # Combine all components
            value = (base_level + trend + yearly_seasonal + weekly_seasonal + 
                    cyclic + ar_process * 0.1 * base_level + exog_effect + static_effect)
            
            # Add noise
            noise = np.random.randn(self.n_timesteps) * self.noise_level * base_level
            value += noise
            
            # Add outliers
            outlier_mask = np.random.rand(self.n_timesteps) < self.outlier_prob
            outlier_values = np.random.randn(outlier_mask.sum()) * 3 * base_level
            value[outlier_mask] += outlier_values
            
            # Ensure positive values (for models that require them)
            value = np.maximum(value, 0.1)
            
            # Create DataFrame
            df = pd.DataFrame({
                'value': value,
                'trend': trend,
                'yearly_seasonal': yearly_seasonal,
                'weekly_seasonal': weekly_seasonal,
                'cyclic': cyclic,
                'noise': noise,
                'is_outlier': outlier_mask.astype(int)
            }, index=self.time_index)
            
            # Add missing values
            missing_mask = np.random.rand(self.n_timesteps) < self.missing_prob
            df.loc[missing_mask, 'value'] = np.nan
            
            series_data[series_id] = df
        
        return series_data
    
    def _convert_to_count(self, series_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Convert continuous series to count data"""
        count_data = {}
        
        for series_id, df in series_data.items():
            # Transform to count data (e.g., integer values)
            count_values = np.round(np.exp(df['value'] / 50)).astype(int)
            count_values = np.maximum(count_values, 0)  # Ensure non-negative
            
            count_df = df.copy()
            count_df['value'] = count_values
            count_data[series_id] = count_df
        
        return count_data
    
    def _convert_to_binary(self, series_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Convert continuous series to binary data"""
        binary_data = {}
        
        for series_id, df in series_data.items():
            # Use a dynamic threshold for each series
            threshold = df['value'].quantile(0.5)
            binary_values = (df['value'] > threshold).astype(int)
            
            binary_df = df.copy()
            binary_df['value'] = binary_values
            binary_data[series_id] = binary_df
        
        return binary_data
    
    def _create_final_dataset(self) -> pd.DataFrame:
        """Combine all data into final dataset format"""
        all_data = []
        
        for i in range(self.n_series):
            series_id = self.series_ids.iloc[i]['series_id']
            
            # Time series data
            ts_data = self.target_series[series_id].copy()
            ts_data['series_id'] = series_id
            ts_data['timestamp'] = ts_data.index
            
            # Add count and binary versions
            ts_data['value_count'] = self.count_series[series_id]['value']
            ts_data['value_binary'] = self.binary_series[series_id]['value']
            
            # Add exogenous variables
            for col in self.exog_vars[series_id].columns:
                ts_data[col] = self.exog_vars[series_id][col]
            
            # Add static variables
            for col in self.static_vars.columns:
                if col != 'series_id':
                    ts_data[col] = self.static_vars.loc[i, col]
            
            # Add time features
            for col in self.time_features.columns:
                ts_data[col] = self.time_features[col]
            
            all_data.append(ts_data)
        
        final_df = pd.concat(all_data, ignore_index=True)
        return final_df
    
    def get_train_test_split(self, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and testing sets"""
        cutoff_idx = int(len(self.time_index) * (1 - test_ratio))
        cutoff_date = self.time_index[cutoff_idx]
        
        train_data = self.data[self.data['timestamp'] < cutoff_date]
        test_data = self.data[self.data['timestamp'] >= cutoff_date]
        
        return train_data, test_data
    
    def get_data_for_model(self, model_type: str) -> Dict[str, any]:
        """
        Get data in format appropriate for different models
        
        Args:
            model_type: One of ['nbeats', 'tft', 'deepar', 'lstm', 'lstm_attention', 
                               'arima', 'random_forest', 'xgboost']
        """
        if model_type in ['nbeats', 'deepar', 'lstm', 'lstm_attention']:
            return self._prepare_sequential_data()
        elif model_type == 'tft':
            return self._prepare_tft_data()
        elif model_type == 'arima':
            return self._prepare_arima_data()
        elif model_type in ['random_forest', 'xgboost']:
            return self._prepare_ml_data()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _prepare_sequential_data(self) -> Dict[str, any]:
        """Prepare data for sequential models (NBeats, DeepAR, LSTM)"""
        # Define sequence and prediction lengths
        seq_length = 60
        pred_length = 7
        
        sequences = []
        targets = []
        static_features = []
        exog_features = []
        series_ids = []
        
        for series_id in self.series_ids['series_id']:
            series_data = self.data[self.data['series_id'] == series_id].copy()
            series_data = series_data.sort_values('timestamp')
            
            # Extract features
            values = series_data['value'].values
            exog_cols = [col for col in series_data.columns if col.startswith('exog_')]
            static_cols = [col for col in series_data.columns if col.startswith('static_')]
            
            # Create sequences
            for i in range(len(values) - seq_length - pred_length + 1):
                sequence = values[i:i+seq_length]
                target = values[i+seq_length:i+seq_length+pred_length]
                
                # Skip if any missing values in sequence or target
                if np.isnan(sequence).any() or np.isnan(target).any():
                    continue
                
                # Extract exogenous features for the sequence and future
                exog_seq = series_data.iloc[i:i+seq_length][exog_cols].values
                exog_future = series_data.iloc[i+seq_length:i+seq_length+pred_length][exog_cols].values
                exog_combined = np.concatenate([exog_seq, exog_future], axis=0)
                
                # Extract static features
                static_feat = series_data.iloc[i][static_cols].values
                
                sequences.append(sequence)
                targets.append(target)
                exog_features.append(exog_combined)
                static_features.append(static_feat)
                series_ids.append(series_id)
        
        return {
            'sequences': np.array(sequences),
            'targets': np.array(targets),
            'exog_features': np.array(exog_features),
            'static_features': np.array(static_features),
            'series_ids': series_ids,
            'seq_length': seq_length,
            'pred_length': pred_length
        }
    
    def _prepare_tft_data(self) -> Dict[str, any]:
        """Prepare data specifically for TFT model"""
        # Similar to sequential data but with TFT-specific formatting
        seq_length = 60
        pred_length = 7
        
        static_inputs = []
        encoder_inputs = []
        decoder_inputs = []
        targets = []
        
        for series_id in self.series_ids['series_id']:
            series_data = self.data[self.data['series_id'] == series_id].copy()
            series_data = series_data.sort_values('timestamp')
            
            for i in range(len(series_data) - seq_length - pred_length + 1):
                # Target values
                target = series_data.iloc[i:i+seq_length+pred_length]['value'].values
                if np.isnan(target).any():
                    continue
                
                # Static inputs (one value per series)
                static_cols = [col for col in series_data.columns if col.startswith('static_')]
                static_input = []
                for col in static_cols:
                    static_input.append(series_data.iloc[i][col])
                
                # Encoder inputs (historical data)
                encoder_cols = [col for col in series_data.columns if col.startswith('exog_')]
                encoder_cols.append('value')
                encoder_input = []
                for col in encoder_cols:
                    encoder_input.append(series_data.iloc[i:i+seq_length][col].values)
                
                # Decoder inputs (known future data)
                decoder_cols = [col for col in series_data.columns if col.startswith('exog_')]
                decoder_input = []
                for col in decoder_cols:
                    decoder_input.append(series_data.iloc[i+seq_length:i+seq_length+pred_length][col].values)
                
                static_inputs.append(static_input)
                encoder_inputs.append(encoder_input)
                decoder_inputs.append(decoder_input)
                targets.append(target)
        
        return {
            'static_inputs': static_inputs,
            'encoder_inputs': encoder_inputs,
            'decoder_inputs': decoder_inputs,
            'targets': np.array(targets),
            'seq_length': seq_length,
            'pred_length': pred_length
        }
    
    def _prepare_arima_data(self) -> Dict[str, pd.DataFrame]:
        """Prepare data for ARIMA models (one series per output)"""
        arima_data = {}
        
        for series_id in self.series_ids['series_id']:
            series_data = self.data[self.data['series_id'] == series_id].copy()
            series_data = series_data.sort_values('timestamp').set_index('timestamp')
            
            # For ARIMA, we primarily use the value column
            arima_data[series_id] = series_data[['value']].fillna(method='ffill').fillna(method='bfill')
        
        return arima_data
    
    def _prepare_ml_data(self) -> Dict[str, any]:
        """Prepare data for tree-based ML models (Random Forest, XGBoost)"""
        # Create feature engineering for ML models
        features = []
        targets = []
        
        for series_id in self.series_ids['series_id']:
            series_data = self.data[self.data['series_id'] == series_id].copy()
            series_data = series_data.sort_values('timestamp')
            
            # Create lag features
            for lag in range(1, 8):
                series_data[f'lag_{lag}'] = series_data['value'].shift(lag)
            
            # Create rolling window features
            for window in [7, 30]:
                series_data[f'rolling_mean_{window}'] = series_data['value'].rolling(window).mean()
                series_data[f'rolling_std_{window}'] = series_data['value'].rolling(window).std()
            
            # Create feature set
            feature_cols = [col for col in series_data.columns if col not in ['value', 'timestamp', 'series_id']]
            X = series_data[feature_cols].fillna(0)
            y = series_data['value']
            
            # Remove rows with NaN in target
            valid_mask = ~y.isna()
            X = X[valid_mask]
            y = y[valid_mask]
            
            features.extend(X.values.tolist())
            targets.extend(y.values.tolist())
        
        return {
            'features': np.array(features),
            'targets': np.array(targets),
            'feature_names': feature_cols
        }
    
    def visualize_samples(self, n_series: int = 3, save_path: Optional[str] = None):
        """Visualize sample time series and components"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 15))
        
        for i in range(min(n_series, self.n_series)):
            series_id = self.series_ids.iloc[i]['series_id']
            series_data = self.target_series[series_id]
            
            # Plot main value
            axes[0].plot(series_data.index, series_data['value'], 
                        label=f'{series_id} (Cat: {self.series_ids.iloc[i]["category"]})', 
                        alpha=0.7)
            
            # Plot components for first series only
            if i == 0:
                axes[1].plot(series_data.index, series_data['trend'], label='Trend', color='red')
                axes[1].plot(series_data.index, series_data['yearly_seasonal'], label='Yearly Seasonal', color='green')
                axes[1].plot(series_data.index, series_data['weekly_seasonal'], label='Weekly Seasonal', color='blue')
                axes[1].plot(series_data.index, series_data['cyclic'], label='Cyclic', color='purple')
                
                # Plot exogenous variables
                exog_data = self.exog_vars[series_id]
                axes[2].plot(exog_data.index, exog_data['exog_1'], label='Exog 1', alpha=0.7)
                axes[2].plot(exog_data.index, exog_data['exog_2'], label='Exog 2', alpha=0.7)
                axes[2].plot(exog_data.index, exog_data['exog_3'], label='Exog 3', alpha=0.7)
        
        axes[0].set_title('Sample Time Series')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Value')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_title('Time Series Components (First Series)')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Component Value')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        axes[2].set_title('Exogenous Variables (First Series)')
        axes[2].set_xlabel('Date')
        axes[2].set_ylabel('Value')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def get_statistics(self) -> Dict[str, any]:
        """Get comprehensive statistics of the dataset"""
        stats = {
            'basic_info': {
                'n_series': self.n_series,
                'n_timesteps': self.n_timesteps,
                'freq': self.freq,
                'time_range': f"{self.time_index[0]} to {self.time_index[-1]}",
                'n_exog_vars': self.n_exog_vars,
                'n_static_vars': self.n_static_vars
            },
            'series_characteristics': {},
            'missing_data': {},
            'hierarchy_info': {}
        }
        
        # Calculate per-series statistics
        for i, series_id in enumerate(self.series_ids['series_id'][:10]):  # First 10 for display
            series_data = self.target_series[series_id]['value']
            stats['series_characteristics'][series_id] = {
                'mean': float(series_data.mean()),
                'std': float(series_data.std()),
                'min': float(series_data.min()),
                'max': float(series_data.max()),
                'missing_count': int(series_data.isna().sum()),
                'missing_percentage': float(series_data.isna().mean() * 100)
            }
        
        # Overall missing data stats
        all_values = []
        all_missing = []
        for series_data in self.target_series.values():
            all_values.extend(series_data['value'].dropna().tolist())
            all_missing.append(series_data['value'].isna().sum())
        
        stats['missing_data'] = {
            'total_missing': sum(all_missing),
            'average_missing_per_series': np.mean(all_missing),
            'overall_missing_percentage': (sum(all_missing) / (self.n_series * self.n_timesteps)) * 100
        }
        
        # Hierarchy information
        if self.hierarchy:
            stats['hierarchy_info'] = {
                'total_levels': len(self.hierarchy),
                'level_sizes': {k: len(v) for k, v in list(self.hierarchy.items())[:5]}  # Show first 5
            }
        
        # Data type statistics
        stats['data_types'] = {
            'continuous': {
                'mean': np.mean(all_values),
                'std': np.std(all_values),
                'range': (min(all_values), max(all_values))
            },
            'count': {
                'mean': np.mean([self.count_series[sid]['value'].mean() for sid in self.series_ids['series_id'][:10]]),
                'max': np.max([self.count_series[sid]['value'].max() for sid in self.series_ids['series_id'][:10]])
            },
            'binary': {
                'positive_ratio': np.mean([self.binary_series[sid]['value'].mean() for sid in self.series_ids['series_id'][:10]])
            }
        }
        
        return stats
    
    def validate_dataset(self) -> Dict[str, bool]:
        """Validate the dataset for common issues"""
        validation_results = {
            'time_index_consistency': True,
            'no_duplicate_series_ids': True,
            'all_features_present': True,
            'reasonable_value_ranges': True,
            'exogenous_alignment': True
        }
        
        # Check time index consistency
        try:
            assert all(self.target_series[sid].index.equals(self.time_index) for sid in self.series_ids['series_id'][:5])
        except:
            validation_results['time_index_consistency'] = False
        
        # Check for duplicate series IDs
        validation_results['no_duplicate_series_ids'] = len(self.series_ids['series_id'].unique()) == self.n_series
        
        # Check all features are present
        required_columns = ['value', 'trend', 'yearly_seasonal', 'weekly_seasonal', 'cyclic', 'noise']
        for i in range(5):  # Check first 5 series
            series_id = self.series_ids.iloc[i]['series_id']
            if not all(col in self.target_series[series_id].columns for col in required_columns):
                validation_results['all_features_present'] = False
                break
        
        # Check value ranges are reasonable (no extreme outliers)
        all_values = []
        for sid in self.series_ids['series_id'][:10]:
            all_values.extend(self.target_series[sid]['value'].dropna().tolist())
        
        mean_val = np.mean(all_values)
        std_val = np.std(all_values)
        validation_results['reasonable_value_ranges'] = all(abs(v - mean_val) < 5 * std_val for v in all_values)
        
        # Check exogenous variable alignment
        for i in range(3):  # Check first 3 series
            series_id = self.series_ids.iloc[i]['series_id']
            if not self.exog_vars[series_id].index.equals(self.time_index):
                validation_results['exogenous_alignment'] = False
                break
        
        return validation_results


# Example usage and demonstration
if __name__ == "__main__":
    # Create comprehensive dataset
    dataset = ComprehensiveSyntheticDataset(
        n_series=100,
        n_timesteps=365,
        freq='D',
        start_date='2020-01-01',
        n_exog_vars=5,
        n_static_vars=8,
        noise_level=0.1,
        missing_prob=0.02,
        outlier_prob=0.01,
        random_seed=42,
        include_hierarchical=True,
        n_levels=3
    )
    
    # Get statistics
    stats = dataset.get_statistics()
    print("Dataset Statistics:", json.dumps(stats, indent=2))
    
    # Validate dataset
    validation = dataset.validate_dataset()
    print("\nDataset Validation:", validation)
    
    # Visualize samples
    dataset.visualize_samples(n_series=3, save_path='dataset_visualization.png')
    
    # Get train/test split
    train_data, test_data = dataset.get_train_test_split(test_ratio=0.2)
    print(f"\nTraining data shape: {train_data.shape}")
    print(f"Testing data shape: {test_data.shape}")
    
    # Prepare data for different models
    model_types = ['nbeats', 'tft', 'deepar', 'lstm', 'arima', 'random_forest']
    
    for model_type in model_types:
        try:
            model_data = dataset.get_data_for_model(model_type)
            print(f"\n{model_type.upper()} data prepared successfully")
            
            if model_type in ['nbeats', 'deepar', 'lstm']:
                print(f"  Sequences shape: {model_data['sequences'].shape}")
                print(f"  Targets shape: {model_data['targets'].shape}")
                print(f"  Exogenous features shape: {model_data['exog_features'].shape}")
                print(f"  Static features shape: {model_data['static_features'].shape}")
            
            elif model_type == 'tft':
                print(f"  Number of static inputs: {len(model_data['static_inputs'])}")
                print(f"  Number of encoder inputs per sample: {len(model_data['encoder_inputs'][0])}")
                print(f"  Number of decoder inputs per sample: {len(model_data['decoder_inputs'][0])}")
                print(f"  Targets shape: {model_data['targets'].shape}")
            
            elif model_type == 'arima':
                print(f"  Number of time series: {len(model_data)}")
                first_key = list(model_data.keys())[0]
                print(f"  First series shape: {model_data[first_key].shape}")
            
            elif model_type == 'random_forest':
                print(f"  Features shape: {model_data['features'].shape}")
                print(f"  Targets shape: {len(model_data['targets'])}")
                print(f"  Number of features: {len(model_data['feature_names'])}")
                print(f"  Feature names: {model_data['feature_names'][:10]}...")
        
        except Exception as e:
            print(f"\nError preparing data for {model_type}: {str(e)}")
    
    # Example: Accessing specific series data
    series_id = dataset.series_ids.iloc[0]['series_id']
    print(f"\nDetails for series {series_id}:")
    print(f"Category: {dataset.series_ids.iloc[0]['category']}")
    print(f"Region: {dataset.series_ids.iloc[0]['region']}")
    print(f"Size: {dataset.series_ids.iloc[0]['size']}")
    
    # Show first few rows of static variables for this series
    static_info = dataset.static_vars[dataset.static_vars['series_id'] == series_id].iloc[0]
    print("\nStatic variables:")
    for col in ['static_1', 'static_2', 'static_3', 'static_4', 'static_5']:
        print(f"  {col}: {static_info[col]:.3f}")
    
    # Show hierarchy information
    if dataset.hierarchy:
        print(f"\nHierarchy levels: {len(dataset.hierarchy)}")
        print("Hierarchy structure (sample):")
        for i, (key, value) in enumerate(list(dataset.hierarchy.items())[:5]):
            print(f"  {key}: {len(value)} series")
    
    print("\nDataset generation completed successfully!")
    print("This dataset includes:")
    print("- Multiple time series with various patterns (trend, seasonality, cycles)")
    print("- Exogenous variables with different characteristics")
    print("- Static variables including categorical features")
    print("- Missing values and outliers")
    print("- Hierarchical structure for aggregation modeling")
    print("- Time features for temporal modeling")
    print("- Different data types (continuous, count, binary)")
    print("\nThe dataset is ready for benchmarking various time series models!")
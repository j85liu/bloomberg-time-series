import os
import sys
import numpy as np
import torch
import pandas as pd  # Needed for handling timestamps
from sklearn.preprocessing import MinMaxScaler

# Ensure correct working directory for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))  
sys.path.append(PROJECT_ROOT)  

def prepare_data(time_series, seq_length=50, train_ratio=0.8):
    """
    Prepares time-series data for training.

    Parameters:
    - time_series (pd.DataFrame or np.array): Raw time-series data (could contain timestamps).
    - seq_length (int): Number of time steps in each training sample.
    - train_ratio (float): Ratio of data to use for training.

    Returns:
    - X_train, Y_train, X_test, Y_test: Training and test tensors
    - scaler: Fitted MinMaxScaler for inverse transformation
    """
    
    # Convert to Pandas DataFrame if it's a NumPy array
    if isinstance(time_series, np.ndarray):
        time_series = pd.DataFrame(time_series)  
    
    # **Ensure Timestamps Are Removed**
    if isinstance(time_series, pd.DataFrame):
        # If first column is a timestamp, remove it
        if pd.api.types.is_datetime64_any_dtype(time_series.iloc[:, 0]):
            print("📅 Detected timestamps. Dropping first column...")
            time_series = time_series.iloc[:, 1:]

        # Convert all columns to numeric (force non-numeric values to NaN)
        time_series = time_series.apply(pd.to_numeric, errors='coerce')

        # Drop rows with NaN values (in case conversion failed)
        time_series = time_series.dropna()

    # Convert DataFrame to NumPy array
    time_series = time_series.values

    # Normalize data (MinMax Scaling)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(time_series)

    # Create sequences
    X, Y = [], []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i + seq_length])
        Y.append(scaled_data[i + seq_length])  
    
    X, Y = np.array(X), np.array(Y)
    
    # Split into train and test
    train_size = int(len(X) * train_ratio)
    X_train, Y_train = X[:train_size], Y[:train_size]
    X_test, Y_test = X[train_size:], Y[train_size:]

    # Convert to PyTorch tensors
    X_train, Y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32)
    X_test, Y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.float32)

    return X_train, Y_train, X_test, Y_test, scaler

# **🚀 Test Fix**
if __name__ == "__main__":
    df = pd.DataFrame({
        "timestamp": pd.date_range(start="2023-01-01", periods=200, freq="D"),
        "price": np.sin(np.linspace(0, 10, 200)) + np.random.normal(0, 0.1, 200)  
    })
    X_train, Y_train, X_test, Y_test, _ = prepare_data(df)
    print(f"✅ Data prepared: Train shape {X_train.shape}, Test shape {X_test.shape}")

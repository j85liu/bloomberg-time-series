import os
import sys
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

# 🔹 Ensure Python can find this file's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # utils/
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))  # Root directory
sys.path.append(PROJECT_ROOT)  # Add root to system path

def prepare_data(time_series, seq_length=50, train_ratio=0.8):
    """
    Prepares time-series data for training.

    Parameters:
    - time_series (array-like): Raw financial or synthetic time series data.
    - seq_length (int): Number of time steps in each training sample.
    - train_ratio (float): Ratio of data to use for training.

    Returns:
    - X_train, Y_train: Training feature/target tensors
    - X_test, Y_test: Testing feature/target tensors
    - scaler: Fitted MinMaxScaler for inverse transformation
    """
    
    # Ensure data is numpy array
    time_series = np.array(time_series).reshape(-1, 1)
    
    # Normalize data (MinMax Scaling)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(time_series)

    # Create sequences
    X, Y = [], []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i + seq_length])
        Y.append(scaled_data[i + seq_length])  # Predict next step
    
    X, Y = np.array(X), np.array(Y)
    
    # Split into train and test
    train_size = int(len(X) * train_ratio)
    X_train, Y_train = X[:train_size], Y[:train_size]
    X_test, Y_test = X[train_size:], Y[train_size:]

    # Convert to PyTorch tensors
    X_train, Y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32)
    X_test, Y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.float32)

    return X_train, Y_train, X_test, Y_test, scaler  # Return scaler for inverse transformation

# 🔹 Quick test to ensure this works
if __name__ == "__main__":
    test_series = np.sin(np.linspace(0, 10, 200))  # Example sine wave data
    X_train, Y_train, X_test, Y_test, _ = prepare_data(test_series)
    print(f"✅ Data prepared: Train shape {X_train.shape}, Test shape {X_test.shape}")

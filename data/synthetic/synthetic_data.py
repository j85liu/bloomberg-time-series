import numpy as np
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt

def generate_gbm_series(T=3650, S0=100, mu=0.05, sigma=0.2, dt=1/252, change_points=None):
    """
    Generates a synthetic financial time-series using Geometric Brownian Motion (GBM).

    Parameters:
        T (int): Number of time steps (3650 for ~10 years of data)
        S0 (float): Initial stock price
        mu (float): Expected annual return (drift)
        sigma (float): Volatility (standard deviation of returns)
        dt (float): Time step (default: 1/252 for daily stock prices)
        change_points (dict): {time_step: (new_mu, new_sigma)} for regime shifts.

    Returns:
        pd.DataFrame: Time series data with columns ['Date', 'Price']
    """
    np.random.seed(None)  # Ensure different randomization each run
    time = np.arange(T)
    S = np.zeros(T)
    S[0] = S0

    for t in range(1, T):
        if change_points and t in change_points:
            mu, sigma = change_points[t]
        dW = np.random.normal(0, np.sqrt(dt))  # Random shock
        S[t] = S[t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)

    df = pd.DataFrame({'Date': pd.date_range(start='2010-01-01', periods=T, freq='D'), 'Price': S})
    return df

def generate_multiple_series(num_series=100, T=3650, save_csv=True, save_path="data/"):
    """
    Generates multiple synthetic financial time-series and saves them.

    Parameters:
        num_series (int): Number of different time-series to generate.
        T (int): Number of time steps per series (3650 for ~10 years).
        save_csv (bool): If True, saves each time-series as CSV.
        save_path (str): Directory where CSV files are stored.

    Returns:
        dict: Dictionary of generated DataFrames.
    """

    # ✅ 1️⃣ DELETE any existing synthetic data
    if os.path.exists(save_path):
        shutil.rmtree(save_path)  # Remove old folder
    os.makedirs(save_path)  # Create a fresh directory

    series_dict = {}

    plt.figure(figsize=(15, 7))  # Initialize plot

    for i in range(num_series):
        mu = np.random.uniform(0.03, 0.07)  # Randomized drift
        sigma = np.random.uniform(0.15, 0.3)  # Randomized volatility
        change_points = {T // 2: (mu * 0.5, sigma * 1.5)}  # Regime shift halfway

        df = generate_gbm_series(T=T, S0=100, mu=mu, sigma=sigma, change_points=change_points)
        series_dict[f"synthetic_series_{i+1}"] = df

        if save_csv:
            df.to_csv(os.path.join(save_path, f"synthetic_series_{i+1}.csv"), index=False)

        if i < 10:  # Only plot first 10 series to avoid clutter
            plt.plot(df["Date"], df["Price"], label=f"Series {i+1}", alpha=0.7)

    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title("Synthetic Stock Price Time Series (GBM) - First 10 Series")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"\n✅ {num_series} synthetic time series saved in '{save_path}'")

    return series_dict

if __name__ == "__main__":
    generate_multiple_series(num_series=100, T=3650)

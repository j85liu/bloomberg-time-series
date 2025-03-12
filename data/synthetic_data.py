import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_gbm_series(T=252, S0=100, mu=0.05, sigma=0.2, dt=1/252, change_points=None, seed=None):
    """
    Generates a synthetic financial time-series using Geometric Brownian Motion (GBM).

    Parameters:
        T (int): Number of time steps (default: 252 trading days ~ 1 year)
        S0 (float): Initial stock price
        mu (float): Expected annual return (drift)
        sigma (float): Volatility (standard deviation of returns)
        dt (float): Time step (default: 1/252 for daily stock prices)
        change_points (dict): Dictionary of {time_step: new_mu, new_sigma} for regime shifts.
        seed (int): Random seed for reproducibility.
    
    Returns:
        pd.DataFrame: Time series data with columns ['Date', 'Price']
    """
    if seed is not None:
        np.random.seed(seed)

    time = np.arange(T)
    S = np.zeros(T)
    S[0] = S0

    # Generate Brownian motion with dynamic drift/sigma
    for t in range(1, T):
        if change_points and t in change_points:
            mu, sigma = change_points[t]  # Apply new drift & volatility

        dW = np.random.normal(0, np.sqrt(dt))  # Random shock
        S[t] = S[t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)

    df = pd.DataFrame({'Date': pd.date_range(start='2023-01-01', periods=T, freq='D'), 'Price': S})
    return df


def generate_multiple_series(num_series=3, T=252, save_csv=False):
    """
    Generates multiple synthetic financial time-series with different GBM parameters.

    Parameters:
        num_series (int): Number of different time-series to generate.
        T (int): Number of time steps per series.
        save_csv (bool): If True, saves each time-series as CSV.

    Returns:
        dict: Dictionary of generated dataframes.
    """
    series_dict = {}

    # Ensure the "data" folder exists
    if save_csv and not os.path.exists("data"):
        os.makedirs("data")

    for i in range(num_series):
        seed = i + 42  # Different seed for each series
        mu = np.random.uniform(0.03, 0.07)  # Randomized drift per series
        sigma = np.random.uniform(0.15, 0.3)  # Randomized volatility per series

        # Introduce unique regime shifts for each series
        change_points = {
            int(T * np.random.uniform(0.3, 0.7)): (mu * np.random.uniform(0.4, 0.8), sigma * np.random.uniform(1.2, 1.8))
        }

        df = generate_gbm_series(T=T, S0=100, mu=mu, sigma=sigma, change_points=change_points, seed=seed)
        series_dict[f"synthetic_series_{i+1}"] = df

        if save_csv:
            df.to_csv(f"data/synthetic_series_{i+1}.csv", index=False)

    return series_dict


if __name__ == "__main__":
    num_series = 3
    synthetic_data = generate_multiple_series(num_series, save_csv=True)

    # Plot each series separately
    for i, (name, df) in enumerate(synthetic_data.items()):
        plt.figure(figsize=(10, 5))
        plt.plot(df["Date"], df["Price"], label=name, color=f"C{i}")
        
        # Formatting
        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.title(f"Synthetic Stock Price Time Series {i+1}")
        plt.legend()
        plt.grid(True)
        plt.show()

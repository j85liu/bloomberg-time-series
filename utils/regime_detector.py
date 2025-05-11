# Create a new file: utils/regime_detector.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

class MarketRegimeDetector:
    """
    Detect and classify market regimes to help with model selection
    
    This class implements multiple approaches to regime detection:
    1. Rules-based approach using volatility and trend
    2. Statistical clustering of market conditions
    3. Change-point detection for regime shifts
    """
    
    def __init__(self, num_regimes=4, method='hybrid', lookback=252):
        """
        Initialize regime detector
        
        Args:
            num_regimes: Number of regimes to identify
            method: Detection method ('rules', 'clustering', 'changepoint', or 'hybrid')
            lookback: Lookback period for statistics
        """
        self.num_regimes = num_regimes
        self.method = method
        self.lookback = lookback
        
        # For clustering method
        self.kmeans = None
        self.pca = None
        
        # For hybrid method
        self.regime_thresholds = None
        
    def rules_based_regimes(self, df):
        """
        Apply rules-based regime detection
        
        Args:
            df: DataFrame with 'close' column for VIX
            
        Returns:
            regime: Array of regime labels
        """
        # Calculate volatility and trend features
        vix = df['close'].values
        vix_returns = np.diff(np.log(vix)) if len(vix) > 1 else np.zeros(1)
        vix_returns = np.append(0, vix_returns)  # Pad first value
        
        # Calculate moving statistics
        window = min(self.lookback, len(vix) - 1)
        if window <= 0:
            return np.zeros(len(vix), dtype=int)
            
        # Initialize arrays
        vix_mean = np.zeros_like(vix)
        vix_std = np.zeros_like(vix)
        
        # Calculate rolling statistics
        for i in range(window, len(vix)):
            vix_mean[i] = np.mean(vix[i-window:i])
            vix_std[i] = np.std(vix[i-window:i])
        
        # Fill initial values
        vix_mean[:window] = vix_mean[window]
        vix_std[:window] = vix_std[window]
        
        # Calculate z-scores
        vix_zscore = (vix - vix_mean) / (vix_std + 1e-8)
        
        # Calculate trend
        trend_window = min(21, len(vix) - 1)
        vix_trend = np.zeros_like(vix)
        
        for i in range(trend_window, len(vix)):
            vix_trend[i] = (vix[i] / vix[i-trend_window]) - 1
            
        # Fill initial values
        vix_trend[:trend_window] = vix_trend[trend_window]
        
        # Define regimes based on z-score and trend
        regime = np.ones_like(vix, dtype=int)  # Default: normal volatility
        
        # Low volatility regime
        regime[vix_zscore < -1.0] = 0
        
        # High volatility regimes
        regime[vix_zscore > 1.0] = 2  # High volatility
        regime[vix_zscore > 2.0] = 3  # Crisis
        
        # Refine with trend
        # In normal regime, differentiate between rising and falling vol
        mask_normal = (regime == 1)
        regime[mask_normal & (vix_trend > 0.2)] = 4  # Rising volatility
        regime[mask_normal & (vix_trend < -0.2)] = 5  # Falling volatility
        
        # Map to desired number of regimes if necessary
        if self.num_regimes < 6:
            # Create simpler mapping (combine similar regimes)
            mapping = {
                0: 0,  # Low volatility
                1: 1,  # Normal volatility
                4: 1,  # Rising volatility (map to normal)
                5: 1,  # Falling volatility (map to normal)
                2: 2,  # High volatility
                3: 3   # Crisis
            }
            
            # Apply mapping
            new_regime = np.zeros_like(regime)
            for old, new in mapping.items():
                new_regime[regime == old] = new
                
            regime = new_regime
            
        return regime
    
    def clustering_regimes(self, df):
        """
        Apply clustering-based regime detection
        
        Args:
            df: DataFrame with market features
            
        Returns:
            regime: Array of regime labels
        """
        # Extract features for clustering
        features = []
        
        # Basic VIX features
        if 'close' in df.columns:
            vix = df['close'].values
            features.append(vix)
            
            # Calculate returns if we have enough data
            if len(vix) > 1:
                vix_returns = np.diff(np.log(vix))
                vix_returns = np.append(0, vix_returns)  # Pad first value
                features.append(vix_returns)
                
                # Volatility of volatility
                vol_window = min(21, len(vix_returns) - 1)
                if vol_window > 0:
                    vol_of_vol = np.zeros_like(vix_returns)
                    for i in range(vol_window, len(vix_returns)):
                        vol_of_vol[i] = np.std(vix_returns[i-vol_window:i])
                    vol_of_vol[:vol_window] = vol_of_vol[vol_window]
                    features.append(vol_of_vol)
        
        # Additional features if available
        feature_names = [
            'vol_5d', 'vol_21d', 'momentum_5d', 'momentum_21d',
            'T10Y2Y', 'STLFSI', 'BAA10Y'
        ]
        
        for feature in feature_names:
            if feature in df.columns:
                features.append(df[feature].values)
        
        # Need at least 2 features for meaningful clustering
        if len(features) < 2:
            # Fallback to rules-based
            return self.rules_based_regimes(df)
        
        # Prepare feature matrix
        X = np.column_stack(features)
        
        # Handle NaNs
        X = np.nan_to_num(X, nan=0.0)
        
        # Standardize features
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_std = (X - mean) / (std + 1e-8)
        
        # Dimensionality reduction if we have many features
        if X_std.shape[1] > 5:
            if self.pca is None:
                self.pca = PCA(n_components=min(5, X_std.shape[1]))
                self.pca.fit(X_std)
            
            X_reduced = self.pca.transform(X_std)
        else:
            X_reduced = X_std
        
        # Apply clustering
        if self.kmeans is None:
            self.kmeans = KMeans(n_clusters=self.num_regimes, random_state=42)
            self.kmeans.fit(X_reduced)
        
        # Get regime labels
        regime = self.kmeans.predict(X_reduced)
        
        # Sort regimes by average volatility
        if 'close' in df.columns:
            # Calculate mean VIX by regime
            regime_vix = {}
            for i in range(self.num_regimes):
                mask = (regime == i)
                if np.any(mask):
                    regime_vix[i] = np.mean(vix[mask])
            
            # Create mapping from current regime to volatility-sorted regime
            sorted_regimes = sorted(regime_vix.items(), key=lambda x: x[1])
            regime_map = {old_regime: new_regime for new_regime, (old_regime, _) in enumerate(sorted_regimes)}
            
            # Apply mapping
            new_regime = np.zeros_like(regime)
            for old, new in regime_map.items():
                new_regime[regime == old] = new
                
            regime = new_regime
        
        return regime
    
    def hybrid_regimes(self, df):
        """
        Apply hybrid approach combining rules and clustering
        
        Args:
            df: DataFrame with market features
            
        Returns:
            regime: Array of regime labels
        """
        # Get regimes from both methods
        rules_regime = self.rules_based_regimes(df)
        cluster_regime = self.clustering_regimes(df)
        
        # Create market features for calibration
        # This helps align the clustering with interpretable rules
        if self.regime_thresholds is None and 'close' in df.columns:
            vix = df['close'].values
            
            # Calculate z-scores using rules-based approach
            window = min(self.lookback, len(vix) - 1)
            if window > 0:
                vix_mean = np.zeros_like(vix)
                vix_std = np.zeros_like(vix)
                
                for i in range(window, len(vix)):
                    vix_mean[i] = np.mean(vix[i-window:i])
                    vix_std[i] = np.std(vix[i-window:i])
                
                # Fill initial values
                vix_mean[:window] = vix_mean[window]
                vix_std[:window] = vix_std[window]
                
                # Calculate z-scores
                vix_zscore = (vix - vix_mean) / (vix_std + 1e-8)
                
                # Calculate average z-score for each cluster regime
                regime_zscore = {}
                for i in range(self.num_regimes):
                    mask = (cluster_regime == i)
                    if np.any(mask):
                        regime_zscore[i] = np.mean(vix_zscore[mask])
                
                # Determine threshold values based on regime z-scores
                thresholds = []
                for i in range(1, self.num_regimes):
                    # Find boundary between regimes
                    r1 = i - 1
                    r2 = i
                    if r1 in regime_zscore and r2 in regime_zscore:
                        threshold = (regime_zscore[r1] + regime_zscore[r2]) / 2
                        thresholds.append(threshold)
                
                self.regime_thresholds = sorted(thresholds)
        
        # Combine methods
        # Rules-based takes precedence for extreme regimes (low and crisis)
        # Clustering helps with the middle regimes
        regime = np.copy(cluster_regime)
        
        # Override with rules-based for extreme regimes
        regime[rules_regime == 0] = 0                    # Low volatility
        regime[rules_regime == self.num_regimes - 1] = self.num_regimes - 1  # Crisis
        
        return regime
    
    def detect_regimes(self, df):
        """
        Detect market regimes based on configured method
        
        Args:
            df: DataFrame with market data
            
        Returns:
            regime: Array of regime labels
            regime_probs: Array of regime probabilities (one-hot encoded)
        """
        # Apply selected method
        if self.method == 'rules':
            regime = self.rules_based_regimes(df)
        elif self.method == 'clustering':
            regime = self.clustering_regimes(df)
        elif self.method == 'hybrid':
            regime = self.hybrid_regimes(df)
        else:
            # Default to rules-based
            regime = self.rules_based_regimes(df)
        
        # Create one-hot encoded regime probabilities
        regime_probs = np.zeros((len(regime), self.num_regimes))
        for i in range(self.num_regimes):
            regime_probs[:, i] = (regime == i).astype(float)
        
        return regime, regime_probs
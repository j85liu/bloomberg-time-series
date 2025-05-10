import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MetaFeatureExtractor(nn.Module):
    """
    Enhanced Meta-feature extractor with financial-specific features and task embeddings.
    
    This module takes raw time series data and extracts both statistical and 
    learned features that can predict which forecasting model will perform best.
    """
    
    def __init__(self, input_dim=None, hidden_dim=64, meta_feature_dim=32, num_tasks=100):
        """
        Initialize the enhanced meta-feature extractor.
        
        Args:
            input_dim: Optional dimension of input time series (not used for 1D series)
            hidden_dim: Dimension of hidden layers
            meta_feature_dim: Dimension of output meta-features
            num_tasks: Number of different time series tasks for embedding
        """
        super().__init__()
        
        # Convolutional feature extraction - learns patterns directly from raw time series
        # Architecture: Conv1D → ReLU → MaxPool → Conv1D → ReLU → MaxPool → Conv1D → ReLU → AvgPool
        self.conv_layers = nn.Sequential(
            # First conv layer: 1 input channel (univariate time series) → 16 output channels
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),  # Reduce sequence length by half
            
            # Second conv layer: 16 input channels → 32 output channels
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),  # Further reduce sequence length
            
            # Third conv layer: 32 input channels → 64 output channels
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            
            # Global average pooling to get fixed-size output regardless of input length
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Statistical feature encoder - processes handcrafted statistical features
        self.stat_encoder = nn.Sequential(
            # 6 statistical features: mean, std, skewness, kurtosis, autocorr, trend
            nn.Linear(6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Financial feature extraction and market regime detection
        # 5 financial features: volatility, trend_strength, momentum, mean_reversion, seasonality
        self.financial_encoder = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Market regime detector - classifies current market state
        self.market_regime_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # 4 common market regimes: trending, mean-reverting, high volatility, low volatility
            nn.Linear(hidden_dim, 4)
        )
        
        # Task embedding layer - learns representations for different time series
        self.task_embedding = nn.Embedding(num_embeddings=num_tasks, embedding_dim=hidden_dim)
        
        # Combined feature encoder - merges all feature types
        self.combined_encoder = nn.Sequential(
            # Combine convolutional, statistical, financial, task, and regime features
            nn.Linear(64 + hidden_dim + hidden_dim + hidden_dim + 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),  # Prevent overfitting
            nn.Linear(hidden_dim, meta_feature_dim)
        )
        
    def extract_statistical_features(self, x):
        """
        Extract statistical features from time series.
        
        Args:
            x: Time series data [batch_size, seq_len]
            
        Returns:
            stats: Statistical features [batch_size, 6]
        """
        # Mean of each time series
        mean = torch.mean(x, dim=1, keepdim=True)
        
        # Standard deviation of each time series
        std = torch.std(x, dim=1, keepdim=True)
        
        # Skewness (third standardized moment)
        # Formula: E[(X - μ)³] / σ³
        centered = x - mean
        skewness = torch.mean(centered**3, dim=1, keepdim=True) / (std**3 + 1e-8)  # Add epsilon to avoid division by zero
        
        # Kurtosis (fourth standardized moment, excess kurtosis formula)
        # Formula: E[(X - μ)⁴] / σ⁴ - 3
        # Subtracted 3 to get excess kurtosis (normal distribution has kurtosis=3)
        kurtosis = torch.mean(centered**4, dim=1, keepdim=True) / (std**4 + 1e-8) - 3.0
        
        # First-order autocorrelation
        # Formula: sum((x_t - μ)(x_{t+1} - μ)) / sum((x_t - μ)²)
        autocorr = torch.sum(centered[:, :-1] * centered[:, 1:], dim=1, keepdim=True) / (
            torch.sum(centered**2, dim=1, keepdim=True) + 1e-8)
        
        # Trend (simple linear regression coefficient)
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Create time indices [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, dtype=torch.float32, device=x.device).view(1, -1).expand(batch_size, -1)
        t_mean = torch.mean(t, dim=1, keepdim=True)
        t_std = torch.std(t, dim=1, keepdim=True) + 1e-8
        x_std = std + 1e-8
        
        # Calculate Pearson correlation between time and values
        # Formula: sum((t - t_mean)(x - x_mean)) / (n * σ_t * σ_x)
        trend = torch.sum((t - t_mean) * (x - mean), dim=1, keepdim=True) / (
            seq_len * t_std * x_std)
            
        # Combine all statistical features
        stats = torch.cat([mean, std, skewness, kurtosis, autocorr, trend], dim=1)
        return stats
    
    def extract_financial_features(self, x):
        """
        Extract financial-specific features from time series.
        
        Args:
            x: Time series data [batch_size, seq_len]
            
        Returns:
            financial_features: Financial features [batch_size, 5]
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Calculate returns: r_t = (p_t / p_{t-1}) - 1
        returns = torch.zeros_like(x)
        returns[:, 1:] = (x[:, 1:] / (x[:, :-1] + 1e-8)) - 1.0
        
        # 1. Volatility (standard deviation of returns)
        volatility = torch.std(returns, dim=1, keepdim=True)
        
        # 2. Trend strength (R² of linear fit)
        # Create time indices
        t = torch.arange(seq_len, dtype=torch.float32, device=x.device).view(1, -1).expand(batch_size, -1)
        
        # Calculate means
        t_mean = torch.mean(t, dim=1, keepdim=True)
        x_mean = torch.mean(x, dim=1, keepdim=True)
        
        # Calculate covariance and variances
        cov_tx = torch.sum((t - t_mean) * (x - x_mean), dim=1, keepdim=True)
        var_t = torch.sum((t - t_mean)**2, dim=1, keepdim=True)
        var_x = torch.sum((x - x_mean)**2, dim=1, keepdim=True)
        
        # Calculate correlation coefficient and R²
        corr = cov_tx / (torch.sqrt(var_t * var_x) + 1e-8)
        r_squared = corr**2
        
        # 3. Momentum (return over past window)
        if seq_len >= 5:
            momentum = (x[:, -1:] / (x[:, -5:-4] + 1e-8)) - 1.0
        else:
            momentum = returns[:, -1:]
        
        # 4. Mean reversion (negative of autocorrelation of returns)
        returns_mean = torch.mean(returns, dim=1, keepdim=True)
        returns_centered = returns - returns_mean
        
        # Mean reversion strength (negative autocorrelation of returns)
        mean_reversion = -torch.sum(returns_centered[:, :-1] * returns_centered[:, 1:], dim=1, keepdim=True) / (
            torch.sum(returns_centered**2, dim=1, keepdim=True) + 1e-8)
        
        # 5. Seasonality strength (using autocorrelation at different lags)
        if seq_len >= 12:
            # Calculate autocorrelation at lag 7 (weekly) and lag 12 (monthly)
            ac7 = torch.sum((x[:, :-7] - x_mean) * (x[:, 7:] - x_mean), dim=1, keepdim=True) / (var_x + 1e-8)
            seasonality = torch.abs(ac7)  # Use absolute value as seasonality strength
        else:
            seasonality = torch.zeros(batch_size, 1, device=x.device)
        
        # Combine financial features
        financial_features = torch.cat([
            volatility, r_squared, momentum, mean_reversion, seasonality
        ], dim=1)
        
        return financial_features
    
    def forward(self, x, task_ids=None):
        """
        Extract enhanced meta-features from time series.
        
        Args:
            x: Time series data [batch_size, seq_len]
            task_ids: Optional tensor of task identifiers [batch_size]
            
        Returns:
            meta_features: Meta-features for model selection [batch_size, meta_feature_dim]
            market_regimes: Market regime probabilities [batch_size, num_regimes]
        """
        batch_size = x.size(0)
        
        # Extract convolutional features
        # First reshape to [batch_size, channels=1, seq_len] for Conv1D
        conv_input = x.view(batch_size, 1, -1)
        # Apply convolutional network and reshape output
        conv_features = self.conv_layers(conv_input).view(batch_size, -1)
        
        # Extract statistical features
        stat_features = self.extract_statistical_features(x)
        # Encode statistical features through MLP
        stat_encoded = self.stat_encoder(stat_features)
        
        # Extract financial features
        financial_features = self.extract_financial_features(x)
        # Encode financial features through MLP
        financial_encoded = self.financial_encoder(financial_features)
        
        # Detect market regime
        regime_logits = self.market_regime_detector(financial_encoded)
        market_regime_probs = F.softmax(regime_logits, dim=1)
        
        # Get task embedding if task IDs provided
        if task_ids is not None:
            task_emb = self.task_embedding(task_ids)
        else:
            # Use zero embedding if no task ID provided
            task_emb = torch.zeros(batch_size, self.task_embedding.embedding_dim, device=x.device)
        
        # Combine all features
        combined = torch.cat([
            conv_features,        # Learned patterns from convolutions
            stat_encoded,         # Statistical properties 
            financial_encoded,    # Financial indicators
            task_emb,             # Task-specific knowledge
            market_regime_probs   # Market regime information
        ], dim=1)
        
        # Final meta-feature encoding
        meta_features = self.combined_encoder(combined)
        
        return meta_features, market_regime_probs


class MetaKnowledgeDatabase:
    """
    Database for storing and retrieving meta-knowledge about model performance.
    
    This class serves as a memory of how different models performed on various
    time series with different characteristics, enabling learning from experience.
    """
    def __init__(self, feature_dim, num_models):
        """
        Initialize meta-knowledge database.
        
        Args:
            feature_dim: Dimension of meta-features
            num_models: Number of models being compared
        """
        self.feature_dim = feature_dim
        self.num_models = num_models
        self.meta_features = []  # List to store meta-features
        self.performance_metrics = []  # List to store corresponding performance metrics
        self.task_ids = []  # List to store task identifiers (if available)
        self.market_regimes = []  # List to store detected market regimes
        
    def add_knowledge(self, features, performance, task_ids=None, market_regimes=None):
        """
        Add new meta-knowledge about model performance.
        
        Args:
            features: Characteristics of the time series [batch_size, feature_dim]
            performance: Performance metrics for each model [batch_size, num_models]
            task_ids: Optional task identifiers [batch_size]
            market_regimes: Optional market regime probabilities [batch_size, num_regimes]
        """
        # Detach tensors from computation graph and move to CPU for storage
        self.meta_features.append(features.detach().cpu())
        self.performance_metrics.append(performance.detach().cpu())
        
        # Store task IDs if provided
        if task_ids is not None:
            self.task_ids.append(task_ids.detach().cpu())
            
        # Store market regimes if provided
        if market_regimes is not None:
            self.market_regimes.append(market_regimes.detach().cpu())
    
    def get_dataset(self):
        """
        Get collected meta-knowledge as a dataset.
        
        Returns:
            X: Meta-features [total_samples, feature_dim]
            y: Performance metrics [total_samples, num_models]
            task_ids: Task identifiers or None
            market_regimes: Market regime probabilities or None
        """
        if not self.meta_features:
            return None, None, None, None
            
        # Concatenate all stored features and metrics
        X = torch.cat(self.meta_features, dim=0)
        y = torch.cat(self.performance_metrics, dim=0)
        
        # Concatenate task IDs if available
        task_ids = torch.cat(self.task_ids, dim=0) if self.task_ids else None
        
        # Concatenate market regimes if available
        market_regimes = torch.cat(self.market_regimes, dim=0) if self.market_regimes else None
        
        return X, y, task_ids, market_regimes
        
    def save(self, path):
        """
        Save meta-knowledge database to disk.
        
        Args:
            path: File path to save the database
        """
        X, y, task_ids, market_regimes = self.get_dataset()
        if X is not None:
            save_dict = {'X': X, 'y': y}
            if task_ids is not None:
                save_dict['task_ids'] = task_ids
            if market_regimes is not None:
                save_dict['market_regimes'] = market_regimes
                
            torch.save(save_dict, path)
    
    def load(self, path):
        """
        Load meta-knowledge database from disk.
        
        Args:
            path: File path to load the database from
        """
        data = torch.load(path)
        self.meta_features = [data['X']]
        self.performance_metrics = [data['y']]
        
        if 'task_ids' in data:
            self.task_ids = [data['task_ids']]
            
        if 'market_regimes' in data:
            self.market_regimes = [data['market_regimes']]
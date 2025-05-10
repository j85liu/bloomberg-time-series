import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MetaFeatureExtractor(nn.Module):
    """
    Extract meta-features from time series data for model selection.
    
    This module takes raw time series data and extracts both statistical and 
    learned features that can predict which forecasting model will perform best.
    """
    
    def __init__(self, input_dim=None, hidden_dim=64, meta_feature_dim=32):
        """
        Initialize the meta-feature extractor.
        
        Args:
            input_dim: Optional dimension of input time series (not used for 1D series)
            hidden_dim: Dimension of hidden layers
            meta_feature_dim: Dimension of output meta-features
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
        # Architecture: Linear → ReLU → Linear
        self.stat_encoder = nn.Sequential(
            # 6 statistical features: mean, std, skewness, kurtosis, autocorr, trend
            nn.Linear(6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Combined feature encoder - merges conv and statistical features
        # Architecture: Linear → ReLU → Linear
        self.combined_encoder = nn.Sequential(
            # Combine 64 conv features and hidden_dim statistical features
            nn.Linear(64 + hidden_dim, hidden_dim),
            nn.ReLU(),
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
        
    def forward(self, x):
        """
        Extract meta-features from time series.
        
        Args:
            x: Time series data [batch_size, seq_len]
            
        Returns:
            meta_features: Meta-features for model selection [batch_size, meta_feature_dim]
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
        
        # Combine convolutional and statistical features
        combined = torch.cat([conv_features, stat_encoded], dim=1)
        # Final encoding to meta-features
        meta_features = self.combined_encoder(combined)
        
        return meta_features


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
        
    def add_knowledge(self, features, performance):
        """
        Add new meta-knowledge about model performance.
        
        Args:
            features: Characteristics of the time series [batch_size, feature_dim]
            performance: Performance metrics for each model [batch_size, num_models]
        """
        # Detach tensors from computation graph and move to CPU for storage
        self.meta_features.append(features.detach().cpu())
        self.performance_metrics.append(performance.detach().cpu())
    
    def get_dataset(self):
        """
        Get collected meta-knowledge as a dataset.
        
        Returns:
            X: Meta-features [total_samples, feature_dim]
            y: Performance metrics [total_samples, num_models]
        """
        if not self.meta_features:
            return None, None
            
        # Concatenate all stored features and metrics
        X = torch.cat(self.meta_features, dim=0)
        y = torch.cat(self.performance_metrics, dim=0)
        return X, y
        
    def save(self, path):
        """
        Save meta-knowledge database to disk.
        
        Args:
            path: File path to save the database
        """
        X, y = self.get_dataset()
        if X is not None:
            torch.save({'X': X, 'y': y}, path)
    
    def load(self, path):
        """
        Load meta-knowledge database from disk.
        
        Args:
            path: File path to load the database from
        """
        data = torch.load(path)
        self.meta_features = [data['X']]
        self.performance_metrics = [data['y']]
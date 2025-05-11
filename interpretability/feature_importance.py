# Create a new file: interpretability/feature_importance.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

class MetaFeatureImportance:
    """
    Analyzes and visualizes the importance of meta-features for model selection.
    """
    
    def __init__(self, meta_feature_extractor, meta_learner):
        """
        Initialize the feature importance analyzer.
        
        Args:
            meta_feature_extractor: Trained meta-feature extractor
            meta_learner: Trained meta-learner
        """
        self.feature_extractor = meta_feature_extractor
        self.meta_learner = meta_learner
        
    def compute_gradient_importance(self, time_series_batch):
        """
        Compute feature importance using gradient-based methods.
        
        Args:
            time_series_batch: Batch of time series [batch_size, seq_len]
            
        Returns:
            feature_importance: Importance of each meta-feature
        """
        # Enable gradients
        time_series_batch.requires_grad_(True)
        
        # Extract meta-features
        meta_features, regimes = self.feature_extractor(time_series_batch)
        
        # Make predictions
        output = self.meta_learner(meta_features, market_regimes=regimes)
        weights = output['weights']
        
        # Compute gradients for each model weight
        importance_per_model = []
        for i in range(weights.size(1)):
            # Zero gradients
            if time_series_batch.grad is not None:
                time_series_batch.grad.zero_()
                
            # Compute gradients for model i
            model_weight = weights[:, i].mean()
            model_weight.backward(retain_graph=True)
            
            # Store gradients
            importance = time_series_batch.grad.abs().mean(dim=0)
            importance_per_model.append(importance)
            
        # Combine importance across models
        overall_importance = torch.stack(importance_per_model).mean(dim=0)
        
        return overall_importance
    
    def visualize_importance(self, importance, feature_names=None):
        """
        Visualize feature importance.
        
        Args:
            importance: Feature importance scores
            feature_names: Optional list of feature names
        """
        # Convert to numpy
        if isinstance(importance, torch.Tensor):
            importance = importance.detach().cpu().numpy()
            
        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f"Feature {i+1}" for i in range(len(importance))]
            
        # Sort by importance
        indices = np.argsort(importance)
        sorted_names = [feature_names[i] for i in indices]
        sorted_importance = importance[indices]
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(sorted_names, sorted_importance)
        plt.xlabel("Importance")
        plt.title("Meta-Feature Importance for Model Selection")
        plt.tight_layout()
        plt.show()
        
    def analyze_regime_dependence(self, time_series_dataset, regimes=['Trending', 'Mean-Reverting', 'High Volatility', 'Low Volatility']):
        """
        Analyze how model selection depends on market regimes.
        
        Args:
            time_series_dataset: Dataset of time series samples
            regimes: Names of market regimes
            
        Returns:
            regime_analysis: Dictionary with model weights per regime
        """
        regime_weights = {regime: [] for regime in regimes}
        
        with torch.no_grad():
            for time_series in time_series_dataset:
                # Extract meta-features and regime probabilities
                meta_features, regime_probs = self.feature_extractor(time_series.unsqueeze(0))
                
                # For each regime, get model weights
                for i, regime in enumerate(regimes):
                    # Create artificial regime vector with 100% probability for this regime
                    regime_vector = torch.zeros_like(regime_probs)
                    regime_vector[:, i] = 1.0
                    
                    # Get model weights for this regime
                    output = self.meta_learner(meta_features, market_regimes=regime_vector)
                    weights = output['weights'].squeeze(0).cpu().numpy()
                    
                    regime_weights[regime].append(weights)
        
        # Convert lists to arrays
        for regime in regimes:
            regime_weights[regime] = np.array(regime_weights[regime])
            
        return regime_weights
        
    def visualize_regime_analysis(self, regime_analysis, model_names):
        """
        Visualize how model selection depends on market regimes.
        
        Args:
            regime_analysis: Output from analyze_regime_dependence
            model_names: Names of the base models
        """
        regimes = list(regime_analysis.keys())
        num_regimes = len(regimes)
        num_models = len(model_names)
        
        # Compute average weights per regime
        avg_weights = np.zeros((num_regimes, num_models))
        for i, regime in enumerate(regimes):
            avg_weights[i] = np.mean(regime_analysis[regime], axis=0)
            
        # Set up plot
        fig, ax = plt.subplots(figsize=(12, 8))
        bar_width = 0.8 / num_models
        
        # Plot bars for each model
        for i in range(num_models):
            x = np.arange(num_regimes) + i * bar_width - (num_models - 1) * bar_width / 2
            ax.bar(x, avg_weights[:, i], width=bar_width, label=model_names[i])
            
        # Add labels and legend
        ax.set_xticks(np.arange(num_regimes))
        ax.set_xticklabels(regimes)
        ax.set_ylabel('Average Model Weight')
        ax.set_title('Model Selection Depends on Market Regime')
        ax.legend()
        
        plt.tight_layout()
        plt.show()
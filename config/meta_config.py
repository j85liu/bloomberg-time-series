# Create a new file: config/meta_config.py

"""
Configuration file for meta-learning framework.
"""

# Meta-feature extractor configuration
META_FEATURE_CONFIG = {
    'hidden_dim': 64,
    'meta_feature_dim': 32,
    'num_tasks': 100,
    'use_convolutions': True,
    'use_financial_features': True,
    'use_regime_detection': True
}

# Meta-learner configuration
META_LEARNER_CONFIG = {
    'hidden_dim': 64,
    'use_regime_info': True,
    'use_financial_metrics': True,
    'use_model_features': True
}

# Training configuration
TRAINING_CONFIG = {
    'meta_epochs': 50,
    'end_to_end_epochs': 100,
    'meta_lr': 0.001,
    'framework_lr': 0.0005,
    'batch_size': 32,
    'early_stopping_patience': 10
}

# Base model weights initialization
# Options: 'random', 'equal', 'performance_based'
MODEL_WEIGHT_INIT = 'equal'

# Model selection strategy
# Options: 'soft_selection', 'hard_selection', 'dynamic'
MODEL_SELECTION_STRATEGY = 'soft_selection'

# Financial performance metrics to track
FINANCIAL_METRICS = ['mse', 'mae', 'mape', 'sharpe_ratio', 'max_drawdown']

# Meta-feature groups for ablation studies
META_FEATURE_GROUPS = {
    'statistical': ['mean', 'std', 'skewness', 'kurtosis', 'autocorrelation'],
    'financial': ['volatility', 'momentum', 'mean_reversion', 'drawdown'],
    'regime': ['regime_probabilities'],
    'learned': ['convolutional_features'],
    'all': 'all'
}
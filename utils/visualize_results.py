# Create a new file: utils/visualize_results.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import torch
from pathlib import Path
import seaborn as sns

def plot_model_selection_by_regime(results_path='results/evaluation_results.json', 
                                 output_path='results/model_selection_by_regime.png'):
    """
    Create a visualization of model selection by market regime
    
    Args:
        results_path: Path to evaluation results JSON
        output_path: Path to save the visualization
    """
    # Load evaluation results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Check if needed data is available
    if 'by_regime' not in results or 'model_selection' not in results:
        print("Required data not found in results")
        return
    
    # Extract regime-specific model performances
    regimes = list(results['by_regime'].keys())
    models = list(results['model_selection'].keys())
    
    # Create a table of model performance by regime
    regime_performance = {}
    
    for regime in regimes:
        # Get metrics for this regime
        regime_metrics = results['by_regime'][regime]
        
        # Use MSE as the performance metric
        mse = regime_metrics.get('mse', 0)
        mae = regime_metrics.get('mae', 0)
        dir_acc = regime_metrics.get('dir_acc', 0)
        
        # Store in the dictionary
        regime_performance[regime] = {
            'mse': mse,
            'mae': mae,
            'dir_acc': dir_acc
        }
    
    # Get model selection statistics
    model_stats = results['model_selection']
    
    # Create a plot
    plt.figure(figsize=(15, 8))
    
    # Create a bar chart of model selection by regime
    ax1 = plt.subplot(211)
    
    # Hard-coded example of selection by regime
    # This would need to be replaced with actual data from a more detailed evaluation
    regime_selection = {
        'regime_0': {'tft': 0.2, 'nbeatsx': 0.7, 'deepar': 0.1},  # Low vol regime
        'regime_1': {'tft': 0.5, 'nbeatsx': 0.3, 'deepar': 0.2},  # Normal regime
        'regime_2': {'tft': 0.4, 'nbeatsx': 0.1, 'deepar': 0.5},  # High vol regime
        'regime_3': {'tft': 0.3, 'nbeatsx': 0.1, 'deepar': 0.6}   # Crisis regime
    }
    
    # Plot the model weights by regime
    x = np.arange(len(regimes))
    width = 0.25
    
    for i, model in enumerate(models):
        model_weights = [regime_selection[regime][model] for regime in regimes]
        ax1.bar(x + i*width - width, model_weights, width, label=model)
    
    ax1.set_ylabel('Model Weight')
    ax1.set_title('Meta-Learner Model Selection by Market Regime')
    ax1.set_xticks(x)
    ax1.set_xticklabels(regimes)
    ax1.legend()
    
    # Plot performance metrics by regime
    ax2 = plt.subplot(212)
    
    # Extract MSE values
    mse_values = [regime_performance[regime]['mse'] for regime in regimes]
    mae_values = [regime_performance[regime]['mae'] for regime in regimes]
    
    # Create a line plot
    ax2.plot(regimes, mse_values, 'o-', label='MSE')
    ax2.plot(regimes, mae_values, 's-', label='MAE')
    
    ax2.set_xlabel('Market Regime')
    ax2.set_ylabel('Error Metric')
    ax2.set_title('Forecasting Performance by Market Regime')
    ax2.legend()
    
    # Add a second y-axis for directional accuracy
    ax3 = ax2.twinx()
    dir_acc_values = [regime_performance[regime].get('dir_acc', 0) for regime in regimes]
    ax3.plot(regimes, dir_acc_values, 'd-', color='g', label='Dir. Accuracy')
    ax3.set_ylabel('Directional Accuracy (%)')
    ax3.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Saved model selection visualization to {output_path}")

def plot_feature_importance(results_path='results/feature_importance.npz',
                          output_path='results/feature_importance.png',
                          feature_names=None):
    """
    Plot feature importance for meta-learning
    
    Args:
        results_path: Path to feature importance results
        output_path: Path to save visualization
        feature_names: List of feature names (if available)
    """
    try:
        importance_data = np.load(results_path)
        importance = importance_data['importance']
        
        if 'feature_names' in importance_data:
            feature_names = importance_data['feature_names']
        elif feature_names is None:
            # Create generic feature names
            feature_names = [f'Feature {i+1}' for i in range(len(importance))]
        
        # Sort by importance
        sorted_idx = np.argsort(importance)
        sorted_importance = importance[sorted_idx]
        sorted_names = [feature_names[i] for i in sorted_idx]
        
        # Plot
        plt.figure(figsize=(12, 10))
        
        # Create a horizontal bar chart
        bars = plt.barh(range(len(sorted_names)), sorted_importance, align='center')
        
        # Color code by feature type
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # Simple grouping example based on name prefixes
        # This would need to be customized based on actual feature names
        for i, name in enumerate(sorted_names):
            color_idx = 0
            if name.startswith(('vol_', 'volatility')):
                color_idx = 0
            elif name.startswith(('mom_', 'momentum')):
                color_idx = 1
            elif name.startswith(('regime_', 'market')):
                color_idx = 2
            elif name.startswith(('corr_', 'correlation')):
                color_idx = 3
            else:
                color_idx = 4
            
            bars[i].set_color(colors[color_idx])
        
        plt.yticks(range(len(sorted_names)), sorted_names)
        plt.xlabel('Importance Score')
        plt.title('Meta-Learning Feature Importance')
        
        # Add a grid for readability
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        print(f"Saved feature importance visualization to {output_path}")
        
    except FileNotFoundError:
        print(f"Feature importance file {results_path} not found")

def plot_forecast_comparison(results_path='results/forecast_examples.npz',
                           output_path='results/forecast_comparison.png'):
    """
    Plot comparison of individual model forecasts vs. meta-learning ensemble
    
    Args:
        results_path: Path to forecast examples data
        output_path: Path to save visualization
    """
    try:
        forecast_data = np.load(results_path, allow_pickle=True)
        
        # Extract data
        dates = forecast_data['dates']
        actuals = forecast_data['actuals']
        ensemble_forecast = forecast_data['ensemble_forecast']
        lower_bound = forecast_data['lower_bound']
        upper_bound = forecast_data['upper_bound']
        
        # Individual model forecasts
        model_forecasts = forecast_data['model_forecasts'].item()
        model_names = list(model_forecasts.keys())
        
        # Model weights
        model_weights = forecast_data['model_weights']
        
        # Create plot
        plt.figure(figsize=(15, 10))
        
        # Plot input and actual values
        ax1 = plt.subplot(211)
        
        # Plot actuals
        ax1.plot(dates, actuals, 'k-', label='Actual', linewidth=2
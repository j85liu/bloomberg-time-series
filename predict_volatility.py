# Create a new file: predict_volatility.py

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse

from models.tft_v3 import TemporalFusionTransformer
from models.nbeatsx_v7 import NBEATSx
from models.deepar_v3 import DeepARModel
from models.meta_learning.enhanced_framework import EnhancedMetaLearningFramework
from utils.regime_detector import MarketRegimeDetector

def load_models(model_path='models/volatility_meta_framework.pt', device='cpu'):
    """
    Load trained models and framework
    
    Args:
        model_path: Path to saved meta-learning framework
        device: PyTorch device
        
    Returns:
        framework: Loaded meta-learning framework
    """
    # Check if model path exists
    if not Path(model_path).exists():
        print(f"Model file not found: {model_path}")
        return None
    
    try:
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Load configuration
        config = checkpoint.get('config', {})
        
        # Initialize base models
        tft_model = TemporalFusionTransformer(
            num_static_vars=1,  # Task ID
            num_future_vars=0,  # Placeholder
            num_past_vars=1,    # Input series
            static_input_sizes=[1],  # Task ID
            encoder_input_sizes=[1],  # Input series
            decoder_input_sizes=[],   # Placeholder
            hidden_dim=config.get('tft_hidden_dim', 64),
            forecast_horizon=21,
            backcast_length=60,
            output_dim=1,
            quantiles=[0.1, 0.5, 0.9]
        ).to(device)
        
        # Load TFT weights if available
        if Path('models/tft_best.pt').exists():
            tft_model.load_state_dict(torch.load('models/tft_best.pt', map_location=device))
        
        # Initialize NBEATSx model
        nbeatsx_model = NBEATSx(
            input_size=60,   # 60 days of history
            forecast_size=21,  # 21 days ahead
            exog_channels=14,  # Example number of features
            stack_types=['trend', 'seasonality', 'generic'],
            num_blocks_per_stack=[3, 3, 1],
            hidden_units=config.get('nbeatsx_hidden_dim', 128),
            layers=config.get('nbeatsx_layers', 4),
            basis_kwargs={
                'degree': 3,
                'harmonics': 5
            },
            dropout=0.1,
            exog_mode='tcn'
        ).to(device)
        
        # Load NBEATSx weights if available
        if Path('models/nbeatsx_best.pt').exists():
            nbeatsx_model.load_state_dict(torch.load('models/nbeatsx_best.pt', map_location=device))
        
        # Initialize DeepAR model
        deepar_model = DeepARModel(
            num_time_features=13,  # Example number of time features
            num_static_features=1,  # Task ID
            embedding_dim=config.get('deepar_embedding_dim', 32),
            hidden_size=config.get('deepar_hidden_size', 64),
            num_layers=config.get('deepar_num_layers', 2),
            dropout=0.1,
            likelihood='gaussian',
            seq_len=60,
            prediction_len=21
        ).to(device)
        
        # Load DeepAR weights if available
        if Path('models/deepar_best.pt').exists():
            deepar_model.load_state_dict(torch.load('models/deepar_best.pt', map_location=device))
        
        # Combine base models
        base_models = {
            'tft': tft_model,
            'nbeatsx': nbeatsx_model,
            'deepar': deepar_model
        }
        
        # Initialize framework
        framework = EnhancedMetaLearningFramework(
            base_models=base_models,
            meta_feature_dim=config.get('meta_feature_dim', 32),
            hidden_dim=config.get('meta_hidden_dim', 64),
            num_regimes=config.get('num_regimes', 4),
            regime_method=config.get('regime_method', 'hybrid'),
            use_model_features=config.get('use_model_features', True)
        )
        
        # Load framework weights
        framework.meta_feature_extractor.load_state_dict(checkpoint['meta_feature_extractor'])
        framework.meta_learner.load_state_dict(checkpoint['meta_learner'])
        framework.temperature.data = checkpoint['temperature']
        
        # Set to evaluation mode
        framework.meta_feature_extractor.eval()
        framework.meta_learner.eval()
        
        return framework
    
    except Exception as e:
        print(f"Error loading models: {e}")
        return None

def predict_volatility(framework, input_data, task_id=0, vix_data=None):
    """
    Make a volatility forecast using the meta-learning framework
    
    Args:
        framework: Trained meta-learning framework
        input_data: Input time series data
        task_id: Task ID (volatility index ID)
        vix_data: Optional VIX data for regime detection
        
    Returns:
        results: Dictionary with forecast and details
    """
    # Ensure input is a torch tensor
    if isinstance(input_data, np.ndarray):
        input_data = torch.tensor(input_data, dtype=torch.float32)
    elif isinstance(input_data, pd.DataFrame) or isinstance(input_data, pd.Series):
        input_data = torch.tensor(input_data.values, dtype=torch.float32)
    
    # Add batch dimension if needed
    if len(input_data.shape) == 1:
        input_data = input_data.unsqueeze(0)
    
    # Add feature dimension if needed (for NBEATSx)
    if len(input_data.shape) == 2:
        # Check if second dimension is features or time
        if input_data.shape[1] < 10:  # Assume it's features if less than 10
            # Already has feature dimension
            pass
        else:
            # Add feature dimension
            input_data = input_data.unsqueeze(-1)
    
    # Create task ID tensor
    task_id_tensor = torch.tensor([task_id], dtype=torch.long)
    
    # Make prediction
    with torch.no_grad():
        results = framework.forecast(
            time_series=input_data,
            task_ids=task_id_tensor,
            vix_data=vix_data,
            return_details=True
        )
    
    return results

def plot_forecast(input_series, forecast_results, dates=None, title='Volatility Forecast',
                output_path='forecast.png'):
    """
    Plot the forecast results
    
    Args:
        input_series: Input time series
        forecast_results: Results from predict_volatility
        dates: Optional dates for x-axis
        title: Plot title
        output_path: Path to save the plot
    """
    # Extract data
    forecast = forecast_results['forecast'].cpu().numpy()
    lower_bound = forecast_results['lower_bound'].cpu().numpy()
    upper_bound = forecast_results['upper_bound'].cpu().numpy()
    
    # Extract individual model forecasts if available
    individual_forecasts = {}
    if 'individual_forecasts' in forecast_results:
        for model_name, model_result in forecast_results['individual_forecasts'].items():
            individual_forecasts[model_name] = model_result['mean'].cpu().numpy()
    
    # Extract model weights
    model_weights = forecast_results['model_weights'].cpu().numpy()[0]
    
    # Convert input to numpy
    if isinstance(input_series, torch.Tensor):
        input_series = input_series.cpu().numpy()
    
    # Handle dimensions
    if len(input_series.shape) == 3:
        # Take first feature if multiple features
        input_series = input_series[0, :, 0]
    elif len(input_series.shape) == 2:
        if input_series.shape[0] == 1:
            input_series = input_series[0]
    
    # Create dates if not provided
    if dates is None:
        total_len = len(input_series) + len(forecast)
        dates = pd.date_range(end=pd.Timestamp.now(), periods=total_len, freq='D')
    
    # Create the plot
    plt.figure(figsize=(15, 8))
    
    # Plot the input series
    input_dates = dates[:len(input_series)]
    plt.plot(input_dates, input_series, 'k-', label='Historical Values', linewidth=2)
    
    # Plot the forecast
    forecast_dates = dates[-len(forecast):]
    plt.plot(forecast_dates, forecast.squeeze(), 'r-', label='Ensemble Forecast', linewidth=2)
    
    # Plot the confidence interval
    plt.fill_between(forecast_dates, lower_bound.squeeze(), upper_bound.squeeze(), 
                     color='r', alpha=0.2, label='Confidence Interval')
    
    # Plot individual model forecasts
    linestyles = ['--', '-.', ':']
    colors = ['blue', 'green', 'purple']
    
    for i, (model_name, model_forecast) in enumerate(individual_forecasts.items()):
        plt.plot(forecast_dates, model_forecast.squeeze(), linestyles[i % len(linestyles)],
                color=colors[i % len(colors)],
                label=f'{model_name} (w={model_weights[i]:.2f})', alpha=0.7)
    
    # Add a vertical line separating historical data from forecast
    plt.axvline(x=input_dates[-1], color='gray', linestyle='--', alpha=0.7)
    plt.text(input_dates[-1], plt.ylim()[1]*0.95, 'Forecast Start', 
             horizontalalignment='center', verticalalignment='top')
    
    # Customize the plot
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Add annotations
    # 1. Market regime
    if 'current_regime' in forecast_results:
        regime_names = ['Low Volatility', 'Normal', 'High Volatility', 'Crisis']
        regime = forecast_results['current_regime']
        plt.text(0.02, 0.02, f"Current Regime: {regime_names[regime]}", 
                transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    # 2. Model weights
    if model_weights is not None:
        weight_text = "Model Weights:\n"
        for i, (model_name, weight) in enumerate(zip(individual_forecasts.keys(), model_weights)):
            weight_text += f"{model_name}: {weight:.2f}\n"
        
        plt.text(0.02, 0.85, weight_text, transform=plt.gca().transAxes, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path)
    
    print(f"Forecast visualization saved to {output_path}")
    
    # Show the plot if in interactive mode
    plt.show()

def main():
    """Main function for volatility prediction"""
    parser = argparse.ArgumentParser(description='Predict volatility using meta-learning framework')
    parser.add_argument('--input_file', type=str, default='data/vix_processed.csv',
                      help='Path to input time series data')
    parser.add_argument('--task', type=int, default=0, help='Task ID (index of volatility series)')
    parser.add_argument('--output', type=str, default='forecast.png', help='Output image path')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input_file).exists():
        print(f"Input file not found: {args.input_file}")
        return
    
    # Load input data
    try:
        data = pd.read_csv(args.input_file, index_col=0, parse_dates=True)
        print(f"Loaded data from {args.input_file} with {len(data)} rows")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Load models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    framework = load_models(device=device)
    
    if framework is None:
        print("Failed to load models")
        return
    
    # Extract input features
    # Simplified version - actual feature extraction would be more complex
    if 'close' in data.columns:
        # Use last 60 days of VIX for input
        input_series = data['close'].values[-60:]
        
        # Create a simple feature matrix
        # In a real application, this would be more sophisticated
        features = np.zeros((1, 60, 14))  # Batch size 1, 60 timesteps, 14 features
        
        # Use VIX as first feature
        features[0, :, 0] = input_series
        
        # Extract other features if available
        feature_names = [
            'vol_5d', 'vol_10d', 'vol_63d', 
            'price_to_ma20', 'price_to_ma50',
            'momentum_5d', 'momentum_10d', 'momentum_21d',
            'day_of_week', 'month', 'quarter', 'year'
        ]
        
        for i, feature in enumerate(feature_names):
            if feature in data.columns:
                features[0, :, i+1] = data[feature].values[-60:]
        
        # Convert to torch tensor
        input_tensor = torch.tensor(features, dtype=torch.float32)
        
        # Make prediction
        forecast_results = predict_volatility(
            framework=framework,
            input_data=input_tensor,
            task_id=args.task,
            vix_data=data  # Use full data for regime detection
        )
        
        # Get dates for plotting
        dates = data.index[-60:].tolist()
        forecast_dates = pd.date_range(
            start=dates[-1] + pd.Timedelta(days=1), 
            periods=21, 
            freq='D'
        ).tolist()
        all_dates = dates + forecast_dates
        
        # Plot the forecast
        plot_forecast(
            input_series=input_series,
            forecast_results=forecast_results,
            dates=all_dates,
            title='Volatility Forecast (Meta-Learning Ensemble)',
            output_path=args.output
        )
        
        # Print summary
        print("\nForecast Summary:")
        
        # Get regime info
        if 'current_regime' in forecast_results:
            regime_names = ['Low Volatility', 'Normal', 'High Volatility', 'Crisis']
            regime = forecast_results['current_regime']
            print(f"Market Regime: {regime_names[regime]}")
        
        # Print model weights
        print("\nModel Weights:")
        for i, model_name in enumerate(forecast_results['individual_forecasts'].keys()):
            print(f"  {model_name}: {forecast_results['model_weights'][0, i].item():.4f}")
        
        # Print forecast statistics
        forecast = forecast_results['forecast'].cpu().numpy()
        print("\nForecast Statistics:")
        print(f"  Mean: {forecast.mean():.4f}")
        print(f"  Min: {forecast.min():.4f}")
        print(f"  Max: {forecast.max():.4f}")
        print(f"  Last Value: {input_series[-1]:.4f}")
        print(f"  Predicted Change: {(forecast[0, 0] - input_series[-1]) / input_series[-1] * 100:.2f}%")
    else:
        print("'close' column not found in data")

if __name__ == "__main__":
    main()
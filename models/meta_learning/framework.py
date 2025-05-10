import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from torch.utils.data import TensorDataset, DataLoader

from .meta_features import MetaFeatureExtractor, MetaKnowledgeDatabase
from .meta_learner import MetaLearner, train_meta_learner

class FinancialMetaLearningFramework:
    """
    Complete meta-learning framework for financial time series forecasting.
    
    This framework:
    1. Extracts meta-features from time series
    2. Collects knowledge about model performance
    3. Learns to predict which model performs best for specific time series
    4. Generates forecasts using model selection or ensemble
    """
    def __init__(self, base_models, meta_feature_dim=32, hidden_dim=64):
        """
        Initialize the meta-learning framework.
        
        Args:
            base_models: Dictionary of base forecasting models (TFT, NBEATSx, DeepAR)
            meta_feature_dim: Dimension of meta-features
            hidden_dim: Hidden dimension for neural networks
        """
        # Store base models dictionary
        self.base_models = base_models  
        self.num_models = len(base_models)
        
        # Initialize meta-feature extractor
        self.meta_feature_extractor = MetaFeatureExtractor(
            input_dim=None,  # Will be determined by data
            hidden_dim=hidden_dim,
            meta_feature_dim=meta_feature_dim
        )
        
        # Initialize meta-learner for model performance prediction
        self.meta_learner = MetaLearner(
            meta_feature_dim=meta_feature_dim,
            hidden_dim=hidden_dim,
            num_models=self.num_models
        )
        
        # Initialize meta-knowledge database
        self.meta_database = MetaKnowledgeDatabase(meta_feature_dim, self.num_models)
        
        # Learnable temperature parameter for softmax (controls sharpness of model selection)
        # Higher temperature = softer distribution, lower temperature = harder selection
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        
    def collect_meta_knowledge(self, time_series, targets, time_features=None, static_features=None, future_time_features=None):
        """
        Collect meta-knowledge by evaluating performance of base models on data.
        
        Args:
            time_series: Input time series [batch_size, seq_len]
            targets: Target values [batch_size, forecast_horizon]
            time_features: Time features [batch_size, seq_len, feat_dim] or None
            static_features: Static features [batch_size, feat_dim] or None
            future_time_features: Future time features [batch_size, forecast_horizon, feat_dim] or None
        """
        with torch.no_grad():  # No gradients needed for evaluation
            # Extract meta-features from input time series
            meta_features = self.meta_feature_extractor(time_series)
            
            # Evaluate each model's performance on this data
            model_performances = []
            
            # Process each model
            for name, model in self.base_models.items():
                # Handle different input formats for different models
                if name == 'tft':
                    # Format inputs for TFT model
                    static_inputs = [static_features] if static_features is not None else None
                    encoder_inputs = [time_series.unsqueeze(-1)]  # Add feature dimension
                    # Add time features if available
                    encoder_inputs += [time_features] if time_features is not None else []
                    # Add future time features if available
                    decoder_inputs = [future_time_features] if future_time_features is not None else []
                    
                    # Get TFT output and attention weights
                    output, _ = model(static_inputs, encoder_inputs, decoder_inputs, return_attention=True)
                    # Extract forecast (median quantile)
                    forecasts = output[:, :, 0, 1]  # [batch, horizon, output_dim=0, quantile=1 (0.5)]
                    
                elif name == 'nbeatsx':
                    # Format inputs for NBEATSx model
                    exog = None
                    # Combine time features if available
                    if time_features is not None and future_time_features is not None:
                        exog = torch.cat([time_features, future_time_features], dim=1)
                    
                    # Get NBEATSx forecast and components
                    forecasts, _ = model(time_series, exog, return_decomposition=True)
                    
                elif name == 'deepar':
                    # Format inputs for DeepAR model
                    output = model(
                        time_series,
                        time_features=time_features,
                        static_features=static_features,
                        future_time_features=future_time_features,
                        training=False
                    )
                    # Extract mean forecast
                    forecasts = output['mean'].squeeze(-1)
                
                # Calculate performance (negative MSE so higher is better)
                # Use element-wise MSE and mean across forecast horizon
                mse_loss = -F.mse_loss(forecasts, targets, reduction='none').mean(dim=1, keepdim=True)
                model_performances.append(mse_loss)
            
            # Combine model performances into a single tensor [batch_size, num_models]
            model_performances = torch.cat(model_performances, dim=1)
            
            # Add to meta-knowledge database
            self.meta_database.add_knowledge(meta_features, model_performances)
            
    def meta_train(self, epochs=50, lr=0.001):
        """
        Train meta-learner on collected meta-knowledge.
        
        Args:
            epochs: Number of training epochs
            lr: Learning rate
            
        Returns:
            bool: Whether training was successful
        """
        return train_meta_learner(
            self.meta_feature_extractor, 
            self.meta_learner, 
            self.meta_database,
            epochs=epochs,
            lr=lr
        )
    
    def forecast(self, time_series, time_features=None, static_features=None, future_time_features=None):
        """
        Generate forecast using meta-learning framework.
        
        Args:
            time_series: Input time series [batch_size, seq_len]
            time_features: Time features [batch_size, seq_len, feat_dim] or None
            static_features: Static features [batch_size, feat_dim] or None
            future_time_features: Future time features [batch_size, forecast_horizon, feat_dim] or None
            
        Returns:
            Dictionary containing:
                - forecast: Ensemble forecast
                - model_weights: Model weights
                - individual_forecasts: Individual model forecasts
                - meta_features: Meta-features used for selection
        """
        with torch.no_grad():  # No gradients for inference
            # Extract meta-features from input time series
            meta_features = self.meta_feature_extractor(time_series)
            
            # Predict model performance using meta-learner
            predicted_performance = self.meta_learner(meta_features)
            
            # Convert to model weights using temperature-scaled softmax
            # Temperature controls how "sharp" the selection is
            model_weights = F.softmax(predicted_performance / self.temperature, dim=1)
            
            # Get forecasts from each model
            individual_forecasts = {}
            model_outputs = []
            
            # Process each model
            for i, (name, model) in enumerate(self.base_models.items()):
                # Handle different input formats for different models
                if name == 'tft':
                    # Format inputs for TFT
                    static_inputs = [static_features] if static_features is not None else None
                    encoder_inputs = [time_series.unsqueeze(-1)]
                    encoder_inputs += [time_features] if time_features is not None else []
                    decoder_inputs = [future_time_features] if future_time_features is not None else []
                    
                    # Get forecast and attention for explainability
                    output, attention = model(static_inputs, encoder_inputs, decoder_inputs, return_attention=True)
                    forecasts = output[:, :, 0, 1]  # [batch, horizon, output_dim=0, quantile=1 (0.5)]
                    
                    # Store both forecast and attention for explainability
                    individual_forecasts[name] = {
                        'mean': forecasts,
                        'attention': attention
                    }
                    
                elif name == 'nbeatsx':
                    # Format inputs for NBEATSx
                    exog = None
                    if time_features is not None and future_time_features is not None:
                        exog = torch.cat([time_features, future_time_features], dim=1)
                    
                    # Get forecast and components for explainability
                    forecast, components = model(time_series, exog, return_decomposition=True)
                    individual_forecasts[name] = {
                        'mean': forecast,
                        'components': components
                    }
                    
                elif name == 'deepar':
                    # Format inputs for DeepAR
                    output = model(
                        time_series,
                        time_features=time_features,
                        static_features=static_features,
                        future_time_features=future_time_features,
                        training=False
                    )
                    # Store mean and scale for probabilistic information
                    individual_forecasts[name] = {
                        'mean': output['mean'].squeeze(-1),
                        'scale': output['scale'].squeeze(-1)
                    }
                
                # Collect model outputs for ensemble
                model_outputs.append(individual_forecasts[name]['mean'].unsqueeze(1))
            
            # Stack model outputs [batch_size, num_models, horizon]
            stacked_outputs = torch.cat(model_outputs, dim=1)
            
            # Compute weighted ensemble
            # Expand weights to match forecast dimensions
            weights_expanded = model_weights.unsqueeze(-1)  # [batch_size, num_models, 1]
            # Weighted sum across models
            ensemble_forecast = torch.sum(stacked_outputs * weights_expanded, dim=1)  # [batch_size, horizon]
            
            # Return comprehensive results
            return {
                'forecast': ensemble_forecast,              # Final ensemble forecast
                'model_weights': model_weights,             # Model selection weights
                'individual_forecasts': individual_forecasts, # Individual model forecasts (for analysis)
                'meta_features': meta_features              # Meta-features (for explainability)
            }


def meta_learning_pipeline(data_processor, train_data, val_data, test_data, base_models, 
                          num_epochs=50, meta_epochs=20, lr=0.001, meta_lr=0.0005):
    """
    Complete meta-learning pipeline for financial forecasting.
    
    Args:
        data_processor: Data processor for scaling/inverse scaling
        train_data: Training data batches
        val_data: Validation data batches
        test_data: Test data batches
        base_models: Dictionary of base forecasting models
        num_epochs: Number of end-to-end training epochs
        meta_epochs: Number of meta-learner training epochs
        lr: Learning rate for end-to-end training
        meta_lr: Learning rate for meta-learner
        
    Returns:
        framework: Trained meta-learning framework
        test_results: Test evaluation results
    """
    # Initialize meta-learning framework
    framework = FinancialMetaLearningFramework(
        base_models=base_models,
        meta_feature_dim=32,
        hidden_dim=64
    )
    
    # Phase 1: Collect meta-knowledge from training data
    print("Phase 1: Collecting meta-knowledge...")
    for batch in train_data:
        framework.collect_meta_knowledge(
            time_series=batch['input'],
            targets=batch['target'],
            time_features=batch.get('time_features'),
            static_features=batch.get('static_features'),
            future_time_features=batch.get('future_time_features')
        )
    
    # Phase 2: Train meta-learner on collected meta-knowledge
    print("Phase 2: Training meta-learner...")
    meta_trained = framework.meta_train(epochs=meta_epochs, lr=meta_lr)
    
    if not meta_trained:
        print("Error: Failed to train meta-learner. Check your meta-knowledge database.")
        return None
    
    # Phase 3: End-to-end training of entire framework
    print("Phase 3: End-to-end training of the whole framework...")
    
    # Setup optimizer for framework components (excluding base models which stay fixed)
    framework_params = list(framework.meta_feature_extractor.parameters()) + \
                      list(framework.meta_learner.parameters()) + \
                      [framework.temperature]
    
    optimizer = optim.Adam(framework_params, lr=lr)
    criterion = nn.MSELoss()
    
    # Setup for early stopping
    best_val_loss = float('inf')
    best_state = None
    patience = 10
    patience_counter = 0
    
    # Main training loop
    for epoch in range(num_epochs):
        # Training mode
        framework.meta_feature_extractor.train()
        framework.meta_learner.train()
        
        total_loss = 0
        
        # Process each training batch
        for batch in train_data:
            # Zero gradients
            optimizer.zero_grad()
            
            # Generate forecast
            output = framework.forecast(
                time_series=batch['input'],
                time_features=batch.get('time_features'),
                static_features=batch.get('static_features'),
                future_time_features=batch.get('future_time_features')
            )
            
            # Compute loss between forecast and target
            loss = criterion(output['forecast'], batch['target'])
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Calculate average training loss
        avg_train_loss = total_loss / len(train_data)
        
        # Validation phase
        framework.meta_feature_extractor.eval()
        framework.meta_learner.eval()
        
        val_loss = 0
        with torch.no_grad():
            for batch in val_data:
                # Generate forecast on validation data
                output = framework.forecast(
                    time_series=batch['input'],
                    time_features=batch.get('time_features'),
                    static_features=batch.get('static_features'),
                    future_time_features=batch.get('future_time_features')
                )
                
                # Compute validation loss
                loss = criterion(output['forecast'], batch['target'])
                val_loss += loss.item()
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_data)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Early stopping logic
        if avg_val_loss < best_val_loss:
            # Save best model state
            best_val_loss = avg_val_loss
            best_state = {
                'meta_feature_extractor': framework.meta_feature_extractor.state_dict(),
                'meta_learner': framework.meta_learner.state_dict(),
                'temperature': framework.temperature.item()
            }
            patience_counter = 0
        else:
            # Increment patience counter if no improvement
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model state
    if best_state:
        framework.meta_feature_extractor.load_state_dict(best_state['meta_feature_extractor'])
        framework.meta_learner.load_state_dict(best_state['meta_learner'])
        framework.temperature.data.fill_(best_state['temperature'])
    
    # Phase 4: Evaluation on test data
    print("Phase 4: Evaluating on test data...")
    mse_scores = []
    mae_scores = []
    
    # Track model selection statistics
    model_selection = {name: [] for name in base_models.keys()}
    
    # Evaluation mode
    framework.meta_feature_extractor.eval()
    framework.meta_learner.eval()
    
    with torch.no_grad():
        for batch in test_data:
            # Generate forecast on test data
            output = framework.forecast(
                time_series=batch['input'],
                time_features=batch.get('time_features'),
                static_features=batch.get('static_features'),
                future_time_features=batch.get('future_time_features')
            )
            
            # Convert to numpy for evaluation
            forecast = output['forecast'].cpu().numpy()
            target = batch['target'].cpu().numpy()
            
            # Inverse transform for actual values
            forecast = data_processor.inverse_scale(forecast, is_target=True)
            target = data_processor.inverse_scale(target, is_target=True)
            
            # Calculate metrics
            mse = np.mean((forecast - target) ** 2)
            mae = np.mean(np.abs(forecast - target))
            
            mse_scores.append(mse)
            mae_scores.append(mae)
            
            # Record model selection weights
            weights = output['model_weights'].cpu().numpy()
            for i, name in enumerate(base_models.keys()):
                model_selection[name].append(weights[:, i].mean())
    
    # Compile test results
    test_results = {
        'mse': np.mean(mse_scores),
        'mae': np.mean(mae_scores),
        'model_selection': {name: np.mean(weights) for name, weights in model_selection.items()}
    }
    
    print(f"Test Results - MSE: {test_results['mse']:.4f}, MAE: {test_results['mae']:.4f}")
    print(f"Average Model Selection: {test_results['model_selection']}")
    
    return framework, test_results
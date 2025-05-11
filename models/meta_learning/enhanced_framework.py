# Create a new file: models/meta_learning/enhanced_framework.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from .meta_features import MetaFeatureExtractor
from .meta_learner import MetaLearner, train_meta_learner

from utils.model_feature_extractor import ModelFeatureExtractor
from utils.regime_detector import MarketRegimeDetector

class EnhancedMetaLearningFramework:
    """
    Enhanced meta-learning framework for financial time series forecasting.
    
    Key improvements:
    1. Internal model representation extraction
    2. Advanced market regime detection 
    3. Multi-task learning across volatility indices
    4. Financial metrics optimization
    """
    def __init__(
        self, 
        base_models, 
        meta_feature_dim=32, 
        hidden_dim=64, 
        num_regimes=4,
        regime_method='hybrid',
        use_model_features=True
    ):
        """
        Initialize the enhanced meta-learning framework.
        
        Args:
            base_models: Dictionary of base forecasting models (TFT, NBEATSx, DeepAR)
            meta_feature_dim: Dimension of meta-features
            hidden_dim: Hidden dimension for neural networks
            num_regimes: Number of market regimes to detect
            regime_method: Method for regime detection ('rules', 'clustering', 'hybrid')
            use_model_features: Whether to use internal model features
        """
        # Store base models dictionary
        self.base_models = base_models  
        self.num_models = len(base_models)
        
        # Initialize regime detector
        self.regime_detector = MarketRegimeDetector(
            num_regimes=num_regimes,
            method=regime_method
        )
        
        # Initialize meta-feature extractor
        self.meta_feature_extractor = MetaFeatureExtractor(
            input_dim=None,  # Will be determined by data
            hidden_dim=hidden_dim,
            meta_feature_dim=meta_feature_dim
        )
        
        # Initialize model feature extractor if enabled
        self.use_model_features = use_model_features
        if use_model_features:
            self.model_feature_extractor = ModelFeatureExtractor(base_models)
        
        # Initialize meta-learner with options for model features
        model_output_dim = hidden_dim if use_model_features else None
        
        self.meta_learner = MetaLearner(
            meta_feature_dim=meta_feature_dim,
            hidden_dim=hidden_dim,
            num_models=self.num_models,
            model_output_dim=model_output_dim,
            use_regime_info=True,
            use_financial_metrics=True
        )
        
        # Enhanced temperature parameters (per regime)
        self.temperature = nn.Parameter(torch.ones(num_regimes) * 1.0)
        
        # Create meta-database
        from .meta_features import MetaKnowledgeDatabase
        self.meta_database = MetaKnowledgeDatabase(meta_feature_dim, self.num_models)
    
    def collect_meta_knowledge(
        self, 
        time_series, 
        targets, 
        time_features=None, 
        static_features=None, 
        future_time_features=None,
        task_ids=None,
        vix_data=None  # For regime detection
    ):
        """
        Enhanced knowledge collection with regime and model features.
        
        Args:
            time_series: Input time series [batch_size, seq_len]
            targets: Target values [batch_size, forecast_horizon]
            time_features: Time features or None
            static_features: Static features or None
            future_time_features: Future time features or None
            task_ids: Task identifiers or None
            vix_data: Optional DataFrame with VIX data for regime detection
        """
        with torch.no_grad():
            # Extract meta-features from input time series
            meta_features, _ = self.meta_feature_extractor(time_series, task_ids)
            
            # Detect market regimes if VIX data provided
            if vix_data is not None:
                _, regime_probs = self.regime_detector.detect_regimes(vix_data)
                # Convert to torch tensor
                regime_probs = torch.tensor(regime_probs, dtype=torch.float32)
            else:
                regime_probs = None
            
            # Format data for different models
            model_batches = {}
            
            # Prepare batch for TFT
            if 'tft' in self.base_models:
                tft_static = [static_features] if static_features is not None else None
                tft_encoder = [time_series.unsqueeze(-1)]  # Add feature dimension
                if time_features is not None:
                    tft_encoder.append(time_features)
                tft_decoder = [future_time_features] if future_time_features is not None else []
                
                model_batches['tft'] = {
                    'static_inputs': tft_static,
                    'encoder_inputs': tft_encoder,
                    'decoder_inputs': tft_decoder
                }
                
            # Prepare batch for NBEATSx
            if 'nbeatsx' in self.base_models:
                exog = None
                if time_features is not None and future_time_features is not None:
                    exog = torch.cat([time_features, future_time_features], dim=1)
                
                model_batches['nbeatsx'] = {
                    'x': time_series,
                    'exog': exog
                }
                
            # Prepare batch for DeepAR
            if 'deepar' in self.base_models:
                model_batches['deepar'] = {
                    'time_series': time_series,
                    'time_features': time_features,
                    'static_features': static_features,
                    'future_time_features': future_time_features
                }
            
            # Extract model features if enabled
            model_features = None
            if self.use_model_features:
                model_features = self.model_feature_extractor.extract_model_features(model_batches)
                if model_features is not None:
                    model_features = torch.tensor(model_features, dtype=torch.float32)
            
            # Evaluate model performances
            model_performances = []
            
            # Process each model
            for name, model in self.base_models.items():
                if name == 'tft':
                    # Get TFT output
                    output, _ = model(
                        model_batches['tft']['static_inputs'],
                        model_batches['tft']['encoder_inputs'],
                        model_batches['tft']['decoder_inputs'],
                        return_attention=True
                    )
                    # Extract forecast (median quantile)
                    forecasts = output[:, :, 0, 1]  # [batch, horizon, output_dim=0, quantile=1 (0.5)]
                    
                elif name == 'nbeatsx':
                    # Get NBEATSx forecast
                    forecasts = model(
                        model_batches['nbeatsx']['x'],
                        model_batches['nbeatsx'].get('exog')
                    )
                    
                elif name == 'deepar':
                    # Get DeepAR forecast
                    output = model(
                        model_batches['deepar']['time_series'],
                        time_features=model_batches['deepar'].get('time_features'),
                        static_features=model_batches['deepar'].get('static_features'),
                        future_time_features=model_batches['deepar'].get('future_time_features'),
                        training=False
                    )
                    # Extract mean forecast
                    forecasts = output['mean'].squeeze(-1)
                
                # Calculate performance metrics
                
                # 1. MSE (negative, so higher is better)
                mse_loss = -F.mse_loss(forecasts, targets, reduction='none').mean(dim=1, keepdim=True)
                
                # 2. MAE (negative)
                mae_loss = -F.l1_loss(forecasts, targets, reduction='none').mean(dim=1, keepdim=True)
                
                # 3. Directional accuracy
                # For each sample, check if the forecast direction matches the actual direction
                forecast_dir = (forecasts[:, -1] > forecasts[:, 0]).float()
                target_dir = (targets[:, -1] > targets[:, 0]).float()
                dir_acc = (forecast_dir == target_dir).float().unsqueeze(1)
                
                # Combine metrics (weighted sum)
                # You can adjust weights based on what's most important
                performance = 0.6 * mse_loss + 0.2 * mae_loss + 0.2 * dir_acc
                
                model_performances.append(performance)
            
            # Combine model performances into a single tensor [batch_size, num_models]
            model_performances = torch.cat(model_performances, dim=1)
            
            # Add to meta-knowledge database
            self.meta_database.add_knowledge(
                features=meta_features,
                performance=model_performances,
                task_ids=task_ids,
                market_regimes=regime_probs
            )
            
    def meta_train(self, epochs=50, lr=0.001):
        """
        Train meta-learner on collected meta-knowledge.
        """
        return train_meta_learner(
            self.meta_feature_extractor, 
            self.meta_learner, 
            self.meta_database,
            epochs=epochs,
            lr=lr
        )

    def forecast(
        self, 
        time_series, 
        time_features=None, 
        static_features=None, 
        future_time_features=None,
        task_ids=None,
        vix_data=None,
        return_details=False
    ):
        """
        Generate enhanced forecasts using meta-learning framework.
        
        Args:
            time_series: Input time series [batch_size, seq_len]
            time_features: Time features or None
            static_features: Static features or None
            future_time_features: Future time features or None
            task_ids: Task identifiers or None
            vix_data: Optional DataFrame with VIX data for regime detection
            return_details: Whether to return detailed information
            
        Returns:
            Dictionary with forecast and additional information
        """
        with torch.no_grad():
            # Extract meta-features
            meta_features, _ = self.meta_feature_extractor(time_series, task_ids)
            
            # Detect market regimes if VIX data provided
            if vix_data is not None:
                regimes, regime_probs = self.regime_detector.detect_regimes(vix_data)
                # Convert to torch tensor
                regime_probs = torch.tensor(regime_probs, dtype=torch.float32)
                # Get current regime (last timestep)
                current_regime = regimes[-1] if len(regimes) > 0 else 0
            else:
                regime_probs = None
                current_regime = 0
            
            # Format data for different models
            model_batches = {}
            
            # Prepare batch for TFT
            if 'tft' in self.base_models:
                tft_static = [static_features] if static_features is not None else None
                tft_encoder = [time_series.unsqueeze(-1)]  # Add feature dimension
                if time_features is not None:
                    tft_encoder.append(time_features)
                tft_decoder = [future_time_features] if future_time_features is not None else []
                
                model_batches['tft'] = {
                    'static_inputs': tft_static,
                    'encoder_inputs': tft_encoder,
                    'decoder_inputs': tft_decoder
                }
                
            # Prepare batch for NBEATSx
            if 'nbeatsx' in self.base_models:
                exog = None
                if time_features is not None and future_time_features is not None:
                    exog = torch.cat([time_features, future_time_features], dim=1)
                
                model_batches['nbeatsx'] = {
                    'x': time_series,
                    'exog': exog
                }
                
            # Prepare batch for DeepAR
            if 'deepar' in self.base_models:
                model_batches['deepar'] = {
                    'time_series': time_series,
                    'time_features': time_features,
                    'static_features': static_features,
                    'future_time_features': future_time_features
                }
            
            # Extract model features if enabled
            model_features = None
            if self.use_model_features:
                model_features = self.model_feature_extractor.extract_model_features(model_batches)
                if model_features is not None:
                    model_features = torch.tensor(model_features, dtype=torch.float32)
            
            # Predict model performance using meta-learner
            if self.use_model_features and model_features is not None:
                meta_output = self.meta_learner.forward_with_model_features(
                    meta_features, model_features, regime_probs
                )
            else:
                meta_output = self.meta_learner(meta_features, market_regimes=regime_probs)
            
            # Get model weights
            # Use regime-specific temperature
            predicted_weights = meta_output['weights']
            
            # Adjust with regime-specific temperature
            temperature = self.temperature[current_regime]
            model_weights = F.softmax(predicted_weights / temperature, dim=1)
            
            # Get forecasts from each model
            individual_forecasts = {}
            model_outputs = []
            
            # Process each model
            for name, model in self.base_models.items():
                if name == 'tft':
                    # Get forecast and attention
                    output, attention = model(
                        model_batches['tft']['static_inputs'],
                        model_batches['tft']['encoder_inputs'],
                        model_batches['tft']['decoder_inputs'],
                        return_attention=True
                    )
                    forecasts = output[:, :, 0, 1]  # [batch, horizon, output_dim=0, quantile=1 (0.5)]
                    
                    # Extract uncertainty bounds
                    lower_bound = output[:, :, 0, 0]  # Lower quantile
                    upper_bound = output[:, :, 0, 2]  # Upper quantile
                    
                    # Store results with interpretability info
                    individual_forecasts[name] = {
                        'mean': forecasts,
                        'lower': lower_bound,
                        'upper': upper_bound,
                        'attention': attention
                    }
                    
                elif name == 'nbeatsx':
                    # Get forecast and components
                    forecast, components = model(
                        model_batches['nbeatsx']['x'],
                        model_batches['nbeatsx'].get('exog'),
                        return_decomposition=True
                    )
                    
                    # Store with components for interpretability
                    individual_forecasts[name] = {
                        'mean': forecast,
                        'components': components
                    }
                    
                elif name == 'deepar':
                    # Get distribution parameters
                    output = model(
                        model_batches['deepar']['time_series'],
                        time_features=model_batches['deepar'].get('time_features'),
                        static_features=model_batches['deepar'].get('static_features'),
                        future_time_features=model_batches['deepar'].get('future_time_features'),
                        training=False
                    )
                    
                    # Extract mean and uncertainty
                    mean = output['mean'].squeeze(-1)
                    scale = output['scale'].squeeze(-1)
                    
                    # Gaussian approximation for bounds
                    lower_bound = mean - 1.96 * scale
                    upper_bound = mean + 1.96 * scale
                    
                    individual_forecasts[name] = {
                        'mean': mean,
                        'lower': lower_bound,
                        'upper': upper_bound,
                        'scale': scale
                    }
                
                # Collect outputs for ensemble
                model_outputs.append(individual_forecasts[name]['mean'].unsqueeze(1))
            
            # Stack outputs [batch_size, num_models, horizon]
            stacked_outputs = torch.cat(model_outputs, dim=1)
            
            # Compute weighted ensemble
            weights_expanded = model_weights.unsqueeze(-1)  # [batch_size, num_models, 1]
            ensemble_forecast = torch.sum(stacked_outputs * weights_expanded, dim=1)  # [batch_size, horizon]
            
            # Compute ensemble uncertainty bounds
            # Method 1: Weighted average of individual bounds
            lower_bounds = []
            upper_bounds = []
            
            for name, forecast in individual_forecasts.items():
                if 'lower' in forecast and 'upper' in forecast:
                    lower_bounds.append(forecast['lower'].unsqueeze(1))
                    upper_bounds.append(forecast['upper'].unsqueeze(1))
            
            if lower_bounds and upper_bounds:
                stacked_lower = torch.cat(lower_bounds, dim=1)
                stacked_upper = torch.cat(upper_bounds, dim=1)
                
                ensemble_lower = torch.sum(stacked_lower * weights_expanded, dim=1)
                ensemble_upper = torch.sum(stacked_upper * weights_expanded, dim=1)
            else:
                # Fallback if bounds not available
                ensemble_std = torch.std(stacked_outputs, dim=1)
                ensemble_lower = ensemble_forecast - 1.96 * ensemble_std
                ensemble_upper = ensemble_forecast + 1.96 * ensemble_std
            
            # Compile results
            results = {
                'forecast': ensemble_forecast,
                'lower_bound': ensemble_lower,
                'upper_bound': ensemble_upper,
                'model_weights': model_weights
            }
            
            # Add financial metrics if available
            if 'financial_metrics' in meta_output:
                results['financial_metrics'] = meta_output['financial_metrics']
            
            # Add detailed information if requested
            if return_details:
                results['individual_forecasts'] = individual_forecasts
                results['meta_features'] = meta_features
                
                if regime_probs is not None:
                    results['regime_probs'] = regime_probs
                    results['current_regime'] = current_regime
                
                if model_features is not None:
                    results['model_features'] = model_features
            
            return results
    
    def evaluate(
        self,
        test_data,
        data_processor=None,
        metrics=['mse', 'mae', 'mape', 'dir_acc'],
        by_regime=True,
        by_task=True
    ):
        """
        Evaluate framework performance on test data.
        
        Args:
            test_data: Test data batches
            data_processor: Optional data processor for inverse scaling
            metrics: List of metrics to compute
            by_regime: Whether to break down performance by regime
            by_task: Whether to break down performance by task
            
        Returns:
            evaluation_results: Dictionary with evaluation metrics
        """
        self.meta_feature_extractor.eval()
        self.meta_learner.eval()
        
        all_forecasts = []
        all_targets = []
        all_regimes = []
        all_tasks = []
        all_weights = []
        
        # Process test batches
        with torch.no_grad():
            for batch in test_data:
                # Generate forecast
                output = self.forecast(
                    time_series=batch['input'],
                    time_features=batch.get('time_features'),
                    static_features=batch.get('static_features'),
                    future_time_features=batch.get('future_time_features'),
                    task_ids=batch.get('task_ids'),
                    vix_data=batch.get('vix_data'),
                    return_details=True
                )
                
                # Convert to numpy and inverse scale if needed
                forecast = output['forecast'].cpu().numpy()
                target = batch['target'].cpu().numpy()
                
                if data_processor is not None:
                    forecast = data_processor.inverse_scale(forecast, is_target=True)
                    target = data_processor.inverse_scale(target, is_target=True)
                
                # Store results
                all_forecasts.append(forecast)
                all_targets.append(target)
                
                # Store model weights
                all_weights.append(output['model_weights'].cpu().numpy())
                
                # Store regime information if available
                if 'current_regime' in output:
                    all_regimes.append(np.ones(len(forecast)) * output['current_regime'])
                elif 'regime' in batch:
                    all_regimes.append(batch['regime'].cpu().numpy())
                
                # Store task information if available
                if 'task_ids' in batch:
                    all_tasks.append(batch['task_ids'].cpu().numpy())
                elif 'task_ids_test' in batch:
                    all_tasks.append(batch['task_ids_test'])
        
        # Combine all results
        all_forecasts = np.vstack(all_forecasts)
        all_targets = np.vstack(all_targets)
        all_weights = np.vstack(all_weights)
        
        if all_regimes:
            all_regimes = np.concatenate(all_regimes)
        if all_tasks:
            all_tasks = np.concatenate(all_tasks)
        
        # Calculate metrics
        results = {}
        
        # Overall metrics
        results['overall'] = self._calculate_metrics(all_forecasts, all_targets, metrics)
        
        # Metrics by regime
        if by_regime and all_regimes is not None and len(all_regimes) > 0:
            regime_results = {}
            unique_regimes = np.unique(all_regimes)
            
            for regime in unique_regimes:
                mask = (all_regimes == regime)
                if np.sum(mask) > 0:
                    regime_results[f'regime_{int(regime)}'] = self._calculate_metrics(
                        all_forecasts[mask], all_targets[mask], metrics
                    )
            
            results['by_regime'] = regime_results
        
        # Metrics by task
        if by_task and all_tasks is not None and len(all_tasks) > 0:
            task_results = {}
            unique_tasks = np.unique(all_tasks)
            
            for task in unique_tasks:
                mask = (all_tasks == task)
                if np.sum(mask) > 0:
                    task_results[f'task_{int(task)}'] = self._calculate_metrics(
                        all_forecasts[mask], all_targets[mask], metrics
                    )
            
            results['by_task'] = task_results
        
        # Model selection statistics
        if len(all_weights) > 0:
            model_names = list(self.base_models.keys())
            weight_stats = {}
            
            for i, name in enumerate(model_names):
                model_weights = all_weights[:, i]
                weight_stats[name] = {
                    'mean': float(np.mean(model_weights)),
                    'std': float(np.std(model_weights)),
                    'max': float(np.max(model_weights)),
                    'min': float(np.min(model_weights))
                }
                
                # Calculate how often this model was selected (had highest weight)
                selections = (np.argmax(all_weights, axis=1) == i).sum()
                weight_stats[name]['selection_rate'] = float(selections / len(all_weights))
            
            results['model_selection'] = weight_stats
        
        return results
    
    def _calculate_metrics(self, forecasts, targets, metric_names):
        """
        Calculate evaluation metrics.
        
        Args:
            forecasts: Forecasts array [batch_size, horizon]
            targets: Targets array [batch_size, horizon]
            metric_names: List of metric names
            
        Returns:
            metrics: Dictionary of metric values
        """
        metrics = {}
        
        for name in metric_names:
            if name.lower() == 'mse':
                # Mean Squared Error
                mse = np.mean((forecasts - targets) ** 2)
                metrics['mse'] = float(mse)
                
                # Root Mean Squared Error
                metrics['rmse'] = float(np.sqrt(mse))
            
            elif name.lower() == 'mae':
                # Mean Absolute Error
                mae = np.mean(np.abs(forecasts - targets))
                metrics['mae'] = float(mae)
            
            elif name.lower() == 'mape':
                # Mean Absolute Percentage Error
                # Avoid division by zero
                non_zero = (np.abs(targets) > 1e-8)
                if np.sum(non_zero) > 0:
                    mape = np.mean(np.abs((targets[non_zero] - forecasts[non_zero]) / targets[non_zero])) * 100
                    metrics['mape'] = float(mape)
                else:
                    metrics['mape'] = float('nan')
            
            elif name.lower() == 'dir_acc':
                # Directional Accuracy
                # Check if forecast direction matches actual direction
                # Compare first and last points of the horizon
                f_dir = (forecasts[:, -1] > forecasts[:, 0])
                t_dir = (targets[:, -1] > targets[:, 0])
                dir_acc = np.mean(f_dir == t_dir) * 100
                metrics['dir_acc'] = float(dir_acc)
            
            elif name.lower() == 'sharpe':
                # Pseudo-Sharpe ratio (simplified for forecasts)
                # Use relative changes from first point as proxy for returns
                f_rets = (forecasts[:, 1:] - forecasts[:, :-1]) / np.abs(forecasts[:, :-1] + 1e-8)
                t_rets = (targets[:, 1:] - targets[:, :-1]) / np.abs(targets[:, :-1] + 1e-8)
                
                # Calculate mean return and volatility
                mean_f_ret = np.mean(f_rets)
                std_f_ret = np.std(f_rets)
                
                if std_f_ret > 0:
                    sharpe = mean_f_ret / std_f_ret
                    metrics['sharpe'] = float(sharpe)
                else:
                    metrics['sharpe'] = float('nan')
            
            elif name.lower() == 'coverage':
                # Prediction interval coverage
                # This requires the prediction intervals from the forecast
                # Not calculated here but could be added if intervals are passed
                pass
        
        return metrics
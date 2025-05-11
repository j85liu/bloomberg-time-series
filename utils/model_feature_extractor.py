# Create a new file: utils/model_feature_extractor.py

import torch
import torch.nn as nn
import numpy as np

class ModelFeatureExtractor:
    """Extract internal representations from base models to inform meta-learning"""
    
    def __init__(self, models):
        """
        Initialize with trained base models
        
        Args:
            models: Dictionary of trained models (TFT, NBEATSx, DeepAR)
        """
        self.models = models
        
    def extract_tft_features(self, data_batch):
        """Extract interpretable features from TFT model"""
        model = self.models.get('tft')
        if model is None:
            return None
            
        with torch.no_grad():
            # Get model outputs with attention
            _, attention = model(
                data_batch['static_inputs'],
                data_batch['encoder_inputs'],
                data_batch['decoder_inputs'],
                return_attention=True
            )
            
            features = {}

            # Variable selection weights
            if 'encoder_weights' in attention:
                features['var_importance'] = attention['encoder_weights'].mean(dim=1).cpu().numpy()
            
            # Extract decoder-encoder attention (future looking at past)
            if 'decoder_encoder_attention' in attention:
                # Average across attention heads for interpretability
                attn = attention['decoder_encoder_attention'].mean(dim=1).cpu().numpy()
                
                # Calculate attention concentration (how focused is attention)
                attn_entropy = -np.sum(attn * np.log(attn + 1e-10), axis=2)
                features['attn_entropy'] = attn_entropy
                
                # Recent attention (how much model focuses on recent vs distant past)
                # Calculate weighted average position of attention
                seq_len = attn.shape[2]
                positions = np.arange(seq_len)
                weighted_pos = np.sum(attn * positions.reshape(1, 1, -1), axis=2)
                features['attn_recency'] = weighted_pos / seq_len
                
                # Attention patterns
                # Summarize into a smaller fixed-size representation
                # We'll divide the attention pattern into regions and average within each
                if seq_len >= 20:
                    # Divide into 4 regions: very recent, recent, mid, distant
                    regions = [
                        slice(int(0.75*seq_len), seq_len),    # very recent (last 25%)
                        slice(int(0.5*seq_len), int(0.75*seq_len)),  # recent (25-50%)
                        slice(int(0.25*seq_len), int(0.5*seq_len)),  # mid (25-50%) 
                        slice(0, int(0.25*seq_len))           # distant (first 25%)
                    ]
                    
                    region_attn = np.zeros((attn.shape[0], attn.shape[1], 4))
                    for i, region in enumerate(regions):
                        region_attn[:, :, i] = attn[:, :, region].mean(axis=2)
                        
                    features['attn_pattern'] = region_attn
        
        return features
    
    def extract_nbeatsx_features(self, data_batch):
        """Extract interpretable features from NBEATSx model"""
        model = self.models.get('nbeatsx')
        if model is None:
            return None
            
        with torch.no_grad():
            # Get forecasts and components
            _, components = model(
                data_batch['x'],
                return_decomposition=True
            )
            
            features = {}
            
            # Calculate relative importance of each component
            if components:
                # Stack components on new axis
                stacked = torch.stack(components, dim=1)  # [batch, n_components, horizon]
                
                # Calculate magnitude (L1 norm) of each component
                magnitudes = stacked.abs().sum(dim=2)  # [batch, n_components]
                
                # Calculate relative contribution
                total = magnitudes.sum(dim=1, keepdim=True)
                rel_contribution = magnitudes / (total + 1e-10)
                
                features['component_importance'] = rel_contribution.cpu().numpy()
                
                # Calculate trend strength
                if len(components) >= 3 and "trend" in model.stacks[0].stack_type:
                    trend_idx = 0  # Assuming trend is first component
                    trend = components[trend_idx]
                    
                    # Calculate monotonicity (how much trend changes direction)
                    # Higher values mean more consistent trend
                    diffs = trend[:, 1:] - trend[:, :-1]
                    sign_changes = ((diffs[:, 1:] * diffs[:, :-1]) < 0).float().mean(dim=1)
                    trend_monotonicity = 1.0 - sign_changes
                    
                    features['trend_monotonicity'] = trend_monotonicity.cpu().numpy()
                
                # Calculate seasonality strength if available
                if len(components) >= 3 and "seasonality" in model.stacks[1].stack_type:
                    seasonality_idx = 1  # Assuming seasonality is second component
                    seasonality = components[seasonality_idx]
                    
                    # Calculate strength as standard deviation
                    seasonality_strength = seasonality.std(dim=1)
                    features['seasonality_strength'] = seasonality_strength.cpu().numpy()
            
            return features
    
    def extract_deepar_features(self, data_batch):
        """Extract interpretable features from DeepAR model"""
        model = self.models.get('deepar')
        if model is None:
            return None
            
        with torch.no_grad():
            # Get distribution parameters
            output = model(
                data_batch['time_series'],
                time_features=data_batch.get('time_features'),
                static_features=data_batch.get('static_features'),
                future_time_features=None,
                training=False
            )
            
            features = {}
            
            # Extract uncertainty information
            if 'mean' in output and 'scale' in output:
                mean = output['mean'].cpu().numpy()
                scale = output['scale'].cpu().numpy()
                
                # Calculate coefficient of variation (uncertainty relative to prediction)
                cv = scale / (np.abs(mean) + 1e-10)
                features['uncertainty'] = cv.mean(axis=1).squeeze()
                
                # Calculate mean prediction trend
                if mean.shape[1] > 1:
                    trend = (mean[:, -1] - mean[:, 0]) / (mean[:, 0] + 1e-10)
                    features['forecast_trend'] = trend.squeeze()
                    
                    # Calculate forecast monotonicity
                    diffs = mean[:, 1:] - mean[:, :-1]
                    sign_changes = ((diffs[:, 1:] * diffs[:, :-1]) < 0)
                    monotonicity = 1.0 - sign_changes.mean(axis=1)
                    features['forecast_monotonicity'] = monotonicity.squeeze()
            
            return features
    
    def extract_model_features(self, data_batch):
        """
        Extract and combine features from all models
        
        Args:
            data_batch: Dictionary with model-specific batches
            
        Returns:
            model_features: Combined model features for meta-learning
        """
        # Extract features from each model
        tft_features = self.extract_tft_features(data_batch.get('tft', {}))
        nbeatsx_features = self.extract_nbeatsx_features(data_batch.get('nbeatsx', {}))
        deepar_features = self.extract_deepar_features(data_batch.get('deepar', {}))
        
        # Process into a format suitable for meta-learning
        # We need a fixed-size representation for each batch item
        
        # Build combined feature vector
        combined_features = []
        
        # Add available features from each model
        if tft_features:
            # Basic variable importance
            if 'var_importance' in tft_features:
                # Ensure fixed size by truncating or padding
                max_vars = min(10, tft_features['var_importance'].shape[1])
                for i in range(tft_features['var_importance'].shape[0]):
                    combined_features.append(tft_features['var_importance'][i, :max_vars])
            
            # Attention recency feature
            if 'attn_recency' in tft_features:
                for i in range(tft_features['attn_recency'].shape[0]):
                    combined_features.append(tft_features['attn_recency'][i])
        
        if nbeatsx_features:
            # Component importance
            if 'component_importance' in nbeatsx_features:
                for i in range(nbeatsx_features['component_importance'].shape[0]):
                    combined_features.append(nbeatsx_features['component_importance'][i])
            
            # Trend and seasonality features
            if 'trend_monotonicity' in nbeatsx_features:
                for i in range(len(nbeatsx_features['trend_monotonicity'])):
                    combined_features.append([nbeatsx_features['trend_monotonicity'][i]])
                    
            if 'seasonality_strength' in nbeatsx_features:
                for i in range(len(nbeatsx_features['seasonality_strength'])):
                    combined_features.append([nbeatsx_features['seasonality_strength'][i]])
        
        if deepar_features:
            # Uncertainty features
            if 'uncertainty' in deepar_features:
                for i in range(len(deepar_features['uncertainty'])):
                    combined_features.append([deepar_features['uncertainty'][i]])
            
            # Forecast trend
            if 'forecast_trend' in deepar_features:
                for i in range(len(deepar_features['forecast_trend'])):
                    combined_features.append([deepar_features['forecast_trend'][i]])
        
        # Convert to numpy array
        if combined_features:
            return np.array(combined_features)
        else:
            return None
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class MetaLearner(nn.Module):
    """
    Enhanced meta-learner module that implements stacked generalization 
    to select or combine base forecasting models.
    
    This model takes meta-features extracted from time series and model outputs
    to predict how well each forecasting model will perform.
    """
    def __init__(self, meta_feature_dim, hidden_dim=64, num_models=3, forecast_horizon=10, 
                 model_output_dim=None, use_regime_info=True, use_financial_metrics=True):
        """
        Initialize the enhanced meta-learner.
        
        Args:
            meta_feature_dim: Dimension of input meta-features
            hidden_dim: Dimension of hidden layers
            num_models: Number of base forecasting models to select from
            forecast_horizon: Length of the forecast horizon
            model_output_dim: Dimension of model outputs for stacked generalization (optional)
            use_regime_info: Whether to use market regime information
            use_financial_metrics: Whether to use financial performance metrics
        """
        super().__init__()
        
        self.use_regime_info = use_regime_info
        self.use_financial_metrics = use_financial_metrics
        
        # Meta-feature based prediction network
        self.meta_network = nn.Sequential(
            nn.Linear(meta_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_models)
        )
        
        # Regime-specific network (if using regime information)
        if use_regime_info:
            self.regime_networks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(meta_feature_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, num_models)
                ) for _ in range(4)  # One network per market regime
            ])
        
        # Stacked generalization network for model outputs
        if model_output_dim is not None:
            # Process each model's output separately
            self.model_processors = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(forecast_horizon, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim // num_models)
                ) for _ in range(num_models)
            ])
            
            # Combine processed outputs
            self.stacking_network = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, num_models)
            )
            
            # Final combination network
            input_dim = num_models * 2  # Meta-network + stacking network
            if use_regime_info:
                input_dim += num_models  # Add regime network output
                
            self.combined_network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, num_models)
            )
        else:
            self.model_processors = None
            self.stacking_network = None
            self.combined_network = None
            
        # Financial performance projection (Sharpe ratio, etc.)
        if use_financial_metrics:
            self.financial_projector = nn.Sequential(
                nn.Linear(num_models, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 4)  # Sharpe ratio, max drawdown, alpha, beta
            )
        
    def forward(self, meta_features, model_outputs=None, market_regimes=None):
        """
        Predict performance of each model based on meta-features and model outputs.
        
        Args:
            meta_features: Meta-features of time series [batch_size, meta_feature_dim]
            model_outputs: Optional forecasts from base models [batch_size, num_models, forecast_horizon]
            market_regimes: Optional market regime probabilities [batch_size, num_regimes]
            
        Returns:
            Dictionary containing:
                - weights: Predicted model weights [batch_size, num_models]
                - financial_metrics: Optional financial performance metrics [batch_size, 4]
        """
        # Get weights from meta-features
        meta_weights = self.meta_network(meta_features)
        
        # Combine with regime-specific predictions if available
        regime_weights = None
        if self.use_regime_info and market_regimes is not None:
            # Get predictions from each regime-specific network
            regime_preds = []
            for i, network in enumerate(self.regime_networks):
                regime_pred = network(meta_features)
                regime_preds.append(regime_pred.unsqueeze(1))
                
            # Stack and weight by regime probabilities
            regime_preds = torch.cat(regime_preds, dim=1)  # [batch_size, num_regimes, num_models]
            regime_probs = market_regimes.unsqueeze(-1)    # [batch_size, num_regimes, 1]
            regime_weights = torch.sum(regime_preds * regime_probs, dim=1)  # [batch_size, num_models]
        
        # Process model outputs for stacked generalization if available
        stacked_weights = None
        if model_outputs is not None and self.stacking_network is not None:
            # Process each model's output separately
            processed_outputs = []
            for i, processor in enumerate(self.model_processors):
                model_i_output = model_outputs[:, i, :]  # [batch_size, forecast_horizon]
                processed = processor(model_i_output)
                processed_outputs.append(processed)
                
            # Concatenate processed outputs
            processed_outputs = torch.cat(processed_outputs, dim=1)  # [batch_size, hidden_dim]
            
            # Generate weights from processed outputs
            stacked_weights = self.stacking_network(processed_outputs)
        
        # Combine different weight predictions if needed
        if stacked_weights is not None:
            if regime_weights is not None:
                combined_input = torch.cat([meta_weights, regime_weights, stacked_weights], dim=1)
            else:
                combined_input = torch.cat([meta_weights, stacked_weights], dim=1)
                
            final_weights = self.combined_network(combined_input)
        elif regime_weights is not None:
            # Simple average of meta and regime weights if no stacking
            final_weights = (meta_weights + regime_weights) / 2.0
        else:
            # Just use meta-features if no other inputs
            final_weights = meta_weights
            
        # Add softmax to get proper weights
        model_weights = F.softmax(final_weights, dim=1)
        
        # Calculate financial metrics if requested
        financial_metrics = None
        if self.use_financial_metrics:
            financial_metrics = self.financial_projector(model_weights)
            
        return {
            'weights': model_weights,
            'financial_metrics': financial_metrics
        }


def train_meta_learner(meta_feature_extractor, meta_learner, meta_database, epochs=50, lr=0.001, batch_size=32):
    """
    Train meta-learner on collected meta-knowledge.
    
    Args:
        meta_feature_extractor: Meta-feature extraction module
        meta_learner: Meta-learner module
        meta_database: Database containing meta-knowledge
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size for training
        
    Returns:
        bool: Whether training was successful
    """
    # Get meta-dataset from database
    meta_features, performances, task_ids, market_regimes = meta_database.get_dataset()
    if meta_features is None:
        print("No meta-knowledge available for training")
        return False
    
    # Create PyTorch DataLoader for batching
    if task_ids is not None and market_regimes is not None:
        dataset = TensorDataset(meta_features, performances, task_ids, market_regimes)
    elif task_ids is not None:
        dataset = TensorDataset(meta_features, performances, task_ids)
    elif market_regimes is not None:
        dataset = TensorDataset(meta_features, performances, market_regimes)
    else:
        dataset = TensorDataset(meta_features, performances)
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Define mean squared error loss
    criterion = nn.MSELoss()
    
    # Setup Adam optimizer with both meta-feature extractor and meta-learner parameters
    optimizer = optim.Adam(list(meta_feature_extractor.parameters()) + 
                          list(meta_learner.parameters()), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        
        # Process each batch
        for batch in dataloader:
            # Parse batch data
            if len(batch) == 4:  # meta_features, performances, task_ids, market_regimes
                features, targets, task_ids_batch, market_regimes_batch = batch
            elif len(batch) == 3:  # meta_features, performances, task_ids/market_regimes
                if task_ids is not None:
                    features, targets, task_ids_batch = batch
                    market_regimes_batch = None
                else:
                    features, targets, market_regimes_batch = batch
                    task_ids_batch = None
            else:  # meta_features, performances
                features, targets = batch
                task_ids_batch = None
                market_regimes_batch = None
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Extract meta-features from input features
            meta_features, extracted_regimes = meta_feature_extractor(features, task_ids_batch)
            
            # Use extracted or provided market regimes
            if market_regimes_batch is None:
                market_regimes_batch = extracted_regimes
            
            # Predict model performance based on meta-features
            output = meta_learner(meta_features, market_regimes=market_regimes_batch)
            pred_performance = output['weights']
            
            # Compute MSE loss
            loss = criterion(pred_performance, targets)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print epoch statistics
        avg_loss = total_loss / len(dataloader)
        print(f'Meta-Learner Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}')
    
    print("Meta-learner training complete")
    return True
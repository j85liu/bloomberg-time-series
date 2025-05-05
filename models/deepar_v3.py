import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, NegativeBinomial
import numpy as np
from typing import Tuple, Optional, List, Dict, Union
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import logging
from pathlib import Path
import json


# ============= Model Classes =============

class GaussianLayer(nn.Module):
    """Gaussian output distribution for continuous data"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.mu_linear = nn.Linear(hidden_size, 1)
        self.sigma_linear = nn.Linear(hidden_size, 1)
        
    def forward(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.mu_linear(hidden_state)
        sigma = F.softplus(self.sigma_linear(hidden_state)) + 1e-5  # Ensure positive
        return mu, sigma
    
    def sample(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        dist = Normal(mu, sigma)
        return dist.sample()
    
    def log_prob(self, y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        dist = Normal(mu, sigma)
        return dist.log_prob(y)


class StudentTLayer(nn.Module):
    """Student-T output distribution for heavy-tailed continuous data"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.mu_linear = nn.Linear(hidden_size, 1)
        self.sigma_linear = nn.Linear(hidden_size, 1)
        self.nu_linear = nn.Linear(hidden_size, 1)
        
    def forward(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.mu_linear(hidden_state)
        sigma = F.softplus(self.sigma_linear(hidden_state)) + 1e-5
        nu = F.softplus(self.nu_linear(hidden_state)) + 2.1  # Ensure >2 for finite variance
        return mu, sigma, nu


class NegativeBinomialLayer(nn.Module):
    """Negative Binomial output distribution for count data"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.mu_linear = nn.Linear(hidden_size, 1)
        self.alpha_linear = nn.Linear(hidden_size, 1)
        
    def forward(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = F.softplus(self.mu_linear(hidden_state)) + 1e-5
        alpha = F.softplus(self.alpha_linear(hidden_state)) + 1e-5
        return mu, alpha
    
    def sample(self, mu: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        # Convert to shape parameter for PyTorch NegativeBinomial
        r = 1.0 / alpha
        p = r / (r + mu)
        dist = NegativeBinomial(total_count=r, probs=p, validate_args=False)
        return dist.sample()
    
    def log_prob(self, y: torch.Tensor, mu: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        r = 1.0 / alpha
        p = r / (r + mu)
        dist = NegativeBinomial(total_count=r, probs=p, validate_args=False)
        return dist.log_prob(y)


class DeepARModel(nn.Module):
    """DeepAR Model faithful to the paper"""
    
    def __init__(
        self,
        num_time_features: int,
        num_static_features: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        likelihood: str = 'gaussian',
        seq_len: int = 60,
        prediction_len: int = 7
    ):
        super().__init__()
        
        self.likelihood = likelihood
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.prediction_len = prediction_len
        
        # Feature embedding layers
        self.lag_embedding = nn.Linear(1, embedding_dim)
        if num_time_features > 0:
            self.time_feature_embeddings = nn.Linear(num_time_features, embedding_dim)
        else:
            self.time_feature_embeddings = None
            
        if num_static_features > 0:
            self.static_embedding = nn.Linear(num_static_features, embedding_dim)
        else:
            self.static_embedding = None
        
        # LSTM network
        lstm_input_size = embedding_dim  # From lag embedding
        if self.time_feature_embeddings:
            lstm_input_size += embedding_dim
        if self.static_embedding:
            lstm_input_size += embedding_dim
            
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        if likelihood == 'gaussian':
            self.output_layer = GaussianLayer(hidden_size)
        elif likelihood == 'studentt':
            self.output_layer = StudentTLayer(hidden_size)
        elif likelihood == 'negbinomial':
            self.output_layer = NegativeBinomialLayer(hidden_size)
        else:
            raise ValueError(f"Unknown likelihood: {likelihood}")
            
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights according to paper specifications"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(
        self,
        y: torch.Tensor,
        time_features: Optional[torch.Tensor] = None,
        static_features: Optional[torch.Tensor] = None,
        future_time_features: Optional[torch.Tensor] = None,
        training: bool = True
    ) -> Dict[str, torch.Tensor]:
        batch_size = y.size(0)
        seq_len = y.size(1)
        
        # Extract static embeddings
        if self.static_embedding and static_features is not None:
            static_emb = self.static_embedding(static_features).unsqueeze(1)
            static_emb = static_emb.expand(-1, seq_len + self.prediction_len, -1)
        else:
            static_emb = None
        
        # Initialize hidden states
        hidden = self._init_hidden(batch_size)
        
        # Lists to store predictions
        means = []
        scales = []
        
        # Auto-regressive loop
        future_y = []
        current_y = torch.zeros(batch_size, 1, 1, device=y.device)  # Shape: (batch_size, seq_len=1, feature_dim=1)
        
        for t in range(seq_len + self.prediction_len):
            # Get input embeddings
            if t < seq_len:
                # Training phase: use ground truth
                if t > 0:
                    current_y = y[:, t-1:t].unsqueeze(-1)  # Add dimension for embedding
                input_features = time_features[:, t:t+1] if time_features is not None else None
            else:
                # Prediction phase: use previous prediction
                t_offset = t - seq_len
                input_features = future_time_features[:, t_offset:t_offset+1] if future_time_features is not None else None
            
            # Get embeddings - now both inputs have correct dimensions
            lag_emb = self.lag_embedding(current_y)
            
            # Combine embeddings
            x = lag_emb
            if self.time_feature_embeddings and input_features is not None:
                time_emb = self.time_feature_embeddings(input_features)
                x = torch.cat([x, time_emb], dim=-1)
            if static_emb is not None:
                static_chunk = static_emb[:, t:t+1]
                x = torch.cat([x, static_chunk], dim=-1)
            
            # LSTM step
            output, hidden = self.lstm(x, hidden)
            
            # Get distribution parameters
            if self.likelihood == 'gaussian':
                mean, scale = self.output_layer(output)
                # For prediction steps, store parameters
                if t >= seq_len:
                    means.append(mean.squeeze(1))
                    scales.append(scale.squeeze(1))
                # Sample next value
                current_y = self.output_layer.sample(mean, scale) if not training or t >= seq_len else y[:, t:t+1].unsqueeze(-1)
                if t >= seq_len:
                    future_y.append(current_y.squeeze(-1).squeeze(1))
            elif self.likelihood == 'studentt':
                mean, scale, nu = self.output_layer(output)
                if t >= seq_len:
                    means.append(mean.squeeze(1))
                    scales.append(scale.squeeze(1))
                # Simplified sampling for now
                current_y = Normal(mean, scale).sample() if not training or t >= seq_len else y[:, t:t+1].unsqueeze(-1)
                if t >= seq_len:
                    future_y.append(current_y.squeeze(-1).squeeze(1))
            elif self.likelihood == 'negbinomial':
                mean, alpha = self.output_layer(output)
                if t >= seq_len:
                    means.append(mean.squeeze(1))
                    scales.append(alpha.squeeze(1))
                current_y = self.output_layer.sample(mean, alpha) if not training or t >= seq_len else y[:, t:t+1].unsqueeze(-1)
                if t >= seq_len:
                    future_y.append(current_y.squeeze(-1).squeeze(1))
        
        # Stack outputs
        result = {
            'mean': torch.stack(means, dim=1) if means else None,
            'scale': torch.stack(scales, dim=1) if scales else None,
            'samples': torch.stack(future_y, dim=1) if future_y else None
        }
        
        return result
    
    def _init_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden and cell states for LSTM"""
        device = next(self.parameters()).device
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return hidden, cell
    
    def loss(
        self,
        y: torch.Tensor,
        time_features: Optional[torch.Tensor] = None,
        static_features: Optional[torch.Tensor] = None,
        future_time_features: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Split input and target
        seq_len = min(y.size(1) - self.prediction_len, self.seq_len)
        input_y = y[:, :seq_len]
        target_y = y[:, seq_len:seq_len + self.prediction_len].unsqueeze(-1)  # Add dimension to match model output
        
        # Split features
        input_time_features = time_features[:, :seq_len] if time_features is not None else None
        future_time_features = time_features[:, seq_len:seq_len + self.prediction_len] if time_features is not None else None
        
        # Forward pass
        outputs = self.forward(input_y, input_time_features, static_features, future_time_features, training=True)
        
        # Compute loss
        if self.likelihood == 'gaussian':
            log_prob = self.output_layer.log_prob(target_y, outputs['mean'], outputs['scale'])
        elif self.likelihood == 'negbinomial':
            log_prob = self.output_layer.log_prob(target_y, outputs['mean'], outputs['scale'])
        else:
            raise NotImplementedError(f"Loss not implemented for {self.likelihood}")
        
        # Apply mask if provided
        if mask is not None:
            mask = mask[:, seq_len:seq_len + self.prediction_len].unsqueeze(-1)  # Add dimension to match log_prob
            log_prob = log_prob * mask
            nll = -log_prob.sum() / mask.sum()
        else:
            nll = -log_prob.mean()
        
        return nll
    
    def sample(
        self,
        y: torch.Tensor,
        num_samples: int,
        time_features: Optional[torch.Tensor] = None,
        static_features: Optional[torch.Tensor] = None,
        future_time_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate multiple samples from the predictive distribution"""
        samples = []
        
        for _ in range(num_samples):
            output = self.forward(y, time_features, static_features, future_time_features, training=False)
            samples.append(output['samples'].unsqueeze(-1))
        
        return torch.cat(samples, dim=-1)
    

# ============= Example Usage =============

if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    from torch.utils.data import Dataset, DataLoader
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create synthetic time series dataset
    class SyntheticTimeSeriesDataset(Dataset):
        def __init__(self, num_series=1000, seq_len=60, pred_len=7):
            self.seq_len = seq_len
            self.pred_len = pred_len
            
            # Generate synthetic data with trend and seasonality
            time_steps = seq_len + pred_len
            time = np.arange(time_steps)
            
            data = []
            for i in range(num_series):
                # Create varying trends and seasonality for each series
                trend = 0.01 * time + np.random.randn() * 0.05
                seasonal = 10 * np.sin(2 * np.pi * time / 12) + 5 * np.cos(2 * np.pi * time / 24)
                noise = np.random.randn(time_steps) * 0.5
                series = trend + seasonal + noise + 10
                data.append(series)
            
            self.data = torch.FloatTensor(data)
            
            # Create time features (e.g., time of day, day of week)
            hour_of_day = np.sin(2 * np.pi * np.arange(time_steps) / 24)
            day_of_week = np.sin(2 * np.pi * np.arange(time_steps) / 7)
            self.time_features = torch.FloatTensor(np.stack([hour_of_day, day_of_week], axis=-1))
            
            # Repeat time features for all series
            self.time_features = self.time_features.unsqueeze(0).repeat(num_series, 1, 1)
            
            # Create static features (random category embeddings)
            self.static_features = torch.randn(num_series, 5)  # 5-dim static features
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return {
                'y': self.data[idx],
                'time_features': self.time_features[idx],
                'static_features': self.static_features[idx]
            }
    
    # Hyperparameters
    config = {
        'num_time_features': 2,
        'num_static_features': 5,
        'embedding_dim': 32,
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.1,
        'likelihood': 'gaussian',  # or 'studentt', 'negbinomial'
        'seq_len': 60,
        'prediction_len': 7,
        'batch_size': 32,
        'num_epochs': 10,
        'learning_rate': 0.001
    }
    
    # Create dataset and dataloader
    dataset = SyntheticTimeSeriesDataset(
        num_series=1000, 
        seq_len=config['seq_len'], 
        pred_len=config['prediction_len']
    )
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    # Initialize model
    model = DeepARModel(**{k: v for k, v in config.items() if k not in ['batch_size', 'num_epochs', 'learning_rate']})
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Training loop
    print("Starting training...")
    for epoch in range(config['num_epochs']):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            y = batch['y']
            time_features = batch['time_features']
            static_features = batch['static_features']
            
            # Debug prints
            if batch_idx == 0 and epoch == 0:
                print(f"y shape: {y.shape}")
                print(f"time_features shape: {time_features.shape}")
                print(f"static_features shape: {static_features.shape}")
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Compute loss
            loss = model.loss(
                y=y,
                time_features=time_features,
                static_features=static_features
            )
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{config['num_epochs']}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"Epoch [{epoch+1}/{config['num_epochs']}] Average Loss: {avg_loss:.4f}\n")
    
    # Make predictions
    print("Making predictions...")
    model.eval()
    
    # Take first batch for prediction
    batch = next(iter(dataloader))
    y = batch['y'][:5]  # Take first 5 series
    time_features = batch['time_features'][:5]
    static_features = batch['static_features'][:5]
    
    with torch.no_grad():
        # Get single prediction
        output = model.forward(
            y[:, :config['seq_len']], 
            time_features[:, :config['seq_len']], 
            static_features,
            future_time_features=time_features[:, config['seq_len']:config['seq_len']+config['prediction_len']],
            training=False
        )
        
        # Get multiple samples
        samples = model.sample(
            y[:, :config['seq_len']], 
            num_samples=100,
            time_features=time_features[:, :config['seq_len']], 
            static_features=static_features,
            future_time_features=time_features[:, config['seq_len']:config['seq_len']+config['prediction_len']]
        )
    
    # Print results
    print(f"Prediction means shape: {output['mean'].shape}")
    print(f"Prediction samples shape: {samples.shape}")
    
    # Visualize results for first series
    print("\nPrediction vs Ground Truth for first series:")
    print("Ground truth:", y[0, config['seq_len']:config['seq_len']+config['prediction_len']].numpy())
    print("Mean prediction:", output['mean'][0].squeeze().numpy())
    print("Sample prediction:", samples[0, :, 0].numpy())  # First sample
    
    # Calculate metrics
    mse = torch.mean((output['mean'].squeeze() - y[:, config['seq_len']:config['seq_len']+config['prediction_len']]) ** 2)
    mae = torch.mean(torch.abs(output['mean'].squeeze() - y[:, config['seq_len']:config['seq_len']+config['prediction_len']]))
    print(f"\nMean Squared Error: {mse.item():.4f}")
    print(f"Mean Absolute Error: {mae.item():.4f}")
    
    # Example of saving and loading the model
    print("\nSaving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }, 'deepar_model.pt')
    
    print("Loading model...")
    checkpoint = torch.load('deepar_model.pt')
    new_model = DeepARModel(**{k: v for k, v in checkpoint['config'].items() if k not in ['batch_size', 'num_epochs', 'learning_rate']})
    new_model.load_state_dict(checkpoint['model_state_dict'])
    new_optimizer = optim.Adam(new_model.parameters(), lr=checkpoint['config']['learning_rate'])
    new_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print("Model loaded successfully!")
    
    # Test with different likelihoods
    print("\nTesting with Negative Binomial likelihood...")
    config_negbin = config.copy()
    config_negbin['likelihood'] = 'negbinomial'
    
    # Convert data to count data for neg binomial
    count_data = torch.round(torch.exp(dataset.data / 5)).clamp(0, 100)  # Convert to counts
    
    # Create a dataset for count data
    count_dataset = SyntheticTimeSeriesDataset(num_series=100)
    count_dataset.data = count_data[:100]  # Use subset for quick test
    count_dataloader = DataLoader(count_dataset, batch_size=16, shuffle=True)
    
    # Initialize and train negative binomial model
    negbin_model = DeepARModel(**{k: v for k, v in config_negbin.items() if k not in ['batch_size', 'num_epochs', 'learning_rate']})
    negbin_optimizer = optim.Adam(negbin_model.parameters(), lr=config['learning_rate'])
    
    # Train for a few epochs
    for epoch in range(3):
        for batch_idx, batch in enumerate(count_dataloader):
            negbin_optimizer.zero_grad()
            loss = negbin_model.loss(
                y=batch['y'],
                time_features=batch['time_features'],
                static_features=batch['static_features']
            )
            loss.backward()
            negbin_optimizer.step()
            
            if batch_idx == 0:
                print(f"NegBin Model - Epoch [{epoch+1}/3], Loss: {loss.item():.4f}")
    
    print("\nExample usage completed successfully!")
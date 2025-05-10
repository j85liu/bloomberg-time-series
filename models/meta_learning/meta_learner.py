import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class MetaLearner(nn.Module):
    """
    Meta-learner module that learns to predict model performance based on time series characteristics.
    
    This model takes meta-features extracted from time series and predicts how well
    each forecasting model will perform on that time series.
    """
    def __init__(self, meta_feature_dim, hidden_dim=64, num_models=3):
        """
        Initialize the meta-learner.
        
        Args:
            meta_feature_dim: Dimension of input meta-features
            hidden_dim: Dimension of hidden layers
            num_models: Number of base forecasting models to select from
        """
        super().__init__()
        
        # Neural network for performance prediction
        # Architecture: Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear
        self.network = nn.Sequential(
            # First layer: meta_features → hidden
            nn.Linear(meta_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),  # Prevent overfitting
            
            # Second layer: hidden → hidden
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),  # Prevent overfitting
            
            # Output layer: hidden → num_models (one score per model)
            nn.Linear(hidden_dim, num_models)
        )
        
    def forward(self, meta_features):
        """
        Predict performance of each model based on meta-features.
        
        Args:
            meta_features: Meta-features of time series [batch_size, meta_feature_dim]
            
        Returns:
            performance: Predicted performance scores for each model [batch_size, num_models]
        """
        return self.network(meta_features)


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
    meta_features, performances = meta_database.get_dataset()
    if meta_features is None:
        print("No meta-knowledge available for training")
        return False
        
    # Create PyTorch DataLoader for batching
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
        for features, targets in dataloader:
            # Zero gradients
            optimizer.zero_grad()
            
            # Extract meta-features from input features
            meta_features = meta_feature_extractor(features)
            
            # Predict model performance based on meta-features
            pred_performance = meta_learner(meta_features)
            
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
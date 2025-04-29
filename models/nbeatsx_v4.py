import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple, Union

# -- Basis functions -- #
class TrendBasis(nn.Module):
    """Polynomial Trend Basis as used in the NBEATSx paper"""
    def __init__(self, degree: int, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        self.degree = degree

        self.backcast_basis = nn.Parameter(self._basis(backcast_size), requires_grad=False)
        self.forecast_basis = nn.Parameter(self._basis(forecast_size), requires_grad=False)

    def _basis(self, size):
        return torch.stack([torch.pow(torch.linspace(0, 1, steps=size), i) for i in range(self.degree + 1)], dim=0)

    def forward(self, theta):
        cut = self.degree + 1
        backcast = torch.einsum('bp,pt->bt', theta[:, cut:], self.backcast_basis)
        forecast = torch.einsum('bp,pt->bt', theta[:, :cut], self.forecast_basis)
        return backcast, forecast

class SeasonalityBasis(nn.Module):
    """Seasonality Basis using Fourier terms (sines and cosines)"""
    def __init__(self, harmonics: int, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        self.harmonics = harmonics

        self.backcast_basis = nn.Parameter(self._fourier_basis(backcast_size), requires_grad=False)
        self.forecast_basis = nn.Parameter(self._fourier_basis(forecast_size), requires_grad=False)

    def _fourier_basis(self, size):
        t = torch.linspace(0, 2 * np.pi, steps=size)
        basis = []
        for i in range(1, self.harmonics + 1):
            basis.append(torch.sin(i * t))
            basis.append(torch.cos(i * t))
        return torch.stack(basis, dim=0)

    def forward(self, theta):
        cut = 2 * self.harmonics
        backcast = torch.einsum('bp,pt->bt', theta[:, cut:], self.backcast_basis)
        forecast = torch.einsum('bp,pt->bt', theta[:, :cut], self.forecast_basis)
        return backcast, forecast

class IdentityBasis(nn.Module):
    """Identity Basis function to handle generic components as described in NBEATSx paper"""
    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        
        # Identity matrices for backcast and forecast
        self.backcast_basis = nn.Parameter(torch.eye(backcast_size), requires_grad=False)
        self.forecast_basis = nn.Parameter(torch.eye(forecast_size), requires_grad=False)
        
    def forward(self, theta):
        cut = self.forecast_size
        forecast = torch.einsum('bp,pt->bt', theta[:, :cut], self.forecast_basis)
        backcast = torch.einsum('bp,pt->bt', theta[:, cut:], self.backcast_basis)
        return backcast, forecast

# -- GRN for exogenous processing -- #
class GRN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, input_dim)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        residual = x
        x = self.dropout(self.elu(self.linear1(x)))
        x = self.dropout(self.elu(self.linear2(x)))
        x = self.linear3(x)
        return self.layer_norm(x + residual)  # Residual connection

# -- Interpretable N-BEATS Block -- #
class NBeatsBlock(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        exog_size: int, 
        theta_size: int, 
        hidden_units: int,
        layers: int,
        basis,
        dropout: float = 0.1
    ):
        super().__init__()
        self.exog_grn = GRN(exog_size, hidden_units, dropout) if exog_size > 0 else None

        # Stack multiple fully connected layers
        fc_layers = []
        fc_layers.append(nn.Linear(input_size + exog_size, hidden_units))
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Dropout(dropout))
        
        for _ in range(layers - 1):
            fc_layers.append(nn.Linear(hidden_units, hidden_units))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout))
            
        self.fc_stack = nn.Sequential(*fc_layers)
        self.fc_theta = nn.Linear(hidden_units, theta_size)  # outputs the basis coefficients
        self.basis = basis

    def forward(self, x, exog=None):
        if self.exog_grn and exog is not None:
            processed_exog = self.exog_grn(exog)
            x = torch.cat((x, processed_exog), dim=1)
        
        x = self.fc_stack(x)
        theta = self.fc_theta(x)
        backcast, forecast = self.basis(theta)
        return backcast, forecast, theta

# -- NBEATSx Stack -- #
class Stack(nn.Module):
    def __init__(
        self,
        input_size: int,
        forecast_size: int,
        exog_size: int,
        stack_type: str,
        num_blocks: int,
        hidden_units: int,
        layers: int,
        degree: int = None,
        harmonics: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.stack_type = stack_type
        
        # Configure basis based on stack type
        if stack_type == 'trend':
            assert degree is not None, "Trend stack requires degree parameter"
            for _ in range(num_blocks):
                basis = TrendBasis(degree, input_size, forecast_size)
                theta_size = 2 * (degree + 1)  # for both forecast and backcast
                self.blocks.append(
                    NBeatsBlock(
                        input_size=input_size,
                        exog_size=exog_size,
                        theta_size=theta_size,
                        hidden_units=hidden_units,
                        layers=layers,
                        basis=basis,
                        dropout=dropout
                    )
                )
        elif stack_type == 'seasonality':
            assert harmonics is not None, "Seasonality stack requires harmonics parameter"
            for _ in range(num_blocks):
                basis = SeasonalityBasis(harmonics, input_size, forecast_size)
                theta_size = 4 * harmonics  # for both forecast and backcast
                self.blocks.append(
                    NBeatsBlock(
                        input_size=input_size,
                        exog_size=exog_size,
                        theta_size=theta_size,
                        hidden_units=hidden_units,
                        layers=layers,
                        basis=basis,
                        dropout=dropout
                    )
                )
        elif stack_type == 'generic':
            for _ in range(num_blocks):
                basis = IdentityBasis(input_size, forecast_size)
                theta_size = input_size + forecast_size
                self.blocks.append(
                    NBeatsBlock(
                        input_size=input_size,
                        exog_size=exog_size,
                        theta_size=theta_size,
                        hidden_units=hidden_units,
                        layers=layers,
                        basis=basis,
                        dropout=dropout
                    )
                )
        else:
            raise ValueError(f"Unknown stack type: {stack_type}")
            
    def forward(self, x, exog=None, return_theta=False):
        residual = x
        forecast = 0
        thetas = []
        
        for block in self.blocks:
            backcast, block_forecast, theta = block(residual, exog)
            residual = residual - backcast
            forecast = forecast + block_forecast
            if return_theta:
                thetas.append(theta)
                
        return residual, forecast, thetas if return_theta else thetas

# -- NBEATSx Interpretable Model -- #
class NBeatsX(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        forecast_size: int,
        exog_size: int = 0, 
        stack_types: List[str] = ['trend', 'seasonality', 'generic'],
        num_blocks_per_stack: List[int] = [3, 3, 1],
        hidden_units: int = 256, 
        layers: int = 4,
        trend_degree: int = 3,
        seasonality_harmonics: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        assert len(stack_types) == len(num_blocks_per_stack), "Number of stacks must match number of block counts"
        
        self.stacks = nn.ModuleList()
        
        for stack_type, num_blocks in zip(stack_types, num_blocks_per_stack):
            self.stacks.append(
                Stack(
                    input_size=input_size,
                    forecast_size=forecast_size,
                    exog_size=exog_size,
                    stack_type=stack_type,
                    num_blocks=num_blocks,
                    hidden_units=hidden_units,
                    layers=layers,
                    degree=trend_degree if stack_type == 'trend' else None,
                    harmonics=seasonality_harmonics if stack_type == 'seasonality' else None,
                    dropout=dropout
                )
            )
            
    def forward(self, x, exog=None, return_thetas=False, return_components=False):
        residual = x
        forecast = 0
        all_thetas = []
        components = []
        
        for i, stack in enumerate(self.stacks):
            _, stack_forecast, thetas = stack(residual, exog, return_theta=True)
            
            if return_components:
                components.append(stack_forecast)
                
            # New (fixed):
            new_residual, stack_forecast, thetas = stack(residual, exog, return_theta=True)
            residual = new_residual  # Use the residual returned from the stack
            forecast = forecast + stack_forecast
            
            if return_thetas:
                all_thetas.append(thetas)
        
        if return_thetas and return_components:
            return forecast, all_thetas, components
        elif return_thetas:
            return forecast, all_thetas
        elif return_components:
            return forecast, components
        else:
            return forecast
            
# Example usage
if __name__ == "__main__":
    # Example parameters
    input_size = 24  # 24 hours of history
    forecast_size = 12  # 12 hours of forecast
    exog_size = 5  # 5 exogenous variables (e.g., temperature, calendar features, etc.)
    
    # Create model with configurable stack architecture
    model = NBeatsX(
        input_size=input_size,
        forecast_size=forecast_size,
        exog_size=exog_size,
        stack_types=['trend', 'seasonality', 'generic'],
        num_blocks_per_stack=[2, 2, 1],
        hidden_units=128,
        layers=3,
        trend_degree=4,
        seasonality_harmonics=8,
        dropout=0.1,
    )
    
    # Generate sample data
    x = torch.randn(32, input_size)  # batch of 32, input sequence of 24
    exog = torch.randn(32, exog_size)  # batch of 32, 5 exog features
    
    # Forward pass
    forecast = model(x, exog)
    print(f"Forecast shape: {forecast.shape}")  # Should be [32, 12]
    
    # Forward pass with component decomposition
    forecast, components = model(x, exog, return_components=True)
    print(f"Components: {len(components)}")  # Should be 3 (one per stack)
    for i, comp in enumerate(components):
        print(f"Component {i} shape: {comp.shape}")  # Each should be [32, 12]
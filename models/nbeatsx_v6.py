import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple, Union

class TCN(nn.Module):
    """Temporal Convolutional Network for processing exogenous variables"""
    def __init__(self, input_channels, output_channels, kernel_size=3, dropout=0.1, dilation_base=2, layers=4):
        super().__init__()
        
        self.network = nn.Sequential()
        
        # First layer has dilation=1
        padding = (kernel_size - 1)
        self.network.add_module(
            "conv_0",
            nn.Conv1d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=1
            )
        )
        self.network.add_module("relu_0", nn.ReLU())
        self.network.add_module("dropout_0", nn.Dropout(dropout))
        
        # Additional layers with increasing dilation
        for i in range(1, layers):
            dilation = dilation_base ** i
            padding = (kernel_size - 1) * dilation
            
            self.network.add_module(
                f"conv_{i}",
                nn.Conv1d(
                    in_channels=output_channels,
                    out_channels=output_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation
                )
            )
            self.network.add_module(f"relu_{i}", nn.ReLU())
            self.network.add_module(f"dropout_{i}", nn.Dropout(dropout))
        
        # Final 1x1 convolution to reduce channel size
        self.final_conv = nn.Conv1d(
            in_channels=output_channels,
            out_channels=output_channels,
            kernel_size=1
        )
    
    def forward(self, x):
        """
        Forward pass of TCN
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, input_channels]
            
        Returns:
            Processed tensor of shape [batch_size, sequence_length, output_channels]
        """
        # Transpose to [batch_size, input_channels, sequence_length] for 1D convolution
        x = x.transpose(1, 2)
        
        # Apply network
        x = self.network(x)
        x = self.final_conv(x)
        
        # Transpose back to [batch_size, sequence_length, output_channels]
        return x.transpose(1, 2)

class TrendBasis(nn.Module):
    """Polynomial Trend Basis"""
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
    """Seasonality Basis using Fourier terms"""
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
    """Identity Basis function"""
    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        
        self.backcast_basis = nn.Parameter(torch.eye(backcast_size), requires_grad=False)
        self.forecast_basis = nn.Parameter(torch.eye(forecast_size), requires_grad=False)
        
    def forward(self, theta):
        cut = self.forecast_size
        forecast = torch.einsum('bp,pt->bt', theta[:, :cut], self.forecast_basis)
        backcast = torch.einsum('bp,pt->bt', theta[:, cut:], self.backcast_basis)
        return backcast, forecast

class NBEATSxBlock(nn.Module):
    """NBEATSx block with TCN for exogenous variables"""
    def __init__(
        self, 
        input_size: int, 
        exog_channels: int, 
        theta_size: int, 
        hidden_units: int,
        layers: int,
        basis,
        dropout: float = 0.1,
        tcn_layers: int = 3
    ):
        super().__init__()
        
        # Flag indicating if we use exogenous variables
        self.has_exog = exog_channels > 0
        
        # TCN for processing exogenous variables
        if self.has_exog:
            self.exog_tcn = TCN(
                input_channels=exog_channels,
                output_channels=hidden_units,
                kernel_size=3,
                dropout=dropout,
                layers=tcn_layers
            )
            
            # Linear layer to reduce hidden units
            self.exog_linear = nn.Linear(hidden_units, hidden_units)
        
        # Main network
        fc_layers = []
        fc_layers.append(nn.Linear(input_size, hidden_units))
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Dropout(dropout))
        
        for _ in range(layers - 1):
            fc_layers.append(nn.Linear(hidden_units, hidden_units))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout))
            
        self.fc_stack = nn.Sequential(*fc_layers)
        self.fc_theta = nn.Linear(hidden_units, theta_size)
        self.basis = basis

    def forward(self, x, exog=None):
        """
        Forward pass through NBEATSx block
        
        Args:
            x: Target time series tensor [batch_size, input_size]
            exog: Exogenous variables tensor [batch_size, input_size, exog_channels]
                  or None if no exogenous variables
        """
        # Get batch size
        batch_size = x.size(0)
        
        # Process main input
        h = self.fc_stack(x)
        
        # Process exogenous variables if available
        if self.has_exog and exog is not None:
            # Apply TCN to get encoded representation
            exog_encoded = self.exog_tcn(exog)
            
            # Average pooling over sequence length
            exog_pooled = torch.mean(exog_encoded, dim=1)
            
            # Project to match hidden size
            exog_hidden = self.exog_linear(exog_pooled)
            
            # Add to main hidden state (residual connection)
            h = h + exog_hidden
        
        # Get theta coefficients
        theta = self.fc_theta(h)
        
        # Apply basis functions
        backcast, forecast = self.basis(theta)
        
        return backcast, forecast, theta

class NBEATSxStack(nn.Module):
    """Stack for NBEATSx with multiple blocks"""
    def __init__(
        self,
        input_size: int,
        forecast_size: int,
        exog_channels: int,
        stack_type: str,
        num_blocks: int,
        hidden_units: int,
        layers: int,
        degree: int = None,
        harmonics: int = None,
        dropout: float = 0.1,
        tcn_layers: int = 3
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
                    NBEATSxBlock(
                        input_size=input_size,
                        exog_channels=exog_channels,
                        theta_size=theta_size,
                        hidden_units=hidden_units,
                        layers=layers,
                        basis=basis,
                        dropout=dropout,
                        tcn_layers=tcn_layers
                    )
                )
        elif stack_type == 'seasonality':
            assert harmonics is not None, "Seasonality stack requires harmonics parameter"
            for _ in range(num_blocks):
                basis = SeasonalityBasis(harmonics, input_size, forecast_size)
                theta_size = 4 * harmonics  # for both forecast and backcast
                self.blocks.append(
                    NBEATSxBlock(
                        input_size=input_size,
                        exog_channels=exog_channels,
                        theta_size=theta_size,
                        hidden_units=hidden_units,
                        layers=layers,
                        basis=basis,
                        dropout=dropout,
                        tcn_layers=tcn_layers
                    )
                )
        elif stack_type == 'generic':
            for _ in range(num_blocks):
                basis = IdentityBasis(input_size, forecast_size)
                theta_size = input_size + forecast_size
                self.blocks.append(
                    NBEATSxBlock(
                        input_size=input_size,
                        exog_channels=exog_channels,
                        theta_size=theta_size,
                        hidden_units=hidden_units,
                        layers=layers,
                        basis=basis,
                        dropout=dropout,
                        tcn_layers=tcn_layers
                    )
                )
        else:
            raise ValueError(f"Unknown stack type: {stack_type}")
            
    def forward(self, x, exog=None, return_theta=False):
        """
        Forward pass through NBEATSx stack
        
        Args:
            x: Target time series tensor [batch_size, input_size]
            exog: Exogenous variables tensor [batch_size, input_size, exog_channels]
                  or None if no exogenous variables
            return_theta: Whether to return the basis coefficients
        """
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

class NBEATSx(nn.Module):
    """Neural Basis Expansion Analysis with Exogenous Variables
    
    Implementation according to the paper:
    Olivares et al. (2023) "Neural basis expansion analysis with exogenous variables:
    Forecasting electricity prices with NBEATSx"
    """
    def __init__(
        self, 
        input_size: int, 
        forecast_size: int,
        exog_channels: int = 0, 
        stack_types: List[str] = ['trend', 'seasonality', 'generic'],
        num_blocks_per_stack: List[int] = [3, 3, 1],
        hidden_units: int = 256, 
        layers: int = 4,
        trend_degree: int = 3,
        seasonality_harmonics: int = 5,
        dropout: float = 0.1,
        tcn_layers: int = 3
    ):
        super().__init__()
        
        assert len(stack_types) == len(num_blocks_per_stack), "Number of stacks must match number of block counts"
        
        self.stacks = nn.ModuleList()
        self.input_size = input_size
        self.exog_channels = exog_channels
        
        for stack_type, num_blocks in zip(stack_types, num_blocks_per_stack):
            self.stacks.append(
                NBEATSxStack(
                    input_size=input_size,
                    forecast_size=forecast_size,
                    exog_channels=exog_channels,
                    stack_type=stack_type,
                    num_blocks=num_blocks,
                    hidden_units=hidden_units,
                    layers=layers,
                    degree=trend_degree if stack_type == 'trend' else None,
                    harmonics=seasonality_harmonics if stack_type == 'seasonality' else None,
                    dropout=dropout,
                    tcn_layers=tcn_layers
                )
            )
            
    def forward(self, x, exog=None, return_thetas=False, return_components=False):
        """
        Forward pass through NBEATSx model
        
        Args:
            x: Target time series tensor [batch_size, input_size]
            exog: Exogenous variables tensor [batch_size, input_size, exog_channels]
                  or None if no exogenous variables
            return_thetas: Whether to return the basis coefficients
            return_components: Whether to return the individual stack outputs
        
        Returns:
            forecast: Forecasted values
            all_thetas: (optional) Basis coefficients
            components: (optional) Individual stack outputs
        """
        # Validate input shapes
        batch_size = x.size(0)
        
        # Handle different input formats
        if len(x.shape) == 3 and x.shape[2] == 1:
            # If input is [batch_size, input_size, 1], reshape to [batch_size, input_size]
            x = x.squeeze(-1)
        
        # Check exogenous variables
        if exog is not None:
            if len(exog.shape) != 3 or exog.shape[1] != self.input_size:
                raise ValueError(f"Expected exogenous variables of shape [batch_size, {self.input_size}, exog_channels], "
                                f"but got {exog.shape}")
            if exog.shape[2] != self.exog_channels and self.exog_channels > 0:
                raise ValueError(f"Expected {self.exog_channels} exogenous channels, "
                               f"but got {exog.shape[2]}")
        
        residual = x
        forecast = 0
        all_thetas = []
        components = []
        
        for i, stack in enumerate(self.stacks):
            # First pass for components
            if return_components:
                _, stack_forecast, _ = stack(residual, exog, return_theta=False)
                components.append(stack_forecast)
            
            # Main pass for forecast
            new_residual, stack_forecast, thetas = stack(residual, exog, return_theta=True)
            residual = new_residual
            forecast = forecast + stack_forecast
            
            if return_thetas:
                all_thetas.append(thetas)
        
        # Return appropriate outputs
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
    input_size = 24  # Input sequence length
    forecast_size = 12  # Forecast horizon
    exog_channels = 5  # Number of exogenous variables
    
    # Create model
    model = NBEATSx(
        input_size=input_size,
        forecast_size=forecast_size,
        exog_channels=exog_channels,
        stack_types=['trend', 'seasonality', 'generic'],
        num_blocks_per_stack=[2, 2, 1],
        hidden_units=128,
        layers=3,
        trend_degree=4,
        seasonality_harmonics=8,
        dropout=0.1,
    )
    
    # Generate sample data
    batch_size = 32
    x = torch.randn(batch_size, input_size)  # Target series input
    exog = torch.randn(batch_size, input_size, exog_channels)  # Exogenous variables
    
    # Forward pass
    forecast = model(x, exog)
    print(f"Forecast shape: {forecast.shape}")  # Should be [batch_size, forecast_size]
    
    # Forward pass with component decomposition
    forecast, components = model(x, exog, return_components=True)
    print(f"Components: {len(components)}")  # Should be 3 (one per stack)
    for i, comp in enumerate(components):
        print(f"Component {i} shape: {comp.shape}")  # Each should be [batch_size, forecast_size]
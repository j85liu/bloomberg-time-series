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

# -- Temporal GRN for time-varying exogenous variables -- #
class TemporalGRN(nn.Module):
    """GRN that processes exogenous variables across time"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Feature-wise processing layers
        self.feature_fc1 = nn.Linear(input_dim, hidden_dim)
        self.feature_fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Temporal attention mechanism
        self.attn_q = nn.Linear(hidden_dim, hidden_dim)
        self.attn_k = nn.Linear(hidden_dim, hidden_dim)
        self.attn_v = nn.Linear(hidden_dim, hidden_dim)
        self.attn_scale = nn.Parameter(torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float32)), requires_grad=False)
        
        # Output projection
        self.output_fc = nn.Linear(hidden_dim, output_dim)
        
        # Other components
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        """
        Args:
            x: Time-varying exogenous variables [batch_size, seq_length, input_dim]
        Returns:
            processed_x: Processed variables [batch_size, output_dim]
        """
        batch_size, seq_len, _ = x.size()
        
        # Process features at each time step
        x_hidden = self.feature_fc1(x)  # [batch_size, seq_len, hidden_dim]
        x_hidden = self.elu(x_hidden)
        x_hidden = self.dropout(x_hidden)
        x_hidden = self.feature_fc2(x_hidden)  # [batch_size, seq_len, hidden_dim]
        x_hidden = self.layer_norm1(x_hidden)
        
        # Self-attention over time
        q = self.attn_q(x_hidden)  # [batch_size, seq_len, hidden_dim]
        k = self.attn_k(x_hidden)  # [batch_size, seq_len, hidden_dim]
        v = self.attn_v(x_hidden)  # [batch_size, seq_len, hidden_dim]
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.attn_scale  # [batch_size, seq_len, seq_len]
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Apply attention weights
        context = torch.matmul(attn_weights, v)  # [batch_size, seq_len, hidden_dim]
        
        # Pool over time dimension (weighted average)
        # We could use last token, mean, or a learnable pooling
        context_pooled = context.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Final output projection
        output = self.output_fc(context_pooled)  # [batch_size, output_dim]
        output = self.layer_norm2(output)
        
        return output

# -- Improved N-BEATS Block with support for time-varying exogenous variables -- #
class ImprovedNBeatsBlock(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        exog_channels: int,  # Number of exogenous features per time step
        theta_size: int, 
        hidden_units: int,
        layers: int,
        basis,
        dropout: float = 0.1,
        use_time_varying_exog: bool = True
    ):
        super().__init__()
        
        # For time-varying exogenous variables
        self.use_time_varying_exog = use_time_varying_exog
        if use_time_varying_exog and exog_channels > 0:
            self.temporal_grn = TemporalGRN(
                input_dim=exog_channels,  
                hidden_dim=hidden_units,
                output_dim=hidden_units,  # Output is added to input representation
                dropout=dropout
            )
            main_input_size = input_size
        else:
            # Traditional static exogenous variable processing
            self.exog_grn = GRN(exog_channels, hidden_units, dropout) if exog_channels > 0 else None
            main_input_size = input_size + exog_channels if exog_channels > 0 else input_size
        
        # Main network
        fc_layers = []
        fc_layers.append(nn.Linear(main_input_size, hidden_units))
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
        """
        Args:
            x: Input time series [batch_size, input_size]
            exog: Exogenous variables, now supported in two formats:
                 - Static: [batch_size, exog_channels]
                 - Time-varying: [batch_size, input_size, exog_channels]
        """
        batch_size = x.size(0)
        
        if self.use_time_varying_exog and exog is not None:
            # Process time-varying exogenous variables
            # exog shape: [batch_size, input_size, exog_channels]
            exog_repr = self.temporal_grn(exog)  # [batch_size, hidden_units]
            
            # Main network forward pass
            x_hidden = self.fc_stack(x)  # [batch_size, hidden_units]
            
            # Add exogenous representation (residual connection)
            hidden = x_hidden + exog_repr  # [batch_size, hidden_units]
        else:
            # Traditional approach for static exogenous variables
            if self.exog_grn and exog is not None:
                # Process static exogenous variables
                processed_exog = self.exog_grn(exog)  # [batch_size, exog_channels]
                combined = torch.cat((x, processed_exog), dim=1)  # [batch_size, input_size + exog_channels]
                hidden = self.fc_stack(combined)  # [batch_size, hidden_units]
            else:
                # No exogenous variables
                hidden = self.fc_stack(x)  # [batch_size, hidden_units]
        
        # Output theta coefficients
        theta = self.fc_theta(hidden)  # [batch_size, theta_size]
        
        # Apply basis functions
        backcast, forecast = self.basis(theta)
        
        return backcast, forecast, theta

# -- Traditional GRN for static exogenous variables -- #
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

# -- Improved NBEATSx Stack with support for time-varying exogenous variables -- #
class ImprovedStack(nn.Module):
    def __init__(
        self,
        input_size: int,
        forecast_size: int,
        exog_channels: int,  # Number of exogenous variables per time step
        stack_type: str,
        num_blocks: int,
        hidden_units: int,
        layers: int,
        degree: int = None,
        harmonics: int = None,
        dropout: float = 0.1,
        use_time_varying_exog: bool = True
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
                    ImprovedNBeatsBlock(
                        input_size=input_size,
                        exog_channels=exog_channels,
                        theta_size=theta_size,
                        hidden_units=hidden_units,
                        layers=layers,
                        basis=basis,
                        dropout=dropout,
                        use_time_varying_exog=use_time_varying_exog
                    )
                )
        elif stack_type == 'seasonality':
            assert harmonics is not None, "Seasonality stack requires harmonics parameter"
            for _ in range(num_blocks):
                basis = SeasonalityBasis(harmonics, input_size, forecast_size)
                theta_size = 4 * harmonics  # for both forecast and backcast
                self.blocks.append(
                    ImprovedNBeatsBlock(
                        input_size=input_size,
                        exog_channels=exog_channels,
                        theta_size=theta_size,
                        hidden_units=hidden_units,
                        layers=layers,
                        basis=basis,
                        dropout=dropout,
                        use_time_varying_exog=use_time_varying_exog
                    )
                )
        elif stack_type == 'generic':
            for _ in range(num_blocks):
                basis = IdentityBasis(input_size, forecast_size)
                theta_size = input_size + forecast_size
                self.blocks.append(
                    ImprovedNBeatsBlock(
                        input_size=input_size,
                        exog_channels=exog_channels,
                        theta_size=theta_size,
                        hidden_units=hidden_units,
                        layers=layers,
                        basis=basis,
                        dropout=dropout,
                        use_time_varying_exog=use_time_varying_exog
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

# -- Improved NBEATSx Model with support for time-varying exogenous variables -- #
class ImprovedNBeatsX(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        forecast_size: int,
        exog_channels: int = 0,  # Number of exogenous features per time step
        stack_types: List[str] = ['trend', 'seasonality', 'generic'],
        num_blocks_per_stack: List[int] = [3, 3, 1],
        hidden_units: int = 256, 
        layers: int = 4,
        trend_degree: int = 3,
        seasonality_harmonics: int = 5,
        dropout: float = 0.1,
        use_time_varying_exog: bool = True
    ):
        super().__init__()
        
        assert len(stack_types) == len(num_blocks_per_stack), "Number of stacks must match number of block counts"
        
        self.stacks = nn.ModuleList()
        self.use_time_varying_exog = use_time_varying_exog
        
        for stack_type, num_blocks in zip(stack_types, num_blocks_per_stack):
            self.stacks.append(
                ImprovedStack(
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
                    use_time_varying_exog=use_time_varying_exog
                )
            )
            
    def forward(self, x, exog=None, return_thetas=False, return_components=False):
        """
        Forward pass of the improved NBEATSx model
        
        Args:
            x: Input time series [batch_size, input_size]
            exog: Exogenous variables, now supported in two formats:
                  - Static: [batch_size, exog_channels]
                  - Time-varying: [batch_size, input_size, exog_channels]
            return_thetas: Whether to return the basis coefficients
            return_components: Whether to return the individual stack outputs
        """
        # Handle input dimensions
        if len(x.shape) == 3 and x.shape[2] == 1:
            # If input is [batch_size, input_size, 1], reshape to [batch_size, input_size]
            x = x.squeeze(-1)
        
        # Validate exogenous variable dimensions if time-varying mode is enabled
        if self.use_time_varying_exog and exog is not None:
            if len(exog.shape) == 2:
                # If exogenous variables are provided as [batch_size, exog_channels],
                # reshape them to [batch_size, input_size, exog_channels]
                batch_size, exog_channels = exog.shape
                exog = exog.unsqueeze(1).expand(-1, x.shape[1], -1)
        
        residual = x
        forecast = 0
        all_thetas = []
        components = []
        
        for i, stack in enumerate(self.stacks):
            # First pass to get components for visualization
            if return_components:
                _, stack_forecast, _ = stack(residual, exog, return_theta=False)
                components.append(stack_forecast)
            
            # Main pass for actual forecasting
            new_residual, stack_forecast, thetas = stack(residual, exog, return_theta=True)
            residual = new_residual
            forecast = forecast + stack_forecast
            
            if return_thetas:
                all_thetas.append(thetas)
        
        # Return appropriate outputs based on flags
        if return_thetas and return_components:
            return forecast, all_thetas, components
        elif return_thetas:
            return forecast, all_thetas
        elif return_components:
            return forecast, components
        else:
            return forecast
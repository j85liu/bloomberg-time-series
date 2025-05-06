import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Union

class Chomp1d(nn.Module):
    """
    Removes the last elements of a time series.
    Used to maintain causality in temporal convolutional networks.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    """
    Temporal block for TCN with residual connections
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size, 
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size, 
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        # Residual connection if input and output dimensions differ
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """Initialize weights with He initialization"""
        nn.init.kaiming_normal_(self.conv1.weight.data, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight.data, nonlinearity='relu')
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight.data, nonlinearity='relu')

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    """
    Enhanced TCN implementation with residual connections
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(
                in_channels, out_channels, kernel_size, stride=1, 
                dilation=dilation_size, padding=(kernel_size-1) * dilation_size,
                dropout=dropout
            )]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: [batch_size, channels, sequence_length]
        """
        return self.network(x)

class WaveNetModule(nn.Module):
    """
    WaveNet-style module for processing exogenous variables
    """
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2, num_levels=4):
        super(WaveNetModule, self).__init__()
        # Shape of (1, num_inputs, 1) to broadcast over batch and time
        self.weight = nn.Parameter(torch.Tensor(1, num_inputs, 1), requires_grad=True)
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(0.5))
        
        layers = []
        # First layer
        padding = (kernel_size - 1) * (2**0)
        layers += [
            nn.Conv1d(num_inputs, num_channels, kernel_size, padding=padding, dilation=2**0),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout)
        ]
        
        # Additional layers with increasing dilation
        for i in range(1, num_levels):
            dilation = 2**i
            padding = (kernel_size - 1) * dilation
            layers += [
                nn.Conv1d(num_channels, num_channels, kernel_size, padding=padding, dilation=dilation),
                Chomp1d(padding),
                nn.ReLU()
            ]
            
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: [batch_size, channels, sequence_length]
        """
        x = x * self.weight  # Element-wise multiplication for variable importance
        return self.network(x)

class TrendBasis(nn.Module):
    """Polynomial Trend Basis"""
    def __init__(self, degree: int, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        self.degree = degree

        # Create polynomial bases
        backcast_basis = []
        forecast_basis = []
        
        for i in range(degree + 1):
            backcast_basis.append((torch.linspace(0, 1, steps=backcast_size) ** i).unsqueeze(0))
            forecast_basis.append((torch.linspace(0, 1, steps=forecast_size) ** i).unsqueeze(0))
            
        self.backcast_basis = nn.Parameter(torch.cat(backcast_basis, dim=0), requires_grad=False)
        self.forecast_basis = nn.Parameter(torch.cat(forecast_basis, dim=0), requires_grad=False)

    def forward(self, theta):
        cut = self.degree + 1
        backcast = torch.einsum('bp,pt->bt', theta[:, cut:], self.backcast_basis)
        forecast = torch.einsum('bp,pt->bt', theta[:, :cut], self.forecast_basis)
        return backcast, forecast

class SeasonalityBasis(nn.Module):
    """Seasonality Basis using Fourier terms"""
    def __init__(self, harmonics: int, backcast_size: int, forecast_size: int):
        super().__init__()
        self.harmonics = harmonics
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        
        # Create Fourier bases
        backcast_basis = []
        forecast_basis = []
        
        backcast_t = torch.linspace(0, 2 * np.pi, steps=backcast_size)
        forecast_t = torch.linspace(0, 2 * np.pi, steps=forecast_size)
        
        for i in range(1, harmonics + 1):
            backcast_basis.append(torch.sin(i * backcast_t).unsqueeze(0))
            backcast_basis.append(torch.cos(i * backcast_t).unsqueeze(0))
            forecast_basis.append(torch.sin(i * forecast_t).unsqueeze(0))
            forecast_basis.append(torch.cos(i * forecast_t).unsqueeze(0))
            
        self.backcast_basis = nn.Parameter(torch.cat(backcast_basis, dim=0), requires_grad=False)
        self.forecast_basis = nn.Parameter(torch.cat(forecast_basis, dim=0), requires_grad=False)

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
        
        # Using identity matrices directly
        self.forecast_basis = nn.Parameter(torch.eye(forecast_size), requires_grad=False)
        self.backcast_basis = nn.Parameter(torch.eye(backcast_size), requires_grad=False)
        
    def forward(self, theta):
        cut = self.forecast_size
        forecast = torch.einsum('bp,pt->bt', theta[:, :cut], self.forecast_basis)
        backcast = torch.einsum('bp,pt->bt', theta[:, cut:], self.backcast_basis)
        return backcast, forecast

class ExogenousBasisInterpretable(nn.Module):
    """
    Interpretable basis for exogenous variables - applies learned weights directly
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, theta, insample_x_t, outsample_x_t):
        # insample_x_t: [batch, channels, time]
        # outsample_x_t: [batch, channels, time]
        # theta: [batch, 2*channels]
        
        backcast_basis = insample_x_t
        forecast_basis = outsample_x_t
        
        cut_point = forecast_basis.shape[1]
        backcast = torch.einsum('bp,bpt->bt', theta[:, cut_point:], backcast_basis)
        forecast = torch.einsum('bp,bpt->bt', theta[:, :cut_point], forecast_basis)
        
        return backcast, forecast

class ExogenousBasisWavenet(nn.Module):
    """
    WaveNet-based basis for exogenous variables
    """
    def __init__(self, out_channels, in_channels, num_levels=4, kernel_size=3, dropout=0.1):
        super().__init__()
        # Learnable weights for each input channel
        self.weight = nn.Parameter(torch.Tensor(1, in_channels, 1), requires_grad=True)
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(0.5))
        
        # First layer (input conv)
        padding = (kernel_size - 1) * (2**0)
        layers = [
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=2**0),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout)
        ]
        
        # Additional layers with increasing dilation
        for i in range(1, num_levels):
            dilation = 2**i
            padding = (kernel_size - 1) * dilation
            layers += [
                nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
                Chomp1d(padding),
                nn.ReLU()
            ]
        
        self.wavenet = nn.Sequential(*layers)
    
    def transform(self, insample_x_t, outsample_x_t):
        # Concatenate along time dimension
        input_size = insample_x_t.shape[2]
        x_t = torch.cat([insample_x_t, outsample_x_t], dim=2)
        
        # Apply channel weighting and WaveNet
        x_t = x_t * self.weight  # Element-wise multiplication with learned weights
        x_t = self.wavenet(x_t)
        
        # Split back into backcast and forecast parts
        backcast_basis = x_t[:, :, :input_size]
        forecast_basis = x_t[:, :, input_size:]
        
        return backcast_basis, forecast_basis
        
    def forward(self, theta, insample_x_t, outsample_x_t):
        backcast_basis, forecast_basis = self.transform(insample_x_t, outsample_x_t)
        
        cut_point = forecast_basis.shape[1]
        backcast = torch.einsum('bp,bpt->bt', theta[:, cut_point:], backcast_basis)
        forecast = torch.einsum('bp,bpt->bt', theta[:, :cut_point], forecast_basis)
        
        return backcast, forecast

class ExogenousBasisTCN(nn.Module):
    """
    TCN-based basis for exogenous variables
    """
    def __init__(self, out_channels, in_channels, num_levels=4, kernel_size=2, dropout=0.1):
        super().__init__()
        # Create TCN with increasing channel size
        tcn_channels = [out_channels] * num_levels
        self.tcn = TemporalConvNet(in_channels, tcn_channels, kernel_size, dropout)
    
    def transform(self, insample_x_t, outsample_x_t):
        # Concatenate along time dimension
        input_size = insample_x_t.shape[2]
        x_t = torch.cat([insample_x_t, outsample_x_t], dim=2)
        
        # Apply TCN
        x_t = self.tcn(x_t)
        
        # Split back into backcast and forecast parts
        backcast_basis = x_t[:, :, :input_size]
        forecast_basis = x_t[:, :, input_size:]
        
        return backcast_basis, forecast_basis
        
    def forward(self, theta, insample_x_t, outsample_x_t):
        backcast_basis, forecast_basis = self.transform(insample_x_t, outsample_x_t)
        
        cut_point = forecast_basis.shape[1]
        backcast = torch.einsum('bp,bpt->bt', theta[:, cut_point:], backcast_basis)
        forecast = torch.einsum('bp,bpt->bt', theta[:, :cut_point], forecast_basis)
        
        return backcast, forecast

class NBEATSxBlock(nn.Module):
    """Enhanced NBEATSx block with sophisticated exogenous variable handling"""
    def __init__(
        self, 
        input_size: int, 
        forecast_size: int,
        exog_channels: int, 
        theta_size: int, 
        hidden_units: int,
        layers: int,
        basis_type: str,
        basis_kwargs: dict = {},
        dropout: float = 0.1,
        exog_mode: str = 'tcn'  # 'tcn', 'wavenet', or 'interpretable'
    ):
        super().__init__()
        
        # Flag indicating if we use exogenous variables
        self.has_exog = exog_channels > 0
        self.exog_mode = exog_mode if self.has_exog else None
        self.basis_type = basis_type
        self.input_size = input_size
        self.forecast_size = forecast_size
        
        # Determine basis function
        if basis_type == 'trend':
            self.basis = TrendBasis(
                degree=basis_kwargs.get('degree', 3),
                backcast_size=input_size,
                forecast_size=forecast_size
            )
        elif basis_type == 'seasonality':
            self.basis = SeasonalityBasis(
                harmonics=basis_kwargs.get('harmonics', 5),
                backcast_size=input_size,
                forecast_size=forecast_size
            )
        elif basis_type == 'generic':
            self.basis = IdentityBasis(
                backcast_size=input_size,
                forecast_size=forecast_size
            )
        else:
            raise ValueError(f"Unknown basis type: {basis_type}")
        
        # Exogenous variable processing
        if self.has_exog:
            if exog_mode == 'tcn':
                self.exog_basis = ExogenousBasisTCN(
                    out_channels=hidden_units,
                    in_channels=exog_channels,
                    num_levels=basis_kwargs.get('tcn_levels', 4),
                    kernel_size=basis_kwargs.get('tcn_kernel_size', 3),
                    dropout=dropout
                )
            elif exog_mode == 'wavenet':
                self.exog_basis = ExogenousBasisWavenet(
                    out_channels=hidden_units,
                    in_channels=exog_channels,
                    num_levels=basis_kwargs.get('wavenet_levels', 4),
                    kernel_size=basis_kwargs.get('wavenet_kernel_size', 3),
                    dropout=dropout
                )
            elif exog_mode == 'interpretable':
                self.exog_basis = ExogenousBasisInterpretable()
            else:
                raise ValueError(f"Unknown exogenous mode: {exog_mode}")
                
            # Additional parameters for exogenous variables
            self.exog_theta_size = 2 * exog_channels if exog_mode == 'interpretable' else 2 * hidden_units
        
        # Main network for target variable
        fc_layers = []
        fc_layers.append(nn.Linear(input_size, hidden_units))
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Dropout(dropout))
        
        for _ in range(layers - 1):
            fc_layers.append(nn.Linear(hidden_units, hidden_units))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout))
            
        self.fc_stack = nn.Sequential(*fc_layers)
        
        # Theta generator for main basis
        self.fc_theta = nn.Linear(hidden_units, theta_size)
        
        # Additional network for exogenous variable theta if needed
        if self.has_exog:
            self.fc_exog_theta = nn.Linear(hidden_units, self.exog_theta_size)

    def forward(self, x, exog=None):
        """
        Enhanced forward pass through NBEATSx block
        
        Args:
            x: Target time series tensor [batch_size, input_size]
            exog: Exogenous variables tensor [batch_size, input_size+forecast_size, exog_channels]
                  or None if no exogenous variables
        
        Returns:
            backcast: Reconstruction of input
            forecast: Prediction for future values
        """
        # Process main input
        h = self.fc_stack(x)
        
        # Get theta coefficients for main basis
        theta = self.fc_theta(h)
        
        # Apply main basis function
        backcast, forecast = self.basis(theta)
        
        # Process exogenous variables if available
        if self.has_exog and exog is not None:
            # Get exogenous part for input and forecast periods
            input_size = self.input_size
            insample_exog = exog[:, :input_size, :].transpose(1, 2)  # [batch, channels, time]
            outsample_exog = exog[:, input_size:, :].transpose(1, 2)  # [batch, channels, time]
            
            # Get theta for exogenous basis
            exog_theta = self.fc_exog_theta(h)
            
            # Apply exogenous basis
            exog_backcast, exog_forecast = self.exog_basis(
                exog_theta, insample_exog, outsample_exog
            )
            
            # Add contributions to main forecast and backcast
            backcast = backcast + exog_backcast
            forecast = forecast + exog_forecast
        
        return backcast, forecast

class NBEATSxStack(nn.Module):
    """Stack of NBEATSx blocks with the same type"""
    def __init__(
        self,
        input_size: int,
        forecast_size: int,
        exog_channels: int,
        stack_type: str,
        num_blocks: int,
        hidden_units: int,
        layers: int,
        basis_kwargs: dict = {},
        dropout: float = 0.1,
        exog_mode: str = 'tcn'
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.stack_type = stack_type
        
        # Determine theta size based on stack type
        if stack_type == 'trend':
            degree = basis_kwargs.get('degree', 3)
            theta_size = 2 * (degree + 1)  # for both forecast and backcast
        elif stack_type == 'seasonality':
            harmonics = basis_kwargs.get('harmonics', 5)
            theta_size = 4 * harmonics  # for both forecast and backcast
        elif stack_type == 'generic':
            theta_size = input_size + forecast_size
        else:
            raise ValueError(f"Unknown stack type: {stack_type}")
            
        # Create blocks
        for _ in range(num_blocks):
            self.blocks.append(
                NBEATSxBlock(
                    input_size=input_size,
                    forecast_size=forecast_size,
                    exog_channels=exog_channels,
                    theta_size=theta_size,
                    hidden_units=hidden_units,
                    layers=layers,
                    basis_type=stack_type,
                    basis_kwargs=basis_kwargs,
                    dropout=dropout,
                    exog_mode=exog_mode
                )
            )
            
    def forward(self, x, exog=None, return_block_outputs=False):
        """
        Forward pass through NBEATSx stack
        
        Args:
            x: Target time series tensor [batch_size, input_size]
            exog: Exogenous variables tensor [batch_size, input_size+forecast_size, exog_channels]
                  or None if no exogenous variables
            return_block_outputs: Whether to return individual block outputs
        """
        residual = x
        forecast = torch.zeros(x.size(0), self.blocks[0].forecast_size, device=x.device)
        block_forecasts = []
        
        for block in self.blocks:
            backcast, block_forecast = block(residual, exog)
            residual = residual - backcast
            forecast = forecast + block_forecast
            
            if return_block_outputs:
                block_forecasts.append(block_forecast)
                
        if return_block_outputs:
            return residual, forecast, block_forecasts
        else:
            return residual, forecast

class NBEATSx(nn.Module):
    """
    Enhanced Neural Basis Expansion Analysis with Exogenous Variables
    
    Integrates sophisticated exogenous variable handling from original N-BEATS model
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
        basis_kwargs: dict = {},
        dropout: float = 0.1,
        exog_mode: str = 'tcn'  # 'tcn', 'wavenet', or 'interpretable'
    ):
        super().__init__()
        
        assert len(stack_types) == len(num_blocks_per_stack), "Number of stacks must match number of block counts"
        
        self.stacks = nn.ModuleList()
        self.input_size = input_size
        self.forecast_size = forecast_size
        self.exog_channels = exog_channels
        
        # Set default values for basis parameters if not provided
        if 'degree' not in basis_kwargs:
            basis_kwargs['degree'] = 3
        if 'harmonics' not in basis_kwargs:
            basis_kwargs['harmonics'] = 5
        
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
                    basis_kwargs=basis_kwargs,
                    dropout=dropout,
                    exog_mode=exog_mode
                )
            )
            
    def forward(self, x, exog=None, return_decomposition=False):
        """
        Forward pass through enhanced NBEATSx model
        
        Args:
            x: Target time series tensor [batch_size, input_size]
            exog: Exogenous variables tensor [batch_size, input_size+forecast_size, exog_channels]
                  or None if no exogenous variables
            return_decomposition: Whether to return the individual stack outputs
        
        Returns:
            forecast: Forecasted values
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
            expected_exog_length = self.input_size + self.forecast_size
            if len(exog.shape) != 3 or exog.shape[1] != expected_exog_length:
                raise ValueError(f"Expected exogenous variables of shape [batch_size, {expected_exog_length}, exog_channels], "
                                f"but got {exog.shape}")
            if exog.shape[2] != self.exog_channels and self.exog_channels > 0:
                raise ValueError(f"Expected {self.exog_channels} exogenous channels, "
                               f"but got {exog.shape[2]}")
        
        residual = x
        forecast = torch.zeros(batch_size, self.forecast_size, device=x.device)
        components = []
        
        # Create copies of the input for decomposition to avoid modifying the original
        original_residual = residual.clone() if return_decomposition else None
        
        for i, stack in enumerate(self.stacks):
            # For decomposition, process each stack independently from original input
            if return_decomposition:
                temp_residual = original_residual.clone()
                _, stack_component = stack(temp_residual, exog)
                components.append(stack_component)
            
            # For the main forecast, process sequentially through stacks
            residual, stack_forecast = stack(residual, exog)
            forecast = forecast + stack_forecast
        
        if return_decomposition:
            return forecast, components
        else:
            return forecast

# Example usage
if __name__ == "__main__":
    # Example parameters
    input_size = 24  # Input sequence length
    forecast_size = 12  # Forecast horizon
    exog_channels = 5  # Number of exogenous variables
    
    # Create model with TCN-based exogenous handling
    model_tcn = NBEATSx(
        input_size=input_size,
        forecast_size=forecast_size,
        exog_channels=exog_channels,
        stack_types=['trend', 'seasonality', 'generic'],
        num_blocks_per_stack=[2, 2, 1],
        hidden_units=128,
        layers=3,
        basis_kwargs={
            'degree': 4,
            'harmonics': 8,
            'tcn_levels': 4,
            'tcn_kernel_size': 3
        },
        dropout=0.1,
        exog_mode='tcn'
    )
    
    # Create model with WaveNet-based exogenous handling
    model_wavenet = NBEATSx(
        input_size=input_size,
        forecast_size=forecast_size,
        exog_channels=exog_channels,
        stack_types=['trend', 'seasonality', 'generic'],
        num_blocks_per_stack=[2, 2, 1],
        hidden_units=128,
        layers=3,
        basis_kwargs={
            'degree': 4,
            'harmonics': 8,
            'wavenet_levels': 4,
            'wavenet_kernel_size': 3
        },
        dropout=0.1,
        exog_mode='wavenet'
    )
    
    # Generate sample data
    batch_size = 32
    x = torch.randn(batch_size, input_size)  # Target series input
    exog = torch.randn(batch_size, input_size + forecast_size, exog_channels)  # Exogenous variables
    
    # Test forward pass with TCN exogenous handling
    forecast_tcn = model_tcn(x, exog)
    print(f"TCN forecast shape: {forecast_tcn.shape}")  # Should be [batch_size, forecast_size]
    
    # Test forward pass with WaveNet exogenous handling
    forecast_wavenet = model_wavenet(x, exog)
    print(f"WaveNet forecast shape: {forecast_wavenet.shape}")  # Should be [batch_size, forecast_size]
    
    # Test decomposition
    forecast, components = model_tcn(x, exog, return_decomposition=True)
    print(f"Components: {len(components)}")  # Should be 3 (one per stack)
    for i, comp in enumerate(components):
        print(f"Component {i} ({model_tcn.stacks[i].stack_type}) shape: {comp.shape}")
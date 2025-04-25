import torch
import torch.nn as nn
import numpy as np

# -- Basis functions -- #
class TrendBasis(nn.Module):
    """Polynomial Trend Basis as used in the N-BEATS paper"""
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


# -- GRN for exogenous processing -- #
class GRN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = self.elu(x)
        x = self.linear2(x)
        return self.layer_norm(x + residual)  # Residual connection


# -- Interpretable N-BEATS Block -- #
class NBeatsBlock(nn.Module):
    def __init__(self, input_size, exog_size, theta_size, hidden_units, basis):
        super().__init__()
        self.exog_grn = GRN(exog_size, hidden_units) if exog_size > 0 else None

        self.fc1 = nn.Linear(input_size + exog_size, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc_theta = nn.Linear(hidden_units, theta_size)  # outputs the basis coefficients
        self.basis = basis
        self.relu = nn.ReLU()

    def forward(self, x, exog):
        if self.exog_grn and exog is not None:
            exog = self.exog_grn(exog)
            x = torch.cat((x, exog), dim=1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        theta = self.fc_theta(x)
        backcast, forecast = self.basis(theta)
        return backcast, forecast


# -- N-BEATSx Interpretable Model -- #
class NBeatsX(nn.Module):
    def __init__(self, input_size, exog_size=0, forecast_size=1, hidden_units=256, num_blocks=3):
        super().__init__()
        degree = 3  # Polynomial degree for trend
        theta_size = 2 * (degree + 1)

        self.blocks = nn.ModuleList([
            NBeatsBlock(
                input_size=input_size,
                exog_size=exog_size,
                theta_size=theta_size,
                hidden_units=hidden_units,
                basis=TrendBasis(degree, input_size, forecast_size)
            ) for _ in range(num_blocks)
        ])

    def forward(self, x, exog=None):
        residual = x
        forecast = 0
        for block in self.blocks:
            backcast, block_forecast = block(residual, exog)
            residual = residual - backcast
            forecast = forecast + block_forecast
        return forecast

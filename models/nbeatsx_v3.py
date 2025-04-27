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
        return backcast, forecast, theta

# -- N-BEATSx Interpretable Model -- #
class NBeatsX(nn.Module):
    def __init__(self, input_size, exog_size=0, forecast_size=1, hidden_units=256, num_blocks=3):
        super().__init__()
        degree = 3  # Polynomial degree for trend
        harmonics = 3  # Number of Fourier harmonics for seasonality

        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            if i % 2 == 0:  # Alternate between trend and seasonality blocks
                basis = TrendBasis(degree, input_size, forecast_size)
                # FIXED: theta_size needs to be 2 * (degree + 1) to account for both forecast and backcast
                theta_size = 2 * (degree + 1)  # degrees + 1 for each of backcast and forecast
            else:
                basis = SeasonalityBasis(harmonics, input_size, forecast_size)
                # FIXED: theta_size needs to be 4 * harmonics to account for both forecast and backcast
                theta_size = 4 * harmonics  # 2 * harmonics for each of backcast and forecast

            self.blocks.append(
                NBeatsBlock(
                    input_size=input_size,
                    exog_size=exog_size,
                    theta_size=theta_size,
                    hidden_units=hidden_units,
                    basis=basis
                )
            )

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
        if return_theta:
            return forecast, thetas
        return forecast
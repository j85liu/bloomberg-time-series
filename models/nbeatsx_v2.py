import torch
import torch.nn as nn

class NBeatsBlock(nn.Module):
    """
    A single interpretable N-BEATSx block that incorporates temporal, spatial,
    seasonal, and exogenous covariates into the forecast.
    """
    def __init__(self, input_size, exog_size, output_size, hidden_units=256):
        super(NBeatsBlock, self).__init__()
        self.fc1 = nn.Linear(input_size + exog_size, hidden_units)
        self.bn1 = nn.BatchNorm1d(hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.bn2 = nn.BatchNorm1d(hidden_units)
        self.fc3 = nn.Linear(hidden_units, hidden_units)
        self.bn3 = nn.BatchNorm1d(hidden_units)
        self.fc4 = nn.Linear(hidden_units, output_size)
        self.relu = nn.ReLU()

    def forward(self, x, exog):
        # Combine main input with exogenous features
        x = torch.cat((x, exog), dim=1)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x

class NBeatsX(nn.Module):
    """
    Interpretable N-BEATSx model with multi-block residual learning and support for multiple exogenous features.
    """
    def __init__(self, input_size, exog_size, output_size=1, num_blocks=3, hidden_units=256):
        super(NBeatsX, self).__init__()
        self.blocks = nn.ModuleList([
            NBeatsBlock(input_size, exog_size, output_size, hidden_units) for _ in range(num_blocks)
        ])
        self.output_size = output_size

    def forward(self, x, exog):
        residual = x.clone()  # Avoid modifying computation graph
        forecast = torch.zeros((x.shape[0], self.output_size), device=x.device)
        for block in self.blocks:
            block_forecast = block(residual, exog)
            forecast += block_forecast
            residual -= block_forecast
        return forecast

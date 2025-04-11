import torch
import torch.nn as nn

class NBeatsBlock(nn.Module):
    """
    A single block of the N-BEATSx model with exogenous variables.
    """

    def __init__(self, input_size, exog_size, output_size, hidden_units=256):
        super(NBeatsBlock, self).__init__()
        self.fc1 = nn.Linear(input_size + exog_size, hidden_units)  # Include exogenous factors
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, hidden_units)
        self.fc4 = nn.Linear(hidden_units, output_size)  # Final output

        self.relu = nn.ReLU()

    def forward(self, x, exog):
        x = torch.cat((x, exog), dim=1)  # Concatenate exogenous variables
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)  # Output prediction
        return x

class NBeatsX(nn.Module):
    def __init__(self, input_size, exog_size, output_size=1, num_blocks=3, hidden_units=256):
        super(NBeatsX, self).__init__()
        print('hi')
        self.blocks = nn.ModuleList([
            NBeatsBlock(input_size, exog_size, output_size, hidden_units) for _ in range(num_blocks)
        ])
        self.output_size = output_size
        

    def forward(self, x, exog):
        residual = x.clone()  # Clone to avoid modifying computation graph
        forecast = torch.zeros_like((x.shape[0], self.output_size))  # Initialize to avoid in-place operations
        
        for block in self.blocks:
            block_forecast = block(residual, exog)
            forecast = forecast + block_forecast  # Avoid in-place modification
            residual = residual - block_forecast  # Avoid in-place modification
        
        return forecast

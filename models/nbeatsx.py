import torch
import torch.nn as nn
import torch.optim as optim

class NBeatsBlock(nn.Module):
    """
    A single block of the N-BEATSx model.
    """

    def __init__(self, input_size, output_size, hidden_units=256):
        super(NBeatsBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, hidden_units)
        self.fc4 = nn.Linear(hidden_units, output_size)  # Final projection

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)  # No activation at output
        return x

class NBeatsX(nn.Module):
    """
    Full N-BEATSx Model with multiple stacks.
    """

    def __init__(self, input_size, output_size, num_blocks=3, hidden_units=256):
        super(NBeatsX, self).__init__()
        self.blocks = nn.ModuleList([NBeatsBlock(input_size, output_size, hidden_units) for _ in range(num_blocks)])

    def forward(self, x):
        residual = x
        forecast = 0
        for block in self.blocks:
            forecast += block(residual)
            residual -= forecast  # Residual learning
        return forecast

# Example usage
if __name__ == "__main__":
    model = NBeatsX(input_size=30, output_size=1)
    test_input = torch.rand(1, 30)
    output = model(test_input)
    print("Example N-BEATSx Output:", output)

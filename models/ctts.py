import torch
import torch.nn as nn
import torch.optim as optim

class CNNFeatureExtractor(nn.Module):
    """
    CNN module to extract short-term patterns.
    """

    def __init__(self, input_channels, hidden_units):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=hidden_units, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class TransformerForecaster(nn.Module):
    """
    Transformer module to learn long-term dependencies.
    """

    def __init__(self, hidden_units, num_heads=4, num_layers=2):
        super(TransformerForecaster, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_units, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_units, 1)

    def forward(self, x):
        x = self.transformer(x)
        x = self.fc_out(x[:, -1, :])  # Take the last time step output
        return x

class CTTS(nn.Module):
    """
    CNN-Transformer Hybrid (CTTS) Model
    """

    def __init__(self, input_channels=1, hidden_units=64):
        super(CTTS, self).__init__()
        self.cnn = CNNFeatureExtractor(input_channels, hidden_units)
        self.transformer = TransformerForecaster(hidden_units)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(2, 0, 1)  # Reshape for Transformer (seq_len, batch, features)
        x = self.transformer(x)
        return x

# Example usage
if __name__ == "__main__":
    model = CTTS()
    test_input = torch.rand(10, 1, 30)  # (batch, channels, sequence)
    output = model(test_input)
    print("Example CTTS Output:", output)

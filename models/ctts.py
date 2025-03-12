import torch
import torch.nn as nn
import torch.optim as optim
import math

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
        """
        Input: (batch, channels, sequence_length)
        Output: (batch, hidden_units, sequence_length)
        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for Transformers (helps retain time information).
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # Shape: (max_len, 1, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Input: (sequence_length, batch_size, d_model)
        Output: (sequence_length, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0), :]
        return x


class TransformerForecaster(nn.Module):
    """
    Transformer module to learn long-term dependencies.
    """

    def __init__(self, hidden_units, num_heads=4, num_layers=2, output_size=1):
        super(TransformerForecaster, self).__init__()
        self.positional_encoding = PositionalEncoding(hidden_units)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_units, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_units, output_size)  # Forecasting output

    def forward(self, x):
        """
        Input: (sequence_length, batch_size, hidden_units)
        Output: (batch_size, output_size)
        """
        x = self.positional_encoding(x)  # Add positional encoding
        x = self.transformer(x)
        x = self.fc_out(x[-1, :, :])  # Take the last time step's output
        return x


class CTTS(nn.Module):
    """
    CNN-Transformer Hybrid (CTTS) Model
    """

    def __init__(self, input_channels=1, hidden_units=64, output_size=1):
        super(CTTS, self).__init__()
        self.cnn = CNNFeatureExtractor(input_channels, hidden_units)
        self.projection = nn.Linear(hidden_units, hidden_units)  # Ensure proper embedding
        self.transformer = TransformerForecaster(hidden_units, output_size=output_size)

    def forward(self, x):
        """
        Input: (batch, channels, sequence_length)
        Output: (batch, output_size)
        """
        x = self.cnn(x)  # CNN output: (batch, hidden_units, sequence_length)
        x = x.permute(2, 0, 1)  # Reshape for Transformer: (seq_len, batch, hidden_units)
        x = self.projection(x)  # Project to d_model for Transformer
        x = self.transformer(x)
        return x


# Example Usage & Testing
if __name__ == "__main__":
    model = CTTS(input_channels=1, hidden_units=64, output_size=1)

    test_input = torch.rand(10, 1, 30)  # (batch_size, channels, sequence_length)
    output = model(test_input)

    print("CTTS Model Output Shape:", output.shape)  # Expected: (batch_size, output_size)

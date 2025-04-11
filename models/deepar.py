import torch
import torch.nn as nn

class DeepAR(nn.Module):
    """
    DeepAR Probabilistic Forecasting Model.
    """

    def __init__(self, input_size, hidden_units=64, num_layers=2):
        super(DeepAR, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_units, num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_units, 1)  # Output 1 value per time step
        print('hi')

    def forward(self, x):
        """
        Forward pass: LSTM processes input sequence, outputs next time step forecast.
        """
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_units)
        return self.fc_out(lstm_out)  # Predict at each time step

# Example usage
if __name__ == "__main__":
    batch_size = 10
    seq_len = 30
    input_size = 1  # Assuming 1 feature per time step

    model = DeepAR(input_size=input_size)
    test_input = torch.rand(batch_size, seq_len, input_size)  # (batch, seq_len, input_size)
    output = model(test_input)

    print("Example DeepAR Output Shape:", output.shape)  # Expected: (batch_size, seq_len, 1)

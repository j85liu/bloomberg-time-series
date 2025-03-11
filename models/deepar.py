import torch
import torch.nn as nn

class DeepAR(nn.Module):
    """
    DeepAR Probabilistic Forecasting Model.
    """

    def __init__(self, input_size, hidden_units=64):
        super(DeepAR, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_units, batch_first=True)
        self.fc_out = nn.Linear(hidden_units, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        return self.fc_out(x[:, -1, :])  # Last step prediction

# Example usage
if __name__ == "__main__":
    model = DeepAR(input_size=30)
    test_input = torch.rand(10, 30)
    output = model(test_input)
    print("Example DeepAR Output:", output)

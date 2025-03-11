import torch
import torch.nn as nn

class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer (TFT) for interpretable time-series forecasting.
    """

    def __init__(self, input_size, hidden_units=64, num_heads=4, num_layers=2):
        super(TemporalFusionTransformer, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_units)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_units, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_units, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc_out(x[:, -1, :])  # Take last time step output

# Example usage
if __name__ == "__main__":
    model = TemporalFusionTransformer(input_size=30)
    test_input = torch.rand(10, 30)
    output = model(test_input)
    print("Example TFT Output:", output)

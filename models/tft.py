import torch
import torch.nn as nn
import torch.nn.functional as F

class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.weights = nn.Linear(input_size, input_size)
        self.context = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        # x: (batch, seq_len, features)
        soft_weights = F.softmax(self.weights(x), dim=-1)
        context_vector = torch.sum(soft_weights * x, dim=-1)
        return context_vector.unsqueeze(-1), soft_weights

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.linear2 = nn.Linear(hidden_size, input_size)
        self.gate = nn.GRUCell(input_size, input_size)

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = self.elu(x)
        x = self.linear2(x)
        x = self.gate(x, residual)
        return x

class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_size, static_size=0, exog_size=0,
                 hidden_units=64, num_heads=4, num_layers=2, output_size=1):
        super(TemporalFusionTransformer, self).__init__()

        # 1. Embedding layers
        self.input_embedding = nn.Linear(input_size, hidden_units)
        self.exog_embedding = nn.Linear(exog_size, hidden_units) if exog_size > 0 else None
        self.static_embedding = nn.Linear(static_size, hidden_units) if static_size > 0 else None

        # 2. Variable Selection Networks
        self.vsn = VariableSelectionNetwork(hidden_units, hidden_units)

        # 3. Static context enrichment (not fully implemented for simplicity)
        self.static_context_grn = GatedResidualNetwork(hidden_units, hidden_units)

        # 4. Local processing via BiLSTM
        self.lstm = nn.LSTM(hidden_units, hidden_units, batch_first=True, bidirectional=True)

        # 5. Temporal processing via Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_units * 2, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 6. Position-wise feedforward
        self.positionwise_ff = nn.Sequential(
            nn.Linear(hidden_units * 2, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_size)
        )

    def forward(self, x, exog=None, static=None):
        # x: (batch, seq_len, input_size)
        x = self.input_embedding(x)

        if self.exog_embedding and exog is not None:
            exog = self.exog_embedding(exog)
            x = x + exog

        # Optional: Add static context
        if self.static_embedding and static is not None:
            static = self.static_embedding(static)
            static_context = self.static_context_grn(static)
            # Static context could be added here

        # Variable selection
        x, _ = self.vsn(x)

        # Local LSTM
        x, _ = self.lstm(x)

        # Transformer expects input (seq_len, batch, features)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)

        # Position-wise FF and forecast head
        output = self.positionwise_ff(x[:, -1, :])  # Use last timestep output
        return output

# Example usage
if __name__ == "__main__":
    batch_size = 10
    seq_len = 30
    input_size = 5
    exog_size = 3
    static_size = 2

    model = TemporalFusionTransformer(input_size, static_size, exog_size)
    x = torch.rand(batch_size, seq_len, input_size)
    exog = torch.rand(batch_size, seq_len, exog_size)
    static = torch.rand(batch_size, static_size)

    output = model(x, exog, static)
    print("Output shape:", output.shape)  # Expected: (batch_size, output_size)

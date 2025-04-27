import torch
import torch.nn as nn
import torch.nn.functional as F

# === 1. Variable Selection Network ===
class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.weights_layer = nn.Linear(input_size, input_size)
        self.context_grn = GatedResidualNetwork(input_size, hidden_size)

    def forward(self, x, context=None):
        # x: (batch, seq_len, input_size)
        soft_weights = F.softmax(self.weights_layer(x), dim=-1)  # (batch, seq_len, input_size)
        if context is not None:
            context = context.unsqueeze(1).expand_as(x)  # (batch, seq_len, input_size)
            x = self.context_grn(x + context)
        weighted_x = torch.sum(soft_weights * x, dim=-1)  # (batch, seq_len)
        return weighted_x.unsqueeze(-1), soft_weights  # (batch, seq_len, 1), (batch, seq_len, input_size)

# === 2. Gated Residual Network ===
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

# === 3. Temporal Fusion Transformer ===
class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_size, static_size=0, exog_size=0,
                 hidden_units=64, num_heads=4, num_layers=2, output_size=1):
        super().__init__()
        self.hidden_units = hidden_units
        self.post_lstm_dim = hidden_units * 2  # Because of BiLSTM

        # Embedding layers
        self.input_embedding = nn.Linear(input_size, hidden_units)
        self.exog_embedding = nn.Linear(exog_size, hidden_units) if exog_size > 0 else None
        self.static_embedding = nn.Linear(static_size, hidden_units) if static_size > 0 else None

        # Static context
        self.static_context_grn = GatedResidualNetwork(hidden_units, hidden_units)

        # Variable Selection Network
        self.vsn = VariableSelectionNetwork(hidden_units, hidden_units)

        # Local sequence modeling: BiLSTM
        self.lstm = nn.LSTM(hidden_units, hidden_units, batch_first=True, bidirectional=True)

        # Temporal processing: Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.post_lstm_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final position-wise feedforward
        self.positionwise_ff = nn.Sequential(
            nn.Linear(self.post_lstm_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_size)
        )

    def forward(self, x, exog=None, static=None):
        batch_size, seq_len, _ = x.size()

        # --- 1. Embed inputs ---
        x = self.input_embedding(x)  # (batch, seq_len, hidden_units)

        if self.exog_embedding and exog is not None:
            exog = self.exog_embedding(exog)  # (batch, seq_len, hidden_units)
            x = x + exog

        # --- 2. Process static context ---
        static_context = None
        if self.static_embedding and static is not None:
            static_embedded = self.static_embedding(static)  # (batch, hidden_units)
            static_context = self.static_context_grn(static_embedded)  # (batch, hidden_units)

        # --- 3. Variable Selection ---
        x, vsn_weights = self.vsn(x, context=static_context)

        # --- 4. LSTM with static context initialization ---
        if static_context is not None:
            h0 = static_context.unsqueeze(0).repeat(2, 1, 1)  # (num_layers * num_directions, batch, hidden_size)
            c0 = torch.zeros_like(h0)
            lstm_out, _ = self.lstm(x, (h0, c0))
        else:
            lstm_out, _ = self.lstm(x)

        # --- 5. Transformer encoding ---
        x = lstm_out.permute(1, 0, 2)  # (seq_len, batch, features)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # (batch, seq_len, features)

        # --- 6. Position-wise FF and final output ---
        output = self.positionwise_ff(x[:, -1, :])  # Use last timestep

        return output, vsn_weights  # prediction, variable attention

# === 4. Example usage ===
if __name__ == "__main__":
    batch_size = 8
    seq_len = 30
    input_size = 5
    exog_size = 3
    static_size = 2

    model = TemporalFusionTransformer(input_size, static_size, exog_size)
    x = torch.rand(batch_size, seq_len, input_size)
    exog = torch.rand(batch_size, seq_len, exog_size)
    static = torch.rand(batch_size, static_size)

    output, vsn_weights = model(x, exog, static)
    print("Output shape:", output.shape)         # (batch_size, output_size)
    print("VSN Weights shape:", vsn_weights.shape)  # (batch_size, seq_len, input_size)

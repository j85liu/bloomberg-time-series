import torch
import torch.nn as nn

class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer (TFT) for interpretable time-series forecasting.
    """

    def __init__(self, input_size, static_size=0, exog_size=0, hidden_units=64, num_heads=4, num_layers=2, output_size=1):
        super(TemporalFusionTransformer, self).__init__()

        # ⚡ Embedding layers for continuous variables
        self.input_embedding = nn.Linear(input_size, hidden_units)
        self.exog_embedding = nn.Linear(exog_size, hidden_units) if exog_size > 0 else None
        self.static_embedding = nn.Linear(static_size, hidden_units) if static_size > 0 else None

        # ⚡ Local LSTM Encoder
        self.lstm = nn.LSTM(hidden_units, hidden_units, batch_first=True, bidirectional=True)

        # ⚡ Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_units*2, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # ⚡ Fully Connected Layers for Prediction
        self.fc_out = nn.Linear(hidden_units*2, output_size)

    def forward(self, x, exog=None, static=None):
        """
        x: Time-series input (batch, seq_len, input_size)
        exog: Exogenous variables (batch, seq_len, exog_size)
        static: Static covariates (batch, static_size)
        """

        # 1️⃣ Apply Input Embedding
        x = self.input_embedding(x)

        # 2️⃣ Include Exogenous Variables (if available)
        if self.exog_embedding and exog is not None:
            exog = self.exog_embedding(exog)
            x = x + exog  # Merge exogenous info

        # 3️⃣ Apply BiLSTM to capture local dependencies
        x, _ = self.lstm(x)  # Output shape: (batch, seq_len, hidden_units*2)

        # 4️⃣ Reshape for Transformer (seq_len, batch, features)
        x = x.permute(1, 0, 2)

        # 5️⃣ Apply Transformer to capture global dependencies
        x = self.transformer(x)

        # 6️⃣ Get the final prediction (last time step)
        x = self.fc_out(x.mean(dim=0))  # Use mean pooling over all timesteps

        return x

# Example usage
if __name__ == "__main__":
    batch_size = 10
    seq_len = 30
    input_size = 5  # Number of input time-series
    exog_size = 3   # Number of exogenous features
    static_size = 2 # Static features like stock category

    model = TemporalFusionTransformer(input_size, static_size, exog_size)

    test_input = torch.rand(batch_size, seq_len, input_size)  # Main time-series input
    test_exog = torch.rand(batch_size, seq_len, exog_size)    # Exogenous variables
    test_static = torch.rand(batch_size, static_size)         # Static features

    output = model(test_input, test_exog, test_static)
    print("Example TFT Output:", output.shape)  # Should output (batch_size, output_size)

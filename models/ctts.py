import torch
import torch.nn as nn
import math
from models.modules import Attention, FeedForward, PreNorm


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        mlp_ratio=4.0,
        attn_dropout=0.0,
        dropout=0.0,
        qkv_bias=True,
        revised=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        assert isinstance(
            mlp_ratio, float
        ), "MLP ratio should be an integer for valid "
        mlp_dim = int(mlp_ratio * dim)

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim,
                                num_heads=heads,
                                qkv_bias=qkv_bias,
                                attn_drop=attn_dropout,
                                proj_drop=dropout,
                            ),
                        ),
                        PreNorm(
                            dim,
                            FeedForward(dim, mlp_dim, dropout_rate=dropout,),
                        )
                        if not revised
                        else FeedForward(
                            dim, mlp_dim, dropout_rate=dropout, revised=True,
                        ),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x

class CTTS(nn.Module):
    def __init__(self, input_channels, cnn_kernel_size, cnn_out_channels, transformer_dim, nhead, num_layers, num_classes=3):
        super().__init__()

        # 1D CNN as token extractor
        self.cnn = nn.Conv1d(
            in_channels=input_channels,
            out_channels=cnn_out_channels,
            kernel_size=cnn_kernel_size,
            stride=1,
            padding=cnn_kernel_size // 2  # same padding
        )

        # Transformer expects input of shape (seq_len, batch_size, embed_dim)
        self.positional_encoding = PositionalEncoding(d_model=cnn_out_channels)

        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=cnn_out_channels,
        #     nhead=nhead,
        #     dim_feedforward=transformer_dim,
        #     batch_first=True
        # )
        # self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.transformer = Transformer(dim=cnn_out_channels, depth=1, heads=nhead)

        # MLP for classification
        self.mlp = nn.Sequential(
            nn.Linear(cnn_out_channels, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # x: (batch_size, input_channels, seq_len)
        tokens = self.cnn(x)  # -> (batch_size, cnn_out_channels, seq_len)
        tokens = tokens.permute(0, 2, 1)  # -> (batch_size, seq_len, cnn_out_channels)
        tokens = self.positional_encoding(tokens)  # add positional encoding
        transformed = self.transformer(tokens)  # -> (batch_size, seq_len, cnn_out_channels)
        pooled = transformed.mean(dim=1)  # global average pooling over tokens
        out = self.mlp(pooled)  # -> (batch_size, num_classes)
        return out


# Example usage
if __name__ == "__main__":
    batch_size = 8
    input_channels = 1
    seq_len = 100

    model = CTTS(
        input_channels=input_channels,
        cnn_kernel_size=5,
        cnn_out_channels=64,
        transformer_dim=128,
        nhead=4,
        num_layers=2,
        num_classes=3
    )

    dummy_input = torch.randn(batch_size, input_channels, seq_len)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # (batch_size, 3)

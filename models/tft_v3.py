import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedLinearUnit(nn.Module):
    """Gated Linear Unit for the gating mechanism in GRN"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)
        
    def forward(self, x):
        x = self.fc(x)
        return x[:, :, :x.size(-1)//2] * torch.sigmoid(x[:, :, x.size(-1)//2:])

class GRN(nn.Module):
    """Gated Residual Network as described in the TFT paper"""
    def __init__(self, input_dim, hidden_dim, output_dim=None, context_dim=None, dropout=0.1):
        super().__init__()
        self.output_dim = output_dim or input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        
        # Optional context integration
        self.context_dim = context_dim
        if context_dim is not None:
            self.context = nn.Linear(context_dim, hidden_dim, bias=False)
            
        # GLU layer for gating mechanism
        self.glu = GatedLinearUnit(hidden_dim, self.output_dim)
        
        # Normalization and dropout
        self.layernorm = nn.LayerNorm(self.output_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection handling
        self.add_skip = input_dim != self.output_dim
        if self.add_skip:
            self.skip = nn.Linear(input_dim, self.output_dim, bias=False)
            
    def forward(self, x, context=None):
        # Main network
        h = self.fc1(x)
        if context is not None and self.context_dim is not None:
            h = h + self.context(context)
        h = self.elu(h)
        
        # Gating and skip connection
        h = self.glu(h)
        h = self.dropout(h)
        
        # Skip connection
        if self.add_skip:
            skip = self.skip(x)
        else:
            skip = x
            
        # Layer norm on output
        return self.layernorm(skip + h)

class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network for selecting the most relevant features"""
    def __init__(self, input_sizes, hidden_dim, dropout=0.1, context_dim=None):
        super().__init__()
        self.input_sizes = input_sizes
        self.num_inputs = len(input_sizes)
        
        # Feature transformation networks - one per input
        self.feature_grns = nn.ModuleList([
            GRN(input_size, hidden_dim, output_dim=hidden_dim, dropout=dropout)
            for input_size in input_sizes
        ])
        
        # Variable selection network
        selector_input_dim = sum(input_sizes)
        self.selector_grn = GRN(
            selector_input_dim, 
            hidden_dim, 
            output_dim=self.num_inputs, 
            context_dim=context_dim, 
            dropout=dropout
        )
        
    def forward(self, x, context=None):
        # x: list of [batch_size, seq_len, input_size_i] tensors
        assert len(x) == self.num_inputs, f"Expected {self.num_inputs} inputs, got {len(x)}"
        
        # Transform each input variable
        transformed_x = [
            self.feature_grns[i](x[i]) for i in range(self.num_inputs)
        ]  # list of [batch_size, seq_len, hidden_dim]
        
        # Concatenate all inputs for variable selection
        flat_x = torch.cat(x, dim=-1)  # [batch_size, seq_len, sum(input_sizes)]
        
        # Compute variable selection weights
        sparse_weights = self.selector_grn(flat_x, context)  # [batch_size, seq_len, num_inputs]
        sparse_weights = torch.softmax(sparse_weights, dim=-1).unsqueeze(-1)  # [batch_size, seq_len, num_inputs, 1]
        
        # Apply weights to transformed inputs
        processed_inputs = torch.stack(transformed_x, dim=2)  # [batch_size, seq_len, num_inputs, hidden_dim]
        combined = (sparse_weights * processed_inputs).sum(dim=2)  # [batch_size, seq_len, hidden_dim]
        
        # Return combined inputs and variable selection weights
        # Return both the combined output and the selection weights
        return combined, sparse_weights.squeeze(-1)

class InterpretableMultiHeadAttention(nn.Module):
    """Interpretable Multi-head attention as described in TFT paper"""
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0, "Hidden dimension must be divisible by number of heads"
        
        # Linear projections for Q, K, V
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Linear projection for output
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.query_proj(query)  # [batch_size, tgt_len, hidden_dim]
        K = self.key_proj(key)      # [batch_size, src_len, hidden_dim]
        V = self.value_proj(value)  # [batch_size, src_len, hidden_dim]
        
        # Reshape for multi-head attention
        head_dim = self.hidden_dim // self.num_heads
        
        # Reshape Q, K, V for multi-head mechanism
        # [batch_size, seq_len, num_heads, head_dim]
        Q = Q.view(batch_size, -1, self.num_heads, head_dim)
        K = K.view(batch_size, -1, self.num_heads, head_dim)
        V = V.view(batch_size, -1, self.num_heads, head_dim)
        
        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(2, 3)) / (head_dim ** 0.5)  # [batch_size, num_heads, tgt_len, src_len]
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Compute attention weights and apply to values
        attn_weights = F.softmax(scores, dim=-1)  # [batch_size, num_heads, tgt_len, src_len]
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attn_weights, V)  # [batch_size, num_heads, tgt_len, head_dim]
        
        # Transpose and reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        
        # Final linear projection
        output = self.out_proj(context)
        
        return output, attn_weights

class TemporalFusionTransformer(nn.Module):
    def __init__(self,
                 num_static_vars=0,
                 num_future_vars=0,
                 num_past_vars=0,
                 static_input_sizes=None,
                 encoder_input_sizes=None,
                 decoder_input_sizes=None,
                 hidden_dim=64,
                 lstm_layers=1,
                 lstm_dropout=0.1,
                 dropout=0.1,
                 num_heads=4,
                 forecast_horizon=1,
                 backcast_length=10,
                 output_dim=1,
                 quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.forecast_horizon = forecast_horizon
        self.backcast_length = backcast_length
        self.num_quantiles = len(quantiles)
        
        # Input sizes
        self.static_input_sizes = static_input_sizes or []
        self.encoder_input_sizes = encoder_input_sizes or []
        self.decoder_input_sizes = decoder_input_sizes or []
        
        # 1. Static Covariate Encoders
        if len(self.static_input_sizes) > 0:
            self.static_vsn = VariableSelectionNetwork(
                self.static_input_sizes, hidden_dim, dropout=dropout
            )
            
        # 2. Encoder and Decoder Variable Selection Networks
        if len(self.encoder_input_sizes) > 0:
            self.encoder_vsn = VariableSelectionNetwork(
                self.encoder_input_sizes, hidden_dim, 
                dropout=dropout, 
                context_dim=hidden_dim if len(self.static_input_sizes) > 0 else None
            )
            
        if len(self.decoder_input_sizes) > 0:
            self.decoder_vsn = VariableSelectionNetwork(
                self.decoder_input_sizes, hidden_dim, 
                dropout=dropout, 
                context_dim=hidden_dim if len(self.static_input_sizes) > 0 else None
            )
        
        # 3. LSTM Encoder-Decoder
        self.lstm_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            dropout=lstm_dropout if lstm_layers > 1 else 0,
            batch_first=True
        )
        
        self.lstm_decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            dropout=lstm_dropout if lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # 4. Self-attention Layer
        self.attention = InterpretableMultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 5. Post-attention GRN with static enrichment
        self.post_attention_grn = GRN(
            hidden_dim, hidden_dim, context_dim=hidden_dim if len(self.static_input_sizes) > 0 else None, dropout=dropout
        )
        
        # 6. Position-wise Feed-Forward
        self.pos_wise_ff = GRN(hidden_dim, hidden_dim, dropout=dropout)
        
        # 7. Quantile output layers
        self.quantile_proj = nn.ModuleList([
            nn.Linear(hidden_dim, output_dim) for _ in range(self.num_quantiles)
        ])
        
    def forward(self, static_inputs=None, encoder_inputs=None, decoder_inputs=None, return_attention=False):
        """
        Args:
            static_inputs: List of static input tensors of shape [batch_size, input_size_i]
            encoder_inputs: List of encoder input tensors of shape [batch_size, backcast_length, input_size_i]
            decoder_inputs: List of decoder input tensors of shape [batch_size, forecast_horizon, input_size_i]
        """
        batch_size = encoder_inputs[0].size(0) if encoder_inputs else decoder_inputs[0].size(0)
        
        # Static variable processing
        static_context = None
        if static_inputs and len(self.static_input_sizes) > 0:
            # Convert static inputs to match sequence dimension for VSN
            # [batch_size, input_size] -> [batch_size, 1, input_size]
            static_inputs_seq = [x.unsqueeze(1) for x in static_inputs]
            static_embedding, _ = self.static_vsn(static_inputs_seq)
            # Remove sequence dimension: [batch_size, 1, hidden_dim] -> [batch_size, hidden_dim]
            static_context = static_embedding.squeeze(1)
        
        # Historical (encoder) variable processing
        encoder_output = None
        if encoder_inputs and len(self.encoder_input_sizes) > 0:
            # Pass static context if available
            static_context_expanded = static_context.unsqueeze(1).expand(-1, self.backcast_length, -1) if static_context is not None else None
            encoder_embedding, encoder_sparse_weights = self.encoder_vsn(encoder_inputs, static_context_expanded)
            
            # LSTM encoder
            encoder_output, (h_n, c_n) = self.lstm_encoder(encoder_embedding)
        else:
            # If no encoder inputs, initialize LSTM states with zeros
            h_n = torch.zeros(1, batch_size, self.hidden_dim, device=decoder_inputs[0].device)
            c_n = torch.zeros(1, batch_size, self.hidden_dim, device=decoder_inputs[0].device)
            
        # Future (decoder) variable processing
        if decoder_inputs and len(self.decoder_input_sizes) > 0:
            # Pass static context if available
            static_context_expanded = static_context.unsqueeze(1).expand(-1, self.forecast_horizon, -1) if static_context is not None else None
            decoder_embedding, decoder_sparse_weights = self.decoder_vsn(decoder_inputs, static_context_expanded)
            
            # LSTM decoder
            decoder_output, _ = self.lstm_decoder(decoder_embedding, (h_n, c_n))
            
            # Self-attention layer - combines past and future representations
            if encoder_output is not None:
                # Create attention mask (only allow attention to past observations)
                seq_len = self.backcast_length + self.forecast_horizon
                # Combine encoder and decoder outputs
                lstm_output = torch.cat([encoder_output, decoder_output], dim=1)
                
                # Apply interpretable multi-head attention
                # When calling the attention layer:
                attn_output, attn_weights = self.attention(lstm_output, lstm_output, lstm_output)
                
                # Static enrichment for decoder steps only
                if static_context is not None:
                    static_context_full = static_context.unsqueeze(1).expand(-1, seq_len, -1)
                    enriched_output = self.post_attention_grn(attn_output, static_context_full)
                else:
                    enriched_output = self.post_attention_grn(attn_output)
                
                # Position-wise feed-forward
                transformer_output = self.pos_wise_ff(enriched_output)
                
                # Extract decoder part - only take the last forecast_horizon steps
                transformer_decoder_output = transformer_output[:, -self.forecast_horizon:, :]
            else:
                # If no encoder output, just use decoder output directly
                transformer_decoder_output = decoder_output
                
            # Compute quantile forecasts
            quantile_forecasts = torch.stack([
                proj(transformer_decoder_output) for proj in self.quantile_proj
            ], dim=-1)  # [batch_size, forecast_horizon, output_dim, num_quantiles]
            
            # At the end:
            if return_attention:
                return quantile_forecasts, attn_weights
            else:
                return quantile_forecasts
        
        # Return none if no decoder inputs were provided
        return None

# Example usage
if __name__ == "__main__":
    batch_size = 32
    backcast_length = 24  # Historical time steps
    forecast_horizon = 12  # Future time steps to predict
    
    # Example input sizes
    static_input_sizes = [5, 3, 2]  # 3 static variables with their sizes
    encoder_input_sizes = [1, 3, 2]  # 3 time-varying past inputs
    decoder_input_sizes = [3, 2]     # 2 time-varying future inputs (known)
    
    # Create model
    model = TemporalFusionTransformer(
        static_input_sizes=static_input_sizes,
        encoder_input_sizes=encoder_input_sizes,
        decoder_input_sizes=decoder_input_sizes,
        hidden_dim=64,
        forecast_horizon=forecast_horizon,
        backcast_length=backcast_length,
        quantiles=[0.1, 0.5, 0.9]
    )
    
    # Create example inputs
    static_inputs = [
        torch.randn(batch_size, size) for size in static_input_sizes
    ]
    
    encoder_inputs = [
        torch.randn(batch_size, backcast_length, size) for size in encoder_input_sizes
    ]
    
    decoder_inputs = [
        torch.randn(batch_size, forecast_horizon, size) for size in decoder_input_sizes
    ]
    
    # Forward pass
    output = model(static_inputs, encoder_inputs, decoder_inputs)
    
    print(f"Output shape: {output.shape}")  # [batch_size, forecast_horizon, output_dim, num_quantiles]
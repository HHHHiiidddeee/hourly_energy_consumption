import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x

class TimeSeriesTransformerModel(nn.Module):
    def __init__(self, input_size, output_channels, kernel_size, output_size, 
                 num_heads, dim_feedforward, num_layers, dropout):
        super().__init__()
        self.conv = nn.Conv1d(input_size, output_channels, kernel_size=kernel_size, 
                              padding=kernel_size//2)
        self.pe = PositionalEncoding(d_model=output_channels)
        self.dropout = nn.Dropout(dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=output_channels, nhead=num_heads, 
                                       dim_feedforward=dim_feedforward, 
                                       dropout=dropout, batch_first=True), 
                                       num_layers=num_layers)
        self.fc = nn.Linear(output_channels, output_size)
    
    def forward(self, x):
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.pe(x)
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        x = self.fc(x[:, -1, :])
        return x

    def get_config(self):
        return {
            "input_size": self.conv.in_channels,
            "output_channels": self.conv.out_channels,
            "kernel_size": self.conv.kernel_size[0],
            "output_size": self.fc.out_features,
            "num_heads": self.transformer_encoder.layers[0].self_attn.num_heads,
            "dim_feedforward": self.transformer_encoder.layers[0].linear1.out_features,
            "num_layers": len(self.transformer_encoder.layers),
            "dropout": self.dropout.p
        }

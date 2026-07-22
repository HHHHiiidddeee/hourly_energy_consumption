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
    def __init__(self, input_size, output_channels, num_conv_layers, kernel_size, output_size, 
                 num_heads, dim_feedforward, num_layers, dropout):
        super().__init__()
        self.input_size = input_size
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.cnn_blocks = nn.ModuleList()
        if num_conv_layers > 0:
            for i in range(num_conv_layers):
                in_channels = input_size if i == 0 else output_channels
                out_channels = output_channels
                self.cnn_blocks.append(
                    nn.Sequential(
                        nn.Conv1d(in_channels, out_channels, kernel_size, stride=2, padding=(kernel_size-2) // 2),
                        nn.BatchNorm1d(out_channels),
                        nn.ReLU(),
                        nn.Dropout(dropout)
                    )
                )
        else:
            self.embedding = nn.Linear(input_size, output_channels)
        self.pe = PositionalEncoding(d_model=output_channels)
        self.dropout = nn.Dropout(dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=output_channels, nhead=num_heads, 
                                       dim_feedforward=dim_feedforward, 
                                       dropout=dropout, batch_first=True), 
                                       num_layers=num_layers)
        self.fc = nn.Linear(output_channels, output_size)
    
    def forward(self, x):
        if len(self.cnn_blocks) > 0:
            for block in self.cnn_blocks:
                x = block(x.transpose(1, 2)).transpose(1, 2)
        else:
            x = self.embedding(x)
        x = self.pe(x)
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        x = self.fc(x[:, -1, :])
        return x

    def get_config(self):
        return {
            "input_size": self.input_size,
            "output_channels": self.output_channels,
            "num_conv_layers": len(self.cnn_blocks),
            "kernel_size": self.kernel_size, 
            "output_size": self.fc.out_features,
            "num_heads": self.transformer_encoder.layers[0].self_attn.num_heads,
            "dim_feedforward": self.transformer_encoder.layers[0].linear1.out_features,
            "num_layers": len(self.transformer_encoder.layers),
            "dropout": self.dropout.p
        }

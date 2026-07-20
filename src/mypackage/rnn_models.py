import torch.nn as nn

RNN_TYPES = ("RNN", "LSTM", "GRU")

class RNNModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float,
                 output_size: int, rnn_type: str = "RNN", bidirectional: bool = False):
        super().__init__()
        if rnn_type not in RNN_TYPES:
            raise ValueError(f"Invalid rnn_type: {rnn_type}. Must be one of {RNN_TYPES}.")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1. Got {num_layers}.")
        if dropout < 0 or dropout >= 1:
            raise ValueError(f"dropout must be in the range [0, 1). Got {dropout}.")
        if bidirectional and rnn_type == "RNN":
            raise ValueError(f"bidirectional=True is not supported for RNN type. Got {rnn_type}.")
        if rnn_type == "RNN":
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, dropout=dropout,
                              bidirectional=bidirectional, batch_first=True)
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout,
                               bidirectional=bidirectional, batch_first=True)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout,
                              bidirectional=bidirectional, batch_first=True)
        self.batch_norm = nn.BatchNorm1d(hidden_size * (2 if bidirectional else 1))
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.batch_norm(out[:, -1, :])
        out = self.fc(out)
        return out

    def get_config(self):
        return {
            "input_size": self.rnn.input_size,
            "hidden_size": self.rnn.hidden_size,
            "num_layers": self.rnn.num_layers,
            "dropout": self.rnn.dropout,
            "output_size": self.fc.out_features,
            "rnn_type": type(self.rnn).__name__,
            "bidirectional": self.rnn.bidirectional
        }

class RNNWithCNNModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, kernel_size: int, num_conv_layers: int,
                 num_rnn_layers: int, dropout: float, output_size: int, rnn_type: str = "RNN",
                 bidirectional: bool = False):
        super().__init__()
        self.input_size = input_size
        if rnn_type not in RNN_TYPES:
            raise ValueError(f"Invalid rnn_type: {rnn_type}. Must be one of {RNN_TYPES}.")
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1. Got {kernel_size}.")
        if num_conv_layers < 1:
            raise ValueError(f"num_conv_layers must be >= 1. Got {num_conv_layers}.")
        if num_rnn_layers < 1:
            raise ValueError(f"num_rnn_layers must be >= 1. Got {num_rnn_layers}.")
        if dropout < 0 or dropout >= 1:
            raise ValueError(f"dropout must be in the range [0, 1). Got {dropout}.")
        if bidirectional and rnn_type == "RNN":
            raise ValueError(f"bidirectional=True is not supported for RNN type. Got {rnn_type}.")
        self.cnn_blocks = nn.ModuleList()
        for i in range(num_conv_layers):
            in_channels = input_size if i == 0 else hidden_size
            out_channels = hidden_size
            self.cnn_blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size, stride=2, padding=(kernel_size-2) // 2),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
        if rnn_type == "RNN":
            self.rnn = nn.RNN(out_channels, hidden_size, num_rnn_layers, dropout=dropout,
                              bidirectional=bidirectional, batch_first=True)
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(out_channels, hidden_size, num_rnn_layers, dropout=dropout,
                               bidirectional=bidirectional, batch_first=True)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(out_channels, hidden_size, num_rnn_layers, dropout=dropout,
                              bidirectional=bidirectional, batch_first=True)
        self.batch_norm = nn.BatchNorm1d(hidden_size * (2 if bidirectional else 1))
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)

    def forward(self, x):
        for block in self.cnn_blocks:
            x = block(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.rnn(x)
        out = self.batch_norm(out[:, -1, :])
        out = self.fc(out)
        return out

    def get_config(self):
        return {
            "input_size": self.input_size,
            "hidden_size": self.rnn.hidden_size,
            "num_conv_layers": len(self.cnn_blocks),
            "num_rnn_layers": self.rnn.num_layers,
            "kernel_size": self.cnn_blocks[0][0].kernel_size[0],
            "dropout": self.rnn.dropout,
            "output_size": self.fc.out_features,
            "rnn_type": type(self.rnn).__name__,
            "bidirectional": self.rnn.bidirectional
        }

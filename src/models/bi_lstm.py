import torch.nn as nn


class BiLSTM(nn.Module):
    """Input -> Bi-LSTM -> Dense -> pollutant predictions."""

    def __init__(self, n_features, hidden, layers, dropout, n_out):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True,
                            bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(2 * hidden, n_out)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])

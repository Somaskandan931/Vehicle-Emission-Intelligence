import torch.nn as nn


class HybridLSTM(nn.Module):
    """H = lambda * H_bi + (1 - lambda) * H_uni, then Dense.

    lambda is a fixed scalar chosen on the validation set (lambda=1 -> Bi-LSTM branch only,
    lambda=0 -> Uni-LSTM branch only). The Bi-LSTM state (2*hidden) is projected to
    `hidden` so both branches live in the same space before fusion.
    """

    def __init__(self, n_features, hidden, layers, dropout, n_out, lam=0.5):
        super().__init__()
        self.lam = float(lam)
        d = dropout if layers > 1 else 0.0
        self.bi = nn.LSTM(n_features, hidden, layers, batch_first=True, bidirectional=True, dropout=d)
        self.uni = nn.LSTM(n_features, hidden, layers, batch_first=True, dropout=d)
        self.proj = nn.Linear(2 * hidden, hidden)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x):
        hb, _ = self.bi(x)
        hu, _ = self.uni(x)
        h = self.lam * self.proj(hb[:, -1]) + (1.0 - self.lam) * hu[:, -1]
        return self.head(h)

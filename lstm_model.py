"""
LSTM Baseline
Reference: Hochreiter & Schmidhuber "Long Short-Term Memory" (1997)
Following the stock forecasting framework:
  1) LSTM encoder processes each stock's time series -> last hidden state (stock embedding)
  2) Linear decoder maps embedding to prediction (no cross-stock aggregation)
Input: (N, T, F) -> Output: (N,)
"""

import torch
import torch.nn as nn
from base_model import SequenceModel


class LSTMNet(nn.Module):
    """
    LSTM sequential encoder + linear decoder.
    Each stock is encoded independently; no graph or cross-stock attention.
    """
    def __init__(
        self,
        d_feat: int,
        d_model: int,
        lstm_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.d_feat = d_feat

        # Step 1: Sequential encoder per stock
        self.lstm = nn.LSTM(
            input_size=d_feat,
            hidden_size=d_model,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.lstm_dropout = nn.Dropout(dropout)
        self.lstm_norm = nn.LayerNorm(d_model)

        # Step 2: Decoder (no GAT / no cross-stock layer)
        self.decoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        x = x[:, :, :self.d_feat]   # (N, T, d_feat)

        # LSTM encoder: extract last hidden state
        out, _ = self.lstm(x)                   # (N, T, d_model)
        h = self.lstm_dropout(out[:, -1, :])     # (N, d_model)
        h = self.lstm_norm(h)

        out = self.decoder(h).squeeze(-1)        # (N,)
        return out


class LSTMModel(SequenceModel):
    def __init__(
        self,
        d_feat: int = 158,
        d_model: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_feat = d_feat
        self.d_model = d_model
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.init_model()

    def init_model(self):
        self.model = LSTMNet(
            d_feat=self.d_feat,
            d_model=self.d_model,
            lstm_layers=self.lstm_layers,
            dropout=self.dropout,
        )
        super().init_model()

"""
TCN Baseline (Temporal Convolutional Network)
Reference: Bai et al. "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling" (2018)
Following the stock forecasting framework:
  1) Causal dilated 1D convolutions encode each stock's time series -> last timestep (stock embedding)
  2) Linear decoder maps embedding to prediction (no cross-stock aggregation)
Input: (N, T, F) -> Output: (N,)
"""

import torch
import torch.nn as nn
from base_model import SequenceModel


class TCNBlock(nn.Module):
    """
    Single TCN block: causal dilated conv -> ReLU -> dropout.
    Output at time t depends only on inputs at times <= t.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.padding_trim = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding_trim,
            dilation=dilation,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv(x)
        if self.padding_trim > 0:
            out = out[:, :, :-self.padding_trim]
        return self.dropout(self.relu(out))


class TCNNet(nn.Module):
    """
    TCN sequential encoder + linear decoder.
    Stack of causal dilated conv layers; each stock is encoded independently.
    """
    def __init__(
        self,
        d_feat: int,
        d_model: int,
        num_layers: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        self.d_feat = d_feat

        # Step 1: Project input (N, T, d_feat) -> (N, d_model, T) and stack TCN blocks
        self.input_proj = nn.Conv1d(d_feat, d_model, 1)
        self.blocks = nn.ModuleList([
            TCNBlock(d_model, d_model, kernel_size, dilation=2 ** i, dropout=dropout)
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Step 2: Decoder
        self.decoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        x = x[:, :, :self.d_feat]   # (N, T, d_feat)
        x = x.transpose(1, 2)       # (N, d_feat, T)

        h = self.input_proj(x)       # (N, d_model, T)
        for block in self.blocks:
            h = h + block(h)         # residual
        h = h[:, :, -1]              # (N, d_model) last timestep
        h = self.norm(h)

        out = self.decoder(h).squeeze(-1)   # (N,)
        return out


class TCNModel(SequenceModel):
    def __init__(
        self,
        d_feat: int = 158,
        d_model: int = 256,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_feat = d_feat
        self.d_model = d_model
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.init_model()

    def init_model(self):
        self.model = TCNNet(
            d_feat=self.d_feat,
            d_model=self.d_model,
            num_layers=self.num_layers,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
        )
        super().init_model()

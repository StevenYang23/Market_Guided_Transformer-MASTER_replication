"""
Transformer Baseline
Reference: Vaswani et al. "Attention Is All You Need" (2017)
Standard Transformer encoder applied along the time axis for stock forecasting.
Input: (N, T, F) -> Output: (N,)
"""

import math
import torch
import torch.nn as nn
from base_model import SequenceModel


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (N, T, D)
        x = x + self.pe[:x.shape[1], :]
        return self.dropout(x)


class TransformerNet(nn.Module):
    """
    Standard Transformer Encoder:
    Linear projection -> Positional Encoding -> Transformer Encoder Layers
    -> take last time step hidden state -> Linear decoder
    """
    def __init__(
        self,
        d_feat: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.d_feat = d_feat
        self.input_proj = nn.Linear(d_feat, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # input shape: (N, T, D)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (N, T, F)
        x = x[:, :, :self.d_feat]       # (N, T, d_feat)
        x = self.input_proj(x)          # (N, T, d_model)
        x = self.pos_enc(x)             # (N, T, d_model)
        x = self.transformer_encoder(x) # (N, T, d_model)
        x = x[:, -1, :]                 # take last time step: (N, d_model)
        out = self.decoder(x).squeeze(-1)  # (N,)
        return out


class TransformerModel(SequenceModel):
    def __init__(
        self,
        d_feat: int = 158,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_feat = d_feat
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.init_model()

    def init_model(self):
        self.model = TransformerNet(
            d_feat=self.d_feat,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        )
        super().init_model()
"""
DTML Baseline
Reference: Yoo et al. "Accurate Multivariate Stock Movement Prediction via Data-Axis 
           Transformer with Multi-Level Contexts" (KDD 2021)
Framework:
  1) Temporal encoder: multi-head self-attention along time axis per stock
  2) Stock correlation mining: attention-based inter-stock aggregation
     using learned query from each stock's temporal representation
  3) Context fusion: gated fusion of individual & collective representations
  4) Linear decoder
Input: (N, T, F) -> Output: (N,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_model import SequenceModel


class TemporalEncoder(nn.Module):
    """
    Multi-head self-attention along time axis (intra-stock).
    Follows the Transformer encoder structure.
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (N, T, d_model)
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x  # (N, T, d_model)


class StockCorrelationAttention(nn.Module):
    """
    Inter-stock attention:
    Each stock's query (last time step) attends to all other stocks' keys/values.
    This mines pairwise stock correlations.
    """
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.temperature = math.sqrt(d_model / nhead)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h):
        """
        h: (N, d_model) — one representation per stock
        Returns: (N, d_model) — context-enriched representation
        """
        N = h.size(0)
        dim = self.d_model // self.nhead

        q = self.q_proj(h)  # (N, d_model)
        k = self.k_proj(h)
        v = self.v_proj(h)

        # Reshape to multi-head: (nhead, N, dim)
        q = q.view(N, self.nhead, dim).permute(1, 0, 2)  # (nhead, N, dim)
        k = k.view(N, self.nhead, dim).permute(1, 0, 2)
        v = v.view(N, self.nhead, dim).permute(1, 0, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.temperature  # (nhead, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)            # (nhead, N, dim)
        out = out.permute(1, 0, 2).contiguous().view(N, self.d_model)  # (N, d_model)
        out = self.out_proj(out)

        # Residual + Norm
        out = self.norm(h + out)
        return out  # (N, d_model)


class GatedFusion(nn.Module):
    """
    Gated fusion of individual temporal representation
    and collective (cross-stock) representation.
    gate = sigmoid(W * [h_ind; h_col]) controls the contribution of each.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h_ind, h_col):
        # h_ind, h_col: (N, d_model)
        gate = torch.sigmoid(self.gate(torch.cat([h_ind, h_col], dim=-1)))
        fused = gate * h_ind + (1 - gate) * h_col
        return self.norm(fused)


class DTMLNet(nn.Module):
    """
    Full DTML model:
      input_proj -> stacked TemporalEncoder -> temporal pooling
      -> StockCorrelationAttention -> GatedFusion -> decoder
    """
    def __init__(
        self,
        d_feat: int,
        d_model: int,
        nhead_temporal: int,
        nhead_stock: int,
        num_temporal_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.d_feat = d_feat  
        # Input projection
        self.input_proj = nn.Linear(d_feat, d_model)

        # Intra-stock temporal encoder (stacked)
        self.temporal_layers = nn.ModuleList([
            TemporalEncoder(
                d_model=d_model,
                nhead=nhead_temporal,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_temporal_layers)
        ])

        # Temporal pooling: weighted sum over time steps (attention pooling)
        self.temporal_attn_pool = nn.Linear(d_model, 1)

        # Inter-stock correlation attention
        self.stock_attn = StockCorrelationAttention(
            d_model=d_model,
            nhead=nhead_stock,
            dropout=dropout,
        )

        # Gated fusion
        self.fusion = GatedFusion(d_model=d_model)

        # Decoder
        self.decoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        x = x[:, :, :self.d_feat]       # (N, T, d_feat)
        x = self.input_proj(x)          # (N, T, d_model)

        # Intra-stock temporal encoding
        for layer in self.temporal_layers:
            x = layer(x)               # (N, T, d_model)

        # Temporal attention pooling -> individual stock representation
        attn_w = F.softmax(
            self.temporal_attn_pool(x).squeeze(-1), dim=-1
        )                               # (N, T)
        h_ind = torch.bmm(
            attn_w.unsqueeze(1), x
        ).squeeze(1)                    # (N, d_model)

        # Inter-stock correlation mining
        h_col = self.stock_attn(h_ind)  # (N, d_model)

        # Gated fusion
        h = self.fusion(h_ind, h_col)   # (N, d_model)

        out = self.decoder(h).squeeze(-1)  # (N,)
        return out


class DTMLModel(SequenceModel):
    def __init__(
        self,
        d_feat: int = 158,
        d_model: int = 256,
        nhead_temporal: int = 4,
        nhead_stock: int = 4,
        num_temporal_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_feat = d_feat
        self.d_model = d_model
        self.nhead_temporal = nhead_temporal
        self.nhead_stock = nhead_stock
        self.num_temporal_layers = num_temporal_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.init_model()

    def init_model(self):
        self.model = DTMLNet(
            d_feat=self.d_feat,
            d_model=self.d_model,
            nhead_temporal=self.nhead_temporal,
            nhead_stock=self.nhead_stock,
            num_temporal_layers=self.num_temporal_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        )
        super().init_model()
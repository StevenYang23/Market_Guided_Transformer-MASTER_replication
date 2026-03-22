"""
GAT Baseline
Reference: Veličković et al. "Graph Attention Networks" (2017)
Following the stock forecasting framework:
  1) Sequential encoder (GRU) encodes each stock's time series -> stock embedding
  2) Graph Attention Network aggregates inter-stock information
     (fully-connected graph, i.e. all stocks on the same day)
  3) Linear decoder outputs prediction
Input: (N, T, F) -> Output: (N,)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from base_model import SequenceModel


class GraphAttentionLayer(nn.Module):
    """
    Single-head Graph Attention Layer.
    Operates on a fully-connected graph over N nodes (stocks).
    """
    def __init__(self, in_features: int, out_features: int, dropout: float, alpha: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout

        self.W = nn.Linear(in_features, out_features, bias=False)
        # attention parameters: concat(Whi, Whj)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, h):
        # h: (N, in_features)
        Wh = self.W(h)                          # (N, out_features)
        N = Wh.size(0)

        # Build attention coefficients on fully-connected graph
        Wh_i = Wh.unsqueeze(1).expand(-1, N, -1)   # (N, N, out_features)
        Wh_j = Wh.unsqueeze(0).expand(N, -1, -1)   # (N, N, out_features)
        concat = torch.cat([Wh_i, Wh_j], dim=-1)   # (N, N, 2*out_features)
        e = self.leakyrelu(self.a(concat).squeeze(-1))  # (N, N)

        attn = F.softmax(e, dim=-1)                 # (N, N)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        # residual connection + aggregation
        out = torch.matmul(attn, Wh) + Wh                 # (N, out_features)
        return F.elu(out)


class MultiHeadGAT(nn.Module):
    """
    Multi-head GAT layer with concatenation (all heads except last),
    last layer uses averaging following the original paper.
    """
    def __init__(self, in_features: int, out_features: int, nhead: int, dropout: float, alpha: float = 0.2, concat: bool = True):
        super().__init__()
        self.concat = concat
        self.heads = nn.ModuleList([
            GraphAttentionLayer(in_features, out_features, dropout, alpha)
            for _ in range(nhead)
        ])
        # If concat, the output dimension is nhead*out_features, we can add a linear layer to project it back to in_features for the next GAT layer.
        if concat:
            self.proj = nn.Linear(nhead * out_features, in_features)


    def forward(self, h):
        if self.concat:
            # concatenate heads (intermediate layer)
            out = torch.cat([head(h) for head in self.heads], dim=-1)  # (N, nhead*out_features)
            return F.elu(self.proj(out)) 
        else:
            # average heads (last layer)
            return torch.mean(torch.stack([head(h) for head in self.heads], dim=0), dim=0)  # (N, out_features)


class GATNet(nn.Module):
    """
    GRU sequential encoder + two-layer GAT + linear decoder
    """
    def __init__(
        self,
        d_feat: int,
        d_model: int,
        gat_hidden: int,
        # gat_out: int,
        nhead: int,
        gru_layers: int,
        dropout: float,
        alpha: float = 0.2,
    ):
        super().__init__()
        # Step 1: Sequential encoder per stock
        self.d_feat = d_feat  
        self.gru = nn.GRU(
            input_size=d_feat,
            hidden_size=d_model,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.gru_dropout = nn.Dropout(dropout)
        self.gru_norm = nn.LayerNorm(d_model)

        # Step 2: Two-layer GAT
        # Layer 1: concat heads -> d_model
        self.gat1 = MultiHeadGAT(
            in_features=d_model,
            out_features=gat_hidden,
            nhead=nhead,
            dropout=dropout,
            alpha=alpha,
            concat=True,
        )
        # Layer 2: average heads -> nhead*gat_hidden -> gat_out
        self.gat2 = MultiHeadGAT(
            in_features=d_model,
            out_features=gat_hidden,
            nhead=nhead,
            dropout=dropout,
            alpha=alpha,
            concat=False,
        )

        # Step 3: Decoder
        self.decoder = nn.Sequential(
            nn.LayerNorm(gat_hidden),
            nn.Linear(gat_hidden, 1),
        )

    def forward(self, x):
        x = x[:, :, :self.d_feat]       # (N, T, d_feat)
        N = x.size(0)

        # GRU encoder: extract last hidden state
        out, _ = self.gru(x)                # (N, T, d_model)
        h = self.gru_dropout(out[:, -1, :]) # (N, d_model)
        
        # If only 1 stock, skip GAT and directly decode
        if h.size(0) == 1:
            return self.decoder(h).squeeze(-1)

        # GAT on fully-connected stock graph
        h = self.gat1(h)   # (N, d_model)
        h = self.gat2(h)   # (N, gat_hidden)

        out = self.decoder(h).squeeze(-1)   # (N,)
        return out


class GATModel(SequenceModel):
    def __init__(
        self,
        d_feat: int = 158,
        d_model: int = 256,
        gat_hidden: int = 64,
        # gat_out: int = 256,
        nhead: int = 4,
        gru_layers: int = 2,
        dropout: float = 0.5,
        alpha: float = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_feat = d_feat
        self.d_model = d_model
        self.gat_hidden = gat_hidden
        # self.gat_out = gat_out
        self.nhead = nhead
        self.gru_layers = gru_layers
        self.dropout = dropout
        self.alpha = alpha
        self.init_model()

    def init_model(self):
        self.model = GATNet(
            d_feat=self.d_feat,
            d_model=self.d_model,
            gat_hidden=self.gat_hidden,
            # gat_out=self.gat_out,
            nhead=self.nhead,
            gru_layers=self.gru_layers,
            dropout=self.dropout,
            alpha=self.alpha,
        )
        super().init_model()
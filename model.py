import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.shape[1], :]
 
class AttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout, transpose_qkv=True):
        super().__init__()
        self.nhead = nhead
        self.dim = d_model // nhead
        self.temperature = math.sqrt(self.dim) if transpose_qkv else None
        self.transpose_qkv = transpose_qkv
        
        self.qkv = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(3)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(nhead)]) if dropout > 0 else None
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model), nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        q, k, v = [proj(x) for proj in self.qkv]
        
        if self.transpose_qkv:
            q, k, v = q.transpose(0,1), k.transpose(0,1), v.transpose(0,1)
        
        outputs = []
        for i in range(self.nhead):
            start, end = i*self.dim, None if i==self.nhead-1 else (i+1)*self.dim
            qh, kh, vh = q[:,:,start:end], k[:,:,start:end], v[:,:,start:end]
            
            attn = torch.softmax(torch.matmul(qh, kh.transpose(1,2)) / (self.temperature if self.temperature else 1.0), dim=-1)
            if self.dropouts: attn = self.dropouts[i](attn)
            outputs.append(torch.matmul(attn, vh))
        
        att_output = torch.concat(outputs, dim=-1)
        if self.transpose_qkv: att_output = att_output.transpose(0,1)
        
        xt = x + att_output
        return xt + self.ffn(self.norm2(xt))

class Gate(nn.Module):
    def __init__(self, d_input, d_output, beta=1.0):
        super().__init__()
        self.linear = nn.Linear(d_input, d_output)
        self.t = beta
        self.d_output = d_output

    def forward(self, x):
        return self.d_output * torch.softmax(self.linear(x) / self.t, dim=-1)

class TemporalAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z):
        h = self.linear(z)
        query = h[:, -1, :].unsqueeze(-1)
        lam = torch.softmax(torch.matmul(h, query).squeeze(-1), dim=1).unsqueeze(1)
        return torch.matmul(lam, z).squeeze(1)
 
class MASTER(nn.Module):
    def __init__(self, config):
        super().__init__()
        m = config['model']
        self.gate = Gate(m['gate_input_end_index'] - m['gate_input_start_index'], m['d_feat'], m['beta'])
        
        self.layers = nn.Sequential(
            nn.Linear(m['d_feat'], m['d_model']),
            PositionalEncoding(m['d_model']), # N, T, D
            AttentionLayer(m['d_model'], m['t_nhead'], m['dropout'], transpose_qkv=False),
            AttentionLayer(m['d_model'], m['s_nhead'], m['dropout'], transpose_qkv=True),
            TemporalAttention(m['d_model']),
            nn.Linear(m['d_model'], 1)
        )
        self.gate_start = m['gate_input_start_index']
        self.gate_end = m['gate_input_end_index']

    def forward(self, x):
        src = x[:, :, :self.gate_start] # N, T, D
        gate_in = x[:, -1, self.gate_start:self.gate_end]
        src = src * self.gate(gate_in).unsqueeze(1)
        return self.layers(src).squeeze(-1)
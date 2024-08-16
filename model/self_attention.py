import math

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, seq_len, hidden_size, nhead, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.attention_size = hidden_size
        self.seq_len = seq_len
        self.nhead = nhead

        self.q = nn.Linear(hidden_size, self.attention_size)
        self.k = nn.Linear(hidden_size, self.attention_size)
        self.v = nn.Linear(hidden_size, self.attention_size)

        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(self.seq_len)
        self.relu = nn.ReLU(inplace=True)

        self.linear = nn.Linear(self.attention_size, self.attention_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, input):
        x = input

        x = x.permute(0, 2, 1)

        batch_size, seq_len, hidden_size = x.size()
        dk = self.attention_size // self.nhead

        x = self.batch_norm(x)

        Q = self.q(x).reshape(batch_size, self.seq_len, self.nhead, dk).transpose(1, 2)
        K = self.k(x).reshape(batch_size, self.seq_len, self.nhead, dk).transpose(1, 2)
        V = self.v(x).reshape(batch_size, self.seq_len, self.nhead, dk).transpose(1, 2)

        attention_scores = torch.matmul(Q, K.transpose(2, 3)) / (math.sqrt(dk))
        attention_scores = torch.softmax(attention_scores, dim=-1)
        attn = attention_scores

        attention_scores = torch.matmul(attention_scores, V)

        attention_scores = attention_scores.transpose(1, 2).reshape(batch_size, self.seq_len, self.attention_size)

        output = self.linear(attention_scores)
        output = self.layer_norm(output + x)
        output = output.transpose(1, 2)

        return output,attn

import torch
from torch import nn
import math
import torch.nn.functional as F

class SiglipAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.embed_dim = config.hidden_size
        self.dropout = config.attention_dropout
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(k.shape[-1])
        weights = F.softmax(scores, dim=-1)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        attn_output = (weights @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(attn_output)

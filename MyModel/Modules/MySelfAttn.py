#coding=utf-8
import torch.nn as nn

class SelfAttention(nn.Module):
    """
    自注意力机制：一个模态作为查询的q、k、v
    # queries的形状：(batch_size，查询的个数=N，d)
    # keys的形状：(batch_size，“键－值”对的个数=N，d)
    # values的形状：(batch_size，“键－值”对的个数=N，值的维度)
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5  # 根号d

        self.W_q = nn.Linear(dim, dim, bias=qkv_bias)  # 缩放点积注意力要求q、k维度相同，因为是要找相似度
        self.W_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.W_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, input):
        query, key, value = input, input, input
        B, N, C = query.shape # B=batch size, N=H*W, C=Channels
        q = self.W_q(query)
        k = self.W_k(key)
        v = self.W_v(value)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

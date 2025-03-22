#coding=utf-8
import torch.nn as nn

class CrossAttention(nn.Module):
    """
    跨尺度注意力机制：一个模态作为查询的q，另一个模态作为k、v
    # queries的形状：(batch_size，查询的个数=N，d)
    # keys的形状：(batch_size，“键－值”对的个数=N，d)
    # values的形状：(batch_size，“键－值”对的个数=N，值的维度)
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5 # 根号d

        #self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.W_q = nn.Linear(dim, dim, bias=qkv_bias) # 缩放点积注意力要求q、k维度相同，因为是要找相似度
        self.W_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.W_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, m_q, m_kv):
        query = m_q
        key, value = m_kv, m_kv
        B, N, C = query.shape # B=batch size, N=H*W, C=Channels
        q = self.W_q(query)
        k = self.W_k(key)
        v = self.W_v(value)

        attn = (q @ k.transpose(-2, -1)) * self.scale # 换成多头注意力？
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MultiHeadCrossAttn(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0):
        super().__init__()
        """
        dim：q、k、v的维度。
        num_heads：多头注意力的头数。
        dropout_rate：dropout比例。
        """
        self.num_heads = num_heads
        self.head_dim = dim // num_heads # head的维度为输入维度除以head的个数
        assert dim % num_heads == 0, "Input dimension must be divisible by the number of heads."
        self.scale = qk_scale or self.head_dim ** -0.5  # 根号d

        self.W_q = nn.Linear(dim, dim, bias=qkv_bias)  # 缩放点积注意力要求q、k维度相同，因为是要找相似度
        self.W_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.W_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, m_q, m_kv):
        query = m_q
        key, value = m_kv, m_kv
        B, N, C = query.shape  # B=batch size, N=H*W, C=Channels
        # 将q、k、v向量拆分成多个头（把C维度分成num_heads个头，每个头维度为head_dim）
        q = self.W_q(query).view(B,N,self.num_heads,self.head_dim)
        k = self.W_k(key).view(B,N,self.num_heads,self.head_dim)
        v = self.W_v(value).view(B,N,self.num_heads,self.head_dim)
        # 交换维度，方便后续和不同头部进行注意力计算
        q = q.transpose(1,2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

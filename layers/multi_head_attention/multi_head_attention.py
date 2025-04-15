import torch
from torch import nn

from utils.device import choose_device


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        :param embed_dim: 嵌入维度
        :param num_heads: 头数
        :param dropout: dropout比例
        """
        super(MultiHeadAttentionLayer, self).__init__()
        assert embed_dim % num_heads == 0

        # 参数
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 线性层
        self.q_linear = nn.Linear(embed_dim, embed_dim, device=choose_device())
        self.k_linear = nn.Linear(embed_dim, embed_dim, device=choose_device())
        self.v_linear = nn.Linear(embed_dim, embed_dim, device=choose_device())

        # 输出线性层
        self.out_linear = nn.Linear(embed_dim, embed_dim, device=choose_device())

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """
        缩放点积注意力
        :param q: query, (batch_size, num_heads, seq_len, head_dim)
        :param k: key, (batch_size, num_heads, seq_len, head_dim)
        :param v: value, (batch_size, num_heads, seq_len, head_dim)
        :param mask: mask, (batch_size, 1, seq_len, seq_len)
        :return: (batch_size, num_heads, seq_len, head_dim)
        """
        # 根据Q、K、V计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim).float()) # (batch_size, num_heads, seq_len, seq_len)

        # 应用Mask
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        # 计算注意力权重
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 加权求和
        output = torch.matmul(attention_weights, v) # （batch_size, num_heads, seq_len, head_dim)

        return output

    def split_heads(self, x):
        """
        将嵌入向量拆分多头
        :param x: 输入张量 (batch_size, seq_len, embed_dim)
        :return: 拆分后的张量 (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, seq_len, embed_dim = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim) # 将第3维拆分到多个heads (batch_size, seq_len, num_heads, head_dim)
        x = x.permute(0, 2, 1, 3) # 调转第1维和第2维 (batch_size, num_heads, seq_len, head_dim)
        return x

    def combine_heads(self, x):
        """
        将多个头拼接回原始形状
        :param x: 输入张量 (batch_size, num_heads, seq_len, head_dim)
        :return: 拼接后的张量 (batch_size, seq_len, embed_dim)
        """
        batch_size, num_heads, seq_len, head_dim = x.size()
        x = x.permute(0, 2, 1, 3) # 调转第1维和第2维 (batch_size, seq_len, num_heads, head_dim)
        x = x.contiguous().view(batch_size, seq_len, self.embed_dim) # 将第3维拼接回embed_dim (batch_size, seq_len, embed_dim)
        return x

    def forward(self, q, k, v, mask=None):
        """
        :param q: (batch_size, seq_len, embed_dim)
        :param k: (batch_size, seq_len, embed_dim)
        :param v: (batch_size, seq_len, embed_dim)
        :param mask: (batch_size, 1, seq_len, seq_len)
        :return: (batch_size, seq_len, embed_dim)
        """
        # 对Q、K、V进行线性变换
        q = self.q_linear(q) #(batch_size, seq_len, embed_dim)
        k = self.k_linear(k) #(batch_size, seq_len, embed_dim)
        v = self.v_linear(v) #(batch_size, seq_len, embed_dim)

        # 对Q、K、V线性变换的结果拆分多头
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # 计算注意力
        attention_output = self.scaled_dot_product_attention(q, k, v, mask) #(batch_size, seq_len, embed_dim)

        # 将多头拼接回原始形状
        attention_output = self.combine_heads(attention_output)

        # 输出线性变换
        output = self.out_linear(attention_output)

        return output
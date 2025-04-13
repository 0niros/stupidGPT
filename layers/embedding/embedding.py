import math

import torch
from torch import nn

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_len=16, dropout=0.1):
        """
        初始化Embedding层
        :param vocab_size: 词库词量大小
        :param embedding_dim: 嵌入维度
        :param max_len: 最大序列长度
        :param dropout: Dropout比例
        """
        super(EmbeddingLayer, self).__init__()

        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)

        # Token embedding: 将token id映射到嵌入向量
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

        # Position Encoding: 添加位置信息
        self.position_encoding = self.create_position_encoding()

    def create_position_encoding(self):
        """
        创建位置编码
        :return: 位置编码张量，形状：(max_len, embedding_dim)
        """
        # 这段代码的功能是生成一个从0到max_len-1的整数序列，并将其转换为列向量。具体功能分解如下：
        # torch.arange(0, max_len)：生成从0到max_len-1的整数张量，形状：(max_len)
        # unsqueeze(1)：将张量在第1维度上增加一维，使其变为列向量，形状：(max_len, 1)
        position = torch.arange(0, self.max_len).unsqueeze(1)

        # TODO
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2) * -(math.log(10000.0) / self.embedding_dim))

        position_encoding = torch.zeros(self.max_len, self.embedding_dim)
        position_encoding[:, 0::2] = torch.sin(position * div_term) # 偶数索引
        position_encoding[:, 1::2] = torch.cos(position * div_term) # 奇数索引

        return position_encoding.unsqueeze(0)

    def forward(self, input_token_ids):
        """
        前向传播
        :param input_token_ids: 输入的token id序列，形状：(batch_size, seq_len)
        :return: 嵌入向量序列，形状：(batch_size, seq_len, embedding_dim)
        """
        # 获取token embedding, 形状：(batch_size, seq_len, embedding_dim)
        token_embed = self.token_embedding(input_token_ids)

        # 缩放嵌入向量,论文中提到的技巧
        token_embed = token_embed * math.sqrt(self.embedding_dim)

        # Position encoding，把输入的位置信息添加到嵌入向量中
        seq_len = input_token_ids.size(1)
        position_encoding = self.position_encoding[:, :seq_len, :] # 截取需要的长度即可

        embedding = token_embed + position_encoding # (batch_size, seq_len, embedding_dim)

        # Dropout, 让模型更好训练
        embedding = self.dropout(embedding)

        return embedding


if __name__ == '__main__':
    embeddingLayer = EmbeddingLayer(vocab_size=100, embedding_dim=128, max_len=16)
    embeddingLayer.forward(torch.randint(0, 12, (1, 10)))

from torch import nn

from layers.add_and_norm.add_and_norm import AddAndNormLayer
from layers.feedforward.feedforward import FeedForwardLayer
from layers.multi_head_attention.multi_head_attention import MultiHeadAttentionLayer


class EncoderLayer(nn.Module):
    """
    单个Encoder层
    """

    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        """
        Init 单个Encoder层
        :param embed_dim: 嵌入维度
        :param num_heads: 注意力头的数量
        :param hidden_dim: FFN的隐藏层维度
        :param dropout: Dropout比例
        """
        super(EncoderLayer, self).__init__()
        # Multi-Head Attention
        self.self_attention = MultiHeadAttentionLayer(embed_dim, num_heads, dropout)

        # Add & Norm, 在self_attention后先残差连接归一化一次
        self.add_and_norm1 = AddAndNormLayer(embed_dim, dropout)

        # FeedForward
        self.feed_forward = FeedForwardLayer(embed_dim, hidden_dim, dropout)

        # Add & Norm, 在feed_forward后残差连接归一化一次
        self.add_and_norm2 = AddAndNormLayer(embed_dim, dropout)

    def forward(self, input_tensor, mask=None):
        """
        Forward 单个Encoder层
        :param input_tensor: 输入张量
        :param mask: 遮罩
        :return: 输出张量
        """
        # Multi-Head Attention
        attention_output = self.self_attention(input_tensor, input_tensor, input_tensor, mask)

        # Add & Norm 1, 在self_attention后先残差连接归一化一次
        attention_output = self.add_and_norm1(input_tensor, attention_output)

        # FeedForward
        ffn_output = self.feed_forward(attention_output)

        # Add & Norm 2, 在feed_forward后残差连接归一化一次
        ffn_output = self.add_and_norm2(attention_output, ffn_output)

        return ffn_output


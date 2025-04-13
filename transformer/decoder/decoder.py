from torch import nn

from layers.add_and_norm.add_and_norm import AddAndNormLayer
from layers.feedforward.feedforward import FeedForwardLayer
from layers.multi_head_attention.multi_head_attention import MultiHeadAttentionLayer


class DecoderLayer(nn.Module):
    """
    单个Decoder层
    """

    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        """
        Init单个Decoder层
        :param embed_dim: 嵌入维度
        :param num_heads: 注意力头数
        :param hidden_dim: FFN的隐藏层数量
        :param dropout: Dropout
        """
        super(DecoderLayer, self).__init__()

        # Masked Multi-Head Attention
        self.self_attention = MultiHeadAttentionLayer(embed_dim, num_heads, dropout)
        self.add_and_norm1 = AddAndNormLayer(embed_dim, dropout)

        # Encoder-Decoder Attention
        self.encoder_decoder_attention = MultiHeadAttentionLayer(embed_dim, num_heads, dropout)
        self.add_and_norm2 = AddAndNormLayer(embed_dim, dropout)

        # FeedForward
        self.feed_forward = FeedForwardLayer(embed_dim, hidden_dim, dropout)
        self.add_and_norm3 = AddAndNormLayer(embed_dim, dropout)

    def forward(self, input_tensor, encoder_output, src_mask=None, tgt_mask=None):
        """
        DecoderLayer前向传播
        :param input_tensor: Decoder输入, 形状:(batch_size, seq_len, embed_dim)
        :param encoder_output: Encoder输出,  形状:(batch_size, seq_len, embed_dim)
        :param src_mask: Encoder Mask
        :param tgt_mask: Decoder Mask
        :return: Decoder输出
        """
        # Masked Multi-Head Attention
        self_attention_output = self.self_attention(input_tensor, input_tensor, input_tensor, mask=tgt_mask)
        self_attention_output = self.add_and_norm1(input_tensor, self_attention_output)

        # Encoder-Decoder Attention
        encoder_decoder_attention_output = self.encoder_decoder_attention(self_attention_output, encoder_output, encoder_output, mask=src_mask)
        encoder_decoder_attention_output = self.add_and_norm2(self_attention_output, encoder_decoder_attention_output)

        # FeedForward
        feed_forward_output = self.feed_forward(encoder_decoder_attention_output)
        feed_forward_output = self.add_and_norm3(encoder_decoder_attention_output, feed_forward_output)

        return feed_forward_output
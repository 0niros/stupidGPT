from torch import nn


class AddAndNormLayer(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        """
        Add & Norm层
        :param embed_dim: 嵌入维度
        :param dropout: Dropout比例
        """
        super(AddAndNormLayer, self).__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, input_tensor, sub_layer_output):
        """
        Forward
        :param input_tensor: 输入张量 (batch_size, seq_len, embed_dim)
        :param sub_layer_output: 子层的输出张量 (batch_size, seq_len, embed_dim)
        :return: 残差链接和归一化后的值 (batch_size, seq_len, embed_dim)
        """
        # 残差连接: 将输入与子层输出相加
        add_output = input_tensor + self.dropout(sub_layer_output)

        # 层归一化
        norm_output = self.layer_norm(add_output)

        return norm_output

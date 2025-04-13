from torch import nn


class FeedForwardLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super(FeedForwardLayer, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        # 第一层线形变换：embed_dim -> hidden_dim
        self.linear1 = nn.Linear(embed_dim, hidden_dim)

        # 激活函数
        self.activation = nn.ReLU()

        # 第二层线形变换：hidden_dim -> embed_dim
        self.linear2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, input_tensor):
        """
        FeedForward前向传播. FF: output = W1(ReLU(W2x + b)) + b2
        :param input_tensor: 输入tensor, 形状(batch_size, seq_len, embed_dim)
        :return: output_tensor, 形状(batch_size, seq_len, embed_dim)
        """
        output_tensor = self.linear1(input_tensor)
        output_tensor = self.activation(output_tensor)
        output_tensor = self.dropout(output_tensor)
        output_tensor = self.linear2(output_tensor)

        return output_tensor
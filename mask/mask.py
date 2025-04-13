import torch


def create_padding_mask(seq, pad_token=0):
    """
    创建 Padding Mask, 目的是屏蔽掉输入序列中的填充部分
    :param seq: 输入序列，形状：(batch_size, seq_len)
    :param pad_token: 填充的TokenId，默认为0
    :return: PaddingMask (batch_size, 1, 1, seq_len)
    """
    mask = (seq.eq(pad_token)).unsqueeze(1).unsqueeze(2)
    return mask

def create_causal_mask(seq_len):
    """
    创建CausalMask, 目的是确保模型在预测的当前位置看不到未来的词
    :param seq_len: 序列长度
    :return: (1, seq_len, seq_len)
    """
    # 调用 torch.triu 提取矩阵的上三角部分（不包括对角线），并将结果转换为布尔类型，形成掩码矩阵。
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool() # (seq_len, seq_len)
    return mask.unsqueeze(0)

def create_combine_mask(padding_mask, causal_mask):
    """
    组合PaddingMask和CausalMask
    :param padding_mask: PaddingMask (batch_size, 1, 1, seq_len)
    :param causal_mask: CausalMask (1, seq_len, seq_len)
    :return: CombineMask (batch_size, 1, seq_len, seq_len)
    """
    # 将PaddingMask扩展到和Causal相同的形状
    padding_mask = padding_mask.expand(-1, -1, causal_mask.size(1), -1) # (batch_size, 1, seq_len, seq_len)

    # 组合
    combine_mask = padding_mask | causal_mask.unsqueeze(0) # (batch_size, 1, seq_len, seq_len)

    return combine_mask

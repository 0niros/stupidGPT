import torch
from torch import nn

from layers.embedding.embedding import EmbeddingLayer
from transformer.encoder.encoder import EncoderLayer
from transformer.decoder.decoder import DecoderLayer
from mask.mask import create_padding_mask, create_combine_mask, create_causal_mask

class TransformerConfig:
    def __init__(self, batch_size, vocab_size, embedding_dim, num_heads, hidden_dim, num_encoder_layers, num_decoder_layers, max_len, dropout=0.1):
        """
        :param vocab_size: 词库大小
        :param embedding_dim: 嵌入维度
        :param num_heads: 头数
        :param hidden_dim: FFN隐藏层维度
        :param num_encoder_layers: Encoder层数
        :param num_decoder_layers: Decoder层数
        :param max_len: 最大序列长度
        :param dropout: Dropout比例
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.max_len = max_len
        self.dropout = dropout
        self.batch_size = batch_size

    def get_train_file_name(self):
        return f"train-{self.vocab_size}_{self.embedding_dim}_{self.num_heads}_{self.hidden_dim}_{self.num_encoder_layers}_{self.num_decoder_layers}_{self.max_len}_{self.dropout}.bin"


class TransformerModel(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(TransformerModel, self).__init__()
        self.config = config

        # Embedding层
        self.enc_embedding = EmbeddingLayer(config.vocab_size, config.embedding_dim, config.max_len, config.dropout)

        # Encoder堆叠
        self.encoder = nn.ModuleList([EncoderLayer(config.embedding_dim, config.num_heads, config.hidden_dim, config.dropout) for _ in range(config.num_encoder_layers)])

        # Embedding层
        self.dec_embedding = EmbeddingLayer(config.vocab_size, config.embedding_dim, config.max_len, config.dropout)

        # Decoder堆叠
        self.decoder = nn.ModuleList([DecoderLayer(config.embedding_dim, config.num_heads, config.hidden_dim, config.dropout) for _ in range(config.num_decoder_layers)])

        # 输出层
        self.linear = nn.Linear(config.embedding_dim, config.vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, enc_input_token_ids, dec_input_token_ids):
        """
        :param dec_input_token_ids: Encoder输入的tokenId序列
        :param enc_input_token_ids: Decoder输入的tokenId序列
        :return: (batch_size, vocab_size) softmax output on vocab
        """
        # Mask
        enc_mask = create_padding_mask(enc_input_token_ids)
        # enc_mask.expand(-1, -1, dec_input_token_ids.size(1), -1)
        dec_mask = create_combine_mask(
            padding_mask=create_padding_mask(dec_input_token_ids),
            causal_mask=create_causal_mask(dec_input_token_ids.size(-1))
        )

        # Encoder Embedding
        embedding_enc_input = self.enc_embedding.forward(enc_input_token_ids)

        # Encoders
        encoder_output = None
        for encoder in self.encoder:
            encoder_output = encoder(embedding_enc_input, enc_mask)

        # Decoder Embedding
        embedding_dec_input = self.dec_embedding.forward(dec_input_token_ids)

        # Decoders
        decoder_output = None
        for decoder in self.decoder:
            decoder_output = decoder(embedding_dec_input, encoder_output, enc_mask, dec_mask)

        # Linear
        linear_output = self.linear(decoder_output)
        # Softmax
        softmax_output = self.softmax(linear_output)

        return softmax_output
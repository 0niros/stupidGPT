import random
from loguru import logger


def random_sample(dataset, batch_size):
    return random.sample(dataset, batch_size)


class SimpleTokenizer:
    """
    简单的Tokenizer
    每行是一个样本, 使用空格分词
    """

    def __init__(self):
        self.vocab = {}
        self.vocab_size = 1
        self.token_to_id = {}
        self.id_to_token = {}

    def add_token(self, token):
        """
        添加token到SimpleTokenizer
        :param token: Token
        """
        if token.lower() not in self.token_to_id:
            self.token_to_id[token.lower()] = self.vocab_size
            self.id_to_token[self.vocab_size] = token.lower()
            self.vocab_size += 1

    def reset_token(self, idx, token):
        """
        调整特殊token位置
        :param idx:
        :param token:
        :return:
        """
        self.token_to_id[token.lower()] = idx
        self.id_to_token[idx] = token.lower()
        self.vocab_size = len(self.token_to_id)

    def fit_on_texts(self, texts):
        """
        根据texts来构建vocab
        :param texts: 输入的原始文本
        """
        for text in texts:
            tokens = text.split()
            for token in tokens:
                self.add_token(token.lower())
        self.reset_token(0, "?")
        logger.info("初始化完成，词库大小:{}", self.vocab_size)

    def encode(self, text, max_length=None, pad=True):
        """
        将texts编码成id序列
        :param text: 输入文本
        :param max_length: 最大长度
        :param pad: 是否需要填充，默认是
        :return: token_ids
        """
        tokens = text.split()
        token_ids = [self.token_to_id.get(token.lower(), 0) for token in tokens]
        if max_length is not None:
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            elif pad and len(token_ids) < max_length:
                token_ids.extend([0] * (max_length - len(token_ids)))
        return token_ids

    def decode(self, token_ids):
        """
        将id序列还原成token
        :param token_ids: id
        :return: tokens
        """
        tokens = [self.id_to_token.get(token_id, ' ') for token_id in token_ids]
        return "".join(tokens)

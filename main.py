from os.path import exists
import numpy as np
import torch
from torch import nn, optim
from loguru import logger

from tokenizer.tokenizer import SimpleTokenizer
from transformer.transformer_model import TransformerConfig, TransformerModel
from utils.device import choose_device

# Transformer 配置
transformer_config = TransformerConfig(
        # Batch size
        batch_size=16,
        # 词库大小
        vocab_size=1200,
        # embedding维度
        embedding_dim=16,
        # 多头注意力的头数
        num_heads=8,
        # 隐藏层维度
        hidden_dim=16,
        # 编码器层数
        num_encoder_layers=16,
        # 解码器层数
        num_decoder_layers=16,
        # 最大长度
        max_len=12,
        # dropout
        # 0.1
        dropout=0.1
)

def infer(train_data_file, prompt):
    """
    模型推理
    :param train_data_file: 模型参数文件
    :param prompt: 提示词
    """
    logger.info(f"🚀 开始推理, 使用{choose_device().type}")
    # 1. 构建模型
    model = TransformerModel(transformer_config)
    if exists(transformer_config.get_train_file_name()):
        model.load_state_dict(torch.load(transformer_config.get_train_file_name()))
        logger.info(f"✅ 加载{transformer_config.get_train_file_name()}成功")

    # 2. 加载数据集
    train_data = None
    tokenizer = SimpleTokenizer()
    with open(train_data_file, 'r') as f:
        train_data = f.readlines()
        train_data = [line.strip() for line in train_data]
    tokenizer.fit_on_texts(train_data)

    # 3. 推理
    model.eval()
    prompt_encoded = tokenizer.encode(prompt, transformer_config.max_len, True)
    encoder_input = torch.from_numpy(np.array(prompt_encoded.copy())).unsqueeze(0)
    decoder_input = torch.from_numpy(np.array(prompt_encoded.copy())).unsqueeze(0)

    input_index = encoder_input.size(0) + 1
    output_words = []
    for i in range(input_index, transformer_config.max_len - 1):
        encoder_input = encoder_input.to(choose_device())
        decoder_input = decoder_input.to(choose_device())
        output = model(encoder_input, decoder_input)
        output = output.argmax(dim=-1)
        output_next_word = output[0][i]
        decoder_input[0][i] = output_next_word
        output_words.append(output_next_word.item())
        if output_next_word.item() == 1 or output_next_word.item() == 0:
            break
    logger.info(f"✅ Prompt:{prompt} 推理完成!")
    logger.info(f"{prompt} {tokenizer.decode(output_words)}")

def train(train_data_file, epochs: int):
    """
    模型训练
    :param train_data_file: 模型参数文件
    :param epochs: 训练Epoch
    """
    logger.info(f"🚀 开始训练, 使用{choose_device().type}")
    # 1. 加载数据集
    train_data = None
    tokenizer = SimpleTokenizer()
    with open(train_data_file, 'r') as f:
        train_data = f.readlines()
        train_data = [line.strip() for line in train_data]
    tokenizer.fit_on_texts(train_data)

    # 2. 构建模型
    model = TransformerModel(transformer_config)
    model = model.to(choose_device())
    if exists(transformer_config.get_train_file_name()):
        model.load_state_dict(torch.load(transformer_config.get_train_file_name()))
        logger.info(f"✅ 加载{transformer_config.get_train_file_name()}成功")
    model.train()

    # 3. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    # 4. 训练
    loss_epoch = 0
    for epoch in range(epochs):
        optimizer.zero_grad()

        # 4.1 取数据
        encoder_input, decoder_input, target_output = tokenizer.random_sample(train_data, transformer_config.batch_size, transformer_config.max_len)
        encoder_input = encoder_input.to(choose_device())
        decoder_input = decoder_input.to(choose_device())
        target_output = target_output.to(choose_device())

        # 4.2 前向传播
        output = model(encoder_input, decoder_input)
        loss = criterion(output.view(-1, output.shape[-1]), target_output.view(-1))

        # 4.3 反向传播
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()

        # 4.4 保存数据
        if epoch % 200 == 0 and epoch != 0:
            logger.info(f"Epoch {epoch}/{epochs}, Avg Loss: {loss_epoch / 200:.4f}")
            torch.save(model.state_dict(), transformer_config.get_train_file_name())
            loss_epoch = 0


# 训练
train('dataset.txt', 600)

# 推理
#infer('dataset.txt', 'The cloud')
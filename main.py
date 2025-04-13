from os.path import exists
import numpy as np
import torch
from torch import nn, optim
from loguru import logger

from tokenizer.tokenizer import SimpleTokenizer
from transformer.transformer_model import TransformerConfig, TransformerModel

transformer_config = TransformerConfig(
        batch_size=16,
        vocab_size=1200,
        embedding_dim=16,
        num_heads=8,
        hidden_dim=16,
        num_encoder_layers=16,
        num_decoder_layers=16,
        max_len=12,
        dropout=0.1
)

def infer(train_data_file, prompt):
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
    print(encoder_input)

    i = tokenizer.get_token_size(transformer_config.max_len, prompt)
    for i in range(transformer_config.max_len - 1):
        output = model(encoder_input, decoder_input)
        output.argmax(dim=-1)

def train(train_data_file, epochs: int):
    # 1. 加载数据集
    train_data = None
    tokenizer = SimpleTokenizer()
    with open(train_data_file, 'r') as f:
        train_data = f.readlines()
        train_data = [line.strip() for line in train_data]
    tokenizer.fit_on_texts(train_data)

    # 2. 构建模型
    model = TransformerModel(transformer_config)
    if exists(transformer_config.get_train_file_name()):
        model.load_state_dict(torch.load(transformer_config.get_train_file_name()))
        logger.info(f"✅ 加载{transformer_config.get_train_file_name()}成功")

    # 3. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 4. 训练
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # 4.1 取数据
        encoder_input, decoder_input, target_output = tokenizer.get_sample(transformer_config.batch_size, train_data, transformer_config.max_len, transformer_config.vocab_size)

        # 4.2 前向传播
        output = model(encoder_input, decoder_input)
        loss = criterion(output, target_output)
        logger.info(f'\rEpoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}')

        # 4.3 反向传播
        loss.backward()
        optimizer.step()

        # 4.4 保存数据
        if epoch % 200 == 0:
            torch.save(model.state_dict(), transformer_config.get_train_file_name())

def choose_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info("🈶 Use cuda")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("🈶 Use metal")
    else:
        device = torch.device('cpu')
        logger.info("🈶 Use cuda")
    return device




train('dataset.txt', 1)
# infer('dataset.txt', 'The ')
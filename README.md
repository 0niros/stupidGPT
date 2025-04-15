# StupidGPT 🤓

> 以下README内容由Cursor生成 🤗

一个基于 Transformer 架构的简单文本生成模型实现。

## 简介

StupidGPT 是一个使用 PyTorch 实现的简化版 Transformer 模型，用于文本生成任务。该项目旨在提供一个简单易懂的 Transformer 实现，帮助理解 Transformer 架构的核心概念。

## 功能

- 基于 Transformer 架构的文本生成模型
- 支持训练和推理功能
- 使用简单的分词器进行文本处理
- 支持 GPU 加速（如果可用）
- 模型参数可配置

## 说明

1. 克隆项目到本地：
```bash
git clone [项目地址]
cd stupidGPT
```

2. 创建并激活虚拟环境（推荐）：
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate  # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

### 训练模型

```python
from main import train

# 训练模型
train('dataset.txt', epochs=10000)
```

### 模型推理

```python
from main import infer

# 使用模型生成文本
infer('dataset.txt', 'The cloud')
```

## 项目结构

```
stupidGPT/
├── main.py              # 主程序入口
├── requirements.txt     # 项目依赖
├── dataset.txt         # 训练数据集
├── model_*.bin         # 模型参数文件
├── tokenizer/          # 分词器模块
├── transformer/        # Transformer 模型实现
├── utils/             # 工具函数
├── layers/            # 神经网络层实现
└── mask/              # 注意力掩码相关
```

## 详细说明

### Tokenizer 模块

Tokenizer 是文本处理的核心组件，负责将原始文本转换为模型可以处理的数字序列。本项目实现了 `SimpleTokenizer` 类，具有以下特点：

#### 主要功能

1. **词汇表管理**
   - 自动构建词汇表（vocabulary）
   - 支持动态添加新词
   - 维护词到ID和ID到词的双向映射
   - 特殊token处理（如空格和句号）

2. **文本编码与解码**
   - 将文本转换为token ID序列
   - 支持最大长度限制
   - 支持填充（padding）
   - 将token ID序列转换回文本

3. **训练数据处理**
   - 支持批量采样
   - 自动处理编码器输入、解码器输入和目标输出
   - 支持序列长度对齐

#### 主要方法

1. **初始化与配置**
   ```python
   tokenizer = SimpleTokenizer()
   ```

2. **构建词汇表**
   ```python
   tokenizer.fit_on_texts(texts)  # texts为文本列表
   ```

3. **文本编码**
   ```python
   # 编码文本，支持最大长度和填充
   token_ids = tokenizer.encode(text, max_length=12, pad=True)
   ```

4. **文本解码**
   ```python
   # 将token ID序列转换回文本
   text = tokenizer.decode(token_ids)
   ```

5. **训练数据采样**
   ```python
   # 获取训练批次数据
   enc_inputs, dec_inputs, outputs = tokenizer.random_sample(dataset, batch_size=16, seq_len=12)
   ```

#### 特殊处理

1. **特殊Token**
   - 空格（ID: 0）
   - 句号（ID: 1）
   - 其他词从ID 2开始分配

2. **大小写处理**
   - 所有词都会被转换为小写
   - 保持词汇表的一致性

3. **填充处理**
   - 使用空格（ID: 0）进行填充
   - 支持截断过长的序列

#### 使用示例

```python
from tokenizer.tokenizer import SimpleTokenizer

# 初始化tokenizer
tokenizer = SimpleTokenizer()

# 训练数据
texts = ["Hello world", "This is a test"]

# 构建词汇表
tokenizer.fit_on_texts(texts)

# 编码文本
encoded = tokenizer.encode("Hello world", max_length=5, pad=True)
# 输出: [token_id1, token_id2, 0, 0, 0]

# 解码
decoded = tokenizer.decode(encoded)
# 输出: "hello world"
```

### Mask 模块

Mask 模块负责处理 Transformer 模型中的注意力掩码，主要包括三种类型的掩码：

#### 1. Padding Mask

用于屏蔽输入序列中的填充部分，确保模型不会关注填充的 token。

```python
def create_padding_mask(seq, pad_token=0):
    """
    创建 Padding Mask
    :param seq: 输入序列，形状：(batch_size, seq_len)
    :param pad_token: 填充的TokenId，默认为0
    :return: PaddingMask (batch_size, 1, 1, seq_len)
    """
```

#### 2. Causal Mask

用于确保解码器在预测当前位置时看不到未来的词，实现自回归生成。

```python
def create_causal_mask(seq_len):
    """
    创建CausalMask
    :param seq_len: 序列长度
    :return: (1, seq_len, seq_len)
    """
```

#### 3. Combine Mask

将 Padding Mask 和 Causal Mask 组合在一起，用于解码器的注意力计算。

```python
def create_combine_mask(padding_mask, causal_mask):
    """
    组合PaddingMask和CausalMask
    :param padding_mask: PaddingMask
    :param causal_mask: CausalMask
    :return: CombineMask
    """
```

### Transformer 模块

Transformer 模块实现了完整的 Transformer 架构，包括编码器、解码器和相关组件。

#### 1. 模型配置

通过 `TransformerConfig` 类配置模型参数：

```python
config = TransformerConfig(
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
```

#### 2. 编码器（Encoder）

编码器由多个 `EncoderLayer` 组成，每个层包含：
- 多头自注意力机制
- 前馈神经网络
- 残差连接和层归一化

```python
class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        self.self_attention = MultiHeadAttentionLayer(embed_dim, num_heads, dropout)
        self.add_and_norm1 = AddAndNormLayer(embed_dim, dropout)
        self.feed_forward = FeedForwardLayer(embed_dim, hidden_dim, dropout)
        self.add_and_norm2 = AddAndNormLayer(embed_dim, dropout)
```

#### 3. 解码器（Decoder）

解码器由多个 `DecoderLayer` 组成，每个层包含：
- 带掩码的多头自注意力机制
- 编码器-解码器注意力机制
- 前馈神经网络
- 残差连接和层归一化

```python
class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        self.self_attention = MultiHeadAttentionLayer(embed_dim, num_heads, dropout)
        self.add_and_norm1 = AddAndNormLayer(embed_dim, dropout)
        self.encoder_decoder_attention = MultiHeadAttentionLayer(embed_dim, num_heads, dropout)
        self.add_and_norm2 = AddAndNormLayer(embed_dim, dropout)
        self.feed_forward = FeedForwardLayer(embed_dim, hidden_dim, dropout)
        self.add_and_norm3 = AddAndNormLayer(embed_dim, dropout)
```

#### 4. 完整模型

`TransformerModel` 类整合了所有组件：

```python
class TransformerModel(nn.Module):
    def __init__(self, config: TransformerConfig):
        self.enc_embedding = EmbeddingLayer(...)
        self.encoder = nn.Sequential([EncoderLayer(...) for _ in range(config.num_encoder_layers)])
        self.dec_embedding = EmbeddingLayer(...)
        self.decoder = nn.Sequential([DecoderLayer(...) for _ in range(config.num_decoder_layers)])
        self.linear = nn.Linear(...)
```

#### 5. 前向传播流程

1. 创建编码器和解码器的掩码
2. 对输入进行嵌入
3. 通过编码器处理输入序列
4. 通过解码器生成输出序列
5. 通过线性层和softmax得到最终输出

## 配置说明

模型的主要配置参数在 `main.py` 中定义，包括：
- 批次大小（batch_size）
- 词库大小（vocab_size）
- 嵌入维度（embedding_dim）
- 注意力头数（num_heads）
- 隐藏层维度（hidden_dim）
- 编码器层数（num_encoder_layers）
- 解码器层数（num_decoder_layers）
- 最大序列长度（max_len）
- Dropout 比率（dropout）

## 依赖项

- torch ~= 2.6.0
- numpy
- loguru

## 注意事项

- 确保有足够的训练数据
- 训练过程可能需要较长时间，建议使用 GPU 加速
- 模型参数文件会自动保存，可以用于后续推理
- Tokenizer 的词汇表大小会影响模型性能和内存使用
- 建议在训练前对文本数据进行预处理，确保质量
- 注意调整模型参数以适应不同的任务需求
- 合理设置序列长度和批次大小以平衡性能和内存使用

## 效果

### 训练
![](https://github.com/0niros/stupidGPT/doc/image/train.png)

### 推理
![](https://github.com/0niros/stupidGPT/doc/image/infer.png)
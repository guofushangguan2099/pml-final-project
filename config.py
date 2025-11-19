# config.py

import torch

# --- 数据和模型路径 ---
DATA_PATH = "data/processed_data.pkl"
MODEL_SAVE_PATH = "saved_models/seq2seq_model.pt"

# --- 模型超参数 ---
# 注意力机制开关
USE_ATTENTION = True

# 词嵌入层维度
EMBEDDING_DIM = 256

# GRU 隐藏层维度
HIDDEN_DIM = 512

# GRU 层数
N_LAYERS = 2

# Dropout 概率 (用于防止过拟合)
DROPOUT = 0.5

# --- 训练超参数 ---
# 批量大小
BATCH_SIZE = 

# 训练轮次 (Epochs)
N_EPOCHS = 20

# 学习率
LEARNING_RATE = 0.001

# 教师强制比率 (Teacher Forcing Ratio)
# 在训练初期，我们有 50% 的概率使用真实的目标词作为解码器的下一个输入，
# 而不是使用解码器自己生成的词。这有助于稳定训练。
TEACHER_FORCING_RATIO = 0.5

# 梯度裁剪阈值 (防止梯度爆炸)
CLIP = 1

# --- 设备配置 ---
# 自动选择可用的设备 (优先使用CUDA GPU)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 打印和保存设置 ---
PRINT_EVERY = 100  # 每隔多少个 batch 打印一次训练状态```

### **为什么这样做？**


# config.py

import torch

DATA_PATH = "data/processed_data.pkl"
MODEL_SAVE_PATH = "saved_models/seq2seq_model.pt"

# --- 模型超参数（大模型配置）---
USE_ATTENTION = True

EMBEDDING_DIM = 256      # 回到大模型
HIDDEN_DIM = 512         # 回到大模型
N_LAYERS = 2             # 2 层（如果原来是 1，这是新的改进）

DROPOUT = 0.5            # 适中的 dropout（不是 0.3 也不是 0.7）

# --- 训练超参数（关键调整）---
BATCH_SIZE = 128
N_EPOCHS = 30            # 给足够的时间

# 核心改进：非常小的学习率 + 学习率调度
LEARNING_RATE = 0.001   # 从 0.001/0.0005 降到 0.0002

# 降低 Teacher Forcing（减少训练测试差异）
TEACHER_FORCING_RATIO = 0.5  # 从 0.5 降到 0.3

CLIP = 1

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRINT_EVERY = 100

# Early Stopping（更宽容）
PATIENCE = 7             # 从 3 增加到 7（给模型更多机会）
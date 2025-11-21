# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import pickle
import time
import math

# --- 导入我们自己的模块 ---
from model import Encoder, Attention, Decoder, Seq2Seq
from config import (
    DATA_PATH, MODEL_SAVE_PATH, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS,
    DROPOUT, BATCH_SIZE, N_EPOCHS, LEARNING_RATE, TEACHER_FORCING_RATIO,
    CLIP, DEVICE, PRINT_EVERY
)

# --- 1. 数据集类和数据加载器 ---
from new_preprocess import Vocabulary
class TranslationDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        src = torch.tensor(self.df.iloc[idx]['modern_numerical'], dtype=torch.long)
        trg = torch.tensor(self.df.iloc[idx]['shakespearean_numerical'], dtype=torch.long)
        return src, trg

def create_collate_fn(pad_idx):
    def collate_fn(batch):
        src_batch, trg_batch = [], []
        for src_sample, trg_sample in batch:
            src_batch.append(src_sample)
            trg_batch.append(trg_sample)
        
        src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
        trg_padded = pad_sequence(trg_batch, batch_first=True, padding_value=pad_idx)
        return src_padded, trg_padded
    return collate_fn

# --- 2. 训练和评估函数 ---

def train_fn(model, dataloader, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, (src, trg) in enumerate(dataloader):
        src, trg = src.to(DEVICE), trg.to(DEVICE)
        
        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio=TEACHER_FORCING_RATIO)
        
        # trg = [batch_size, trg_len]
        # output = [batch_size, trg_len, output_dim]
        output_dim = output.shape[-1]
        
        # 调整形状以计算损失 (去掉<sos>标记)
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
        
        if (i + 1) % PRINT_EVERY == 0:
            print(f"  Batch {i+1}/{len(dataloader)} | Loss: {loss.item():.4f}")

    return epoch_loss / len(dataloader)

def evaluate_fn(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, (src, trg) in enumerate(dataloader):
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            
            # 关闭教师强制进行评估
            output = model(src, trg, teacher_forcing_ratio=0) 
            
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            
            loss = criterion(output, trg)
            epoch_loss += loss.item()
            
    return epoch_loss / len(dataloader)

# --- 辅助函数 ---
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# --- 3. 主训练流程 ---
def run_training():
    print("--- Starting Training Process ---")
    print(f"Device: {DEVICE}")
    
    # 1. 加载数据与词汇表
    print("Loading data and vocabularies...")
    df = pd.read_pickle(DATA_PATH)
    
    with open('data/modern_vocab.pkl', 'rb') as f:
        modern_vocab = pickle.load(f)
    with open('data/shakespearean_vocab.pkl', 'rb') as f:
        shakespearean_vocab = pickle.load(f)
        
    # 根据词汇表设置输入输出维度
    INPUT_DIM = modern_vocab.n_words
    OUTPUT_DIM = shakespearean_vocab.n_words
    PAD_IDX = modern_vocab.word2idx['<pad>']
    
    print(f"Vocab Sizes - Input: {INPUT_DIM}, Output: {OUTPUT_DIM}")

    # 2. 数据划分 (70% Train, 15% Val, 15% Test)
    #先打乱数据，确保随机性
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    n_total = len(df)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    # 剩下的给测试集
    
    train_df = df.iloc[:n_train]
    valid_df = df.iloc[n_train : n_train + n_val]
    test_df = df.iloc[n_train + n_val :]
    
    print(f"\nData Split Result:")
    print(f"  Training Set:   {len(train_df)} samples")
    print(f"  Validation Set: {len(valid_df)} samples")
    print(f"  Test Set:       {len(test_df)} samples")

    # 3. 创建 DataLoader
    train_dataset = TranslationDataset(train_df)
    valid_dataset = TranslationDataset(valid_df)
    test_dataset = TranslationDataset(test_df) # 新增测试集

    collate_fn = create_collate_fn(PAD_IDX)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # 4. 初始化模型
    print("\nInitializing model...")
    attn = Attention(HIDDEN_DIM)
    enc = Encoder(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT)
    dec = Decoder(OUTPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT, attn)
    model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
    
    # 初始化参数权重 (推荐步骤，有助于模型收敛)
    def init_weights(m):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)
    model.apply(init_weights)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    # 5. 定义优化器和损失函数
    # ✅ 关键修改：添加 weight_decay=1e-5 进行正则化
    optimizer = optim.Adam(
    model.parameters(), 
    lr=LEARNING_RATE,
    weight_decay=1e-4,      # L2 正则化（原来是 1e-5，增强到 5e-5）
    betas=(0.9, 0.98),      # Adam 的动量参数（更适合 NLP）
    eps=1e-9                # 数值稳定性
)
    criterion = nn.CrossEntropyLoss(
    ignore_index=PAD_IDX,
    # label_smoothing=0.1     # 防止模型过度自信
)
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',              # 监控最小化指标（loss）
    factor=0.5,              # LR 衰减因子（每次减半）
    patience=4,              # 4 个 epoch 没改善就降低学习率
    min_lr=1e-6              # 最小学习率
)
    # 6. 训练循环 (含早停机制)
    best_valid_loss = float('inf')
    PATIENCE = 7      # ✅ 早停耐心值：如果验证集3次没有变好就停止
    patience_counter = 0
    
    print("\n--- Starting Epochs (with Early Stopping) ---")
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        
        print(f"Epoch {epoch+1}/{N_EPOCHS}")
        
        # 训练和验证
        train_loss = train_fn(model, train_dataloader, optimizer, criterion, CLIP)
        valid_loss = evaluate_fn(model, valid_dataloader, criterion)  # ← 只调用一次
        
        # ✅ 更新学习率
        scheduler.step(valid_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 计算时间
        end_time = time.time()  # ← 只计算一次
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        # 计算 PPL
        train_ppl = math.exp(train_loss)
        val_ppl = math.exp(valid_loss)
        
        # --- 早停与模型保存逻辑 ---
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0 # 重置计数器
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            save_msg = "✅ Model Saved (New Best)"
        else:
            patience_counter += 1
            save_msg = f"⚠️ No Improvement ({patience_counter}/{PATIENCE})"
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s | LR: {current_lr:.6f} | {save_msg}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {train_ppl:7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {val_ppl:7.3f}')
        
        # 触发早停
        if patience_counter >= PATIENCE:
            print(f"\n🛑 Early stopping triggered! Best validation loss was {best_valid_loss:.3f}")
            break

    # 7. 最终测试 (Test Set Evaluation)
    print("\n--- Training Finished. Evaluating on Independent Test Set ---")
    # 加载保存的最佳模型 (防止使用的是最后一次过拟合的参数)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    
    test_loss = evaluate_fn(model, test_dataloader, criterion)
    test_ppl = math.exp(test_loss)
    
    print(f"{'='*40}")
    print(f"FINAL TEST RESULTS")
    print(f"{'='*40}")
    print(f"Test Loss: {test_loss:.3f}")
    print(f"Test PPL:  {test_ppl:.3f}")
    print(f"{'='*40}")
    
    return model


if __name__ == '__main__':
    run_training()

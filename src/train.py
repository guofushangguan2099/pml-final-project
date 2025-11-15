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
from .model import Encoder, Attention, Decoder, Seq2Seq
from config import (
    DATA_PATH, MODEL_SAVE_PATH, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS,
    DROPOUT, BATCH_SIZE, N_EPOCHS, LEARNING_RATE, TEACHER_FORCING_RATIO,
    CLIP, DEVICE, PRINT_EVERY
)

# --- 1. 数据集类和数据加载器 ---
from .preprocess import Vocabulary
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

    # 加载数据和词汇表
    print("Loading data and vocabularies...")
    df = pd.read_pickle(DATA_PATH)
    with open('data/modern_vocab.pkl', 'rb') as f:
        modern_vocab = pickle.load(f)
    with open('data/shakespearean_vocab.pkl', 'rb') as f:
        shakespearean_vocab = pickle.load(f)

    INPUT_DIM = modern_vocab.n_words
    OUTPUT_DIM = shakespearean_vocab.n_words
    PAD_IDX = modern_vocab.word2idx['<pad>']

    # 创建 DataLoader
    # 为了演示，我们简单地将数据分为训练集和验证集 (80/20)
    train_size = int(0.8 * len(df))
    train_df = df[:train_size]
    valid_df = df[train_size:]

    train_dataset = TranslationDataset(train_df)
    valid_dataset = TranslationDataset(valid_df)
    
    collate_fn = create_collate_fn(PAD_IDX)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    print(f"Train batches: {len(train_dataloader)}, Validation batches: {len(valid_dataloader)}")

    # 初始化模型
    print("Initializing model...")
    attn = Attention(HIDDEN_DIM)
    enc = Encoder(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT)
    dec = Decoder(OUTPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT, attn)
    model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
    
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX) # 忽略填充标记的损失

    best_valid_loss = float('inf')

    # 开始训练循环
    print("\n--- Starting Epochs ---")
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        
        print(f"Epoch {epoch+1}/{N_EPOCHS}")
        train_loss = train_fn(model, train_dataloader, optimizer, criterion, CLIP)
        valid_loss = evaluate_fn(model, valid_dataloader, criterion)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        # 如果当前验证损失是最好的，则保存模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    print("\n--- Training Finished ---")
    print(f"Best validation loss: {best_valid_loss:.3f}")
    print(f"Model saved to {MODEL_SAVE_PATH}")


if __name__ == '__main__':
    run_training()

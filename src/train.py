# src/train.py
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
import pickle
import time
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau

"""
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 获取项目根目录 (src 的上一级) -> .../pml-final-project
project_root = os.path.dirname(current_dir)
print(project_root)
# 3. 将根目录加入路径 -> 解决 "No module named 'config'"
sys.path.append(project_root)

# 4. 将 src 目录加入路径 -> 解决 "No module named 'model'"
sys.path.append(current_dir)
"""

from model import Encoder, Attention, Decoder, Seq2Seq
from config import (
    DATA_PATH, MODEL_SAVE_PATH, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS,
    DROPOUT, BATCH_SIZE, N_EPOCHS, LEARNING_RATE, TEACHER_FORCING_RATIO,
    CLIP, DEVICE, PRINT_EVERY
)

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
    """Create collate function with padding"""
    def collate_fn(batch):
        src_batch, trg_batch = [], []
        for src_sample, trg_sample in batch:
            src_batch.append(src_sample)
            trg_batch.append(trg_sample)
        
        src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
        trg_padded = pad_sequence(trg_batch, batch_first=True, padding_value=pad_idx)
        return src_padded, trg_padded
    return collate_fn

def train_fn(model, dataloader, optimizer, criterion, clip):
    """Single training epoch"""
    model.train()
    epoch_loss = 0
    for i, (src, trg) in enumerate(dataloader):
        src, trg = src.to(DEVICE), trg.to(DEVICE)
        
        optimizer.zero_grad()
        output, *rest = model(src, trg, teacher_forcing_ratio=TEACHER_FORCING_RATIO)
        
        # output: [batch_size, trg_len, output_dim]
        output_dim = output.shape[-1]
        
        # Reshape for loss calculation, skip <sos> token
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
    """Evaluate model on validation/test set"""
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, (src, trg) in enumerate(dataloader):
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            
            # No teacher forcing during evaluation
            output, *rest = model(src, trg, teacher_forcing_ratio=0) 
            
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            
            loss = criterion(output, trg)
            epoch_loss += loss.item()
            
    return epoch_loss / len(dataloader)

def epoch_time(start_time, end_time):
    """Calculate elapsed time"""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def load_pretrained_embeddings(vocab, glove_file_path, emb_dim):
    """
    1. 读取 GloVe 文件
    2. 创建一个形状为 (vocab_size, emb_dim) 的矩阵
    3. 如果单词在 GloVe 中存在，填入 GloVe 向量；否则保留随机初始化
    """
    print(f"Loading GloVe embeddings from {glove_file_path}...")
    embeddings_index = {}
    
    # 1. 读取 GloVe 向量到内存
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            # 确保转换不出错
            try:
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            except ValueError:
                continue

    # 2. 初始化嵌入矩阵
    # 使用正态分布或均匀分布初始化，保持与 PyTorch 默认初始化相近的方差
    # 这样对于 GloVe 中没有的生僻词（OOV），它们会有合理的初始值
    vocab_size = vocab.n_words
    embedding_matrix = np.random.normal(scale=0.6, size=(vocab_size, emb_dim))
    
    # 3. 匹配你的词汇表
    hits = 0
    for word, idx in vocab.word2idx.items():
        # 尝试直接匹配
        embedding_vector = embeddings_index.get(word)
        # 如果直接匹配失败，尝试小写匹配（GloVe 通常是全小写）
        if embedding_vector is None:
            embedding_vector = embeddings_index.get(word.lower())
            
        if embedding_vector is not None:
            embedding_matrix[idx] = embedding_vector
            hits += 1

    print(f"Loaded {hits} words from GloVe (Coverage: {hits/vocab_size:.1%})")
    
    # 转为 PyTorch Tensor
    return torch.from_numpy(embedding_matrix).float()

def run_training():
    """Main training pipeline with early stopping"""
    print("--- Starting Training Process ---")
    print(f"Device: {DEVICE}")
    
    # Load data and vocabularies
    print("Loading data and vocabularies...")
    df = pd.read_pickle(DATA_PATH)
    
    with open('data/modern_vocab.pkl', 'rb') as f:
        modern_vocab = pickle.load(f)
    with open('data/shakespearean_vocab.pkl', 'rb') as f:
        shakespearean_vocab = pickle.load(f)
        
    INPUT_DIM = modern_vocab.n_words
    OUTPUT_DIM = shakespearean_vocab.n_words
    PAD_IDX = modern_vocab.word2idx['<pad>']
    
    print(f"Vocab Sizes - Input: {INPUT_DIM}, Output: {OUTPUT_DIM}")

    # Split data: 70% train, 15% val, 15% test
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    n_total = len(df)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    
    train_df = df.iloc[:n_train]
    valid_df = df.iloc[n_train : n_train + n_val]
    test_df = df.iloc[n_train + n_val :]
    
    print(f"\nData Split Result:")
    print(f"  Training Set:   {len(train_df)} samples")
    print(f"  Validation Set: {len(valid_df)} samples")
    print(f"  Test Set:       {len(test_df)} samples")

    # Create DataLoaders
    train_dataset = TranslationDataset(train_df)
    valid_dataset = TranslationDataset(valid_df)
    test_dataset = TranslationDataset(test_df)

    collate_fn = create_collate_fn(PAD_IDX)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # Initialize model
    print("\nInitializing model...")
    attn = Attention(HIDDEN_DIM)
    enc = Encoder(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT)
    dec = Decoder(OUTPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT, attn)
    model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)

    def init_weights(m):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)
    
    model.apply(init_weights)

    GLOVE_PATH = 'data/glove.6B.300d.txt' 

    # 1. 为 Encoder (现代英语) 加载预训练向量
    # 这是一个巨大的提升点，因为 Encoder 需要理解输入的语义
    if os.path.exists(GLOVE_PATH):
        # 这里使用 mod_vocab (现代英语词汇表)
        pretrained_weights = load_pretrained_embeddings(modern_vocab, GLOVE_PATH, EMBEDDING_DIM)
        
        # 注入权重
        model.encoder.embedding.weight.data.copy_(pretrained_weights)
        
        # 选项：你可以选择是否同时也为 Decoder (莎士比亚英语) 加载
        # 虽然莎士比亚英语有些词 GloVe 没有，但大部分基础词汇是通用的
        # decoder_weights = load_pretrained_embeddings(shk_vocab, GLOVE_PATH, EMBEDDING_DIM)
        # model.decoder.embedding.weight.data.copy_(decoder_weights)
        
        print("✅ Successfully initialized model with Pre-trained Embeddings!")
        
        # 关键决策：是否冻结 Embedding 层？
        # 建议：设为 False (不冻结)。
        # 让预训练向量作为“极好的初始值”，随着训练微调，适应莎士比亚风格。
        model.encoder.embedding.weight.requires_grad = True
        print("Loading Decoder embeddings...")
        # 注意：这里传入的是 shakespearean_vocab
        dec_weights = load_pretrained_embeddings(shakespearean_vocab, GLOVE_PATH, EMBEDDING_DIM)
        model.decoder.embedding.weight.data.copy_(dec_weights)
        model.decoder.embedding.weight.requires_grad = True
        print("✅ Successfully initialized BOTH Encoder and Decoder with GloVe!") 
    else:
        print(f"⚠️ GloVe file not found at {GLOVE_PATH}. Training from scratch.")


    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    # Optimizer with L2 regularization
    optimizer = optim.Adam(
        model.parameters(), 
        lr=LEARNING_RATE,
        weight_decay=1e-5,
        betas=(0.9, 0.98),
        eps=1e-9
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.9,
        patience=10,
        min_lr=1e-6
    )
    
    # Training loop with early stopping
    best_valid_loss = float('inf')
    PATIENCE = 15
    patience_counter = 0
    
    print("\n--- Starting Epochs (with Early Stopping) ---")
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        
        print(f"Epoch {epoch+1}/{N_EPOCHS}")
        
        train_loss = train_fn(model, train_dataloader, optimizer, criterion, CLIP)
        valid_loss = evaluate_fn(model, valid_dataloader, criterion)
        
        scheduler.step(valid_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        train_ppl = math.exp(train_loss)
        val_ppl = math.exp(valid_loss)
        
        # Early stopping logic
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            save_msg = "✅ Model Saved (New Best)"
        else:
            patience_counter += 1
            save_msg = f"⚠️ No Improvement ({patience_counter}/{PATIENCE})"
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s | LR: {current_lr:.6f} | {save_msg}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {train_ppl:7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {val_ppl:7.3f}')
        
        if patience_counter >= PATIENCE:
            print(f"\n🛑 Early stopping triggered! Best validation loss was {best_valid_loss:.3f}")
            break

    # Final test evaluation
    print("\n--- Training Finished. Evaluating on Independent Test Set ---")
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
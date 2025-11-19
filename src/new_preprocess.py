import pandas as pd
import re
import pickle
import os
from collections import Counter

# --- 配置 ---
# 请确保 final.csv 和这个脚本在同一个文件夹，或者修改这里的路径
CSV_PATH = 'data\\final.csv' 
SAVE_DIR = 'data'
MIN_FREQ = 2  # 关键！去掉只出现一次的词，大幅降低噪音

# --- 1. 强力清洗函数 ---
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    # 关键改进：把冒号、分号、括号、破折号等全部切分开
    # 例如: "king:" -> "king :"
    text = re.sub(r"([?.!,:;\"'()\-])", r" \1 ", text)
    # 把多余空格缩减
    text = re.sub(r'[" "]+', " ", text)
    return f"<s> {text.strip()} </s>"

# --- 2. 词汇表构建类 (带频率过滤) ---
class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.word2idx = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<s>", 2: "</s>", 3: "<unk>"}
        self.word_count = Counter()
        self.n_words = 4

    def build_vocab(self, sentences, min_freq=1):
        # 1. 统计所有词频
        print(f"Building vocab for {self.name}...")
        temp_counter = Counter()
        for sentence in sentences:
            temp_counter.update(sentence.split())
        
        # 2. 只添加超过 min_freq 的词
        for word, count in temp_counter.items():
            if count >= min_freq:
                self.word2idx[word] = self.n_words
                self.idx2word[self.n_words] = word
                self.n_words += 1
            # 否则这个词以后会被转为 <unk>
        
        print(f"  Original tokens: {len(temp_counter)}")
        print(f"  Kept tokens (>= {min_freq}): {self.n_words}")
        print(f"  Filtered out: {len(temp_counter) - self.n_words + 4}")

    def numericalize(self, sentence):
        # 把句子转成数字 ID，不在词表里的转为 <unk>
        return [
            self.word2idx.get(word, self.word2idx["<unk>"]) 
            for word in sentence.split()
        ]

# --- 3. 主处理流程 ---
def run_preprocessing():
    print("--- 1. Loading Data ---")
    
    # 检查文件是否存在
    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: '{CSV_PATH}' not found!")
        print(f"   Current working directory is: {os.getcwd()}")
        print("   Please make sure final.csv is in this folder.")
        return
    
    try:
        # 增加 encoding='utf-8' 防止 Windows 读取报错
        df = pd.read_csv(CSV_PATH, encoding='utf-8')
    except UnicodeDecodeError:
        print("⚠️ UTF-8 failed, trying latin-1...")
        df = pd.read_csv(CSV_PATH, encoding='latin-1')
    
    # 自动处理列名
    if 'og' in df.columns and 't' in df.columns:
        df.rename(columns={'t': 'modern', 'og': 'shakespearean'}, inplace=True)
    elif 'Modern English' in df.columns: 
        df.rename(columns={'Modern English': 'modern', 'Shakespeare English': 'shakespearean'}, inplace=True)
    
    print(f"Loaded {len(df)} rows.")
    
    # 清洗文本
    print("\n--- 2. Cleaning Text (Deep Clean) ---")
    df['modern_clean'] = df['modern'].apply(clean_text)
    df['shakespearean_clean'] = df['shakespearean'].apply(clean_text)
    
    print("Example Modern:", df['modern_clean'].iloc[0])
    print("Example Shakespeare:", df['shakespearean_clean'].iloc[0])
    
    # 构建词汇表
    print("\n--- 3. Building Vocabulary ---")
    modern_vocab = Vocabulary('modern')
    shakespeare_vocab = Vocabulary('shakespearean')
    
    # 仅使用频率 >= 2 的词
    modern_vocab.build_vocab(df['modern_clean'], min_freq=MIN_FREQ)
    shakespeare_vocab.build_vocab(df['shakespearean_clean'], min_freq=MIN_FREQ)
    
    # 数字化 (Numericalization)
    print("\n--- 4. Converting to Numbers ---")
    df['modern_numerical'] = df['modern_clean'].apply(modern_vocab.numericalize)
    df['shakespearean_numerical'] = df['shakespearean_clean'].apply(shakespeare_vocab.numericalize)
    
    # 保存文件
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    print("\n--- 5. Saving Files ---")
    df.to_pickle(os.path.join(SAVE_DIR, 'processed_data.pkl'))
    
    with open(os.path.join(SAVE_DIR, 'modern_vocab.pkl'), 'wb') as f:
        pickle.dump(modern_vocab, f)
        
    with open(os.path.join(SAVE_DIR, 'shakespearean_vocab.pkl'), 'wb') as f:
        pickle.dump(shakespeare_vocab, f)
        
    print("✅ Done! New data is ready.")
    print(f"Modern Vocab Size: {modern_vocab.n_words}")
    print(f"Shakespeare Vocab Size: {shakespeare_vocab.n_words}")

if __name__ == "__main__":
    run_preprocessing()
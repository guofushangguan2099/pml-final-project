import pandas as pd
import re
from collections import Counter

# --- 第1步：文本清洗与规范化 ---
def normalize_sentence(s):
    """
    对句子进行清洗和规范化处理
    """
    # 转换为小写
    s = s.lower().strip()
    # 在标点符号周围添加空格
    s = re.sub(r"([?.!,])", r" \1 ", s)
    # 将多个空格替换为单个空格
    s = re.sub(r'[" "]+', " ", s)
    # 添加句首和句尾标记
    s = f"<s> {s} </s>"
    return s

# --- 第2步：构建词汇表 ---
class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.word2idx = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<s>", 2: "</s>", 3: "<unk>"}
        
        # <--- 修改：在这里初始化特殊标记的计数
        self.word_count = {"<pad>": 0, "<s>": 0, "</s>": 0, "<unk>": 0}
        
        self.n_words = 4  # 计算 PAD, SOS, EOS, UNK

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2idx:
            # Word is new, add it
            self.word2idx[word] = self.n_words
            self.word_count[word] = 1
            self.idx2word[self.n_words] = word
            self.n_words += 1
        else:
            # Word is already known (including special tokens)
            self.word_count[word] += 1

# --- 主处理函数 ---
def preprocess_data(df):
    """
    对包含 't' (modern) 和 'og' (shakespearean) 列的DataFrame进行完整的预处理
    """
    # 1. 清洗所有句子
    # <--- 修改：使用 't' 列 (现代英文)
    df['modern_clean'] = df['t'].apply(normalize_sentence)
    # <--- 修改：使用 'og' 列 (莎士比亚式英文)
    df['shakespearean_clean'] = df['og'].apply(normalize_sentence)
    print("Sentences cleaned and normalized.")
    print("\nExample of cleaned sentence:")
    # <--- 修改：使用 't' 列显示原始示例
    print(f"Original: {df['t'][0]}")
    print(f"Cleaned: {df['modern_clean'][0]}")

    # 2. 创建和构建词汇表
    modern_vocab = Vocabulary('modern')
    shakespearean_vocab = Vocabulary('shakespearean')

    for index, row in df.iterrows():
        modern_vocab.add_sentence(row['modern_clean'])
        shakespearean_vocab.add_sentence(row['shakespearean_clean'])

    print(f"\nVocabularies built.")
    print(f"Modern English vocabulary size: {modern_vocab.n_words}")
    print(f"Shakespearean English vocabulary size: {shakespearean_vocab.n_words}")

    # 3. 数值化句子
    df['modern_numerical'] = df['modern_clean'].apply(
        lambda s: [modern_vocab.word2idx.get(word, modern_vocab.word2idx['<unk>']) for word in s.split(' ')]
    )
    df['shakespearean_numerical'] = df['shakespearean_clean'].apply(
        lambda s: [shakespearean_vocab.word2idx.get(word, shakespearean_vocab.word2idx['<unk>']) for word in s.split(' ')]
    )
    print("\nSentences converted to numerical sequences.")
    print("\nExample of numericalized sentence:")
    print(f"Cleaned sentence tokens: {df['modern_clean'][0].split(' ')}")
    print(f"Numerical sequence: {df['modern_numerical'][0]}")

    return df, modern_vocab, shakespearean_vocab

# 假设我们从 dataset.py 导入了加载函数
# 注意：您可能也需要解决 'src' 导入问题
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.dataset import load_and_prepare_data


if __name__ == '__main__':
    # ... (前面的代码不变) ...
    dataframe = load_and_prepare_data('data\\final.csv')
    if dataframe is not None:
        processed_df, mod_vocab, shk_vocab = preprocess_data(dataframe)
        
        print("\n--- Preprocessing Complete ---")
        if not os.path.exists('data'):
            os.makedirs('data')
            
        processed_df.to_pickle('data/processed_data.pkl')
        print("Processed DataFrame saved to 'data/processed_data.pkl'")

        # --- 新增部分：保存词汇表 ---
        import pickle
        with open('data/modern_vocab.pkl', 'wb') as f:
            pickle.dump(mod_vocab, f)
        with open('data/shakespearean_vocab.pkl', 'wb') as f:
            pickle.dump(shk_vocab, f)
        print("Vocabularies saved to 'data/' directory.")
import pandas as pd
import re
from collections import Counter

def normalize_sentence(s):
    """Clean and normalize sentence with special tokens"""
    s = s.lower().strip()
    # Add spacing around punctuation
    s = re.sub(r"([?.!,])", r" \1 ", s)
    s = re.sub(r'[" "]+', " ", s)
    # Wrap with start/end tokens
    s = f"<s> {s} </s>"
    return s

class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.word2idx = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<s>", 2: "</s>", 3: "<unk>"}
        self.word_count = {"<pad>": 0, "<s>": 0, "</s>": 0, "<unk>": 0}
        self.n_words = 4  # PAD, SOS, EOS, UNK

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.word_count[word] = 1
            self.idx2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word_count[word] += 1

def preprocess_data(df):
    """
    Process DataFrame with 't' (modern) and 'og' (shakespearean) columns
    Returns cleaned data and vocabularies
    """
    # Clean sentences
    df['modern_clean'] = df['t'].apply(normalize_sentence)
    df['shakespearean_clean'] = df['og'].apply(normalize_sentence)
    print("Sentences cleaned and normalized.")
    print("\nExample of cleaned sentence:")
    print(f"Original: {df['t'][0]}")
    print(f"Cleaned: {df['modern_clean'][0]}")

    # Build vocabularies
    modern_vocab = Vocabulary('modern')
    shakespearean_vocab = Vocabulary('shakespearean')

    for index, row in df.iterrows():
        modern_vocab.add_sentence(row['modern_clean'])
        shakespearean_vocab.add_sentence(row['shakespearean_clean'])

    print(f"\nVocabularies built.")
    print(f"Modern English vocabulary size: {modern_vocab.n_words}")
    print(f"Shakespearean English vocabulary size: {shakespearean_vocab.n_words}")

    # Convert to numerical sequences
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

# Setup path for imports
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.dataset import load_and_prepare_data


if __name__ == '__main__':
    dataframe = load_and_prepare_data('data\\final.csv')
    if dataframe is not None:
        processed_df, mod_vocab, shk_vocab = preprocess_data(dataframe)
        
        print("\n--- Preprocessing Complete ---")
        if not os.path.exists('data'):
            os.makedirs('data')
            
        processed_df.to_pickle('data/processed_data.pkl')
        print("Processed DataFrame saved to 'data/processed_data.pkl'")

        # Save vocabularies
        import pickle
        with open('data/modern_vocab.pkl', 'wb') as f:
            pickle.dump(mod_vocab, f)
        with open('data/shakespearean_vocab.pkl', 'wb') as f:
            pickle.dump(shk_vocab, f)
        print("Vocabularies saved to 'data/' directory.")
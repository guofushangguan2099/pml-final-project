# src/test.py

import torch
import pickle
import re
import sys
import os
import math
import argparse
from collections import Counter

# Vocabulary class for token mapping
class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.word2idx = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<s>", 2: "</s>", 3: "<unk>"}
        self.word_count = Counter()
        self.n_words = 4

    def build_vocab(self, sentences, min_freq=1):
        temp_counter = Counter()
        for sentence in sentences:
            temp_counter.update(sentence.split())
        for word, count in temp_counter.items():
            if count >= min_freq:
                self.word2idx[word] = self.n_words
                self.idx2word[self.n_words] = word
                self.n_words += 1

    def numericalize(self, sentence):
        return [
            self.word2idx.get(word, self.word2idx["<unk>"]) 
            for word in sentence.split()
        ]

# Pickle compatibility fix
# If vocab was saved from new_preprocess module, inject our class there
try:
    import new_preprocess
except ImportError:
    import types
    new_preprocess = types.ModuleType('new_preprocess')
    sys.modules['new_preprocess'] = new_preprocess

new_preprocess.Vocabulary = Vocabulary

# Setup import paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from model import Encoder, Attention, Decoder, Seq2Seq
    from config import *
except ImportError:
    print("❌ Error: Cannot find model.py or config.py")
    sys.exit(1)

def load_vocab():
    """Load vocabulary files with pickle compatibility patch"""
    vocab_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    with open(os.path.join(vocab_dir, 'modern_vocab.pkl'), 'rb') as f:
        modern_vocab = pickle.load(f)
    with open(os.path.join(vocab_dir, 'shakespearean_vocab.pkl'), 'rb') as f:
        shakespearean_vocab = pickle.load(f)
    return modern_vocab, shakespearean_vocab

def load_model(modern_vocab, shakespearean_vocab, device):
    """Initialize model and load trained weights"""
    INPUT_DIM = modern_vocab.n_words
    OUTPUT_DIM = shakespearean_vocab.n_words
    
    attn = Attention(HIDDEN_DIM)
    enc = Encoder(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT)
    dec = Decoder(OUTPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT, attn)
    model = Seq2Seq(enc, dec, device).to(device)
    
    model_path = os.path.join(os.path.dirname(__file__), '..', MODEL_SAVE_PATH.replace('models/', ''))
    if not os.path.exists(model_path):
        model_path = MODEL_SAVE_PATH
        
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Model loaded: {model_path}")
    else:
        print(f"❌ Error: Model file not found at {model_path}")
        sys.exit(1)
        
    model.eval()
    return model

def preprocess_text(text):
    """Normalize and add special tokens"""
    text = text.lower().strip()
    text = re.sub(r"([?.!,:;\"'()\-])", r" \1 ", text)
    text = re.sub(r'[" "]+', " ", text)
    return f"<s> {text.strip()} </s>"

def translate_sentence_beam(sentence, model, modern_vocab, shakespearean_vocab, device, max_len=50, beam_width=5):
    """Beam search translation"""
    model.eval()
    processed = preprocess_text(sentence)
    tokens = [modern_vocab.word2idx.get(word, modern_vocab.word2idx['<unk>']) for word in processed.split()]
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
    
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)
        start_token = shakespearean_vocab.word2idx['<s>']
        end_token = shakespearean_vocab.word2idx['</s>']
        
        # (score, hidden, sequence)
        candidates = [(0.0, hidden, [start_token])]
        
        for _ in range(max_len):
            new_candidates = []
            for score, curr_hidden, seq in candidates:
                if seq[-1] == end_token:
                    new_candidates.append((score, curr_hidden, seq))
                    continue
                
                input_token = torch.LongTensor([seq[-1]]).to(device)
                output, next_hidden = model.decoder(input_token, curr_hidden, encoder_outputs)
                probs = torch.softmax(output, dim=1)
                
                topk_probs, topk_ids = torch.topk(probs, beam_width * 2)
                
                for i in range(beam_width * 2):
                    word_idx = topk_ids[0][i].item()
                    word_prob = topk_probs[0][i].item()
                    new_score = score + math.log(word_prob + 1e-10)
                    new_candidates.append((new_score, next_hidden, seq + [word_idx]))
            
            # Keep top beam_width candidates
            candidates = sorted(new_candidates, key=lambda x: x[0], reverse=True)[:beam_width]
            if all(c[2][-1] == end_token for c in candidates): break
                
        best_seq = candidates[0][2]
        trg_tokens = [shakespearean_vocab.idx2word[i] for i in best_seq]
        result = [t for t in trg_tokens if t not in ['<s>', '</s>', '<pad>']]
        return ' '.join(result)

def calculate_bleu(n_samples=100):
    """Evaluate model with BLEU scores"""
    print("="*70)
    print(f"📊 Calculating BLEU (Beam Width: 5)")
    print("="*70)
    
    try:
        from nltk.translate.bleu_score import corpus_bleu
        import nltk
        try: nltk.data.find('tokenizers/punkt')
        except LookupError: nltk.download('punkt', quiet=True)
    except ImportError:
        print("⚠️ Please install nltk: pip install nltk")
        return

    import pandas as pd
    modern_vocab, shakespearean_vocab = load_vocab()
    model = load_model(modern_vocab, shakespearean_vocab, DEVICE)
    
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed_data.pkl')
    df = pd.read_pickle(data_path)
    
    # Use last 15% as test set
    test_start_idx = int(0.85 * len(df))
    test_df = df[test_start_idx:]
    test_samples = test_df.sample(min(n_samples, len(test_df)))
    
    references, hypotheses = [], []
    print(f"Processing {len(test_samples)} samples...")
    
    for idx, row in test_samples.iterrows():
        modern = row['modern_clean'].replace('<s>', '').replace('</s>', '').strip()
        shake = row['shakespearean_clean'].replace('<s>', '').replace('</s>', '').strip()
        
        pred = translate_sentence_beam(modern, model, modern_vocab, shakespearean_vocab, DEVICE)
        
        references.append([shake.split()])
        hypotheses.append(pred.split())
        
        if len(hypotheses) % 50 == 0: print(f"  Completed {len(hypotheses)}")
    
    # Calculate all BLEU scores
    b1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    b2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
    b3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
    b4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    
    print("\n" + "="*30)
    print(f"✅ Final Results")
    print("="*30)
    print(f"BLEU-1: {b1:.4f}")
    print(f"BLEU-2: {b2:.4f}")
    print(f"BLEU-3: {b3:.4f}")
    print(f"BLEU-4: {b4:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='bleu', choices=['bleu', 'interactive'])
    parser.add_argument('--n_samples', type=int, default=200)
    args = parser.parse_args()
    
    if args.mode == 'bleu':
        calculate_bleu(args.n_samples)
    elif args.mode == 'interactive':
        mod, shk = load_vocab()
        mdl = load_model(mod, shk, DEVICE)
        while True:
            s = input("\nInput: ")
            if s == 'q': break
            print("Output:", translate_sentence_beam(s, mdl, mod, shk, DEVICE))
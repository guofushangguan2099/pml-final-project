import torch
import pickle
import re
import sys
import os
import math
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import Encoder, Attention, Decoder, Seq2Seq
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

def load_vocab():
    """Load modern and Shakespearean vocabularies"""
    vocab_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    with open(os.path.join(vocab_dir, 'modern_vocab.pkl'), 'rb') as f:
        modern_vocab = pickle.load(f)
    with open(os.path.join(vocab_dir, 'shakespearean_vocab.pkl'), 'rb') as f:
        shakespearean_vocab = pickle.load(f)
    
    return modern_vocab, shakespearean_vocab

def load_model(modern_vocab, shakespearean_vocab, device):
    """Initialize and load trained model weights"""
    INPUT_DIM = modern_vocab.n_words
    OUTPUT_DIM = shakespearean_vocab.n_words
    
    attn = Attention(HIDDEN_DIM)
    enc = Encoder(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT)
    dec = Decoder(OUTPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT, attn)
    model = Seq2Seq(enc, dec, device).to(device)
    
    model_path = os.path.join(os.path.dirname(__file__), '..', MODEL_SAVE_PATH.replace('models/', ''))
    if not os.path.exists(model_path):
        model_path = MODEL_SAVE_PATH
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model

def preprocess_text(text):
    """Add special tokens and normalize spacing"""
    text = text.lower().strip()
    text = re.sub(r"([?.!,:;\"'()\-])", r" \1 ", text)
    text = re.sub(r'[" "]+', " ", text)
    return f"<s> {text.strip()} </s>"

def translate_sentence_beam(sentence, model, modern_vocab, shakespearean_vocab, device, max_len=50, beam_width=5):
    """
    Beam search translation with pruning
    Maintains top-k hypotheses at each timestep
    """
    model.eval()
    
    processed = preprocess_text(sentence)
    tokens = [modern_vocab.word2idx.get(word, modern_vocab.word2idx['<unk>']) 
              for word in processed.split()]
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
    
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)
        
        start_token = shakespearean_vocab.word2idx['<s>']
        end_token = shakespearean_vocab.word2idx['</s>']
        
        # (score, hidden_state, token_sequence)
        candidates = [(0.0, hidden, [start_token])]
        
        for _ in range(max_len):
            new_candidates = []
            
            for score, curr_hidden, seq in candidates:
                # Skip finished sequences
                if seq[-1] == end_token:
                    new_candidates.append((score, curr_hidden, seq))
                    continue
                
                input_token = torch.LongTensor([seq[-1]]).to(device)
                output, next_hidden = model.decoder(input_token, curr_hidden, encoder_outputs)
                probs = torch.softmax(output, dim=1)
                
                # Expand top candidates
                topk_probs, topk_ids = torch.topk(probs, beam_width * 2)
                
                for i in range(beam_width * 2):
                    word_idx = topk_ids[0][i].item()
                    word_prob = topk_probs[0][i].item()
                    new_score = score + math.log(word_prob + 1e-10)
                    new_candidates.append((new_score, next_hidden, seq + [word_idx]))
            
            # Keep top beam_width paths
            candidates = sorted(new_candidates, key=lambda x: x[0], reverse=True)[:beam_width]
            
            if all(c[2][-1] == end_token for c in candidates):
                break
                
        if not candidates:
            return ""
            
        best_score, _, best_seq = candidates[0]
        trg_tokens = [shakespearean_vocab.idx2word[i] for i in best_seq]
        
        result = []
        for t in trg_tokens:
            if t not in ['<s>', '</s>', '<pad>']:
                result.append(t)
                
        return ' '.join(result)

def test_examples():
    """Run translation on sample sentences"""
    print("="*70)
    print("🎭 Shakespeare Translator - Model Testing (Beam Search)")
    print("="*70)
    
    modern_vocab, shakespearean_vocab = load_vocab()
    model = load_model(modern_vocab, shakespearean_vocab, DEVICE)
    
    test_sentences = [
        "Hello, how are you?",
        "I love you very much.",
        "What are you doing today?",
        "The weather is beautiful.",
        "I am going to the market.",
        "Where are you going?",
        "I don't understand.",
        "Please help me.",
    ]
    
    for i, sentence in enumerate(test_sentences, 1):
        translation = translate_sentence_beam(
            sentence, model, modern_vocab, shakespearean_vocab, DEVICE, beam_width=5
        )
        print(f"\n{i}. Modern: {sentence}")
        print(f"   Shakespeare: {translation}")
        print("-" * 70)

def interactive_mode():
    """Interactive CLI for real-time translation"""
    print("="*70)
    print("🎭 Interactive Mode (type 'quit' to exit)")
    print("="*70)
    
    modern_vocab, shakespearean_vocab = load_vocab()
    model = load_model(modern_vocab, shakespearean_vocab, DEVICE)
    
    while True:
        try:
            user_input = input("\nModern English > ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            if not user_input: continue
            
            translation = translate_sentence_beam(
                user_input, model, modern_vocab, shakespearean_vocab, DEVICE, beam_width=5
            )
            print(f"Shakespeare > {translation}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def calculate_bleu(n_samples=100):
    """Evaluate model with BLEU metrics on test set"""
    import pandas as pd
    
    try:
        from nltk.translate.bleu_score import corpus_bleu
        import nltk
        try: nltk.data.find('tokenizers/punkt')
        except LookupError: nltk.download('punkt', quiet=True)
    except ImportError:
        print("Need nltk. Please install it.")
        return

    print("="*70)
    print(f"📊 Calculating BLEU (samples: {n_samples}, Beam Width: 5)")
    print("="*70)
    
    modern_vocab, shakespearean_vocab = load_vocab()
    model = load_model(modern_vocab, shakespearean_vocab, DEVICE)
    
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed_data.pkl')
    df = pd.read_pickle(data_path)
    
    # Use last 15% as test set to match train split
    test_start_idx = int(0.85 * len(df))
    test_df = df[test_start_idx:]
    test_samples = test_df.sample(min(n_samples, len(test_df)))
    
    references = []
    hypotheses = []
    
    print(f"Translating {len(test_samples)} sentences...")
    
    for idx, row in test_samples.iterrows():
        modern_text = row['modern_clean'].replace('<s>', '').replace('</s>', '').strip()
        true_shakespeare = row['shakespearean_clean'].replace('<s>', '').replace('</s>', '').strip()
        
        pred_shakespeare = translate_sentence_beam(
            modern_text, model, modern_vocab, shakespearean_vocab, DEVICE, beam_width=5
        )
        
        references.append([true_shakespeare.split()])
        hypotheses.append(pred_shakespeare.split())
        
        if len(hypotheses) % 20 == 0:
            print(f"  ...processed {len(hypotheses)}/{len(test_samples)}")
    
    bleu_1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu_2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu_3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
    bleu_4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    
    print("\n📈 Final Results (Beam Search):")
    print(f"BLEU-1: {bleu_1:.4f}")
    print(f"BLEU-2: {bleu_2:.4f}")
    print(f"BLEU-3: {bleu_3:.4f}")
    print(f"BLEU-4: {bleu_4:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='examples', 
                       choices=['examples', 'interactive', 'bleu'])
    parser.add_argument('--n_samples', type=int, default=100)
    args = parser.parse_args()
    
    if args.mode == 'examples': test_examples()
    elif args.mode == 'interactive': interactive_mode()
    elif args.mode == 'bleu': calculate_bleu(args.n_samples)
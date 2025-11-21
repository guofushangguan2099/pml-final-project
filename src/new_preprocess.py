import torch
import pickle
import re
import sys
import os
import math  # 新增: 用于计算 log 概率
import argparse

# 确保能导入其他模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import Encoder, Attention, Decoder, Seq2Seq
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

# --- 基础辅助函数 ---

def load_vocab():
    """加载词汇表"""
    vocab_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    with open(os.path.join(vocab_dir, 'modern_vocab.pkl'), 'rb') as f:
        modern_vocab = pickle.load(f)
    with open(os.path.join(vocab_dir, 'shakespearean_vocab.pkl'), 'rb') as f:
        shakespearean_vocab = pickle.load(f)
    
    return modern_vocab, shakespearean_vocab

def load_model(modern_vocab, shakespearean_vocab, device):
    """加载训练好的模型"""
    INPUT_DIM = modern_vocab.n_words
    OUTPUT_DIM = shakespearean_vocab.n_words
    
    attn = Attention(HIDDEN_DIM)
    enc = Encoder(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT)
    dec = Decoder(OUTPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT, attn)
    model = Seq2Seq(enc, dec, device).to(device)
    
    # 加载权重
    model_path = os.path.join(os.path.dirname(__file__), '..', MODEL_SAVE_PATH.replace('models/', ''))
    if not os.path.exists(model_path):
        model_path = MODEL_SAVE_PATH
    
    # map_location 确保在 CPU/GPU 间切换不报错
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model

def preprocess_text(text):
    """预处理文本"""
    text = text.lower().strip()
    text = re.sub(r"([?.!,:;\"'()\-])", r" \1 ", text)
    text = re.sub(r'[" "]+', " ", text)
    return f"<s> {text.strip()} </s>"

# --- 核心升级：Beam Search ---

def translate_sentence_beam(sentence, model, modern_vocab, shakespearean_vocab, device, max_len=50, beam_width=5):
    """
    使用 Beam Search 进行翻译 (能显著提升 BLEU 分数)
    """
    model.eval()
    
    # 1. 预处理输入
    processed = preprocess_text(sentence)
    tokens = [modern_vocab.word2idx.get(word, modern_vocab.word2idx['<unk>']) 
              for word in processed.split()]
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # 2. 编码器一次性运行
        encoder_outputs, hidden = model.encoder(src_tensor)
        
        # Beam Search 初始化
        # 列表中的元素结构: (累计分数, 当前hidden, 累计生成的token索引列表)
        # 分数用 log probability，初始是 0
        # sequence 初始包含 <s>
        start_token = shakespearean_vocab.word2idx['<s>']
        end_token = shakespearean_vocab.word2idx['</s>']
        
        candidates = [(0.0, hidden, [start_token])]
        
        # 3. 循环生成
        for _ in range(max_len):
            new_candidates = []
            
            # 遍历当前的每一条候选路径
            for score, curr_hidden, seq in candidates:
                
                # 如果这条路径已经结束，直接保留
                if seq[-1] == end_token:
                    new_candidates.append((score, curr_hidden, seq))
                    continue
                
                # 准备 Decoder 输入 (上一个词)
                input_token = torch.LongTensor([seq[-1]]).to(device)
                
                # 解码一步
                output, next_hidden = model.decoder(input_token, curr_hidden, encoder_outputs)
                # output: [1, vocab_size] -> 转成概率分布
                probs = torch.softmax(output, dim=1)
                
                # 取出这一步最好的 top k 个词 (避免计算整个词表，太慢)
                # 我们这里取 beam_width * 2 个，确保够选
                topk_probs, topk_ids = torch.topk(probs, beam_width * 2)
                
                # 扩展路径
                for i in range(beam_width * 2):
                    word_idx = topk_ids[0][i].item()
                    word_prob = topk_probs[0][i].item()
                    
                    # 累加对数概率 (log probability)
                    new_score = score + math.log(word_prob + 1e-10)
                    
                    new_candidates.append((new_score, next_hidden, seq + [word_idx]))
            
            # --- 剪枝 (Pruning) ---
            # 对所有扩展出来的路径按分数排序 (大到小)，只保留最好的 beam_width 个
            candidates = sorted(new_candidates, key=lambda x: x[0], reverse=True)[:beam_width]
            
            # 如果所有候选路径都已经是结束符结尾，那就提前退出
            if all(c[2][-1] == end_token for c in candidates):
                break
                
        # 4. 选择最终结果 (分数最高的那条)
        if not candidates:
            return ""
            
        best_score, _, best_seq = candidates[0]
        
        # 转回文字
        trg_tokens = [shakespearean_vocab.idx2word[i] for i in best_seq]
        
        # 去掉 <s> 和 </s>
        result = []
        for t in trg_tokens:
            if t not in ['<s>', '</s>', '<pad>']:
                result.append(t)
                
        return ' '.join(result)

# --- 测试功能模块 ---

def test_examples():
    """测试一些例句"""
    print("="*70)
    print("🎭 莎士比亚翻译器 - 模型测试 (Beam Search)")
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
        print(f"\n{i}. 现代: {sentence}")
        print(f"   莎翁: {translation}")
        print("-" * 70)

def interactive_mode():
    """交互式翻译模式"""
    print("="*70)
    print("🎭 交互模式 (输入 quit 退出)")
    print("="*70)
    
    modern_vocab, shakespearean_vocab = load_vocab()
    model = load_model(modern_vocab, shakespearean_vocab, DEVICE)
    
    while True:
        try:
            user_input = input("\n现代英语 > ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            if not user_input: continue
            
            # 使用 Beam Search
            translation = translate_sentence_beam(
                user_input, model, modern_vocab, shakespearean_vocab, DEVICE, beam_width=5
            )
            print(f"莎士比亚 > {translation}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def calculate_bleu(n_samples=100):
    """计算 BLEU 分数"""
    import pandas as pd
    
    # 自动安装/导入 nltk
    try:
        from nltk.translate.bleu_score import corpus_bleu
        import nltk
        try: nltk.data.find('tokenizers/punkt')
        except LookupError: nltk.download('punkt', quiet=True)
    except ImportError:
        print("Need nltk. Please install it.")
        return

    print("="*70)
    print(f"📊 计算 BLEU (样本数: {n_samples}, Beam Width: 5)")
    print("="*70)
    
    modern_vocab, shakespearean_vocab = load_vocab()
    model = load_model(modern_vocab, shakespearean_vocab, DEVICE)
    
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed_data.pkl')
    df = pd.read_pickle(data_path)
    
    # 只用测试集
    train_size = int(0.8 * len(df)) # 这里必须和你 train.py 的划分比例一致
    # 为保险起见，直接取最后 15%
    test_start_idx = int(0.85 * len(df))
    test_df = df[test_start_idx:]
    
    test_samples = test_df.sample(min(n_samples, len(test_df)))
    
    references = []
    hypotheses = []
    
    print(f"正在翻译 {len(test_samples)} 个句子...")
    
    for idx, row in test_samples.iterrows():
        modern_text = row['modern_clean'].replace('<s>', '').replace('</s>', '').strip()
        true_shakespeare = row['shakespearean_clean'].replace('<s>', '').replace('</s>', '').strip()
        
        # 使用 Beam Search!
        pred_shakespeare = translate_sentence_beam(
            modern_text, model, modern_vocab, shakespearean_vocab, DEVICE, beam_width=5
        )
        
        references.append([true_shakespeare.split()])
        hypotheses.append(pred_shakespeare.split())
        
        if len(hypotheses) % 20 == 0:
            print(f"  ...已处理 {len(hypotheses)}/{len(test_samples)}")
    
    # 计算分数
    bleu_1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu_2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu_3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
    bleu_4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    
    print("\n📈 最终结果 (Beam Search):")
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
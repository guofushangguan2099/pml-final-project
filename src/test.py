# test.py - 莎士比亚翻译测试脚本

import torch
import pickle
import re
import sys
import os

# 确保能导入其他模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import Encoder, Attention, Decoder, Seq2Seq
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from new_preprocess import Vocabulary 

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
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model

def preprocess_text(text):
    """预处理文本（和训练时一样）"""
    text = text.lower().strip()
    text = re.sub(r"([?.!,:;\"'()\-])", r" \1 ", text)
    text = re.sub(r'[" "]+', " ", text)
    return f"<s> {text.strip()} </s>"

def translate_sentence(sentence, model, modern_vocab, shakespearean_vocab, device, max_len=50):
    """
    翻译一个现代英语句子到莎士比亚英语
    """
    model.eval()
    
    # 预处理
    processed = preprocess_text(sentence)
    
    # 转换成数字 ID
    tokens = [modern_vocab.word2idx.get(word, modern_vocab.word2idx['<unk>']) 
              for word in processed.split()]
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Encoder
        encoder_outputs, hidden = model.encoder(src_tensor)
        
        # Decoder 逐词生成
        trg_indexes = [shakespearean_vocab.word2idx['<s>']]
        
        for _ in range(max_len):
            trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
            
            output, hidden = model.decoder(
                trg_tensor, hidden, encoder_outputs
            )
            
            pred_token = output.argmax(1).item()
            trg_indexes.append(pred_token)
            
            # 遇到结束符停止
            if pred_token == shakespearean_vocab.word2idx['</s>']:
                break
    
    # 转回文字
    trg_tokens = [shakespearean_vocab.idx2word[i] for i in trg_indexes]
    
    # 去掉 <s> 和 </s>
    return ' '.join(trg_tokens[1:-1])

def test_examples():
    """测试一些例句"""
    print("="*70)
    print("🎭 莎士比亚翻译器 - 模型测试")
    print("="*70)
    
    # 加载
    print("\n📚 加载词汇表...")
    modern_vocab, shakespearean_vocab = load_vocab()
    print(f"  现代英语词汇: {modern_vocab.n_words}")
    print(f"  莎士比亚词汇: {shakespearean_vocab.n_words}")
    
    print("\n🤖 加载模型...")
    model = load_model(modern_vocab, shakespearean_vocab, DEVICE)
    print(f"  设备: {DEVICE}")
    print("✅ 模型加载成功！\n")
    
    # 测试句子
    test_sentences = [
        "Hello, how are you?",
        "I love you very much.",
        "What are you doing today?",
        "The weather is beautiful.",
        "I am going to the market.",
        "Where are you going?",
        "I don't understand.",
        "Please help me.",
        "Good morning, my friend.",
        "I am very happy to see you."
    ]
    
    print("="*70)
    print("🎬 翻译示例")
    print("="*70)
    
    for i, sentence in enumerate(test_sentences, 1):
        translation = translate_sentence(
            sentence, model, modern_vocab, shakespearean_vocab, DEVICE
        )
        print(f"\n{i}. 现代英语: {sentence}")
        print(f"   莎士比亚: {translation}")
        print("-" * 70)

def interactive_mode():
    """交互式翻译模式"""
    print("="*70)
    print("🎭 莎士比亚翻译器 - 交互模式")
    print("="*70)
    
    # 加载
    print("\n加载中...")
    modern_vocab, shakespearean_vocab = load_vocab()
    model = load_model(modern_vocab, shakespearean_vocab, DEVICE)
    print("✅ 准备就绪！\n")
    
    print("输入现代英语，我会翻译成莎士比亚风格")
    print("输入 'quit' 或 'exit' 退出\n")
    print("-" * 70)
    
    while True:
        try:
            user_input = input("\n现代英语 > ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 再见！")
                break
            
            if not user_input:
                continue
            
            translation = translate_sentence(
                user_input, model, modern_vocab, shakespearean_vocab, DEVICE
            )
            print(f"莎士比亚 > {translation}")
            
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 错误: {e}")

def test_from_dataset():
    """从测试集中随机测试"""
    import pandas as pd
    
    print("="*70)
    print("📊 从测试集评估")
    print("="*70)
    
    # 加载
    modern_vocab, shakespearean_vocab = load_vocab()
    model = load_model(modern_vocab, shakespearean_vocab, DEVICE)
    
    # 加载数据
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed_data.pkl')
    df = pd.read_pickle(data_path)
    
    # 测试集
    train_size = int(0.8 * len(df))
    test_df = df[train_size:]
    
    print(f"\n测试集大小: {len(test_df)} 条")
    print("随机选择 10 条测试:\n")
    
    samples = test_df.sample(10)
    
    for i, (idx, row) in enumerate(samples.iterrows(), 1):
        modern = row['modern_clean'].replace('<s>', '').replace('</s>', '').strip()
        true_shakespeare = row['shakespearean_clean'].replace('<s>', '').replace('</s>', '').strip()
        
        pred_shakespeare = translate_sentence(
            modern, model, modern_vocab, shakespearean_vocab, DEVICE
        )
        
        print(f"{i}. 现代英语: {modern}")
        print(f"   真实莎翁: {true_shakespeare}")
        print(f"   模型翻译: {pred_shakespeare}")
        print("-" * 70)

def calculate_bleu(n_samples=100):
    """
    计算测试集的 BLEU 分数
    
    Args:
        n_samples: 要评估的样本数量
    """
    import pandas as pd
    
    print("="*70)
    print("📊 BLEU 分数评估")
    print("="*70)
    
    # 先安装 nltk
    try:
        from nltk.translate.bleu_score import corpus_bleu
        import nltk
    except ImportError:
        print("\n⚠️ 需要安装 nltk，正在安装...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'nltk'])
        from nltk.translate.bleu_score import corpus_bleu
        import nltk
    
    # 下载必要的数据
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("下载 nltk 数据...")
        nltk.download('punkt', quiet=True)
    
    # 加载模型和数据
    print("\n加载模型...")
    modern_vocab, shakespearean_vocab = load_vocab()
    model = load_model(modern_vocab, shakespearean_vocab, DEVICE)
    
    # 加载测试集
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed_data.pkl')
    df = pd.read_pickle(data_path)
    
    train_size = int(0.8 * len(df))
    test_df = df[train_size:]
    
    # 随机采样
    test_samples = test_df.sample(min(n_samples, len(test_df)))
    
    print(f"评估 {len(test_samples)} 个样本...")
    print("-" * 70)
    
    references = []
    hypotheses = []
    
    for idx, row in test_samples.iterrows():
        # 获取原始句子
        modern_text = row['modern_clean'].replace('<s>', '').replace('</s>', '').strip()
        true_shakespeare = row['shakespearean_clean'].replace('<s>', '').replace('</s>', '').strip()
        
        # 模型翻译
        try:
            pred_shakespeare = translate_sentence(
                modern_text, model, modern_vocab, shakespearean_vocab, DEVICE
            )
        except Exception as e:
            print(f"⚠️ 翻译失败，跳过: {e}")
            continue
        
        # BLEU 需要的格式：reference 是列表的列表，hypothesis 是列表
        references.append([true_shakespeare.split()])
        hypotheses.append(pred_shakespeare.split())
    
    # 计算 BLEU 分数
    bleu_1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu_2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu_3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
    bleu_4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    
    print("\n" + "="*70)
    print("📈 BLEU 分数结果")
    print("="*70)
    print(f"BLEU-1: {bleu_1:.4f}")
    print(f"BLEU-2: {bleu_2:.4f}")
    print(f"BLEU-3: {bleu_3:.4f}")
    print(f"BLEU-4: {bleu_4:.4f}")
    print("="*70)
    
    print("\n💡 BLEU 分数说明:")
    print("  - 0.0-0.1: 很差")
    print("  - 0.1-0.2: 较差")
    print("  - 0.2-0.3: 一般")
    print("  - 0.3-0.4: 良好")
    print("  - 0.4-0.5: 很好")
    print("  - 0.5+:    优秀")
    
    return bleu_1, bleu_2, bleu_3, bleu_4

# ← 所有函数定义在这里之上
# ↓ if __name__ 块在这里（只有一个！）

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='莎士比亚翻译器测试')
    parser.add_argument('--mode', type=str, default='examples', 
                       choices=['examples', 'interactive', 'dataset', 'bleu'],
                       help='测试模式: examples(示例), interactive(交互), dataset(测试集), bleu(BLEU评分)')
    parser.add_argument('--n_samples', type=int, default=100,
                       help='BLEU 评估的样本数量（默认100）')
    
    args = parser.parse_args()
    
    if args.mode == 'examples':
        test_examples()
    elif args.mode == 'interactive':
        interactive_mode()
    elif args.mode == 'dataset':
        test_from_dataset()
    elif args.mode == 'bleu':
        calculate_bleu(args.n_samples)
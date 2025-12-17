import os

# ==========================================
# 0.【关键修复】解决 OMP Error #15
# ==========================================
# 这行代码必须放在 import torch 之前！
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as F
import pickle
import sys
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ==========================================
# 1. 环境与路径设置
# ==========================================
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from model import Encoder, Decoder, Attention, Seq2Seq
    from new_preprocess import preprocess_text, Vocabulary
except ImportError as e:
    print("❌ 导入错误: 请确保 'src' 文件夹下有 model.py 和 new_preprocess.py")
    sys.exit(1)

# ==========================================
# 2. 全局配置 (冠军模型配置)
# ==========================================
EMBEDDING_DIM = 300
HIDDEN_DIM = 512
N_LAYERS = 1      # ✅ 单层
DROPOUT = 0.5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Running on: {DEVICE}")

# ==========================================
# 3. 加载模型与词表
# ==========================================
def load_model_and_vocabs():
    print("正在加载词表...")
    try:
        with open('data/modern_vocab.pkl', 'rb') as f:
            modern_vocab = pickle.load(f)
        with open('data/shakespearean_vocab.pkl', 'rb') as f:
            shk_vocab = pickle.load(f)
    except FileNotFoundError:
        print("❌ 错误: 在 data/ 目录下找不到 .pkl 词表文件。")
        sys.exit(1)

    print("正在初始化模型架构...")
    INPUT_DIM = modern_vocab.n_words
    OUTPUT_DIM = shk_vocab.n_words
    attn = Attention(HIDDEN_DIM)
    enc = Encoder(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT)
    dec = Decoder(OUTPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT, attn)
    
    model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)

    print("正在加载模型权重...")
    model_path = 'saved_models/seq2seq_model.pt'
    if not os.path.exists(model_path):
        print(f"❌ 错误: 找不到模型文件 '{model_path}'")
        sys.exit(1)
        
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval() 
    print("✅ 模型加载成功！")
    return model, modern_vocab, shk_vocab

# ==========================================
# 4. Beam Search 翻译函数
# ==========================================
def translate_sentence(sentence, model, modern_vocab, shk_vocab, max_len=50, beam_size=5, alpha=1.0):
    model.eval()
    processed_text = preprocess_text(sentence)
    tokens = [modern_vocab.word2idx.get(t, modern_vocab.word2idx['<unk>']) 
              for t in processed_text.split()]
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)
    start_token = shk_vocab.word2idx['<s>']
    end_token = shk_vocab.word2idx['</s>']
    beams = [(0.0, [start_token], hidden)]
    
    for _ in range(max_len):
        new_beams = []
        for score, tokens, h in beams:
            if tokens[-1] == end_token:
                new_beams.append((score, tokens, h))
                continue
            trg_tensor = torch.LongTensor([tokens[-1]]).to(DEVICE)
            with torch.no_grad():
                output, new_h, _ = model.decoder(trg_tensor, h, encoder_outputs)
            log_probs = F.log_softmax(output, dim=1).squeeze(0)
            topk_probs, topk_ids = log_probs.topk(beam_size)
            for k in range(beam_size):
                sym = topk_ids[k].item()
                prob = topk_probs[k].item()
                new_beams.append((score + prob, tokens + [sym], new_h))
        def get_beam_score(beam_tuple):
            sc, toks, _ = beam_tuple
            length_penalty = len(toks) ** alpha
            return sc / length_penalty
        new_beams.sort(key=get_beam_score, reverse=True)
        beams = new_beams[:beam_size]
        if all(b[1][-1] == end_token for b in beams):
            break
            
    best_score, best_tokens, _ = beams[0]
    trg_words = []
    for idx in best_tokens:
        if idx == start_token: continue
        if idx == end_token: break
        trg_words.append(shk_vocab.idx2word[idx])
    return " ".join(trg_words)

# ==========================================
# 5. Attention 可视化专用函数
# ==========================================
def get_attention_matrix_and_translation(sentence, model, modern_vocab, shk_vocab):
    model.eval()
    processed_text = preprocess_text(sentence)
    tokens = [modern_vocab.word2idx.get(t, modern_vocab.word2idx['<unk>']) 
              for t in processed_text.split()]
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)
    
    trg_indexes = [shk_vocab.word2idx['<s>']]
    attentions = []
    
    for i in range(50):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(DEVICE)
        with torch.no_grad():
            output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs)
        attentions.append(attention.cpu())
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        if pred_token == shk_vocab.word2idx['</s>']:
            break
            
    trg_tokens = [shk_vocab.idx2word[i] for i in trg_indexes]
    src_tokens = [modern_vocab.idx2word[i] for i in tokens]
    trg_tokens = trg_tokens[1:] 
    
    if len(attentions) > 0:
        attention_matrix = torch.cat(attentions).detach().numpy()
    else:
        attention_matrix = np.zeros((1, len(src_tokens)))
        
    return src_tokens, trg_tokens, attention_matrix

def plot_attention(sentence, model, modern_vocab, shk_vocab):
    src_tokens, trg_tokens, attention_matrix = get_attention_matrix_and_translation(
        sentence, model, modern_vocab, shk_vocab
    )
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    
    cax = ax.matshow(attention_matrix, cmap='viridis')
    fig.colorbar(cax)
    
    # 修复 Warning: 先设置 Locators，再设置 Labels
    ax.xaxis.set_major_locator(ticker.FixedLocator(range(len(src_tokens) + 1)))
    ax.yaxis.set_major_locator(ticker.FixedLocator(range(len(trg_tokens) + 1)))
    
    # +1 是为了错位，让 label 对齐格子中心（有时候需要微调，这里先用最简单的）
    # 通常 heatmap 的 label 对应 index
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    ax.set_xticklabels([''] + src_tokens, rotation=45, fontsize=12)
    ax.set_yticklabels([''] + trg_tokens, fontsize=12)
    
    plt.xlabel("Modern Input", fontsize=14)
    plt.ylabel("Shakespearean Output", fontsize=14)
    plt.title(f"Attention Alignment: {sentence[:20]}...", fontsize=16, pad=20)
    
    safe_filename = sentence.replace(" ", "_").replace("?", "").replace(".", "")[:20] + ".png"
    plt.savefig(safe_filename, dpi=300, bbox_inches='tight')
    plt.show() 
    print(f"✅ 热力图已保存: {safe_filename}")

# ==========================================
# 6. 主程序入口
# ==========================================
if __name__ == "__main__":
    model, modern_vocab, shk_vocab = load_model_and_vocabs()
    
    # -------------------------------------------------
    # 第一步：先画图
    # -------------------------------------------------
    print("\n" + "="*60)
    print("🎨 生成 Attention Heatmap")
    print("="*60)
    
    plot_sentences = [
        "Where are you going?", 
        "I will kill you if you lie to me."
    ]
    for s in plot_sentences:
        plot_attention(s, model, modern_vocab, shk_vocab)

    # -------------------------------------------------
    # 第二步：跑文字翻译 (Beam Search)
    # -------------------------------------------------
    ALPHA = 1.2
    BEAM_SIZE = 5
    print("\n" + "="*60)
    print(f"🧪 风格迁移展示 (Beam={BEAM_SIZE}, Alpha={ALPHA})")
    print("="*60)
    
    showcase_sentences = [
        "Where are you going?",
        "I will kill you if you lie to me.",
        "She is very beautiful.",
        "Get out of my sight!",
        "My heart is heavy.",
        "Fortune is a cruel woman.",
        "I do not think so."
    ]

    for s in showcase_sentences:
        trans = translate_sentence(s, model, modern_vocab, shk_vocab, 
                                 beam_size=BEAM_SIZE, alpha=ALPHA)
        print(f"Modern:  {s}")
        print(f"Shakes:  {trans}")
        print("-" * 60)

    # -------------------------------------------------
    # 第三步：交互模式
    # -------------------------------------------------
    print("\n" + "="*60)
    print("⌨️  交互模式 (输入 'q' 退出)")
    print("="*60)
    while True:
        try:
            s = input("\nInput Modern English: ")
            if s.lower() in ['q', 'quit']: break
            t = translate_sentence(s, model, modern_vocab, shk_vocab, 
                                 beam_size=BEAM_SIZE, alpha=ALPHA)
            print(f">>> {t}")
        except Exception as e:
            print(f"Error: {e}")
            break
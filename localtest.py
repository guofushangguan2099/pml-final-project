import torch
import torch.nn.functional as F
import pickle
import sys
import os
import math

# ==========================================
# 1. 环境与路径设置
# ==========================================
# 将 src 目录加入路径，确保能找到 model.py 和 new_preprocess.py
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from model import Encoder, Decoder, Attention, Seq2Seq
    # 必须导入 Vocabulary，否则 pickle.load 会报错
    from new_preprocess import preprocess_text, Vocabulary
except ImportError as e:
    print("❌ 导入错误: 请确保 'src' 文件夹下有 model.py 和 new_preprocess.py")
    print(f"详细错误: {e}")
    sys.exit(1)

# ==========================================
# 2. 全局配置 (必须与训练时一致)
# ==========================================
EMBEDDING_DIM = 300
HIDDEN_DIM = 512
N_LAYERS = 1
DROPOUT = 0.5

# 自动选择设备
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
    model.eval() # 关闭 Dropout，进入评估模式
    
    print("✅ 模型加载成功！")
    return model, modern_vocab, shk_vocab

# ==========================================
# 4. 核心翻译函数 (Beam Search)
# ==========================================
def translate_sentence(sentence, model, modern_vocab, shk_vocab, max_len=50, beam_size=5, alpha=0.7):
    """
    使用集束搜索翻译句子。
    :param beam_size: 束宽 (3-10)。越大越准，但越慢。
    :param alpha: 长度惩罚因子 (0.0-1.0)。
                  alpha 越大，越鼓励生成长句子 (解决 "whither?" 问题)。
                  alpha = 0.0 表示不惩罚。
    """
    model.eval()
    
    # --- 1. 预处理与编码 ---
    processed_text = preprocess_text(sentence)
    tokens = [modern_vocab.word2idx.get(t, modern_vocab.word2idx['<unk>']) 
              for t in processed_text.split()]
    
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)
        
    # --- 2. Beam Search 初始化 ---
    # 结构: (累积得分, [生成的token列表], hidden_state)
    # 初始得分是 0
    start_token = shk_vocab.word2idx['<s>']
    end_token = shk_vocab.word2idx['</s>']
    
    beams = [(0.0, [start_token], hidden)]
    
    # --- 3. 解码循环 ---
    for _ in range(max_len):
        new_beams = []
        
        for score, tokens, h in beams:
            # 如果该路径已经结束 (遇到 </s>)，直接保留，不继续展开
            if tokens[-1] == end_token:
                new_beams.append((score, tokens, h))
                continue
            
            # 运行 Decoder 一步
            # 输入必须是 [batch=1]
            trg_tensor = torch.LongTensor([tokens[-1]]).to(DEVICE)
            
            with torch.no_grad():
                # Decoder 返回: prediction, hidden, attention
                output, new_h, _ = model.decoder(trg_tensor, h, encoder_outputs)
            
            # 获取概率分布 (log_softmax 得到负数分数，越接近0越好)
            # output: [1, output_dim]
            log_probs = F.log_softmax(output, dim=1).squeeze(0)
            
            # 选出这一步概率最大的 beam_size 个词
            topk_probs, topk_ids = log_probs.topk(beam_size)
            
            for k in range(beam_size):
                sym = topk_ids[k].item()
                prob = topk_probs[k].item()
                
                # 更新分数和路径
                new_beams.append((score + prob, tokens + [sym], new_h))
        
        # --- 4. 筛选最优路径 (带长度惩罚) ---
        def get_beam_score(beam_tuple):
            sc, toks, _ = beam_tuple
            # 如果只看 sc，短句子分数天然高 (因为累加的负数少)
            # 所以除以 (长度^alpha) 来进行归一化
            length_penalty = len(toks) ** alpha
            return sc / length_penalty
            
        # 按调整后的分数排序，取前 beam_size 个
        new_beams.sort(key=get_beam_score, reverse=True)
        beams = new_beams[:beam_size]
        
        # 如果前 beam_size 个路径全部都结束了，那就提前停止
        if all(b[1][-1] == end_token for b in beams):
            break
            
    # --- 5. 取出第一名并转换回文本 ---
    best_score, best_tokens, _ = beams[0]
    
    trg_words = []
    for idx in best_tokens:
        if idx == start_token: continue
        if idx == end_token: break
        trg_words.append(shk_vocab.idx2word[idx])
        
    return " ".join(trg_words)

# ==========================================
# 5. 主程序入口
# ==========================================
if __name__ == "__main__":
    # 1. 加载
    model, modern_vocab, shk_vocab = load_model_and_vocabs()
    
    # 2. 预设句子测试
    print("\n" + "="*40)
    print("🧪 标准基准测试 (Beam Search enabled)")
    print("="*40)
    
    sentences = [
        "Where are you going?",
        "I do not think so.",
        "Can you help me?",
        "Love is a beautiful thing.",
        "He is my brother, and I love him."
    ]
    
    # 关键：在这里调整 alpha 可以改变句子长度
    # alpha=0.6: 倾向短句
    # alpha=1.0: 倾向长句
    ALPHA = 0.6
    BEAM_SIZE = 2

    for s in sentences:
        trans = translate_sentence(s, model, modern_vocab, shk_vocab, 
                                 beam_size=BEAM_SIZE, alpha=ALPHA)
        print(f"Modern:  {s}")
        print(f"Shakes:  {trans}")
        print("-" * 40)

    # 3. 交互模式
    print("\n" + "="*40)
    print("⌨️  交互模式 (输入 'q' 退出)")
    print(f"当前设置: Beam Size = {BEAM_SIZE}, Alpha = {ALPHA}")
    print("="*40)
    
    while True:
        try:
            sentence = input("\n请输入现代英语句子: ")
            if sentence.lower() in ['q', 'quit', 'exit']:
                break
            
            # 这里也可以动态调参测试
            translation = translate_sentence(sentence, model, modern_vocab, shk_vocab, 
                                           beam_size=BEAM_SIZE, alpha=ALPHA)
            print(f">>> {translation}")
            
        except KeyboardInterrupt:
            print("\n退出中...")
            break
        except Exception as e:
            print(f"❌ 出错: {e}")
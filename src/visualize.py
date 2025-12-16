# src/visualize.py
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
from config import DEVICE

def save_attention_plot(sentence, predicted_sentence, attention, index, save_dir='plots'):
    """
    sentence: 原句 (string)
    predicted_sentence: 翻译句 (string)
    attention: Attention 矩阵 (Tensor or Numpy)
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    
    # 画热力图
    cax = ax.matshow(attention.cpu().detach().numpy(), cmap='bone')
    fig.colorbar(cax)

    # 设置轴标签
    ax.set_xticklabels([''] + sentence.split() + ['</s>'], rotation=90, fontsize=12)
    ax.set_yticklabels([''] + predicted_sentence.split(), fontsize=12)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.title(f"Attention Map - Sample {index}")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/attention_{index}.png")
    print(f"✅ Attention map saved to {save_dir}/attention_{index}.png")
    plt.close()

# 你可以在 test.py 里调用这个函数
# src/model.py

import torch
import torch.nn as nn
import random

# --- 模块 1: 编码器 (Encoder) ---
# 职责：读取并理解输入句子，将其压缩成一个“上下文向量”。
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        # 词嵌入层：将输入的词索引转换为密集向量
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        # GRU层：处理序列数据，捕捉上下文信息
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        
        # Dropout层：防止过拟合
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src [batch_size, src_len]
        
        embedded = self.dropout(self.embedding(src))
        # embedded [batch_size, src_len, emb_dim]
        
        # GRU返回所有时间步的顶层隐藏状态和最后一个时间步的所有层隐藏状态
        outputs, hidden = self.rnn(embedded)
        # outputs [batch_size, src_len, hid_dim]
        # hidden [n_layers, batch_size, hid_dim]
        
        return outputs, hidden

# --- 模块 2: 注意力机制 (Attention) ---
# 职责：计算注意力权重，决定解码器应该“关注”输入句子的哪个部分。
class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        # 注意力计算层
        self.attn = nn.Linear(hid_dim * 2, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        # hidden [n_layers, batch_size, hid_dim] -> 我们只取顶层
        # encoder_outputs [batch_size, src_len, hid_dim]
        
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # 将解码器的顶层隐藏状态重复 src_len 次，以便与编码器的输出进行对齐
        hidden = hidden[-1,:,:].unsqueeze(1).repeat(1, src_len, 1)
        # hidden [batch_size, src_len, hid_dim]
        
        # 计算能量（energy）
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy [batch_size, src_len, hid_dim]
        
        # 计算注意力得分
        attention = self.v(energy).squeeze(2)
        # attention [batch_size, src_len]
        
        # 返回经过softmax归一化的注意力权重
        return torch.softmax(attention, dim=1)

# --- 模块 3: 解码器 (Decoder) ---
# 职责：根据上下文向量和注意力权重，一次生成一个词，构建出翻译后的句子。
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        # GRU的输入维度是 嵌入维度 + 隐藏层维度（因为要拼接注意力上下文向量）
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        
        # 全连接层：将GRU的输出映射到词汇表大小，以进行预测
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        # input [batch_size]
        # hidden [n_layers, batch_size, hid_dim]
        # encoder_outputs [batch_size, src_len, hid_dim]
        
        input = input.unsqueeze(1)
        # input [batch_size, 1]
        
        embedded = self.dropout(self.embedding(input))
        # embedded [batch_size, 1, emb_dim]
        
        # 计算注意力权重
        a = self.attention(hidden, encoder_outputs).unsqueeze(1)
        # a [batch_size, 1, src_len]
        
        # 计算加权的上下文向量
        weighted = torch.bmm(a, encoder_outputs)
        # weighted [batch_size, 1, hid_dim]
        
        # 将嵌入向量和加权上下文向量拼接起来，作为GRU的输入
        rnn_input = torch.cat((embedded, weighted), dim=2)
        # rnn_input [batch_size, 1, emb_dim + hid_dim]
        
        output, hidden = self.rnn(rnn_input, hidden)
        # output [batch_size, 1, hid_dim]
        # hidden [n_layers, batch_size, hid_dim]
        
        # 将嵌入向量、GRU输出和上下文向量拼接，用于最终预测
        prediction = self.fc_out(torch.cat((output.squeeze(1), weighted.squeeze(1), embedded.squeeze(1)), dim=1))
        # prediction [batch_size, output_dim]
        
        return prediction, hidden

# --- 模块 4: Seq2Seq 模型 ---
# 职责：封装整个编码器-解码器架构，协调数据流。
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src [batch_size, src_len]
        # trg [batch_size, trg_len]
        
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        # 创建一个张量来存储解码器的所有输出
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # 编码输入句子
        encoder_outputs, hidden = self.encoder(src)
        
        # 解码器的第一个输入是 <sos> 词元
        input = trg[:, 0]
        
        # 循环生成目标句子的每一个词
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[:, t] = output
            
            # 决定是否使用 "教师强制"
            teacher_force = random.random() < teacher_forcing_ratio
            
            # 获取预测概率最高的词
            top1 = output.argmax(1)
            
            # 如果是教师强制，则使用真实的下一个词作为输入；否则使用模型自己的预测
            input = trg[:, t] if teacher_force else top1
            
        return outputs
# src/model.py
import torch
import torch.nn as nn
import random

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        # ✅ 双向 GRU
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        
        # ✅ 将双向 hidden 压缩为单向，传给 Decoder 做初始状态
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        
    def forward(self, src):
        # src: [batch_size, src_len]
        embedded = self.dropout(self.embedding(src))
        
        # outputs: [batch_size, src_len, hid_dim * 2] (双向特征)
        outputs, hidden = self.rnn(embedded)
        
        # 处理 hidden state: 取最后两层(正向+反向)，拼接并压缩
        # hidden shape: [n_layers * 2, batch, hid]
        
        # 取最后两层的 hidden state (Forward + Backward)
        hidden_forward = hidden[-2,:,:]
        hidden_backward = hidden[-1,:,:]
        
        # 压缩: [batch, hid_dim * 2] -> [batch, hid_dim]
        hidden_final = torch.tanh(self.fc(torch.cat((hidden_forward, hidden_backward), dim=1)))
        
        # 复制层数以适配 Decoder: [n_layers, batch, hid_dim]
        # 注意：这里我们简单地将压缩后的状态复制给所有层，这是一种常见的初始化策略
        hidden_final = hidden_final.unsqueeze(0).repeat(self.n_layers, 1, 1)
        
        return outputs, hidden_final

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        # ✅ 关键修改点 1: 输入维度变大了
        # Encoder输出 (hid*2) + Decoder隐状态 (hid) = hid * 3
        self.attn = nn.Linear(hid_dim * 3, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        # hidden: [n_layers, batch, hid] -> 取最后一层 [batch, hid]
        # encoder_outputs: [batch, src_len, hid * 2]
        
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # 将 Decoder hidden 重复 src_len 次以便拼接
        hidden = hidden[-1].unsqueeze(1).repeat(1, src_len, 1)
        
        # 拼接: [batch, src_len, hid] + [batch, src_len, hid*2] -> [..., hid*3]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        
        # 计算注意力分数
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        # ✅ 关键修改点 2: RNN 输入维度变大了
        # Embedding (emb) + Context Vector (hid*2, 因为是双向Encoder的加权和)
        self.rnn = nn.GRU(emb_dim + hid_dim * 2, hid_dim, n_layers, dropout=dropout, batch_first=True)
        
        # ✅ 关键修改点 3: 输出层维度变大了
        # RNN输出(hid) + Context(hid*2) + Embedding(emb)
        self.fc_out = nn.Linear(emb_dim + hid_dim * 3, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        
        # 1. 计算注意力权重
        a = self.attention(hidden, encoder_outputs).unsqueeze(1)
        
        # 2. 计算 Context Vector (Encoder 输出的加权和)
        # a: [batch, 1, src_len], encoder_outputs: [batch, src_len, hid*2]
        # weighted: [batch, 1, hid*2]
        weighted = torch.bmm(a, encoder_outputs)
        
        # 3. RNN Step
        # 输入: Embedding + Context Vector
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        
        # 4. 预测下一个词
        # 拼接: RNN输出 + Context Vector + Embedding
        prediction = self.fc_out(torch.cat((output.squeeze(1), weighted.squeeze(1), embedded.squeeze(1)), dim=1))
        
        return prediction, hidden, a.squeeze(1)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        src_len = src.shape[1]
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        attentions = torch.zeros(batch_size, trg_len, src_len).to(self.device)
        
        encoder_outputs, hidden = self.encoder(src)
        input = trg[:, 0]
        
        for t in range(1, trg_len):
            output, hidden, attention = self.decoder(input, hidden, encoder_outputs)
            outputs[:, t] = output
            attentions[:, t] = attention
            
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
            
        return outputs, attentions
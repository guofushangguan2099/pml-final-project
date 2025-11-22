# src/model.py

import torch
import torch.nn as nn
import random

# Encoder: Processes input sequence and compresses it into context vectors
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        # Word embedding layer
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        # GRU for sequence processing
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src: [batch_size, src_len]
        
        embedded = self.dropout(self.embedding(src))
        # embedded: [batch_size, src_len, emb_dim]
        
        # outputs: all timesteps, hidden: final hidden state
        outputs, hidden = self.rnn(embedded)
        # outputs: [batch_size, src_len, hid_dim]
        # hidden: [n_layers, batch_size, hid_dim]
        return outputs, hidden

# Attention: Computes attention weights over encoder outputs
class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim * 2, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        # hidden: [n_layers, batch_size, hid_dim] - use top layer only
        # encoder_outputs: [batch_size, src_len, hid_dim]
        
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Repeat decoder hidden state src_len times
        hidden = hidden[-1,:,:].unsqueeze(1).repeat(1, src_len, 1)
        # hidden: [batch_size, src_len, hid_dim]
        
        # Calculate energy
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy: [batch_size, src_len, hid_dim]
        
        # Calculate attention scores
        attention = self.v(energy).squeeze(2)
        # attention: [batch_size, src_len]
        
        return torch.softmax(attention, dim=1)

# Decoder: Generates output sequence one token at a time
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        # GRU input: embedding + context vector
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        
        # Output layer
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        # input: [batch_size]
        # hidden: [n_layers, batch_size, hid_dim]
        # encoder_outputs: [batch_size, src_len, hid_dim]
        
        input = input.unsqueeze(1)
        # input: [batch_size, 1]
        
        embedded = self.dropout(self.embedding(input))
        # embedded: [batch_size, 1, emb_dim]
        
        # Calculate attention weights
        a = self.attention(hidden, encoder_outputs).unsqueeze(1)
        # a: [batch_size, 1, src_len]
        
        # Compute weighted context vector
        weighted = torch.bmm(a, encoder_outputs)
        # weighted: [batch_size, 1, hid_dim]
        
        # Concatenate embedding and context vector
        rnn_input = torch.cat((embedded, weighted), dim=2)
        # rnn_input: [batch_size, 1, emb_dim + hid_dim]
        
        output, hidden = self.rnn(rnn_input, hidden)
        # output: [batch_size, 1, hid_dim]
        # hidden: [n_layers, batch_size, hid_dim]
        
        # Combine embedding, output, and context for prediction
        prediction = self.fc_out(torch.cat((output.squeeze(1), weighted.squeeze(1), embedded.squeeze(1)), dim=1))
        # prediction: [batch_size, output_dim]
        
        return prediction, hidden

# Seq2Seq: Complete encoder-decoder architecture
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]
        
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # Encode input sequence
        encoder_outputs, hidden = self.encoder(src)
        
        # First input is <sos> token
        input = trg[:, 0]
        
        # Generate each target token
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[:, t] = output
            
            # Apply teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            
            # Use ground truth or prediction as next input
            input = trg[:, t] if teacher_force else top1
            
        return outputs
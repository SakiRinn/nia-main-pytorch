import random

import torch
import torch.nn as nn


class BilinearAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(BilinearAttention, self).__init__()
        # Size
        self.hidden_dim = hidden_dim
        # Layer
        self.W = nn.Parameter(torch.empty(hidden_dim, hidden_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.W)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden, enc_outputs):
        # hidden, enc_outputs: (N, 1, hidden), (N, T, hidden)
        score = hidden.matmul(self.W).matmul(enc_outputs.transpose(1, 2)).squeeze(1)    # (N, T)
        T = score.size(-1)
        attn = (self.softmax(score).unsqueeze(2) * hidden.repeat(1, T, 1)).sum(1)    # (N, hidden)
        return attn


class Encoder(nn.Module):
    def __init__(self, input_vocab_len, embedding_dim, hidden_dim, num_layers=1, dropout=0.5):
        super(Encoder, self).__init__()
        # Size
        self.input_vocab_len = input_vocab_len
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Layer
        self.embedding = nn.Embedding(input_vocab_len, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)

    def forward(self, src, state=None):
        # x: (N, T)
        embedded = self.embedding(src)    # (N, T, embed)
        out, state = self.lstm(embedded, state)    # (N, T, 2*hidden), (2*layer, N, hidden)
        # Avg bidirectional outputs -> (N, T, hidden)
        out = (out[:, :, :self.hidden_dim] + out[:, :, self.hidden_dim:]) / 2
        return out, state


class Decoder(nn.Module):
    def __init__(self, output_vocab_len, embedding_dim, hidden_dim, num_layers=1, dropout=0.2):
        super(Decoder, self).__init__()
        # Size
        self.output_vocab_len = output_vocab_len
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Layer
        self.embedding = nn.Sequential(
            nn.Embedding(output_vocab_len, embedding_dim),
            nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=False)
        self.attention = BilinearAttention(hidden_dim)
        self.dense = nn.Linear(hidden_dim, output_vocab_len)

    def forward(self, enc_outputs, trg, state=None):
        # enc_outputs, x: (N, T, hidden), (N, T)
        embedded = self.embedding(trg)    # (N, 1, embed)
        out, state = self.lstm(embedded, state)    # (N, 1, hidden)
        out = self.attention(out[:, -1, :].unsqueeze(1), enc_outputs)    # (N, 1, hidden)
        out = self.dense(out)    # (N, 1, embed)
        return out, state


class Seq2Seq(nn.Module):
    def __init__(self, input_vocab_len, output_vocab_len, embedding_dim, hidden_dim,
                 num_layers=1, encoder_dropout=0.5, decoder_dropout=0.2, teacher_forcing_ratio=0.5):
        super(Seq2Seq, self).__init__()
        # Size
        self.input_vocab_len = input_vocab_len
        self.output_vocab_len = output_vocab_len
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Model
        self.encoder = Encoder(input_vocab_len, embedding_dim, hidden_dim, num_layers, encoder_dropout)
        self.decoder = Decoder(output_vocab_len, embedding_dim, hidden_dim, num_layers, decoder_dropout)
        # Option
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, src, trg):
        # src, trg: (N, T_src), (N, T_trg)
        batch_size = src.size(0)
        max_len = trg.size(1)
        outs = torch.zeros(batch_size, max_len, self.output_vocab_len).to(src.device)

        enc_outputs, state = self.encoder(src)
        state = tuple([s[:self.decoder.num_layers] for s in state])
        out = trg[:, 0].unsqueeze(1)

        for t in range(1, max_len):
            out, state = self.decoder(enc_outputs, out, state)
            pred = out.argmax(1).to(torch.long)
            outs[:, t, :] = out
            # Teacher force
            is_teacher = random.random() < self.teacher_forcing_ratio
            out = trg[:, t].unsqueeze(1) if is_teacher else pred.unsqueeze(1)

        return outs

    def predict(self, src, sos_idx=0, eos_idx=0, *, max_len=25):
        # src: (1, T)
        out = torch.tensor(sos_idx).expand(1, 1).to(src.device)
        outs = torch.zeros(1, max_len, self.output_vocab_len).to(src.device)
        preds = torch.zeros(1, max_len)

        enc_outputs, state = self.encoder(src)
        state = tuple([s[:self.decoder.num_layers] for s in state])

        for t in range(max_len):
            out, state = self.decoder(enc_outputs, out, state)
            outs[:, t, :] = out
            pred = out.argmax(-1).to(torch.long)
            preds[:, t] = pred
            # No teacher force
            out = pred.unsqueeze(0)
            if pred.item() == eos_idx:
                break

        return outs, preds

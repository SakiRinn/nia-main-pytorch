import os
import numpy as np
import random

import sys
sys.path.insert(0, '.')
import config
import datasets.encoding as encoding

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(act, dim=0):
    try:
        return eval('nn.' + act)(dim=0)
    except TypeError:
        return eval('nn.' + act)()


def find_checkpoint_file(path):
    checkpoint_file = [path + f for f in os.listdir(path) if 'weights' in f]
    if len(checkpoint_file) == 0:
        return []
    modified_time = [os.path.getmtime(f) for f in checkpoint_file]
    return checkpoint_file[np.argmax(modified_time)]


class BilinearAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BilinearAttention, self).__init__()
        # Size
        self.hidden_size = hidden_size
        # Layer
        self.W = nn.Parameter(torch.empty(hidden_size, hidden_size), requires_grad=True)
        nn.init.xavier_uniform_(self.W)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden, encoder_outputs):
        # hidden, encoder_outputs: (N, 1, hidden), (N, T, hidden)
        score = hidden.matmul(self.W).matmul(encoder_outputs.transpose(1, 2)).squeeze()    # (N, T)
        T = score.size(1)
        attn = (self.softmax(score).unsqueeze(2) * hidden.repeat(1, T, 1)).sum(1)    # (N, hidden)
        return attn


class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers=1, dropout=0.5):
        super(Encoder, self).__init__()
        # Size
        self.input_size = input_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Layer
        self.embedding = nn.Embedding(input_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)

    def forward(self, inputs, state=None):
        # inputs: (N, T, input)
        embedded = self.embedding(inputs)    # (N, T, embed)
        outputs, state = self.lstm(embedded, state)    # (N, T, 2*hidden), (2*layer, N, hidden)
        # Avg bidirectional outputs -> (N, T, hidden)
        outputs = (outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]) / 2
        return outputs, state


class Decoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size,
                 num_layers=1, dropout=0.2, activation='Softmax'):
        super(Decoder, self).__init__()
        # Size
        self.input_size = input_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        # Layer
        self.embedding = nn.Sequential(
            nn.Embedding(input_size, embed_size),
            nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=False)
        self.attention = BilinearAttention(hidden_size)
        self.dense = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            get_activation(activation, dim=-1)
        )

    def forward(self, encoder_outputs, input, state=None):
        # encoder_outputs, input: (N, T, hidden), (N, 1, input)
        embedded = self.embedding(input)    # (N, 1, embed)
        output, state = self.lstm(embedded, state)    # (N, 1, hidden)
        output = self.attention(output[:, -1, :].unsqueeze(1), encoder_outputs)    # (N, 1, hidden)
        output = self.dense(output)    # (N, 1, output)
        return output, state


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder,
                 training=False, output_word_to_index=None):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.training = training
        self.output_word_to_index = output_word_to_index

        if self.training:
            assert self.output_word_to_index is not None
        assert encoder.hidden_size == decoder.hidden_size, \
            "Hidden dimensions of encoder and decoder must be equal!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src, trg: (N, T, en_input), (T, max_len)
        batch_size = src.size(0)
        max_len = trg.size(1)
        vocab_size = self.decoder.output_size
        outputs = torch.zeros(max_len, batch_size, vocab_size).cuda()

        encoder_outputs, state = self.encoder(src)
        state = tuple([s[:self.decoder.num_layers] for s in state])
        output = trg[:, 0, :]
        for t in range(1, max_len):
            output, state = self.decoder(encoder_outputs, output, state)
            outputs[t] = output
            pred = encoding.vectorize(output.argmax(1).unsqueeze(1), self.output_word_to_index)
            pred = pred.squeeze().to(torch.long).cuda()
            if self.training:
                is_teacher = random.random() < teacher_forcing_ratio
                output = (trg[:, t, :] if is_teacher else pred)
            else:
                output = pred
        return outputs


if __name__ == '__main__':
    from datasets import ResDataset
    from torch.utils.data import DataLoader

    dataset = ResDataset()
    dl = DataLoader(dataset, config.MODEL_BATCH_SIZE, shuffle=True)
    iterator = iter(dl)
    sample = next(iterator)
    print(sample[0].device, sample[1].device)
    encoder = Encoder(dataset.input_vocab_len, config.MODEL_EMBEDDING_DIM,
                      config.MODEL_HIDDEN_DIM, config.MODEL_HIDDEN_LAYERS)
    decoder = Decoder(dataset.input_vocab_len, config.MODEL_EMBEDDING_DIM,
                      config.MODEL_HIDDEN_DIM, dataset.output_vocab_len, config.MODEL_HIDDEN_LAYERS)
    seq2seq = Seq2Seq(encoder, decoder, True, dataset.output_word_to_index).cuda()
    print(seq2seq(sample[0], sample[1]))
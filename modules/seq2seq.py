from modules.tools import get_activation
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

    def forward(self, hidden, encoder_outputs):
        # hidden, encoder_outputs: (N, 1, hidden), (N, T, hidden)
        score = hidden.matmul(self.W).matmul(encoder_outputs.transpose(1, 2)).squeeze()    # (N, T)
        T = score.size(1)
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

    def forward(self, x, state=None):
        # x: (N, T)
        embedded = self.embedding(x)    # (N, T, embed)
        outputs, state = self.lstm(embedded, state)    # (N, T, 2*hidden), (2*layer, N, hidden)
        # Avg bidirectional outputs -> (N, T, hidden)
        outputs = (outputs[:, :, :self.hidden_dim] + outputs[:, :, self.hidden_dim:]) / 2
        return outputs, state


class Decoder(nn.Module):
    def __init__(self, output_vocab_len, embedding_dim, hidden_dim,
                 num_layers=1, dropout=0.2, activation='Softmax'):
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
        self.dense = nn.Sequential(
            nn.Linear(hidden_dim, output_vocab_len),
            get_activation(activation, dim=-1)
        )

    def forward(self, encoder_outputs, x, state=None):
        # encoder_outputs, x: (N, T, hidden), (N, T)
        embedded = self.embedding(x)    # (N, 1, embed)
        out, state = self.lstm(embedded, state)    # (N, 1, hidden)
        out = self.attention(out[:, -1, :].unsqueeze(1), encoder_outputs)    # (N, 1, hidden)
        out = self.dense(out)    # (N, 1, output)
        return out, state


class Seq2Seq(nn.Module):
    def __init__(self, input_vocab_len, output_vocab_len, embedding_dim, hidden_dim,
                 num_layers=1, encoder_dropout=0.5, decoder_dropout=0.2, activation='Softmax', training=False):
        super(Seq2Seq, self).__init__()
        # Model
        self.encoder = Encoder(input_vocab_len, embedding_dim, hidden_dim, num_layers, encoder_dropout)
        self.decoder = Decoder(output_vocab_len, embedding_dim, hidden_dim, num_layers, decoder_dropout, activation)
        # Option
        self.training = training

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src, trg: (N, T), train (N, T_max) / eval (N, 1)
        batch_dim = src.size(0)
        max_len = trg.size(1)
        preds = torch.zeros(batch_dim, max_len).to(src.device)

        encoder_outputs, state = self.encoder(src)
        state = tuple([s[:self.decoder.num_layers] for s in state])
        output = trg[:, 0].unsqueeze(1)
        for t in range(1, max_len):
            output, state = self.decoder(encoder_outputs, output, state)
            pred = output.argmax(1).to(torch.long)
            preds[:, t] = pred
            if self.training:
                is_teacher = random.random() < teacher_forcing_ratio
                output = trg[:, t].unsqueeze(1) if is_teacher else pred.unsqueeze(1)
            else:
                output = pred.unsqueeze(1)
        return preds


if __name__ == '__main__':
    import yaml
    def load_yaml(path: str):
        output = None
        with open(path, 'r') as f:
            output = yaml.load(f, yaml.FullLoader)
        if 'seq2seq' in output.keys():
            output['seq2seq']['encoder_dropout'] = output['seq2seq']['dropout']['encoder']
            output['seq2seq']['decoder_dropout'] = output['seq2seq']['dropout']['decoder']
            del output['seq2seq']['dropout']
        return output

    import sys
    sys.path.append('.')
    from datasets import ResDataset
    from torch.utils.data import DataLoader

    config = load_yaml('config/model.yaml')
    run_param = load_yaml('config/run.yaml')

    dataset = ResDataset()
    dl = DataLoader(dataset, run_param['train']['batch_size'], shuffle=True)
    iterator = iter(dl)
    sample = next(iterator)
    print(sample[0].device, sample[1].device)
    seq2seq = Seq2Seq(dataset.input_vocab_len, dataset.output_vocab_len, **config['seq2seq']).cuda()
    print(seq2seq(sample[0], sample[1]))
from utils import get_activation

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim):
        super(PositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim

    def forward(self, input):
        # input: (N, T, embed)
        pos = torch.arange(input.size(1)).expand(input.size()[:-1]).unsqueeze(-1).expand(input.size())
        idx = torch.arange(self.embedding_dim).expand(input.size())
        angles = self.get_angles(pos, idx).to(input.device)    # (N, T, embed)

        angles[:, 0::2] = torch.sin(angles[:, 0::2])    # 2i
        angles[:, 1::2] = torch.cos(angles[:, 1::2])    # 2i+1

        return input + angles

    def get_angles(self, pos, idx):
        """pos * 1/(10000^(2i/d))

        Args:
            pos (Tensor[int]): Position of the word, 0 <= pos < T.
            idx (Tensor[int]): Index of input Tensor, i.e. 2*i (odd) or 2*i+1 (even).

        Returns:
            Tensor[int]: Radian angle.
        """
        angle_rates = 1 / torch.pow(10000, torch.floor(idx.to(torch.float)) / self.embedding_dim)
        return pos * angle_rates


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, forward_dim, num_heads=3, dropout=0.1, activation='Softmax'):
        super(EncoderLayer, self).__init__()

        self.mha = nn.MultiheadAttention(embedding_dim, num_heads, dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, forward_dim),
            get_activation(activation),
            nn.Linear(forward_dim, embedding_dim),
            nn.Dropout(dropout)
        )

        self.layerNorm1 = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.layerNorm2 = nn.LayerNorm(embedding_dim, eps=1e-6)

    def forward(self, x, key_padding_mask=None):
        # x, key_padding_mask: (N, T, embed), (N, T)
        attn, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask)
        attn = self.layerNorm1(x + attn)

        out = self.ffn(attn)
        out = self.layerNorm2(attn + out)    # (N, T, embed)

        return out


class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, forward_dim, num_heads=3, dropout=0.1, activation='Softmax'):
        super(DecoderLayer, self).__init__()

        self.masked_mha = nn.MultiheadAttention(embedding_dim, num_heads, dropout, batch_first=True)
        self.mha = nn.MultiheadAttention(embedding_dim, num_heads, dropout, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, forward_dim),
            get_activation(activation),
            nn.Linear(forward_dim, embedding_dim),
            nn.Dropout(dropout)
        )

        self.layerNorm1 = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.layerNorm2 = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.layerNorm3 = nn.LayerNorm(embedding_dim, eps=1e-6)

    def forward(self, enc_outputs, x, key_padding_mask=None, attn_mask=None):
        # x, key_padding_mask, attn_mask: (N, T), (N, T), (N*num_heads, T, T)
        attn1, attn1_weight = self.masked_mha(x, x, x,
                                              key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        attn1 = self.layerNorm1(x + attn1)

        attn2, attn2_weight = self.mha(attn1, enc_outputs, enc_outputs, key_padding_mask=key_padding_mask)
        attn2 = self.layerNorm2(attn1 + attn2)

        ffn_output = self.ffn(attn2)
        out = self.layerNorm3(attn2 + ffn_output)    # (N, T, embed)

        return out, attn1_weight, attn2_weight


class Encoder(nn.Module):
    def __init__(self, input_vocab_len, embedding_dim, forward_dim,
                 num_layers=6, num_heads=3, dropout=0.1, activation='Softmax'):
        super(Encoder, self).__init__()
        # Size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.forward_dim = forward_dim
        # Layer
        self.embedding = nn.Embedding(input_vocab_len, embedding_dim)
        self.pos_encoding = nn.Sequential(
            PositionalEncoding(embedding_dim),
            nn.Dropout(dropout)
        )
        self.enc_layers = nn.ModuleList([EncoderLayer(embedding_dim, forward_dim, num_heads, dropout, activation)
                                         for _ in range(num_layers)])

    def forward(self, src, key_padding_mask=None):
        # src, key_padding_mask: (N, T), (N, T)
        embedded = self.embedding(src)
        out = self.pos_encoding(embedded)

        for i in range(self.num_layers):
            out = self.enc_layers[i](out, key_padding_mask)

        return out


class Decoder(nn.Module):
    def __init__(self, output_vocab_len, embedding_dim, forward_dim,
                 num_layers=6, num_heads=3, dropout=0.1, activation='Softmax'):
        super(Decoder, self).__init__()
        # Size
        self.embedding_dim = embedding_dim
        self.forward_dim = forward_dim
        self.num_layers = num_layers
        # Layer
        self.embedding = nn.Embedding(output_vocab_len, embedding_dim)
        self.pos_encoding = nn.Sequential(
            PositionalEncoding(embedding_dim),
            nn.Dropout(dropout)
        )
        self.dec_layers = nn.ModuleList([DecoderLayer(embedding_dim, forward_dim, num_heads, dropout, activation)
                                         for _ in range(num_layers)])

    def forward(self, enc_outputs, trg, key_padding_mask=None, attn_mask=None):
        # trg, key_padding_mask, attn_mask: (N, T), (N, T), (N*num_heads, T, T)
        embedded = self.embedding(trg)
        embedded = torch.tensor(self.embedding_dim).sqrt().to(trg.device) * embedded
        out = self.pos_encoding(embedded)

        attn1_weights = []
        attn2_weights = []
        for i in range(self.num_layers):
            out, attn1_weight, attn2_weight = self.dec_layers[i](enc_outputs, out, key_padding_mask, attn_mask)
            attn1_weights.append(attn1_weight)
            attn2_weights.append(attn2_weight)

        return out, attn1_weights, attn2_weights


class Transformer(nn.Module):
    def __init__(self, input_vocab_len, output_vocab_len, embedding_dim, forward_dim,
                 num_layers=6, num_heads=3, encoder_dropout=0.1, decoder_dropout=0.1, activation='Softmax'):
        super(Transformer, self).__init__()
        # Model
        self.encoder = Encoder(input_vocab_len, embedding_dim, forward_dim,
                               num_layers, num_heads, encoder_dropout, activation)
        self.decoder = Decoder(output_vocab_len, embedding_dim, forward_dim,
                               num_layers, num_heads, decoder_dropout, activation)
        self.dense = nn.Sequential(
            nn.Linear(embedding_dim, output_vocab_len),
            get_activation(activation)
        )

    def forward(self, src, trg, src_key_padding_mask=None, trg_key_padding_mask=None, attn_mask=None):
        enc_outputs = self.encoder(src, src_key_padding_mask)
        dec_outputs, _, _ = self.decoder(enc_outputs, trg, trg_key_padding_mask, attn_mask)
        outs = self.dense(dec_outputs)
        return outs

    def predict(self, src, trg, mask=None):
        outs = self(src, trg, mask)    # (N, T, output)
        preds = outs.argmax(-1).to(torch.long)    # (N, T)
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
        if 'transformer' in output.keys():
            output['transformer']['encoder_dropout'] = output['transformer']['dropout']['encoder']
            output['transformer']['decoder_dropout'] = output['transformer']['dropout']['decoder']
            del output['transformer']['dropout']
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
    transformer = Transformer(dataset.input_vocab_len, dataset.output_vocab_len, **config['transformer']).cuda()
    print(transformer(sample[0], sample[1]))
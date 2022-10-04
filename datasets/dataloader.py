import utils
import datasets.encoding as encoding

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class ResDataset(Dataset):
    def __init__(self):
        super(ResDataset, self).__init__()
        input_words, output_words = utils.read()

        self.input_index_to_word, self.input_vocab_len, self.input_word_to_index = encoding.build_index(input_words)
        self.output_index_to_word, self.output_vocab_len, self.output_word_to_index = encoding.build_index(output_words)

        self.input_words = encoding.index(input_words, self.input_word_to_index)
        self.output_words = encoding.index(output_words, self.output_word_to_index)

        self.input_max_len = max([len(sentence) for sentence in self.input_words])
        self.output_max_len = max([len(sentence) for sentence in self.output_words])

        self.input_words = pad_sequence(self.input_words).T.to(torch.long)
        self.output_words = pad_sequence(self.output_words).T.to(torch.long)

    def __len__(self):
        return min(len(self.input_words), len(self.output_words))

    def __getitem__(self, idx: int):
        return self.input_words[idx], self.output_words[idx]

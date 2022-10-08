import datasets.encoding as encoding
import datasets

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class ResDataset(Dataset):
    def __init__(self, training=False, validate_split=0.2):
        super(ResDataset, self).__init__()
        input_words, output_words = datasets.read()

        self._input_index_to_word, self._input_vocab_len, self._input_word_to_index = encoding.build_index(input_words)
        self._output_index_to_word, self._output_vocab_len, self._output_word_to_index = encoding.build_index(output_words)

        self._input_words = encoding.index(input_words, self._input_word_to_index)
        self._output_words = encoding.index(output_words, self._output_word_to_index)

        self._input_max_len = max([len(sentence) for sentence in self._input_words])
        self._output_max_len = max([len(sentence) for sentence in self._output_words])

        self._input_words = pad_sequence(self._input_words).T.to(torch.long)
        self._output_words = pad_sequence(self._output_words).T.to(torch.long)

        split_idx = int(min(len(self._input_words), len(self._output_words)) * validate_split)
        if training:
            # Train set
            self._input_words = self._input_words[split_idx:]
            self._output_words = self._output_words[split_idx:]
        else:
            # Test set
            self._input_words = self._input_words[:split_idx]
            self._output_words = self._output_words[:split_idx]

    def __len__(self):
        return min(len(self._input_words), len(self._output_words))

    def __getitem__(self, idx: int):
        return self._input_words[idx], self._output_words[idx]

    def index_to_word(self):
        return self._input_index_to_word, self._output_index_to_word

    def words(self):
        return self._input_words, self._output_words

    def sos_index(self):
        return self._input_word_to_index['SOS'], self._output_word_to_index['SOS']

    def eos_index(self):
        return self._input_word_to_index['EOS'], self._output_word_to_index['EOS']

    def vocab_len(self):
        return self._input_vocab_len, self._output_vocab_len
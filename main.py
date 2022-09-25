import numpy as np

import datasets
import datasets.encoding as encoding
import datasets.tools as tools
import modules
import config

from torch.nn.utils.rnn import pad_sequence


def train():
    encoder = modules.Encoder(config.MODEL_EMBEDDING_DIM)


if __name__ == "__main__":
    train()

import numpy as np

import datasets
import datasets.encoding as encoding
import datasets.tools as tools
import modules
import config

from torch.nn.utils.rnn import pad_sequence


def get_model(name):
    return eval('modules.' + name)


def train():
    ...


def test():
    ...


if __name__ == "__main__":
    train()

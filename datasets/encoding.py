from utils.fileIO import load_config
from nltk import FreqDist
import numpy as np
import torch

dataset_cfg = load_config('config/dataset.yaml')


# vectorize each word of each sentence as a binary array
def vectorize(sentences, word_index):
    # Vectorizing each element in each sequence
    sentence_max_len = max([len(sentence) for sentence in sentences])
    sequences = torch.zeros(len(sentences), sentence_max_len, len(word_index))
    for i, sentence in enumerate(sentences):
        for j, word in enumerate(sentence):
            sequences[i, j, word] = 1.
    return sequences


# replace words in sentences with index values of it
def index(sentences, word_index):
    # Converting each word to its index value
    outputs = []
    for sentence in sentences:
        output = torch.empty(len(sentence))
        for j, word in enumerate(sentence):
            if word in word_index:
                output[j] = word_index[word]
            else:
                output[j] = word_index['UNK']
        outputs.append(output)
    return outputs


def build_index(words):
    # Creating the vocabulary set with the most common words
    dist = FreqDist(np.hstack(words))
    vocab = dist.most_common(dataset_cfg['vocab_size'] - 1)

    word_index = [word[0] for word in vocab]  # Creating an array of words from the vocabulary set, we will use this array as index-to-word dictionary
    word_index.insert(0, 'ZERO')  # Adding the word "ZERO" to the beginning of the array
    word_index.append('UNK')  # Adding the word 'UNK' to the end of the array (stands for UNKNOWN words)
    index_to_word = {word: idx for idx, word in enumerate(word_index)}  # Creating the word-to-index dictionary from the array created above

    return word_index, len(vocab) + 2, index_to_word

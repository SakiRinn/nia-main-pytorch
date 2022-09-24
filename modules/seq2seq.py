import os
import re
import numpy as np

import config
import datasets.encoding as encoding
from . import tools

from keras.layers import (Activation, Dense, Embedding, RepeatVector,
                          TimeDistributed, recurrent)
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences


class AttentionSeq2Seq:
    def __init__(self, input_words, output_words):
        # mapping between word and index. - list, int, dict
        # e.g.
        # index_to_word: ['ZERO', '@qos', '@location', '@hour', ...]
        # word_to_index: {'ZERO': 0, '@qos': 1, '@location': 2, '@hour': 3, ...}
        self.input_index_to_word, self.input_vocab_len, self.input_word_to_index = encoding.build_index(input_words)
        self.output_index_to_word, self.output_vocab_len, self.output_word_to_index = encoding.build_index(output_words)

        # Indexed words - list
        # e.g. [6, 2, 4, 4, 3, 11, 7]
        self.input_words = encoding.index(input_words, self.input_word_to_index)
        self.output_words = encoding.index(output_words, self.output_word_to_index)

        # Finding the length of the longest sequence. - int
        self.input_max_len = max([len(sentence) for sentence in self.input_words])
        self.output_max_len = max([len(sentence) for sentence in self.output_words])

        # Right-aligned zero padding. - ndarray(sentences_num, max_len)
        self.input_words = pad_sequences(self.input_words, maxlen=self.input_max_len, dtype='int32')
        self.output_words = pad_sequences(self.output_words, maxlen=self.output_max_len, dtype='int32')

        print('[INFO] Creating model...')
        self.model = Sequential()

        print('[INFO] Creating encoder...')
        # Creating encoder network
        self.model.add(Embedding(self.input_vocab_len, 1000, input_length=self.input_max_len, mask_zero=True))
        self.model.add(LSTM(config.MODEL_HIDDEN_DIM))

        print('[INFO] Creating decoder...')
        # Creating decoder network
        self.model.add(RepeatVector(self.output_max_len))
        for i in range(config.MODEL_HIDDEN_LAYERS):
            self.model.add(LSTM(config.MODEL_HIDDEN_DIM, return_sequences=True))
        self.model.add(TimeDistributed(Dense(self.output_vocab_len)))
        self.model.add(Activation(config.MODEL_ACTIVATION))

        print('[INFO] Compiling model...')
        self.model.compile(loss=config.MODEL_LOSS, optimizer=config.MODEL_OPTIMIZER,  metrics=config.MODEL_METRICS)

    def train(self):

        # Load checkpoints if exists.
        self.saved_weights = tools.find_checkpoint_file(config.MODEL_DIR)
        k_start = 1
        if len(self.saved_weights) != 0:
            print('[INFO] Saved weights found, loading...', self.saved_weights)
            epoch = self.saved_weights[self.saved_weights.rfind('_') + 1:self.saved_weights.rfind('.')]
            self.model.load_weights(self.saved_weights)
            k_start = int(epoch) + 1

        print('[INFO] Training...')
        # => EPOCH
        for epoch in range(k_start, config.MODEL_EPOCHS + 1):

            # Shuffling the training data to avoid local minima.
            indices = np.arange(len(self.input_words))
            np.random.shuffle(indices)
            self.input_words = self.input_words[indices]
            self.output_words = self.output_words[indices]

            # => STEP
            i_end = 0
            for step in range(0, len(self.input_words), config.MODEL_BATCH_SIZE):

                # Vectorize the mini-batch data.
                if step + len(self.input_words) >= len(self.input_words):
                    i_end = len(self.input_words)
                else:
                    i_end = step + config.MODEL_BATCH_SIZE
                output_sequences = encoding.vectorize(self.output_words[step:i_end], self.output_max_len, self.output_word_to_index)

                # Train one step.
                print(f'[INFO] Training model: epoch {epoch}th {step}/{self.input_words} samples')
                self.model.fit(self.input_words[step:i_end], output_sequences, batch_size=config.MODEL_BATCH_SIZE, validation_split=config.MODEL_VALIDATION_SPLIT, epochs=1, verbose=2)

                # Save checkpoints.
                while len(os.listdir(config.MODEL_DIR)) >= config.MODEL_MAX_CHECKPOINTS:
                    nums = [int(re.split('[_.]', file)[-2]) for file in os.listdir(config.MODEL_DIR)]
                    os.remove(config.MODEL_WEIGHTS_PATH.format(min(nums)))
                self.model.save_weights(config.MODEL_WEIGHTS_PATH.format(epoch))

    def test(self):
        self.saved_weights = tools.find_checkpoint_file()
        if len(self.saved_weights) != 0:
            print('[INFO] Saved weights found, loading...', self.saved_weights)
            epoch = self.saved_weights[self.saved_weights.rfind('_') + 1:self.saved_weights.rfind('.')]
            self.model.load_weights(self.saved_weights)

        if len(self.saved_weights) == 0:
            print("The network hasn't been trained! Program will exit...")
            exit()
        else:
            print('[INFO] Testing...')
            self.model.load_weights(self.saved_weights)

            result = self.model.predict(self.input_words)
            predictions = np.argmax(result, axis=2)

            inputs = []
            outputs = []
            for input_sequence, prediction in zip(self.input_words, predictions):

                entities = ' '.join([self.input_index_to_word[index] for index in input_sequence if index > 0])
                sequence = ' '.join([self.output_index_to_word[index] for index in prediction if index > 0])
                print(prediction)
                print(entities)
                print(sequence)
                inputs.append(entities)
                outputs.append(sequence)

            np.savetxt(config.MODEL_TEST_INPUT_PATH, inputs, fmt='%s')
            np.savetxt(config.MODEL_TEST_RESULT_PATH, outputs, fmt='%s')

    def predict(self, entities):
        sequence = ''
        self.saved_weights = tools.find_checkpoint_file()
        if len(self.saved_weights) != 0:
            print('[INFO] Saved weights found, loading...', self.saved_weights)
            epoch = self.saved_weights[self.saved_weights.rfind('_') + 1:self.saved_weights.rfind('.')]
            self.model.load_weights(self.saved_weights)

        if len(self.saved_weights) == 0:
            print("The network hasn't been trained! Program will exit...")
        else:
            print('[INFO] Testing...')
            self.model.load_weights(self.saved_weights)

            entities = encoding.index(entities, self.input_word_to_index)
            predictions = np.argmax(self.model.predict([entities]), axis=2)
            sequence = ' '.join([self.output_index_to_word[index] for index in predictions if index > 0])

        return sequence


if __name__ == '__main__':
    ...
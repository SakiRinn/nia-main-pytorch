import numpy as np

import datasets
import datasets.tools as tools
import modules
from modules.seq2seq import AttentionSeq2Seq

train = True
test = True


def predict(username, origin, destination, targets, middleboxes, qos, start, end, allow, block):
    global seq2seq
    entities = tools.anonymize(username, origin, destination, targets,
                               middleboxes, qos, start, end, allow, block)
    intent = AttentionSeq2Seq.predict(entities)
    print('intent', intent)
    result = tools.deanonymize(intent, username, origin, destination, targets,
                               middleboxes, qos, start, end, allow, block)
    print('result', result)

    return result


def main():
    global seq2seq, train, test
    input_words, output_words = tools.read()

    # Creating the network model
    seq2seq = modules.AttentionSeq2Seq(input_words, output_words)
    if train:
        seq2seq.train()
        train = False

    if test:
        seq2seq.test()


if __name__ == "__main__":
    main()

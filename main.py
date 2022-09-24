import numpy as np

import datasets
import datasets.tools as tools
import modules

train = True
test = True

seq2seq = None


def predict(username, origin, destination, targets, middleboxes, qos, start, end, allow, block):
    global seq2seq
    entities = tools.anonymize(username, origin, destination, targets, middleboxes, qos, start, end, allow, block)
    intent = seq2seq.predict(entities)
    print('intent', intent)
    result = tools.deanonymize(deanonymize, username, origin, destination, targets, middleboxes, qos, start, end, allow, block)
    print('result', result)

    return result


def main():
    global seq2seq, train, test
    input_words, output_words = datasets.read()

    # Creating the network model
    seq2seq = modules.AttentionSeq2Seq(input_words, output_words)
    if train:
        seq2seq.train()
        train = False

    if test:
        seq2seq.test()


if __name__ == "__main__":
    main()

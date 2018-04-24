import json
import os
import numpy as np
from glob import glob
from model.reader import load_data_and_labels
from model import wrapper

VOCAB_PATH = 'embedding/vocabs.json'
EMBEDDING_PATH = 'embedding/word_embeddings.npy'

def main(train_dir, dev_dir, test_dir):
    print('Loading data...')
    x_valid, y_valid = load_data_and_labels(dev_dir)
    x_test, y_test = load_data_and_labels(test_dir)
    print(len(x_valid), 'valid sequences')
    print(len(x_test), 'test sequences')

    embeddings = np.load(EMBEDDING_PATH)
    vocabs = json.load(open(VOCAB_PATH, "r", encoding="utf8"))

    # Use pre-trained word embeddings
    model = wrapper.Sequence(max_epoch=20, embeddings=embeddings, vocab_init=vocabs, patience=5, log_dir="log")

    # for train_path in train_paths:
    x_train, y_train = load_data_and_labels(train_dir)
    print(len(x_train), 'train sequences')
    model.train(x_train, y_train, x_valid, y_valid)
    model.eval(x_test, y_test)
import os
import itertools
from unicodedata import category

import numpy as np
import re

from utils import ObjectDict, make_dict, json_load, get_name, parse, read, pkl_load, json_dump
from glob import glob

UNK = '<UNK>'
PAD = '<PAD>'

class Counter:
    def __init__(self):
        self.max_sen_len = 0
        self.longest_sen = None
        self.max_word_len = 0
        self.word_vocab = set()
        self.char_vocab = set()
        self.pos_tags = set()
        self.ner_tags = set()
        self.kb_words = ObjectDict()

    def update(self, sen):
        self.max_sen_len = max(self.max_sen_len, len(sen))
        if self.max_sen_len == len(sen):
            self.longest_sen = sen
        self.max_word_len = max(self.max_word_len, sen.max_word_len)
        self.word_vocab |= sen.word_vocab
        self.char_vocab |= sen.char_vocab
        self.pos_tags |= sen.pos_tags
        self.ner_tags |= sen.ner_tags
        for tag, words in sen.kb_words.items():
            if tag not in self.kb_words:
                self.kb_words[tag] = set()
            self.kb_words[tag] |= words

    def longest_word(self):
        sort = sorted(self.word_vocab, key=lambda w: len(w))
        return sort[-1]

    def __json__(self):
        return {
            "word_vocab": list(self.word_vocab),
            "char_vocab": list(self.char_vocab),
            "pos_tags": list(self.pos_tags),
            "ner_tags": list(self.ner_tags),
            "max_sen_len": self.max_sen_len,
            "max_word_len": self.max_word_len,
        }

    def __str__(self):
        data = {
            "max_sen_len": self.max_sen_len,
            "max_word_len": self.max_word_len,
            "word_vocab_size": len(self.word_vocab),
            "char_vocab_size": len(self.char_vocab),
        }
        return str(data)


class Word:
    def __init__(self, data):
        self.word, self.pos, self.ner = np.array(data)[[0, 1, 3]]
        self.normalized = normalize_word(self.word)
        self.chars = [c for c in self.word]
        self.char_vocab = set(self.chars)

    def __len__(self):
        return len(self.chars)

    def __str__(self):
        return self.word

    def should_ignore(self):
        return all(category(c)[0] in ["P", "N"] for c in self.chars)


class Sentence:
    def __init__(self, words):
        self.words = [Word(w) for w in words]
        self.max_word_len = max(len(w) for w in self.words)
        self.word_vocab = set(w.normalized for w in self.words)
        self.char_vocab = [w.char_vocab for w in self.words]
        self.char_vocab = itertools.chain(*self.char_vocab)
        self.char_vocab = set(self.char_vocab)
        self.pos_tags = set(w.pos for w in self.words)
        self.ner_tags = set(w.ner for w in self.words)
        self.kb_words = ObjectDict()
        for w, pre_w in zip(self.words[1:], self.words[:-1]):
            if w.ner != 'O' and pre_w.ner == 'O' and not pre_w.should_ignore():
                ner = w.ner[-3:]
                if ner not in self.kb_words:
                    self.kb_words[ner] = set()
                self.kb_words[ner].add(pre_w.normalized)

    def __len__(self):
        return len(self.words)

    def __str__(self):
        return " ".join(map(str, self.words))


def read_file(path, counter):
    sentences = parse(read(path), ["\n\n", "\n", "\t"])
    for sentence in sentences:
        sentence = Sentence(sentence)
        counter.update(sentence)
    print("read %d sentences" % len(sentences))
    return len(sentences)


def normalize_word(word):
    word = word.lower()
    word = "".join(['0' if category(c).startswith("N") else c for c in word])
    return word


def construct_word_embeddings(word_vocab):
    unknown_dir = "embedding/unknown.npy"
    vectors_dir = "embedding/vectors.npy"
    words_dir = "embedding/words.json"
    word_embeddings_dir = "embedding/word_embeddings.npy"

    print("read word embedding")
    vectors = np.load(vectors_dir)
    unknown = np.load(unknown_dir).reshape([-1])
    words = json_load(words_dir)
    word2idx = {PAD: 0, UNK: 1}
    word_embeddings = [
        np.zeros(unknown.shape),
        unknown
    ]
    for w, v in zip(words, vectors):
        if w in word_vocab:
            word2idx[w] = len(word2idx)
            word_embeddings.append(v)
    np.save(word_embeddings_dir, word_embeddings)
    return word2idx


def construct_char_embeddings(char_vocab):
    char2idx = {
        PAD: 0,
        UNK: 1,
        **{c: i + 2 for i, c in enumerate(char_vocab)},
    }
    return char2idx


def construct_pos_embeddings(pos_tags):
    pos2idx = {
        PAD: 0,
        **{t: i + 1 for i, t in enumerate(pos_tags)}
    }
    return pos2idx


def construct_ner_embeddings(ner_tags):
    ner2idx = {
        PAD: 0,
        **{t: i + 1 for i, t in enumerate(ner_tags)},
    }
    return ner2idx


def main(train_dir, dev_dir, test_dir):
    # input_dir = "data/train/*.muc"
    vocabs_dir = "embedding/vocabs.json"

    counter = Counter()


    # num_sens = 0
    read_file(train_dir, counter)
    read_file(dev_dir, counter)
    read_file(test_dir, counter)
    # file_train_paths = [
    #     "data/train/%s.muc" % s for s in ["-Doi_song"]
    # ]
    # for path in file_train_paths:
    #     print("read %s" % path)
    #     num_sens += read_file(path, counter)
    # file_paths = [
    #     "data/dev/-Doi_song.muc",
    #     "data/test/-Doi_song.muc"
    # ]
    # for path in file_paths:
    #     print("read %s" % path)
    #     read_file(path, counter)


    print(counter)
    # print("Num sent train: %s" % num_sens)
    print("longest sentence: %s" % str(counter.longest_sen))
    print("longest word: %s" % counter.longest_word())

    kb_words = {k: list(v) for k, v in counter.kb_words.items()}
    json_dump(kb_words, "embedding/kb_words.json")

    word2idx = construct_word_embeddings(counter.word_vocab)
    char2idx = construct_char_embeddings(counter.char_vocab)
    pos2idx = construct_pos_embeddings(counter.pos_tags)
    ner2idx = construct_ner_embeddings(counter.ner_tags)

    vocabs = ObjectDict(make_dict(word2idx, char2idx, ner2idx, pos2idx), max_sen_len=counter.max_sen_len, max_word_len=counter.max_word_len)
    vocabs.save(vocabs_dir)
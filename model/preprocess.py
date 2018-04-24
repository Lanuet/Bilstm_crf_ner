# -*- coding: utf-8 -*-
import itertools
import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib
from unicodedata import category

UNK = '<UNK>'
PAD = '<PAD>'


class WordPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self,
                 lowercase=True,
                 num_norm=True,
                 char_feature=True,
                 pre_word_feature=True,
                 vocab_init=None,
                 padding=True,
                 return_lengths=True):

        self.lowercase = lowercase
        self.num_norm = num_norm
        self.char_feature = char_feature
        self.pre_word_feature = pre_word_feature
        self.padding = padding
        self.return_lengths = return_lengths
        self.vocab_word = vocab_init["word2idx"]
        self.vocab_char = vocab_init["char2idx"]
        self.vocab_tag  = vocab_init["ner2idx"]
        self.vocab_pos = vocab_init["pos2idx"]

    def transform(self, X, kb_avg, y=None):
        """transforms input(s)

        Args:
            X: list of list of words
            y: list of list of tags

        Returns:
            numpy array: sentences
            numpy array: tags

        Examples:
            >>> X = [['President', 'Obama', 'is', 'speaking']]
            >>> print(self.transform(X))
            [
                [
                    [1999, 1037, 22123, 48388],       # word ids
                ],
                [
                    [
                        [1, 2, 3, 4, 5, 6, 7, 8, 9],  # list of char ids
                        [1, 2, 3, 4, 5, 0, 0, 0, 0],  # 0 is a pad
                        [1, 2, 0, 0, 0, 0, 0, 0, 0],
                        [1, 2, 3, 4, 5, 6, 7, 8, 0]
                    ]
                ]
            ]
        """
        words = []
        poss = []
        chars = []
        pre_words = []
        lengths = []
        for sent in X:
            word_ids = []
            pos_ids = []
            char_ids = []
            pre_words_ids = []
            lengths.append(len(sent))
            for w, pos, pre_w in sent:
                if self.char_feature:
                    char_ids.append(self._get_char_ids(w))

                w = self._lower(w)
                w = self._normalize_num(w)
                if w in self.vocab_word:
                    word_id = self.vocab_word[w]
                else:
                    word_id = self.vocab_word[UNK]
                word_ids.append(word_id)
                pos_ids.append(self.vocab_pos[pos])

                if pre_w is None:
                    pre_word_id = self.vocab_word[PAD]
                else:
                    pre_w = self._lower(pre_w)
                    pre_w = self._normalize_num(pre_w)
                    if pre_w in self.vocab_word:
                        pre_word_id = self.vocab_word[pre_w]
                    else:
                        pre_word_id = self.vocab_word[UNK]
                pre_words_ids.append(pre_word_id)

            words.append(word_ids)
            poss.append(pos_ids)
            if self.char_feature:
                chars.append(char_ids)
            if self.pre_word_feature:
                pre_words.append(pre_words_ids)

        if y is not None:
            y = [[self.vocab_tag[t] for t in sent] for sent in y]

        if self.padding:
            sents, y = self.pad_sequence(words, poss, chars, pre_words, kb_avg, y)
        else:
            sents = [words, poss, chars, pre_words]

        if self.return_lengths:
            lengths = np.asarray(lengths, dtype=np.int32)
            lengths = lengths.reshape((lengths.shape[0], 1))
            sents.append(lengths)

        return (sents, y) if y is not None else sents

    def transform_kb(self, kb_words):
        kb_words = list(sorted(kb_words.items(), key=lambda x: x[0]))
        kb_words = [x[1] for x in kb_words]
        kb_words = [[self.vocab_word.get(w, self.vocab_word[UNK]) for w in words] for words in
                    kb_words]
        kb_words, _ = pad_sequences(kb_words, self.vocab_word[PAD])
        kb_words = np.asarray(kb_words)
        kb_words = np.expand_dims(kb_words, 0)
        return kb_words


    def inverse_transform(self, y):
        indice_tag = {i: t for t, i in self.vocab_tag.items()}
        return [indice_tag[y_] for y_ in y]

    def _get_char_ids(self, word):
        return [self.vocab_char.get(c, self.vocab_char[UNK]) for c in word]

    def _lower(self, word):
        return word.lower() if self.lowercase else word

    def _normalize_num(self, word):
        if self.num_norm:
            # return re.sub(r'[0-9０１２３４５６７８９]', r'0', word)
            return "".join(['0' if category(c).startswith('N') else c for c in word])
        else:
            return word

    def pad_sequence(self, word_ids, pos_ids, char_ids, pre_word_ids, kb_avg, labels=None):
        if labels:
            labels, _ = pad_sequences(labels, 0)
            labels = np.asarray(labels)
            labels = dense_to_one_hot(labels, len(self.vocab_tag), nlevels=2)

        word_ids, sequence_lengths = pad_sequences(word_ids, 0)
        word_ids = np.asarray(word_ids)

        pos_ids, _ = pad_sequences(pos_ids, 0)
        pos_ids = np.asarray(pos_ids)

        x = [word_ids, pos_ids]

        if self.char_feature:
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0, nlevels=2)
            char_ids = np.asarray(char_ids)
            x.append(char_ids)
        if self.pre_word_feature:
            pre_word_ids, _ = pad_sequences(pre_word_ids, 0)
            pre_word_ids = np.asarray(pre_word_ids)
            x.append(pre_word_ids)

        kb = np.repeat(np.expand_dims(kb_avg, 0), word_ids.size, 0).reshape((*word_ids.shape, kb_avg.size))
        x.append(kb)
        return x, labels

    def save(self, file_path):
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path):
        p = joblib.load(file_path)
        return p


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple.
        pad_tok: the char to pad with.

    Returns:
        a list of list where each sublist has same length.
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple.
        pad_tok: the char to pad with.

    Returns:
        a list of list where each sublist has same length.
    """
    if nlevels == 1:
        max_length = len(max(sequences, key=len))
        sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_length)
    elif nlevels == 2:
        max_length_word = max(len(max(seq, key=len)) for seq in sequences)
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x: len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok] * max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)
    else:
        raise ValueError('nlevels can take 1 or 2, not take {}.'.format(nlevels))

    return sequence_padded, sequence_length


def dense_to_one_hot(labels_dense, num_classes, nlevels=1):
    """Convert class labels from scalars to one-hot vectors."""
    if nlevels == 1:
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes), dtype=np.int32)
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot
    elif nlevels == 2:
        # assume that labels_dense has same column length
        num_labels = labels_dense.shape[0]
        num_length = labels_dense.shape[1]
        labels_one_hot = np.zeros((num_labels, num_length, num_classes), dtype=np.int32)
        layer_idx = np.arange(num_labels).reshape(num_labels, 1)
        # this index selects each component separately
        component_idx = np.tile(np.arange(num_length), (num_labels, 1))
        # then we use `a` to select indices according to category label
        labels_one_hot[layer_idx, component_idx, labels_dense] = 1
        return labels_one_hot
    else:
        raise ValueError('nlevels can take 1 or 2, not take {}.'.format(nlevels))


def prepare_preprocessor(vocab_init=None):
    p = WordPreprocessor(vocab_init=vocab_init)
    return p


def filter_embeddings(embeddings, vocab, dim):
    """Loads GloVe vectors in numpy array.

    Args:
        embeddings (dict): a dictionary of numpy array.
        vocab (dict): word_index lookup table.

    Returns:
        numpy array: an array of word embeddings.
    """
    _embeddings = np.zeros([len(vocab), dim])
    for word in vocab:
        if word in embeddings:
            word_idx = vocab[word]
            _embeddings[word_idx] = embeddings[word]

    return _embeddings
from collections import defaultdict

import numpy as np

from model.metrics import get_entities
from model.reader import batch_iter
from lifelong import update


class Tagger(object):

    def __init__(self, model, kb_miner, preprocessor=None, lifelong_threshold=3):
        self.model = model
        self.kb_miner = kb_miner
        self.preprocessor = preprocessor
        self.lifelong_threshold = lifelong_threshold

    def predict(self, words):
        length = np.array([len(words)])
        X = self.preprocessor.transform([words])
        pred = self.model.predict(X, length)

        return pred

    def _get_tags(self, pred):
        pred = np.argmax(pred, -1)
        tags = self.preprocessor.inverse_transform(pred[0])

        return tags

    def _get_prob(self, pred):
        prob = np.max(pred, -1)[0]

        return prob

    def _build_response(self, words, tags, prob):
        res = {
            'words': words,
            'entities': [

            ]
        }
        chunks = get_entities(tags)

        for chunk_type, chunk_start, chunk_end in chunks:
            entity = {
                'text': ' '.join(words[chunk_start: chunk_end]),
                'type': chunk_type,
                'score': float(np.average(prob[chunk_start: chunk_end])),
                'beginOffset': chunk_start,
                'endOffset': chunk_end
            }
            res['entities'].append(entity)

        return res

    def analyze(self, words):
        assert isinstance(words, list)

        pred = self.predict(words)
        tags = self._get_tags(pred)
        prob = self._get_prob(pred)
        res = self._build_response(words, tags, prob)

        return res

    def tag(self, sents, kb_words):
        """Tags a sentence named entities.

        Args:
            sent: a sentence

        Return:
            labels_pred: list of (word, tag) for a sentence
        """
        # print("compute kb avg")
        kb_avg = self.preprocessor.transform_kb(kb_words)
        kb_avg = self.kb_miner.predict(kb_avg)
        kb_avg = kb_avg.reshape((-1,))

        # print("preprocess data")
        data = self.preprocessor.transform(sents, kb_avg)
        sequence_lengths = data[-1]
        sequence_lengths = np.reshape(sequence_lengths, (-1,))
        y_pred = self.model.predict(data, len(sequence_lengths))
        y_pred = np.argmax(y_pred, -1)
        y_pred = [self.preprocessor.inverse_transform(y[:l]) for y, l in zip(y_pred, sequence_lengths)]

        sentences = []
        for s, labels in zip(sents, y_pred):
            sen = []
            for w, tag in zip(s, labels):
                w = self.preprocessor.normalize(w[0])
                sen.append((w, tag))
            sentences.append(sen)

        new_kb, new_words = update(kb_words, sentences, min_count=self.lifelong_threshold)

        return new_kb, new_words

    def get_entities(self, words):
        """Gets entities from a sentence.

        Args:
            sent: a sentence

        Return:
            labels_pred: dict of entities for a sentence
        """
        assert isinstance(words, list)

        pred = self.predict(words)
        entities = self._get_chunks(words, pred)

        return entities

    def _get_chunks(self, words, tags):
        """
        Args:
            words: sequence of word
            tags: sequence of labels

        Returns:
            dict of entities for a sequence

        Example:
            words = ['President', 'Obama', 'is', 'speaking', 'at', 'the', 'White', 'House', '.']
            tags = ['O', 'B-Person', 'O', 'O', 'O', 'O', 'B-Location', 'I-Location', 'O']
            result = {'Person': ['Obama'], 'LOCATION': ['White House']}
        """
        chunks = get_entities(tags)
        res = defaultdict(list)
        for chunk_type, chunk_start, chunk_end in chunks:
            res[chunk_type].append(' '.join(words[chunk_start: chunk_end]))  # todo delimiter changeable

        return res

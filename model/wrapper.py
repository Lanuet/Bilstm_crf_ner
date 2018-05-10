import os
import numpy as np
from keras.optimizers import Adam, SGD
from model.config import ModelConfig, TrainingConfig
from model.evaluator import Evaluator
from model.models import SeqLabeling, KBMiner
from model.preprocess import prepare_preprocessor, WordPreprocessor, filter_embeddings
from model.tagger import Tagger
from model.trainer import Trainer


class Sequence(object):
    config_file = 'config.json'
    weight_file = 'model_weights.h5'
    preprocessor_file = 'preprocessor.pkl'

    def __init__(self, char_emb_size=30, word_emb_size=100, char_lstm_units=30,
                 word_lstm_units=100, dropout=0.5, char_feature=True, crf=True,
                 batch_size=20, optimizer='adam', learning_rate=0.001, lr_decay=0.9,
                 clip_gradients=5.0, max_epoch=20, early_stopping=True, patience=3,
                 train_embeddings=True, max_checkpoints_to_keep=5, log_dir=None,
                 embeddings=(), vocab_init=None, pre_word_feature_size=100):

        self.model_config = ModelConfig(char_emb_size, word_emb_size, char_lstm_units,
                                        word_lstm_units, dropout, char_feature, crf, pre_word_feature_size)
        self.training_config = TrainingConfig(batch_size, optimizer, learning_rate,
                                              lr_decay, clip_gradients, max_epoch,
                                              early_stopping, patience, train_embeddings,
                                              max_checkpoints_to_keep)
        self.model = None
        self.p = None
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.embeddings = embeddings

        self.p = WordPreprocessor(vocab_init=vocab_init)
        self.model_config.vocab_size = len(self.p.vocab_word)
        self.model_config.char_vocab_size = len(self.p.vocab_char)
        self.model_config.pos_vocab_size = len(self.p.vocab_pos)

        self.model = SeqLabeling(self.model_config, self.embeddings, len(self.p.vocab_tag))
        if self.training_config.optimizer == 'sgd':
            opt = SGD(lr=self.training_config.learning_rate)
        elif self.training_config.optimizer == 'adam':
            opt = Adam(lr=self.training_config.learning_rate)
        self.model.compile(loss=self.model.crf.loss, optimizer=opt)

        self.kb_miner = KBMiner(self.model_config, self.embeddings, 4)
        self.kb_miner.compile(optimizer=opt, loss='sparse_categorical_crossentropy')


    def train(self, x_train, kb_words, y_train, x_valid=None, y_valid=None):
        trainer = Trainer(self.model, self.kb_miner,
                          self.training_config,
                          checkpoint_path=self.log_dir,
                          preprocessor=self.p)
        trainer.train(x_train, kb_words, y_train, x_valid, y_valid)

    def eval(self, x_test, kb_words, y_test):
        if self.model:
            evaluator = Evaluator(self.model, self.kb_miner, preprocessor=self.p)
            evaluator.eval(x_test, kb_words, y_test)
        else:
            raise (OSError('Could not find a model. Call load(dir_path).'))

    def tag(self, sents, kb_words, max_retry=10):
        if self.model:
            tagger = Tagger(self.model, self.kb_miner, preprocessor=self.p, lifelong_threshold=3)
            new_words = None
            counter = 0
            while new_words is None or new_words > 0:
                kb_words, new_words = tagger.tag(sents, kb_words)
                if new_words > 0:
                    counter += 1
                    print("added %d words" % new_words)
                if max_retry is not None and counter > max_retry:
                    break
            return kb_words
        else:
            raise (OSError('Could not find a model. Call load(dir_path).'))

    def analyze(self, words):
        if self.model:
            tagger = Tagger(self.model, preprocessor=self.p)
            return tagger.analyze(words)
        else:
            raise (OSError('Could not find a model. Call load(dir_path).'))

    def save(self, dir_path):
        self.p.save(os.path.join(dir_path, self.preprocessor_file))
        self.model_config.save(os.path.join(dir_path, self.config_file))
        self.model.save(os.path.join(dir_path, self.weight_file))

    @classmethod
    def load(cls, dir_path):
        self = cls()
        self.p = WordPreprocessor.load(os.path.join(dir_path, cls.preprocessor_file))
        config = ModelConfig.load(os.path.join(dir_path, cls.config_file))
        dummy_embeddings = np.zeros((config.vocab_size, config.word_embedding_size), dtype=np.float32)
        self.model = SeqLabeling(config, dummy_embeddings, ntags=len(self.p.vocab_tag))
        self.model.load(filepath=os.path.join(dir_path, cls.weight_file))

        return self

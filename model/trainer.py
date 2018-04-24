from model.reader import batch_iter
from keras.optimizers import Adam

from model.metrics import get_callbacks


class Trainer(object):

    def __init__(self,
                 model, kb_miner,
                 training_config,
                 checkpoint_path='',
                 save_path='',
                 tensorboard=True,
                 preprocessor=None,
                 ):

        self.model = model
        self.kb_miner = kb_miner
        self.training_config = training_config
        self.checkpoint_path = checkpoint_path
        self.save_path = save_path
        self.tensorboard = tensorboard
        self.preprocessor = preprocessor

    def train(self, x_train, kb_words, y_train, x_valid=None, y_valid=None):

        kb_words = self.preprocessor.transform_kb(kb_words)
        kb_avg = self.kb_miner.predict(kb_words)
        kb_avg = kb_avg.reshape((-1,))

        # Prepare training and validation data(steps, generator)
        train_steps, train_batches = batch_iter(x_train, kb_avg,
                                                y_train,
                                                self.training_config.batch_size,
                                                preprocessor=self.preprocessor)
        valid_steps, valid_batches = batch_iter(x_valid, kb_avg,
                                                y_valid,
                                                self.training_config.batch_size,
                                                preprocessor=self.preprocessor)

        # self.model.compile(loss=self.model.crf.loss,
        #                    optimizer=Adam(lr=self.training_config.learning_rate),
        #                    )

        # Prepare callbacks
        callbacks = get_callbacks(log_dir=self.checkpoint_path,
                                  tensorboard=self.tensorboard,
                                  eary_stopping=self.training_config.early_stopping,
                                  valid=(valid_steps, valid_batches, self.preprocessor))

        # Train the model
        self.model.fit_generator(generator=train_batches,
                                 steps_per_epoch=train_steps,
                                 epochs=self.training_config.max_epoch,
                                 callbacks=callbacks)
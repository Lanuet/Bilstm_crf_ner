from model.reader import batch_iter

from model.metrics import F1score


class Evaluator(object):

    def __init__(self,
                 model, kb_miner,
                 preprocessor=None):

        self.model = model
        self.kb_miner = kb_miner
        self.preprocessor = preprocessor

    def eval(self, x_test, kb_words, y_test):
        kb_words = self.preprocessor.transform_kb(kb_words)
        kb_avg = self.kb_miner.predict(kb_words)
        kb_avg = kb_avg.reshape((-1,))

        # Prepare test data(steps, generator)
        train_steps, train_batches = batch_iter(x_test, kb_avg,
                                                y_test,
                                                batch_size=20,  # Todo: if batch_size=1, eval does not work.
                                                shuffle=False,
                                                preprocessor=self.preprocessor)

        # Build the evaluator and evaluate the model
        f1score = F1score(train_steps, train_batches, self.preprocessor)
        f1score.model = self.model
        f1score.on_epoch_end(epoch=-1)  # epoch takes any integer.
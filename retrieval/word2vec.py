from gensim.models import Word2Vec
from .base import RetrievalBase
import numpy as np


class Word2vecRetrieval(RetrievalBase):
    def __init__(self, input='content', fname=None, use_phrases=False,
                 **kwargs):
        self.fname = fname
        self.model = None
        self._init_params()

    def fit(self, X, y=None):
        self._fit(X, y)
        fname = self.fname
        model = Word2Vec.load(fname)

        # train on raw documents
        model.train(sentences, total_words=None, word_count=0, queue_factor=2,
                    report_delay=1)

        if fname:
            self.model.save(fname)

    def query(self, X, k=1):
        model = self.model
        for q in X:
            matched = self.matching(q)
            n_ret = min(len(matched), k)
            scores = np.apply_along_axis(lambda d: model.wmdistance(q, d),
                                         matched)
            ind = np.argpartition(scores, n_ret)
            labels = self._y[ind]
            yield labels

# from gensim.models import Doc2Vec
from .base import RetrievalBase, RetriEvalMixin
from .word2vec import Word2VecRetrieval

class Doc2VecRetrieval(RetrievalBase, RetriEvalMixin, Word2VecRetrieval):
    def __init__(self, model, name=None, **kwargs):
        self.model = model
        self.verbose = kwargs.get('verbose', 0)
        if name is None:
            name = "paragraph-vectors"

        self._init_params(name=name, **kwargs)
        self.analyzer = self._cv.build_analyzer()

    def fit(self, X, y):
        pass

    def query(self, query, k=1):
        docs, labels = self._matching(query)

        y = labels
        return y

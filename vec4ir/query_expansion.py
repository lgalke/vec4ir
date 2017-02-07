from sklearn.base import BaseEstimator, TransformerMixin
from .utils import filter_vocab


class EmbeddingBasedQueryLanguageModels(BaseEstimator, TransformerMixin):
    """Embedding-based Query Language Models by Zamani and Croft 2016 """

    def __init__(self, embedding, m=10, analyzer=None):
        """
        Initializes the embedding based query language model query expansion
        technique
        """
        BaseEstimator.__init__(self)
        self._embedding = embedding
        self._analyzer = analyzer

    def fit(self, X, y=None):
        """ Learns how to expand query with respect to corpus X """
        E = self._embedding
        if self.analyzer is not None:
            # X is expected to be raw documents
            X_ = (self.analyzer(row) for row in X)
            X_ = (filter_vocab(E, doc) for doc in X_)
        else:
            # X is expected to be vectorized already
            pass

    def transform(self, X, y=None):
        """ Transorms a query into an expanded version of the query.
        """
        pass

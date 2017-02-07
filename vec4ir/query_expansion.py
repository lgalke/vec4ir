from sklearn.base import BaseEstimator, TransformerMixin
from .utils import filter_vocab
from .core import EmbeddedVectorizer
from sklearn.metrics.pairwise import pairwise_distances
from scipy.special import expit
import numpy as np


def delta(X, Y=None, n_jobs=-1):
    """Pairwise delta function: cosine and sigmoid

    :X: TODO
    :returns: TODO

    """
    X_dists = pairwise_distances(X, Y, metric="cosine", n_jobs=n_jobs)
    X_dists = expit(X_dists)
    return X_dists


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
        self._ev = EmbeddedVectorizer(embedding, analyzer=analyzer)

    def fit(self, raw_docs, y=None):
        """ Learns how to expand query with respect to corpus X """
        E, ev = self._embedding, self._ev
        X_ = ev.fit_transform(raw_docs)
        common_words = ev.inverse_transform(np.unique(X_.nonzero()))
        X_ = np.vstack([E[word] for word in common_words])
        deltas = delta(X_, X_)
        priors = np.sum(deltas, axis=0)

        self.deltas = deltas
        self.priors = priors

    def transform(self, query, y=None):
        """ Transorms a query into an expanded version of the query.
        """
        X_q = ev.transform(query)  # index 0?
        <++ZYOMG++>



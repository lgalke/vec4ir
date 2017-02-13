from sklearn.base import BaseEstimator, TransformerMixin
from .utils import filter_vocab
from .core import EmbeddedVectorizer, embed
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.special import expit
import scipy.sparse as sp
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
        # <++ZYOMG++>


class CentroidExpansion(BaseEstimator):

    def __init__(self, embedding, analyzer, m=10, verbose=0,
                 **neighbor_params):
        self.embedding = embedding
        self.m = m
        self.neighbors = NearestNeighbors(n_neighbors=m, **neighbor_params)
        self.common = None
        self.vect = CountVectorizer(analyzer=analyzer,
                                    vocabulary=embedding.index2word)
        BaseEstimator.__init__(self)

    def fit(self, docs, y=None):
        """ Fit effective vocab even if embedding contains more words """
        index2word = self.embedding.index2word
        syn0 = self.embedding.syn0

        # find unique words
        X_tmp = self.vect.fit_transform(docs)
        __, cols, __ = sp.find(X_tmp)

        features = np.unique(cols)
        print('features:', features)

        # reduce vocabulary and vectors
        common_words = np.array([index2word[f] for f in features])
        common_vectors = syn0[features]
        print('CE: common vectors shape', common_vectors.shape)

        # fit nearest neighbors with vectors
        self.common = common_words
        self.neighbors.fit(common_vectors)

        return self

    def transform(self, query, y=None):
        """ Expands query by nearest tokens from collection """
        E, vect = self.embedding, self.vect
        qt = vect.transform([query])
        emb = embed(qt, E.syn0)

        ind = self.neighbors.kneighbors(emb, return_distance=False,
                                        n_neighbors=self.m)

        exp_words = self.common[ind.ravel()]

        expanded_query = query + ' ' + ' '.join(exp_words)
        print("Expanded query:", expanded_query)

        return expanded_query

    def fit_transform(self, X, y):
        raise NotImplemented('fit_transform does not make sense for expansion')

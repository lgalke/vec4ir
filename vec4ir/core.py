#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.exceptions import NotFittedError
import numpy as np
import scipy.sparse as sp
try:
    from .base import RetriEvalMixin
except SystemError:
    from base import RetriEvalMixin
from sklearn.decomposition import PCA


class Retrieval(BaseEstimator, MetaEstimatorMixin, RetriEvalMixin):

    """Meta estimator for an end to end information retrieval process"""

    def __init__(self, retrieval_model, matching=None,
                 query_expansion=None, name='RM'):
        """TODO: to be defined1.

        :retrieval_model: TODO
        :vectorizer: TODO
        :matching: TODO
        :query_expansion: TODO

        """
        BaseEstimator.__init__(self)

        self._retrieval_model = retrieval_model
        self._matching = matching
        self._query_expansion = query_expansion
        self.name = name
        self.labels_ = None

    def fit(self, X, y=None):
        """ Fit vectorizer to raw_docs, transform them and fit the
        retrieval_model.  Matching and Query expansion are fit separatly on the
        `raw_docs` to allow dedicated analysis.

        """
        assert y is None or len(X) == len(y)
        self.labels_ = np.asarray(y) if y is not None else np.arange(len(X))
        matching = self._matching
        query_expansion = self._query_expansion
        retrieval_model = self._retrieval_model

        if query_expansion:
            query_expansion.fit(X)

        if matching:
            matching.fit(X)

        retrieval_model.fit(X)

    def query(self, q, k=None):
        labels = self.labels_
        if labels is None:
            raise NotFittedError
        matching = self._matching
        retrieval_model = self._retrieval_model
        query_expansion = self._query_expansion

        if query_expansion:
            q = query_expansion.transform(q)

        if matching:
            ind = matching.predict(q)
            print('{} documents matched.'.format(len(ind)))
            if len(ind) == 0:
                return []
            labels = labels[ind]  # Reduce our own view
        else:
            ind = None

        # pass matched indices to query method of retrieval model
        # The retrieval model is assumed to reduce its representation of X
        # to the given indices and the returned indices are relative to the
        # reduction
        retrieved_indices = retrieval_model.query(q, k=k, indices=ind)
        if k is not None:
            # Just assert that it did not cheat
            retrieved_indices = retrieved_indices[:k]

        return labels[retrieved_indices]  # Unfold retrieved indices


class EmbeddedVectorizer(TfidfVectorizer):

    """Embedding-aware vectorizer"""

    def __init__(self, embedding, **kwargs):
        """TODO: to be defined1. """
        # list of words in the embedding
        vocabulary = embedding.index2word
        self.embedding = embedding
        print("Embedding shape:", embedding.syn0.shape)
        TfidfVectorizer.__init__(self, vocabulary=vocabulary, **kwargs)

    def fit(self, raw_docs, y=None):
        super().fit(raw_docs)
        return self

    def transform(self, raw_documents, y=None):
        Xt = super().transform(raw_documents)
        syn0 = self.embedding.syn0
        # Xt is sparse counts
        return (Xt @ syn0)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)


def embed(X, E):
    """
    This is effectively: X @ E, just slower... by foot
    Arguments:
        - X  -- (n_samples, n_features)
        - E  -- (n_features, n_dims)
        - X  -- @ E (n_samples, n_dims)
    """
    raise DeprecationWarning("This is slow, use X @ syn0 instead.")
    embedded = np.zeros((X.shape[0], E.shape[1]), dtype=E.dtype)
    for (row, col, val) in zip(*sp.find(X)):
        update = val * E[col, :]
        embedded[row, :] += update
    return embedded


def all_but_the_top(v, D):
    """
    All-but-the-Top: Simple and Effective Postprocessing for Word
    Representations
    https://arxiv.org/abs/1702.01417

    Arguments:
        :v: word vectors of shape (n_words, n_dimensions)
        :D: number of principal components to subtract

    """
    print("All but the top")
    # 1. Compute the mean for v
    mu = np.mean(v, axis=0)
    v_tilde = v - mu  # broadcast hopefully works

    # 2. Compute the PCA components

    pca = PCA(n_components=D)
    u = pca.fit_transform(v.T)

    # 3. Postprocess the representations
    for w in range(v_tilde.shape[0]):
        v_tilde[w, :] -= np.sum([(u[:, i] * v[w]) * u[:, i].T for i in
                                 range(D)],
                                axis=0)

    return v_tilde

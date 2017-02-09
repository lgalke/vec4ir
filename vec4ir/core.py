#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import scipy.sparse as sp


class Retrieval(BaseEstimator):

    """Meta estimator for an end to end information retrieval process"""

    def __init__(self, retrieval_model, vectorizer=None, matching=None,
                 query_expansion=None):
        """TODO: to be defined1.

        :retrieval_model: TODO
        :vectorizer: TODO
        :matching: TODO
        :query_expansion: TODO

        """
        BaseEstimator.__init__(self)

        self._retrieval_model = retrieval_model
        self._vectorizer = vectorizer
        self._matching = matching
        self._query_expansion = query_expansion

    def fit(self, raw_docs, y):
        """ Fit vectorizer to raw_docs, transform them and fit the
        retrieval_model.  Matching and Query expansion are fit separatly on the
        `raw_docs` to allow dedicated analysis.

        """
        matching = self._matching
        vect = self._vectorizer
        retrieval_model = self._retrieval_model
        query_expansion = self._query_expansion

        if vect:
            X = vect.fit_transform(raw_docs)
        else:
            X = raw_docs

        if query_expansion:
            query_expansion.fit(X, y)

        if matching:
            matching.fit(X, y)

        retrieval_model.fit(X, y)

    def query(self, raw_query, k=None):
        matching = self._matching
        vect = self._vectorizer
        retrieval_model = self._retrieval_model
        query_expansion = self._query_expansion

        if vect:
            X_q = vect.transform(raw_query)
        else:
            X_q = raw_query

        if query_expansion:
            X_q = query_expansion.transform(X_q)

        if matching:
            ind = matching.predict(X_q)
        else:
            ind = None

        y_pred = retrieval_model.query(X_q, k=k, matched_indices=ind)

        return y_pred


class EmbeddedVectorizer(CountVectorizer):

    """Embedding-aware vectorizer"""

    def __init__(self, embedding, **kwargs):
        """TODO: to be defined1. """
        # list of words in the embedding
        vocabulary = embedding.index2word
        self.embedding = embedding
        CountVectorizer.__init__(self, vocabulary=vocabulary, **kwargs)

    def transform(raw_documents):
        super().transform(raw_documents)


class CentroidEmbedder(BaseEstimator, TransformerMixin):

    """ Embeds a BOW-Representation of Documents as the respective centroid of
    the word vectors
    """
    def __init__(self, syn0):
        """TODO: to be defined1. """
        BaseEstimator.__init__(self)
        self.syn0 = syn0
        print('syn0 of shape', syn0.shape)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        X is a BOW-like representation of documents, with
        X[:,i] = wv.index2word[i].  This can be achieved by passing
        vocabulary=wv.index2word to your {count,tfidf}vectorizer
        """
        syn0 = self.syn0
        n_samples, n_dimensions, dtype = X.shape[0], syn0.shape[1], syn0.dtype
        centroids = np.zeros((n_samples, n_dimensions), dtype=dtype)
        # N = np.zeros(n_samples, dtype=np.int64)
        rows, cols, values = sp.find(X)
        for (row, col, val) in zip(rows, cols, values):
            # cumulative moving average
            # N[row] += 1
            # centroids[row] += ((val * syn0[col] - centroids[row]) / N[row])
            centroids[row] += val * syn0[col]
        # BOOM
        print('centroids shape', centroids.shape)
        return centroids

#!/usr/bin/env python3
# coding: utf-8

try:
    from .base import RetrievalBase, RetriEvalMixIn, TfidfRetrieval
    from .utils import argtopk
except (SystemError, ValueError):
    from base import RetrievalBase, RetriEvalMixIn, TfidfRetrieval
    from utils import argtopk
from scipy.spatial.distance import cosine
from scipy.special import expit
import numpy as np


def delta(u, v):
    """ cosine ° sigmoid
    >>> delta([0.2], [0.3])
    0.5
    >>> delta([0.3], [0.2])
    0.5
    >>> delta([0.1,0.9], [-0.9,0.1]) == delta([-0.9,0.1], [0.1,0.9])
    True
    """
    # TODO scale with a and c
    return expit(cosine(u, v))


def eqe1(E, query, vocabulary, priors):
    """
    Arguments:
        E - word embedding
        Q - list of query terms
        vocabulary -- list of relevant words
        priors - precomputed priors with same indices as vocabulary
    >>> E = dict()
    >>> E['a'] = np.asarray([0.5,0.5])
    >>> E['b'] = np.asarray([0.2,0.8])
    >>> E['c'] = np.asarray([0.9,0.1])
    >>> E['d'] = np.asarray([0.8,0.2])
    >>> q = "a b".split()
    >>> vocabulary = "a b c".split()
    >>> priors = np.asarray([0.25,0.5,0.25])
    >>> posterior = eqe1(E, q, vocabulary, priors)
    >>> vocabulary[np.argmax(posterior)]
    'c'
    """
    posterior = [priors[i] *
                 np.product([delta(E[qi], E[w]) / priors[i] for qi in query])
                 for i, w in enumerate(vocabulary)]

    return np.asarray(posterior)


def expand(posterior, vocabulary, m=10):
    """
    >>> vocabulary = "a b c".split()
    >>> posterior = [0.9, 0.1, 0.42]
    >>> expand(posterior, vocabulary, 0)
    []
    >>> expand(posterior, vocabulary, 1)
    ['a']
    >>> expand(posterior, vocabulary, 2)
    ['a', 'c']
    """
    if m <= 0:
        return []
    vocabulary = np.asarray(vocabulary)
    expansions = vocabulary[argtopk(posterior, m)]
    return list(expansions)


class EQLM(RetrievalBase, RetriEvalMixIn):

    def __init__(self,
                 retrieval_model,
                 embedding,
                 analyzer=None,
                 name="EQLM",
                 m=10,
                 eqe=1,
                 erm=False,
                 verbose=0,
                 **kwargs):
        if eqe not in [1, 2]:
            raise ValueError("eqe parameter must be either 1 or 2")
        self.eqe = eqe
        self.erm = erm
        self.embedding = embedding
        self.m = m
        self.name = name
        self.verbose = verbose
        self.retrieval_model = TfidfRetrieval()
        self._init_params(**kwargs)
        if analyzer:
            self.analyzer = analyzer
        else:
            self.analyzer = retrieval_model.analyzer

    def fit(self, docs, labels):
        E, RM = self.embedding, self.retrieval_model
        # analyze = self.analyzer
        # self._fit(docs, labels)
        RM.fit(docs, labels)
        V = self.vocabulary = RM._cv.vocabulary_

        priors = np.asarray([sum(delta(E[w], E[v]) for v in V) for w in V])
        if self.verbose > 0:
            print("priors.shape:", priors.shape)

        self.priors = priors

    def query(self, query):
        E, m = self.embedding, self.m
        V = self.vocabulary
        priors = self.priors

        q = self.analyzer(query)
        posterior = eqe1(E, q, priors)
        expansion = V[argtopk(posterior, m)]

        expanded_query = " ".join(q + expansion)
        if self.verbose > 0:
            print("[eqlm] Expanded query: '{}'".format(expanded_query))

        # employ retrieval model
        return self.retrieval_model.query(expanded_query)

if __name__ == "__main__":
    import doctest
    doctest.testmod()

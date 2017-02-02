#!/usr/bin/env python3
# coding: utf-8

try:
    from .base import RetrievalBase, RetriEvalMixin
    from .utils import argtopk
except (SystemError, ValueError):
    from base import RetrievalBase, RetriEvalMixin
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


RetriEvalMixin

if __name__ == "__main__":
    import doctest
    doctest.testmod()

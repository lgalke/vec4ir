#!/usr/bin/env python3
# coding: utf-8
import numpy as np
from collections import Counter


def flatten(l):
    """
    flattens a list of list structure... nothing else.
    """
    return [item for sublist in l for item in sublist]


def filter_vocab(model, words, oov=None):
    filtered = []
    for word in words:
        if word in model:
            filtered.append(word)
        elif oov:
            filtered.append(oov)
    return filtered


def argtopk(A, k=None, sort=True):
    """ Get the the top k elements (in sorted order)
    >>> A = np.asarray([5,4,3,6,7,8,9,0])
    >>> A[argtopk(A, 3)]
    array([9, 8, 7])
    >>> argtopk(A, 1)
    array([6])
    >>> argtopk(A, 6)
    array([6, 5, 4, 3, 0, 1])
    >>> argtopk(A, 10)
    array([6, 5, 4, 3, 0, 1, 2, 7])
    >>> argtopk(A, 28)
    array([6, 5, 4, 3, 0, 1, 2, 7])
    >>> argtopk(A, None)
    array([6, 5, 4, 3, 0, 1, 2, 7])
    >>> X = np.arange(20)
    >>> argtopk(X, 10)
    array([19, 18, 17, 16, 15, 14, 13, 12, 11, 10])
    """
    A = np.asarray(A)
    if len(A.shape) > 1:
        raise ValueError('argtopk only defined for 1-d slices')
    axis = -1
    if k is None or k >= A.size:
        # if list is too short or k is None, return all in sort order
        if sort:
            return np.argsort(A, axis=axis)[::-1]
        else:
            return np.arange(A.shape[0])

    assert k > 0
    # now 0 < k < len(A)
    ind = np.argpartition(A, -k, axis=axis)[-k:]
    if sort:
        # sort according to values in A
        # argsort is always from lowest to highest, so reverse
        ind = ind[np.argsort(A[ind], axis=axis)][::-1]

    return ind


def collection_statistics(embedding, documents, analyzer=None, topn=None):
    # print(embedding, analyzer, documents, sep='\n')
    c = Counter(n_tokens=0, n_embedded=0, n_oov=0)
    f = Counter()
    for document in documents:
        words = analyzer(document) if analyzer is not None else document
        for word in words:
            c['n_tokens'] += 1
            if word in embedding:
                c['n_embedded'] += 1
            else:
                f[word] += 1
                c['n_oov'] += 1

    d = dict(c)
    d['oov_ratio'] = c['n_oov'] / c['n_tokens']
    if topn:
        return d, f.most_common(topn)
    return d


if __name__ == "__main__":
    import doctest
    doctest.testmod()

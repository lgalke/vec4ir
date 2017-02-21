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


def argtopk(A, k, axis=-1, sort=True):
    """ Get the the top k elements (in sorted order)
    >>> A = np.asarray([5,4,3,6,7,8,9,0])
    >>> argtopk(A, 3)
    array([6, 5, 4])
    >>> argtopk(A, 1)
    array([6])
    >>> argtopk(A, 6)
    array([6, 5, 4, 3, 0, 1])
    >>> argtopk(A, 10)
    array([6, 5, 4, 3, 0, 1, 2, 7])
    >>> argtopk(A, 28)
    array([6, 5, 4, 3, 0, 1, 2, 7])
    >>> B = np.asarray([[1,2,3],[4,3,2],[-9,-2,-7]])
    >>> B.shape
    (3, 3)
    >>> r = argtopk(B, 2)
    >>> r[0]
    array([2,1])
    >>> r[1]
    array([0,1])
    >>> r[2]
    array([1,2])
    """
    A = np.asarray(A)
    if k is None or k >= len(A):
        # catch this for more convenience, if list is too short
        return np.argsort(A, axis=axis)[::-1]

    assert k > 0
    # now 0 < k < len(A)
    ind = np.argpartition(A, -k, axis=axis)
    import sys
    # select k highest, so fancy for multi dimensional arrays
    ind = ind[..., -k:]  # <++TODO++> only works for single dim
    print(ind, file=sys.stderr)

    if sort:
        # sort according to values in A
        ind = ind[np.argsort(A[ind], axis=axis)]
        # argsort is always from lowest to highest, so reverse
        ind = ind[::-1]

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

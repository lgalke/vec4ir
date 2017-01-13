#!/usr/bin/env python3
# coding: utf-8
import numpy as np


def filter_vocab(model, words, oov=None):
    """ if analyze is given, analyze words first (split string) """
    filtered = []
    for word in words:
        if word in model:
            filtered.append(word)
        elif oov is not None:
            filtered.append(oov)
    return filtered

def argtopk(A, k, axis=-1, sort=True):
    """
    >>> A = [5,4,3,6,7,8,9,0]
    >>> argtopk(A, 3)
    array([6, 5, 4])
    >>> argtopk(A, 1)
    array([6])
    >>> argtopk(A, -3)
    array([7, 2, 1])
    >>> argtopk(A, -1)
    array([7])
    >>> argtopk(A, -6)
    array([7, 2, 1, 0, 3, 4])
    >>> argtopk(A, 6)
    array([6, 5, 4, 3, 0, 1])
    >>> argtopk(A, 10)
    array([6, 5, 4, 3, 0, 1, 2, 7])
    """
    k = min(len(A), k)
    A = np.asarray(A)
    if k == 0:
        raise UserWarning("k <= 0? result [] may be undesired.")
        return []
    ind = np.argpartition(A, -k, axis=axis)
    ind = ind[-k:] if k > 0 else ind[:-k]

    if sort:
        ind = ind[np.argsort(A[ind])]
        if k > 0:
            ind = ind[::-1]

    return ind

if __name__ == "__main__":
    import doctest
    doctest.testmod()

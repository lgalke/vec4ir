#!/usr/bin/env python
# -*- coding: utf-8 -*-
from timeit import default_timer as timer
from timeit import timedelta
from tfidf_retriever import TfidfRetriever
from word2vec_retriever import Word2vecRetriever


def ir_eval(irmodel, X, Xtest, Ytest, metrics=None):
    """
    irmodel
    X : iterator of documents
    Xtest  : iterator of queries
    Ytest  : iterator of doc,relevance pairs
    """
    t0 = timer()
    print("Fitting", irmodel.name, "...")
    irmodel.fit(X)
    print(timedelta(seconds=timer()-t0))
    print("Transforming", irmodel.name, "...")
    Y = irmodel.predict(Xtest)
    result = dict()
    for metric in metrics:
        result[metric.__name__] = metric(Ytest, Y)

    return result


def main():
    """TODO: Docstring for main.
    :returns: TODO

    """
    tfidf = TfidfRetriever()
    word2vec = Word2vecRetriever()
    data = None
    ir_eval(tfidf, data)
    ir_eval(word2vec, data)


if __name__ == "__main__":
    main()


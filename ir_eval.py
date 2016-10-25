#!/usr/bin/env python
# -*- coding: utf-8 -*-
import timeit

from tfidf_retriever import TfidfRetriever
from word2vec_retriever import Word2vecRetriever


def ir_eval(irmodel, data):



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


#!/usr/bin/env python
# -*- coding: utf-8 -*-
from timeit import default_timer as timer
from datetime import timedelta
from datasets import NTCIR
import sys
import pandas as pd

from retrieval.base import TfidfRetrieval


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
    from argparse import ArgumentParser, FileType
    parser = ArgumentParser()
    parser.add_argument("-f",
                        "--field",
                        choices=['title', 'content'],
                        default='title')
    parser.add_argument("-r", "--rels", type=int, default=1, choices=[1, 2])
    parser.add_argument("-o", "--outfile", default=None,
                        type=FileType('a'))
    args = parser.parse_args()
    tfidf = TfidfRetrieval()
    # word2vec = Word2vecRetriever()
    ntcir2 = NTCIR("../data/NTCIR2/")
    print("Loading NTCIR2 documents...")
    docs_df = ntcir2.docs(kaken=True, gakkai=True)
    print("Loaded {:d} documents.".format(len(docs_df)))
    documents = docs_df[args.field].values
    print(documents)
    print("Fit...")
    tfidf.fit(documents, docs_df.index)
    print("Loading queries and relevancies")
    topics = ntcir2.topics()['title']
    rels = ntcir2.rels(args.rels)['relevance']
    queries = zip(topics.index, topics)
    print("Evaluating...")
    scores = tfidf.evaluate(queries, rels, verbose=1)
    import pprint
    pprint.pprint(scores)
    if args.outfile:
        print(args, file=args.outfile)
        print(scores, file=args.outfile)

    # ir_eval(tfidf, data)
    # ir_eval(word2vec, data)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from timeit import default_timer as timer
from datetime import timedelta
from vec4ir.datasets import NTCIR
from vec4ir.base import TfidfRetrieval
from vec4ir.word2vec import StringSentence, Word2VecRetrieval
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
import sys
import pprint
# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
#                     level=logging.INFO)


def ir_eval(irmodel, documents, labels, queries, rels, metrics=None,
            verbose=3):
    """
    irmodel
    X : iterator of documents
    Xtest  : iterator of queries
    Ytest  : iterator of doc,relevance pairs
    """
    if verbose > 0:
        print("=" * 79)
        t0 = timer()
        print("Fit", irmodel.name, "... ", end='')
        irmodel.fit(documents, labels)
        t1 = timer()
        print("took {} seconds.".format(timedelta(seconds=t1-t0)))
        print("Evaluating", irmodel.name, "...")
    if verbose > 1:
        print("-" * 79)
    values = irmodel.evaluate(queries, rels, verbose=verbose-1)
    if verbose > 1:
        print("-" * 79)
    if verbose > 0:
        print("Average time per query:",
              timedelta(seconds=(timer()-t1)/len(queries)))
        pprint.pprint(values)
        print("=" * 79)

    return values


def main():
    """TODO: Docstring for main.
    :returns: TODO
    """
    from argparse import ArgumentParser, FileType
    parser = ArgumentParser()
    parser.add_argument("-f",
                        "--field",
                        choices=['title', 'content', 'both'],
                        default='title',
                        help="field to use (defaults to 'title')")
    parser.add_argument("-r", "--rels", type=int, default=1, choices=[1, 2],
                        help="relevancies to use (defaults to 1)")
    parser.add_argument("-o", "--outfile", default=sys.stdout,
                        type=FileType('a'))
    parser.add_argument("-v", "--verbose", default=2, type=int,
                        help="verbosity level")
    args = parser.parse_args()
    tfidf = TfidfRetrieval()
    ntcir2 = NTCIR("../data/NTCIR2/", ".cache")
    print("Loading NTCIR2 documents...")
    docs_df = ntcir2.docs(kaken=True, gakkai=True)
    print("Loaded {:d} documents.".format(len(docs_df)))
    if args.field == 'both':
        docs_df['both'] = \
            docs_df[['title', 'content']].apply(lambda x: ' '.join(x), axis=1)
    documents = docs_df[args.field].values
    labels = docs_df.index.values
    # print("Fit...")
    # tfidf.fit(documents, docs_df.index.values)
    print("Loading queries and relevancies")
    topics = ntcir2.topics()['title']
    n_queries = len(topics)
    print("Using {:d} queries".format(n_queries))
    rels = ntcir2.rels(args.rels)['relevance']
    n_rels = len(rels.nonzero()[0])
    print("With {:.1f} relevant documents per query".format(n_rels/n_queries))
    queries = list(zip(topics.index, topics))
    # scores = tfidf.evaluate(queries, rels, verbose=1)

    def evaluation(m):
        return ir_eval(m, documents, labels, queries, rels,
                       verbose=args.verbose)

    results = {}
    results[tfidf.name] = evaluation(tfidf)
    del tfidf
    stop = CountVectorizer(stop_words='english').build_analyzer()
    print("Training word2vec model...")
    model = Word2Vec(StringSentence(documents, stop), min_count=1)
    model.init_sims(replace=True)  # model becomes read-only but saves memory
    print("Done.")
    n_similarity = Word2VecRetrieval(model, analyzer=stop,
                                     method='n_similarity')
    results[n_similarity.name] = evaluation(n_similarity)
    del n_similarity
    wmdistance = Word2VecRetrieval(model, analyzer=stop, method='wmdistance')
    results[wmdistance.name] = evaluation(wmdistance)
    del wmdistance

    pprint.pprint(results, stream=args.outfile)

if __name__ == "__main__":
    main()

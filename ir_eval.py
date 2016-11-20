#!/usr/bin/env python
# -*- coding: utf-8 -*-
from timeit import default_timer as timer
from datetime import timedelta
from vec4ir.datasets import NTCIR
from vec4ir.base import TfidfRetrieval
from vec4ir.word2vec import StringSentence, Word2VecRetrieval
from gensim.models import Word2Vec
from gensim.models import Phrases
from sklearn.feature_extraction.text import CountVectorizer
import sys
import pprint
# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
#                     level=logging.INFO)


def ir_eval(irmodel, documents, labels, queries, rels, metrics=None, k=20,
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
    values = irmodel.evaluate(queries, rels, verbose=verbose-1, k=k)
    if verbose > 1:
        print("-" * 79)
    if verbose > 0:
        print("Average time per query:",
              timedelta(seconds=(timer()-t1)/len(queries)))
        pprint.pprint(values)
        print("=" * 79)

    ndcgs = values['ndcg_at_k']
    values['ndcg_at_k'] = (ndcgs.mean(), ndcgs.std())

    values['params'] = irmodel.get_params(deep=True)

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
    parser.add_argument("-t", "--topics", type=str, default='title',
                        choices=['title'],
                        help="topics' field to use (defaults to 'title')")
    parser.add_argument("-r", "--rels", type=int, default=1, choices=[1, 2],
                        help="relevancies to use (defaults to 1)")
    parser.add_argument("-o", "--outfile", default=sys.stdout,
                        type=FileType('a'))
    parser.add_argument("-k", dest='k', default=20, type=int,
                        help="number of documentss to retrieve")
    parser.add_argument("-m", "--model", default=None, type=str,
                        help="use precomputed word2vec model")
    parser.add_argument("-v", "--verbose", default=2, type=int,
                        help="verbosity level")
    parser.add_argument("-c", "--lowercase", default=False,
                        action='store_true',
                        help="Case insensitive analysis")
    args = parser.parse_args()
    ntcir2 = NTCIR("../data/NTCIR2/", ".cache")
    print("Loading NTCIR2 documents...")
    docs_df = ntcir2.docs(kaken=True, gakkai=True)
    print("Loaded {:d} documents.".format(len(docs_df)))
    docs_df['both'] = docs_df[['title', 'content']]\
        .apply(lambda x: ' '.join(x), axis=1)
    documents = docs_df[args.field].values
    labels = docs_df.index.values
    print("Loading queries and relevancies")
    topics = ntcir2.topics()[args.topics]  # could be variable
    n_queries = len(topics)
    print("Using {:d} queries".format(n_queries))
    rels = ntcir2.rels(args.rels)['relevance']
    n_rels = len(rels.nonzero()[0])
    print("With {:.1f} relevant documents per query".format(n_rels / n_queries))
    queries = list(zip(topics.index, topics))
    analyzer = CountVectorizer(stop_words='english',
                               lowercase=args.lowercase).build_analyzer()

    def evaluation(m):
        return ir_eval(m, documents, labels, queries, rels,
                       verbose=args.verbose, k=args.k)

    results = {}
    results['args'] = args

    tfidf = TfidfRetrieval(analyzer=analyzer)
    results[tfidf.name] = evaluation(tfidf)
    del tfidf

    if args.model:
        print("Loading word2vec model: {}".format(args.model))
        model = Word2Vec.load_word2vec_format(args.model, binary=True)

    else:
        print("Training word2vec model on all available data...")
        model = Word2Vec(StringSentence(docs_df['both'].values, analyzer),
                         min_count=1, iter=10)
        model.init_sims(replace=True)  # model becomes read-only

    print("Done.")

    n_similarity = Word2VecRetrieval(model, analyzer=analyzer, wmd=False,
                                     verbose=args.verbose)
    results[n_similarity.name] = evaluation(n_similarity)
    del n_similarity
    wmdistance = Word2VecRetrieval(model,
                                   analyzer=analyzer,
                                   wmd=True,
                                   verbose=args.verbose)
    results[wmdistance.name] = evaluation(wmdistance)
    del wmdistance
 
    # for wmd in [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]:
    #     name="w2v+wcd+"+str(wmd)+"wmd"
    #     wmdistance = Word2VecRetrieval(model, analyzer=analyzer, wmd=wmd,
    #                                    name=name, verbose=args.verbose)
    #     results[wmdistance.name] = evaluation(wmdistance)
    #     del wmdistance

    # for i in [1, 2, 3, 5, 10]:
    #     name="w2v+wcd+wmd+"+str(i)+"exps"
    #     wmdistance = Word2VecRetrieval(model, analyzer=analyzer, wmd=False,
    #                                    name=name, verbose=args.verbose,
    #                                    n_expansions=i)
    #     results[wmdistance.name] = evaluation(wmdistance)
    #     del wmdistance

    pprint.pprint(results, stream=args.outfile)

if __name__ == "__main__":
    main()

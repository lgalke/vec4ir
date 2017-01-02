#!/usr/bin/env python
# -*- coding: utf-8 -*-
from timeit import default_timer as timer
from datetime import timedelta
from vec4ir.datasets import NTCIR
from vec4ir.base import TfidfRetrieval
from vec4ir.word2vec import StringSentence, Word2VecRetrieval
from vec4ir.doc2vec import Doc2VecRetrieval
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
import sys
import os
import pprint
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def ir_eval(irmodel, documents, labels, queries, rels, metrics=None, k=20,
            verbose=3, replacement="zero"):
    """
    irmodel
    X : iterator of documents
    Xtest  : iterator of queries
    Ytest  : iterator of doc,relevance pairs
    """
    if verbose > 0:
        print("=" * 79)
        t0 = timer()
        print("Fit", irmodel.name, "... ")
        irmodel.fit(documents, labels)
        t1 = timer()
        print("took {} seconds.".format(timedelta(seconds=t1 - t0)))
        print("Evaluating", irmodel.name, "...")
    if verbose > 1:
        print("-" * 79)
    values = irmodel.evaluate(queries, rels, verbose=verbose - 1, k=k, replacement=replacement)
    if verbose > 1:
        print("-" * 79)
    if verbose > 0:
        pprint.pprint(values)
        print("=" * 79)

    ndcgs = values['ndcg_at_k']
    values['ndcg_at_k'] = (ndcgs.mean(), ndcgs.std())

    values['params'] = irmodel.get_params(deep=True)

    return values


def smart_load_word2vec(model_path):
    print("Smart loading", model_path)
    _, ext = os.path.splitext(model_path)
    if model_path is None:
        return None
    if ext == ".gnsm":  # Native format
        print("Loading word2vec model in native gensim format: {}"
              .format(model_path))
        model = Word2Vec.load(model_path)
    else:  # either word2vec text or word2vec binary format
        binary = ".bin" in model_path
        print("Loading {}word2vec model: {}" .format("binary " if binary else "", model_path))
        model = Word2Vec.load_word2vec_format(model_path, binary=binary)

    # FIXME catch the occasional exception?

    return model


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
                        choices=['title', 'description'],
                        help="topics' field to use (defaults to 'title')")
    parser.add_argument("-r", "--rels", type=int, default=1, choices=[1, 2],
                        help="relevancies to use (defaults to 1)")
    parser.add_argument("-R", "--replacement-strategy", type=str,
                        dest='repstrat', default="zero",
                        choices=['drop', 'zero'],
                        help="Out of relevancy file document ids,\
                        default is to use zero relevancy")
    parser.add_argument("-o", "--outfile", default=sys.stdout,
                        type=FileType('a'))
    parser.add_argument("-k", dest='k', default=20, type=int,
                        help="number of documents to retrieve")
    parser.add_argument("-m", "--model", default=None, type=str,
                        help="use precomputed embedding model")
    parser.add_argument("-v", "--verbose", default=2, type=int,
                        help="verbosity level")
    parser.add_argument("-c", "--lowercase", default=False,
                        action='store_true',
                        help="Case insensitive matching analysis \
                        (also relevant for baseline tfidf)")
    parser.add_argument("-l", "--try-lowercase", default=False,
                        action='store_true',
                        help="For embedding-based models, try lowercasing \
                        when there is no initial vocabulary match!")
    parser.add_argument("-M", "--oov", default=None, type=str,
                        help="token for out-of-vocabulary words, \
                        default is ignoreing out-of-vocabulary words")
    args = parser.parse_args()

    ntcir2 = NTCIR("../data/NTCIR2/", ".cache")
    print("Loading NTCIR2 documents...")
    docs_df = ntcir2.docs(kaken=True, gakkai=True)
    print("Loaded {:d} documents.".format(len(docs_df)))
    documents = docs_df[args.field].values
    labels = docs_df.index.values

    print("Loading topics...")
    topics = ntcir2.topics(names=args.topics)[args.topics]  # could be variable
    n_queries = len(topics)
    print("Using {:d} queries".format(n_queries))

    print("Loading relevances...")
    rels = ntcir2.rels(args.rels)['relevance']
    n_rels = len(rels.nonzero()[0])
    print("With {:.1f} relevant docs per query".format(n_rels / n_queries))
    queries = list(zip(topics.index, topics))
    analyzer = CountVectorizer(stop_words='english',
                               lowercase=args.lowercase).build_analyzer()
    cased_analyzer = CountVectorizer(stop_words='english',
                                     lowercase=False).build_analyzer()
    repl = { "drop": None, "zero": 0 }[args.repstrat]


    def evaluation(m):
        return ir_eval(m,
                       documents,
                       labels,
                       queries,
                       rels,
                       verbose=args.verbose,
                       k=args.k,
                       replacement=repl)

    results = dict()
    results['args'] = args

    tfidf = TfidfRetrieval(lowercase=args.lowercase, stop_words='english')
    results[tfidf.name] = evaluation(tfidf)
    del tfidf

    # mpath = smart_load_word2vec(args.model)
    # if not mpath:
    #     print("Training word2vec model on all available data...")
    #     model = Word2Vec(StringSentence(documents,
    #                                     cased_analyzer),
    #                      min_count=1, iter=10)
    #     model.init_sims(replace=True)  # model becomes read-only

    # print("Done.")

    # n_similarity = Word2VecRetrieval(model, wmd=False,
    #                                  analyzer=analyzer,
    #                                  vocab_analyzer=cased_analyzer,
    #                                  try_lowercase=args.try_lowercase,
    #                                  oov=args.oov,
    #                                  stop_words='english',
    #                                  verbose=args.verbose)
    # results[n_similarity.name] = evaluation(n_similarity)
    # del n_similarity

    # wmdistance = Word2VecRetrieval(model,
    #                                analyzer=analyzer,
    #                                vocab_analyzer=cased_analyzer,
    #                                try_lowercase=args.try_lowercase,
    #                                wmd=True,
    #                                oov=args.oov,
    #                                stop_words='english',
    #                                verbose=args.verbose)
    # results[wmdistance.name] = evaluation(wmdistance)
    # del wmdistance
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

    pvdm = Doc2VecRetrieval(name="pvdm", analyzer=analyzer, verbose=2,
                            vocab_analyzer=analyzer)
    results[pvdm.name] = evaluation(pvdm)
    del pvdm

    pprint.pprint(results, stream=args.outfile)

if __name__ == "__main__":
    main()

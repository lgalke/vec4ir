#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from timeit import default_timer as timer
from datetime import timedelta
from vec4ir.datasets import NTCIR
from vec4ir.base import TfidfRetrieval
from vec4ir.word2vec import StringSentence, Word2VecRetrieval
from vec4ir.doc2vec import Doc2VecRetrieval
from vec4ir.eqlm import EQLM
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
import sys
import os
import pprint
import numpy as np
import matplotlib.pyplot as plt
# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
#                     level=logging.INFO)
import pandas as pd
MODEL_KEYS = ['tfidf', 'wcd', 'wmd', 'pvdm', 'eqlm']


def mean_std(array_like):
    array_like = np.asarray(array_like)
    return array_like.mean(), array_like.std()


def plot_precision_recall_curves(path, results, plot_f1=False):
    colors = "b g r c m y k".split()
    keys = sorted(results.keys())
    for name, c in zip(keys, colors):
        plt.plot(results[name]["precision"], color=c, marker="1", linestyle='dashed')
        plt.plot(results[name]["recall"], color=c, marker="2", linestyle='dotted')
        plt.plot(results[name]["f1_score"], color=c, marker="*", linestyle='dashdot')

    plt.legend(keys)
    plt.savefig(plotfile)



def is_embedded(sentence, embedding, analyzer):
    """
    >>> embedding = ["a", "b", "c"]
    >>> queries =  ["a b c", "a", "b", "c", "a b c d", "d", "a b c"  ]
    >>> analyzer = lambda x: x.split()
    >>> [query for query in queries if is_embedded(query, embedding, analyzer)]
    ['a b c', 'a', 'b', 'c', 'a b c']
    >>> analyzer = CountVectorizer().build_analyzer()
    >>> [query for query in queries if is_embedded(query, embedding, analyzer)]
    ['a b c', 'a', 'b', 'c', 'a b c']
    """
    for word in analyzer(sentence):
        if word not in embedding:
            print("Dropping:", sentence, file=sys.stderr)
            return False

    return True


def ir_eval(irmodel, documents, labels, queries, rels, metrics=None, k=20,
            verbose=3, replacement=0):
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
    values = irmodel.evaluate(queries, rels, verbose=verbose - 1, k=k,
                              replacement=replacement)
    if verbose > 1:
        print("-" * 79)
    if verbose > 0:
        pprint.pprint(values)
        print("=" * 79)

    return values


def smart_load_word2vec(model_path):
    print("Smart loading", model_path)
    if model_path is None:
        return None
    _, ext = os.path.splitext(model_path)
    if ext == ".gnsm":  # Native format
        print("Loading embeddings in native gensim format: {}"
              .format(model_path))
        model = Word2Vec.load(model_path)
    else:  # either word2vec text or word2vec binary format
        binary = ".bin" in model_path
        print("Loading embeddings in word2vec format: {}".format(model_path))
        model = Word2Vec.load_word2vec_format(model_path, binary=binary)

    # FIXME catch the occasional exception?

    return model


def _ir_eval_parser():
    from argparse import ArgumentParser, FileType
    parser = ArgumentParser()
    parser.add_argument("--doctest", action='store_true',
                        help="Perform doctest on this module")
    parser.add_argument("-f",
                        "--field",
                        choices=['title', 'content'],
                        default='title',
                        help="field to use (defaults to 'title')")
    parser.add_argument("-F", "--focus", nargs='+',
                        choices=MODEL_KEYS, default=None)
    parser.add_argument("-Q", "--filter-queries", action='store_true',
                        help="Filter queries without complete embedding")
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
    parser.add_argument("-p", "--plot", default=None, type=FileType('w'),
                        metavar="PLOTFILE",
                        help="Save precision-recall curves in PLOTFILE")

    # FIXME this will be model specific soon
    parser.add_argument("-c", "--lowercase", default=False,
                        action='store_true',
                        help="Case insensitive matching analysis \
                        (also relevant for baseline tfidf)")
    parser.add_argument("-l", "--try-lowercase", default=False,
                        action='store_true',
                        help="For embedding-based models, try lowercasing \
                        when there is no initial vocabulary match!")
    # FIXME this will be model specific soon END
    parser.add_argument("-M", "--oov", default=None, type=str,
                        help="token for out-of-vocabulary words, \
                        default is ignoreing out-of-vocabulary words")
    parser.add_argument("-T", "--train", default=False, action='store_true',
                        help="Train a whole new word2vec model")
    return parser


def main():
    """TODO: Docstring for main.
    :returns: TODO
    """
    parser = _ir_eval_parser()
    args = parser.parse_args()
    if args.doctest:
        import doctest
        doctest.testmod()
        exit(int(0))

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
    focus = set([f.lower() for f in args.focus]) if args.focus else None
    repl = {"drop": None, "zero": 0}[args.repstrat]

    model = smart_load_word2vec(args.model)
    if not model and args.train:
        print("Training word2vec model on all available data...")
        sentences = StringSentence(documents, analyzer)
        model = Word2Vec(sentences,
                         min_count=1,
                         iter=20)
        model.init_sims(replace=True)  # model becomes read-only

    if args.filter_queries is True:
        old = len(queries)
        queries = [(nb, query) for nb, query in queries if
                   is_embedded(query, model, analyzer)]
        print("Retained {} (of {}) queries".format(len(queries), old))

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
    tfidf = TfidfRetrieval(analyzer=analyzer)

    RMs = {"tfidf": tfidf,
           "wcd": Word2VecRetrieval(model, wmd=False,
                                    analyzer=analyzer,
                                    oov=args.oov,
                                    verbose=args.verbose,
                                    try_lowercase=args.try_lowercase),
           "wmd": Word2VecRetrieval(model, wmd=True,
                                    analyzer=analyzer,
                                    oov=args.oov,
                                    verbose=args.verbose,
                                    try_lowercase=args.try_lowercase),
           "pvdm": Doc2VecRetrieval(analyzer=analyzer,
                                    verbose=args.verbose),
           "eqlm": EQLM(tfidf, model, m=10, eqe=1, analyzer=analyzer,
                        verbose=args.verbose)
           }

    if focus:
        print("Focussing on:", " ".join(focus))
        for f in focus:
            RM = RMs[f]
            results[RM.name] = evaluation(RMs[f])
            del RM, RMs[f]
    else:
        for key, RM in RMs.items():
            results[RM.name] = evaluation(RM)
            del RM, RMs[key]

    if args.plot:
        plot_precision_recall_curves(args.plot, results)

    # reduce to (mean, std) AFTER plotting precision and recall
    results = {name: {metric: mean_std(values) for metric, values in
               scores.items()} for name, scores in results.items()}

    pprint.pprint(args, args.outfile)
    pprint.pprint(results, args.outfile)
    pd.DataFrame(results).to_latex(args.outfile)
    print("Done.")

if __name__ == "__main__":
    main()

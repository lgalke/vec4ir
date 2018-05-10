#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from timeit import default_timer as timer
from datetime import timedelta
from vec4ir.datasets import NTCIR, QuadflorLike
from argparse import ArgumentParser, FileType
from vec4ir.core import Retrieval, all_but_the_top
from vec4ir.base import TfidfRetrieval
from vec4ir.base import Tfidf, Matching
# from vec4ir.base import Matching
from vec4ir.word2vec import Word2VecRetrieval, WordCentroidRetrieval
from vec4ir.word2vec import FastWordCentroidRetrieval, WordMoversRetrieval
from vec4ir.word2vec import WmdSimilarityRetrieval
from vec4ir.doc2vec import Doc2VecRetrieval, Doc2VecInference
from vec4ir.query_expansion import CentroidExpansion
from vec4ir.query_expansion import EmbeddedQueryExpansion
from vec4ir.word2vec import WordCentroidDistance, WordMoversDistance
from vec4ir.postprocessing import uptrain
from vec4ir.eqlm import EQLM
from vec4ir.utils import collection_statistics, build_analyzer
from gensim.models import Word2Vec, Doc2Vec
from sklearn.feature_extraction.text import CountVectorizer
from operator import itemgetter
from textwrap import indent
from nltk.tokenize import word_tokenize
import sys
import pprint
import os
import numpy as np
import pandas as pd
import yaml
import matplotlib
import logging
matplotlib.use('Agg')  # compat on non-gui uis, must be set before pyplot
import matplotlib.pyplot as plt
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

QUERY_EXPANSION = ['ce', 'eqe1', 'eqe2']
RETRIEVAL_MODEL = ['tfidf', 'wcd', 'wmd', 'd2v']
D2V_RETRIEVAL = ['d2v']
MODEL_KEYS = ['tfidf', 'wcd', 'wmd', 'pvdm', 'eqlm', 'legacy-wcd',
              'legacy-wmd', 'cewcd', 'cetfidf', 'wmdnom', 'wcdnoidf',
              'gensim-wmd', 'eqe1tfidf', 'eqe1wcd']


def mean_std(array_like):
    array_like = np.asarray(array_like)
    return array_like.mean(), array_like.std()


def plot_precision_recall_curves(results, path=None, plot_f1=False):
    """
    Plots a precision recall curve to `path`.
    :results:
        dict of dicts containing strategies as first-level keys and
        'precision', 'recall' as second level keys.
    :path:
        Write plot to this file.
    """
    colors = "b g r c m y k".split()

    keys = sorted(results.keys())
    # patches = []
    for name, c in zip(keys, colors):
        values = results[name]
        # patches.append(mpatches.Patch(color=c, label=name))
        precision_recall = zip(values["precision"], values["recall"])
        precision, recall = zip(*list(sorted(precision_recall,
                                             key=itemgetter(0), reverse=True)))
        recall = [0.] + list(recall) + [1.]
        precision = [1.] + list(precision) + [0.]
        plt.plot(recall, precision, color=c, label=name)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc='lower left')
    if path is None:
        plt.show()
    else:
        plt.savefig(path)


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
            verbose=3, replacement=0, n_jobs=1):
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
    values = irmodel.evaluate(queries, rels, verbose=verbose, k=k,
                              replacement=replacement, n_jobs=n_jobs)
    if verbose > 1:
        print("-" * 79)
    if verbose > 0:
        pprint.pprint(pd.DataFrame(values).describe())
        print("=" * 79)

    return values


def smart_load_embedding(model_path, doc2vec=False):
    print("Smart loading", model_path)
    if model_path is None:
        return None
    _, ext = os.path.splitext(model_path)
    if doc2vec:
        print("Loading Doc2Vec model:", model_path)
        model = Doc2Vec.load(model_path)
    elif ext == ".gnsm":  # Native format
        print("Loading embeddings in native gensim format: {}"
              .format(model_path))
        model = Word2Vec.load(model_path)
    else:  # either word2vec text or word2vec binary format
        binary = ".bin" in model_path
        print("Loading embeddings in word2vec format: {}".format(model_path))
        model = Word2Vec.load_word2vec_format(model_path, binary=binary)
    return model


def _ir_eval_parser(config):
    valid_embedding_keys = config["embeddings"].keys()  # TODO document it
    valid_data_keys = config["data"].keys()
    parser = ArgumentParser()
    parser.add_argument("--doctest", action='store_true',
                        help="Perform doctest on this module")
    parser.add_argument("-d", "--dataset", type=str, default="ntcir2",
                        choices=valid_data_keys,
                        help="Specify dataset to use (as in config file)")
    parser.add_argument("-e", "--embedding", type=str, default=None,
                        choices=valid_embedding_keys,
                        help="Specify embedding to use (as in config file)")
    parser.add_argument("-q", "--query-expansion", type=str, default=None,
                        choices=QUERY_EXPANSION,
                        help="Choose query expansion technique.")
    parser.add_argument("-m", type=int, default=10,
                        help='number of terms to expand in QE')
    parser.add_argument("-r", "--retrieval-model", type=str, default=None,
                        choices=RETRIEVAL_MODEL,
                        help="Choose query expansion technique.")
    parser.add_argument("-f", "--focus", nargs='+',
                        choices=MODEL_KEYS, default=None)
    parser.add_argument("-j", "--jobs", type=int, default=8,
                        help="How many jobs to use, default=8 (one per core)")

    emb_opt = parser.add_argument_group("Embedding options")
    emb_opt.add_argument("-n", "--normalize", action='store_true',
                         default=False,
                         help='normalize word vectors before anything else')
    emb_opt.add_argument("-a", "--all-but-the-top", dest='subtract', type=int,
                         default=None,
                         help='Apply all but the top embedding postprocessing')

    ret_opt = parser.add_argument_group("Retrieval Options")
    ret_opt.add_argument("-I", "--no-idf", action='store_false', dest='idf',
                         default=True,
                         help='Do not use IDF when aggregating word vectors')
    ret_opt.add_argument("-w", "--wmd", type=float, dest='wmd', default=1.0,
                         help="WMD Completeness factor, defaults to 1.0")

    # OPTIONS FOR OUTPUT
    output_options = parser.add_argument_group("Output options")
    output_options.add_argument("-o", "--outfile", default=sys.stdout,
                                type=FileType('a'))
    output_options.add_argument("-v", "--verbose", default=2, type=int,
                                help="verbosity level")
    output_options.add_argument('-s',
                                '--stats',
                                default=False,
                                action='store_true',
                                help="Print statistics for analyzed tokens")
    output_options.add_argument("-p",
                                "--plot",
                                default=None,
                                type=str,
                                metavar="PLOTFILE",
                                help="Save P-R curves in PLOTFILE")

    # OPTIONS FOR EVALUATION
    evaluation_options = parser.add_argument_group("Evaluation options")
    evaluation_options.add_argument("-k", dest='k', default=20, type=int,
                                    help="number of documents to retrieve")
    evaluation_options.add_argument("-Q",
                                    "--filter-queries",
                                    action='store_true',
                                    help="Filter queries w/o compl. embedding")
    evaluation_options.add_argument("-R", "--replacement",
                                    dest='repstrat', default="zero",
                                    choices=['drop', 'zero'],
                                    help=("Out of relevancy file document ids"
                                          "default is to use zero relevancy"))

    # OPTIONS FOR THE MATCHING OPERATION
    matching_options = parser.add_argument_group('Matching options')
    matching_options.add_argument("-C",
                                  "--cased",
                                  dest="lowercase",
                                  default=True,
                                  action='store_false',
                                  help="Case sensitive analysis")
    matching_options.add_argument("-T",
                                  "--tokenizer",
                                  default='sklearn',
                                  type=str,
                                  help=("Specify tokenizer for the matching"
                                        "operation" "defaults to 'sword'"
                                        "which removes"
                                        "punctuation but keeps single"
                                        "characterwords."
                                        "'sklearn' is similar but requires"
                                        "at" "least 2 characters for a word"
                                        "'nltk' retains punctuation as"
                                        "seperate" "tokens (use 'nltk' for"
                                        "glove models)"),
                                  choices=['sklearn', 'sword', 'nltk'])
    matching_options.add_argument("-S", "--dont-stop", dest='stop_words',
                                  default=True, action='store_false',
                                  help="Do NOT use stopwords")
    matching_options.add_argument("-M", "--no-matching", dest='matching',
                                  default=True, action='store_false',
                                  help="Do NOT apply matching operation.")

    parser.add_argument("-t", "--train", default=None, type=int,
                        help="Number of epochs to train")
    return parser


def init_dataset(data_config, default='quadflorlike'):
    """
    Given some dataset configuguration ("type" and **kwargs for the
    initializer), return the initialized data set object.  The returned object
    provides the properties `docs`, `rels`, and `topics`.
    """
    kwargs = dict(data_config)  # we assume dict
    # special type value to determine constructor
    T = kwargs.pop('type', default).lower()
    constructor = {"quadflorlike": QuadflorLike, "ntcir": NTCIR}[T]
    dataset = constructor(**kwargs)  # expand dict to kwargs
    return dataset




def build_query_expansion(key, embedding, analyzer='word', m=10, verbose=0,
                          n_jobs=1, use_idf=True):
    if key is None:
        return None
    QEs = {'ce': CentroidExpansion(embedding, analyzer=analyzer, m=m,
                                   use_idf=use_idf),
           'eqe1': EmbeddedQueryExpansion(embedding, analyzer=analyzer, m=m,
                                          verbose=verbose, eqe=1,
                                          n_jobs=n_jobs, a=1, c=0),
           'eqe2': EmbeddedQueryExpansion(embedding, analyzer=analyzer, m=m,
                                          verbose=verbose, eqe=2,
                                          n_jobs=n_jobs, a=1, c=0)}
    return QEs[key]


def build_retrieval_model(key, embedding, analyzer, use_idf=True,
                          wmd_factor=1.0):
    """
    Arguments:
    :key: the key which specifies the selected retrieval model
    :embedding: word vectors (or document vectors for doc2vec)
    :analyzer: analyzer function
    :use_idf: Usage of inverse document frequency
    :wmd_factor: Completeness factor for word movers distance

    """
    RMs = {
        'tfidf': Tfidf(analyzer=analyzer, use_idf=use_idf),
        'wcd': WordCentroidDistance(embedding=embedding,
                                    analyzer=analyzer,
                                    use_idf=use_idf),
        'wmd': WordMoversDistance(embedding, analyzer,
                                  complete=wmd_factor,
                                  use_idf=use_idf),
        'd2v': Doc2VecInference(embedding, analyzer)
    }
    return RMs[key]


def print_dict(d, header=None, stream=sys.stdout, commentstring="% "):
    if header:
        print(indent(header, commentstring), file=stream)
    s = pprint.pformat(d, 2, 80 - len(commentstring))
    print(indent(s, commentstring), file=stream)
    return


def main():
    """Main Evalation Procedure"""
    # parse command line arguments and read config file
    meta_parser = ArgumentParser(add_help=False)
    meta_parser.add_argument('-c',
                             '--config',
                             type=FileType('r'),
                             default='config.yml',
                             help="Specify configuration file")
    meta_args, remaining_args = meta_parser.parse_known_args()
    config = yaml.load(meta_args.config)  # we need meta parser
    parser = _ir_eval_parser(config)
    args = parser.parse_args(remaining_args)
    print(args)
    if args.doctest:
        import doctest
        doctest.testmod()
        exit(int(0))

    print("Preparing analyzer")
    # Set up matching analyzer                                      Defaults
    matching_analyzer = build_analyzer(tokenizer=args.tokenizer,    # sklearn
                                       stop_words=args.stop_words,  # true
                                       lowercase=args.lowercase)    # true
    analyzed = matching_analyzer  # alias
    print("Selecting data set: {}".format(args.dataset))
    dataset = init_dataset(config['data'][args.dataset])
    print("Loading Data...")
    documents, labels, queries, rels = dataset.load(verbose=args.verbose)
    print("Done")

    # Set up embedding specific analyzer
    if args.embedding in config['embeddings']:
        print('Found Embedding key in config file:', args.embedding)
        embedding_config = config['embeddings'][args.embedding]
        model_path = embedding_config['path']
        if "oov_token" in embedding_config:
            embedding_oov_token = embedding_config["oov_token"]
    else:
        print('Using', args.embedding, 'as model path')
        model_path = args.embedding

    doc2vec = args.retrieval_model in D2V_RETRIEVAL

    # Now model path is either:
    # 1. filename from config
    # 2. raw string argument passed to script
    # 3. None, so NO pre-trained embedding will be used
    if args.retrieval_model in ['tfidf']:
        embedding = None
    elif args.train is not None or model_path is None:
        sents = [analyzed(doc) for doc in documents]
        embedding = uptrain(sents, model_path=model_path,
                            binary=('bin' in model_path),
                            lockf=0.0,
                            min_count=1,  # keep all the words
                            workers=max(1, args.jobs),  # no -1
                            iter=args.train,  # number of epochs
                            negative=50,  # number of negative samples
                            sg=1,  # skip gram!
                            sorted_vocab=1,  # why not
                            alpha=0.25,  # initial learning rate
                            sample=0.001,
                            min_alpha=0.005,  # linear decay target
                            size=300  # vector size
                            )
    else:
        embedding = smart_load_embedding(model_path, doc2vec=doc2vec)

    if embedding:
        print("Top 10 frequent words:", embedding.index2word[:10])

    if args.subtract:
        print('Subtracting first %d principal components' % args.subtract)
        syn0 = embedding.wv.syn0
        embedding.wv.syn0 = all_but_the_top(syn0, args.subtract)
    if args.normalize:
        print('Normalizing word vectors')
        embedding.init_sims(replace=True)

    # embedding_config = config["embeddings"][args.embedding]
    # embedding_analyzer_config = embedding_config["analyzer"]
    # embedding_analyzer = build_analyzer(**embedding_analyzer_config)
    if args.stats:
        # out source
        print("Computing collection statistics...")
        stats, mcf = collection_statistics(embedding=embedding,
                                           documents=documents,
                                           analyzer=matching_analyzer,
                                           topn=50)
        header = ("Statistics: {a.dataset} x {a.embedding}"
                  " x {a.tokenizer} x lower: {a.lowercase}"
                  " x stop_words: {a.stop_words}")
        header = header.format(a=args)
        print_dict(stats, header=header, stream=args.outfile)
        print("Most common fails:", *mcf, sep='\n')
        exit(0)

    repl = {"drop": None, "zero": 0}[args.repstrat]

    if args.filter_queries is True:
        old = len(queries)
        queries = [(nb, query) for nb, query in queries if
                   is_embedded(query, embedding, matching_analyzer)]
        print("Retained {} (of {}) queries".format(len(queries), old))

    def evaluation(m):
        return ir_eval(m,
                       documents,
                       labels,
                       queries,
                       rels,
                       verbose=args.verbose,
                       k=args.k,
                       replacement=repl,
                       n_jobs=args.jobs)

    results = dict()
    if args.retrieval_model is not None:
        query_expansion = build_query_expansion(args.query_expansion,
                                                embedding, analyzed, m=args.m,
                                                verbose=args.verbose,
                                                n_jobs=args.jobs,
                                                use_idf=args.idf)
        retrieval_model = build_retrieval_model(args.retrieval_model,
                                                embedding, analyzed,
                                                use_idf=args.idf,
                                                wmd_factor=args.wmd)
        match_op = Matching(analyzer=matching_analyzer)
        rname = '+'.join(
            (args.embedding,
             args.query_expansion if args.query_expansion else '',
             args.retrieval_model)
        )
        if args.normalize:
            rname = 'norm-' + rname
        ir = Retrieval(retrieval_model, query_expansion=query_expansion,
                       name=rname, matching=match_op)
        results[rname] = evaluation(ir)
    else:

        tfidf = TfidfRetrieval(analyzer=matching_analyzer)
        matching = {"analyzer": matching_analyzer} if args.matching else None

        WCD = FastWordCentroidRetrieval(name="wcd", embedding=embedding,
                                        analyzer=matching_analyzer,
                                        matching=matching,
                                        n_jobs=args.jobs)
        WCDnoidf = FastWordCentroidRetrieval(name="wcd-noidf",
                                             embedding=embedding,
                                             analyzer=matching_analyzer,
                                             matching=matching, use_idf=False,
                                             n_jobs=args.jobs)

        eqe1 = EmbeddedQueryExpansion(embedding,
                                      analyzer=matching_analyzer,
                                      m=10,
                                      eqe=1,
                                      n_jobs=args.jobs)
        eqe1_tfidf = Retrieval(query_expansion=eqe1, retrieval_model=tfidf)
        eqe1_wcd = Retrieval(query_expansion=eqe1, retrieval_model=WCD)

        # matching_estimator = Matching(**matching)
        CE = CentroidExpansion(embedding, matching_analyzer, m=10,
                               verbose=args.verbose)
        CE_WCD = Retrieval(retrieval_model=WCD, matching=None,
                           query_expansion=CE, name='CE+wcd')

        CE_TFIDF = Retrieval(retrieval_model=tfidf, matching=None,
                             query_expansion=CE, name='CE+tfidf')

        RMs = {"tfidf": tfidf,
               "nsim": Word2VecRetrieval(embedding, wmd=False,
                                         analyzer=matching_analyzer,
                                         vocab_analyzer=matching_analyzer,
                                         oov=embedding_oov_token,
                                         verbose=args.verbose),
               "legacy-wcd": WordCentroidRetrieval(embedding,
                                                   name="legacy-mwcd",
                                                   matching=matching,
                                                   analyzer=matching_analyzer,
                                                   oov=embedding_oov_token,
                                                   verbose=args.verbose,
                                                   normalize=False,
                                                   algorithm='brute',
                                                   metric='cosine',
                                                   n_jobs=args.jobs),
               "wcd": WCD,
               "wcdnoidf": WCDnoidf,
               "legacy-wmd": Word2VecRetrieval(embedding, wmd=True,
                                               analyzer=matching_analyzer,
                                               vocab_analyzer=analyzed,
                                               oov=embedding_oov_token,
                                               verbose=args.verbose),
               "wmd": WordMoversRetrieval(embedding=embedding,
                                          analyzer=matching_analyzer,
                                          matching_params=matching,
                                          oov=None,
                                          verbose=args.verbose,
                                          n_jobs=args.jobs),
               "pvdm": Doc2VecRetrieval(analyzer=matching_analyzer,
                                        matching=matching,
                                        n_jobs=args.jobs,
                                        metric="cosine",
                                        algorithm="brute",
                                        alpha=0.25,
                                        min_alpha=0.05,
                                        n_epochs=20,
                                        verbose=args.verbose),
               "cewcd": CE_WCD,
               "cetfidf": CE_TFIDF,
               'wmdnom': WordMoversRetrieval(embedding=embedding,
                                             analyzer=matching_analyzer,
                                             matching_params=None,
                                             oov=embedding_oov_token,
                                             verbose=args.verbose,
                                             n_jobs=args.jobs),
               "eqlm": EQLM(tfidf, embedding, m=10, eqe=1,
                            analyzer=matching_analyzer, verbose=args.verbose),
               "gensim-wmd": WmdSimilarityRetrieval(embedding,
                                                    matching_analyzer, args.k),
               'eqe1tfidf': eqe1_tfidf,
               'eqe1wcd': eqe1_wcd
               }

        focus = [f.lower() for f in args.focus] if args.focus else None
        if focus:
            print("Focussing on:", " ".join(focus))
            for f in focus:
                RM = RMs[f]
                results[RM.name] = evaluation(RMs[f])
                del RM, RMs[f]  # clean up
        else:
            for key, RM in RMs.items():
                results[RM.name] = evaluation(RM)
                del RM, RMs[key]  # clean up

    if args.plot:
        plot_precision_recall_curves(results, path=args.plot)
    results = {name: {metric: mean_std(values) for metric, values in
               scores.items()} for name, scores in results.items()}
    print_dict(config, header="CONFIG", stream=args.outfile)
    print_dict(args, header="ARGS", stream=args.outfile)

    pd.DataFrame(results).to_latex(args.outfile)

    print("Done.")


if __name__ == "__main__":
    main()
